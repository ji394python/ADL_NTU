import os
import json
import pickle
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np

from dataset import SeqTagDataset
from utils import Vocab
from model import SeqTagging

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
EPS = 1e-9

def _train(args, model, train_loader, optimizer):
    model.train()

    epoch_size = 0
    epoch_train_loss = 0
    epoch_train_token_cur = 0
    epoch_train_token_tot = 0
    epoch_train_joint_cur = 0
    epoch_train_joint_tot = 0

    
    bar = tqdm(train_loader)
    for i, batch in enumerate(bar):
        batch['tokens'] = batch['tokens'].to(args.device)
        batch['tags'] = batch['tags'].to(args.device)
        batch['mk'] = batch['mk'].to(args.device)
        bs_size = batch['tokens'].size(0)

        output_dict = model(batch)
        #Total Loss 
        epoch_size += bs_size
        epoch_train_loss += output_dict['loss']*bs_size
        
        # accurancy
        ground = batch['tags'].cpu()
        mediatek = batch['mk'].cpu()
        mediatek = mediatek[:, :ground.size(1)]
        batch_cor = (ground.eq(output_dict['pred_labels'].cpu().view_as(ground)) * mediatek).sum(-1)
        seq_len = mediatek.sum(-1)
        
        epoch_train_token_cur += batch_cor.sum().long().item()
        epoch_train_joint_cur += batch_cor.eq(seq_len).sum().item()
        epoch_train_token_tot += mediatek.sum().long().item()
        epoch_train_joint_tot += len(ground)
        bar.set_postfix(loss=output_dict['loss'].item(), iter=i)

        optimizer.zero_grad()
        output_dict['loss'].backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

    Train_loss = epoch_train_loss / epoch_size + EPS
    Train_token_cur = epoch_train_token_cur / (epoch_train_token_tot + EPS)
    Train_joint_cur = epoch_train_joint_cur / (epoch_train_joint_tot + EPS)
    print(f'Train Loss: {Train_loss:3.4f} Joint Acc: {Train_joint_cur:3.4f} ({epoch_train_joint_cur}/{epoch_train_joint_tot}) Token Acc: {Train_token_cur:6.4f} ({epoch_train_token_cur}/{epoch_train_joint_cur})')

    return None

@torch.no_grad() 
def _validation(args, model, val_loader):
    model.eval()

    epoch_size = 0
    epoch_val_loss = 0
    epoch_val_token_cur = 0
    epoch_val_token_tot = 0
    epoch_val_joint_cur = 0
    epoch_val_joint_tot = 0

    for batch in val_loader:
        batch['tokens'] = batch['tokens'].to(args.device)
        batch['tags'] = batch['tags'].to(args.device)
        batch['mk'] = batch['mk'].to(args.device)
        bs_size = batch['tokens'].size(0)
        output_dict = model(batch)
        
        #Total Loss 
        epoch_size += bs_size
        epoch_val_loss += output_dict['loss']*bs_size
        
        # accurancy
        ground = batch['tags'].cpu()
        mediatek = batch['mk'].cpu()
        mediatek = mediatek[:, :ground.size(1)]
        batch_cor = (ground.eq(output_dict['pred_labels'].cpu().view_as(ground)) * mediatek).sum(-1)
        seq_len = mediatek.sum(-1)
        
        epoch_val_token_cur += batch_cor.sum().long().item()
        epoch_val_joint_cur += batch_cor.eq(seq_len).sum().item()
        epoch_val_token_tot += mediatek.sum().long().item()
        epoch_val_joint_tot += len(ground)
        
    val_loss = epoch_val_loss / epoch_size + EPS
    val_token_cur = epoch_val_token_cur / (epoch_val_token_tot + EPS)
    val_joint_cur = epoch_val_joint_cur / (epoch_val_joint_tot + EPS)
    print(f'Val Loss: {val_loss:3.4f} Joint Acc: {val_joint_cur:3.4f} ({epoch_val_joint_cur}/{epoch_val_joint_tot}) Token Acc: {val_token_cur:6.4f} ({epoch_val_token_cur}/{epoch_val_joint_cur})')

    
    return val_joint_cur

def save_best(model, ckp_dir, epoch):
    ckp_path = ckp_dir / '{}-model.pth'.format(epoch + 1)
    best_ckp_path = ckp_dir / 'best-model.pth'.format(epoch + 1)
    torch.save(model.state_dict(), ckp_path)
    torch.save(model.state_dict(), best_ckp_path)
    print('Saved model checkpoints into {}...'.format(ckp_path))

def main(args):
    torch.manual_seed(args.seed)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    ckpt_dir = args.ckpt_dir / f'{args.name}_{args.seed}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTagDataset] = {
        split: SeqTagDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # create DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) for split, split_dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    

    # init model and move model to target device(cpu / gpu)
    model = SeqTagging(embeddings, args.layers_cnn_nums, args.hidden_size, args.layers_bilstm_nums, args.dropout, args.bidirectional, datasets[TRAIN].num_classes).to(args.device)
    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    max_acc = 0.0
    for epoch in range(args.num_epoch):

        print("EPOCH: %d" % (epoch))

        # Training loop - iterate over train dataloader and update model weights
        _train(args, model, dataloaders[TRAIN], optimizer)

        # Evaluation loop - calculate accuracy and save model weights.
        val_joi_acc = _validation(args, model, dataloaders[DEV])

        if val_joi_acc > max_acc:
            max_acc = val_joi_acc
            save_best(model, ckpt_dir, epoch)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=48)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--layers_cnn_nums", type=int, default=1)
    parser.add_argument("--layers_bilstm_nums", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-4)

    # loss
    parser.add_argument('--grad_clip', default = 5., type=float, help='max gradient')

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    # Track 
    parser.add_argument('--seed', default=101, type=int, help="Model seed")
    parser.add_argument('--name', default='', type=str, help='Model Name')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
