import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
import random
import torch

from dataset import SeqClsDataset
from torch.utils.data import DataLoader
from model import SeqClassifier
from utils import Vocab
from tqdm import trange, tqdm

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
ERP = 1e-10

def _train(args, model, train_loader, optimizer):
    model.train()
    epoch_size = 0
    epoch_train_loss = 0
    epoch_train_correct = 0

    bar = tqdm(train_loader)
    for i, batch in enumerate(bar):
        
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)
        bs_size = batch['intent'].size(0)

        optimizer.zero_grad()
        output_dict = model(batch)

        bar.set_postfix(loss=output_dict['loss'].item(), iter=i)

        #Total Loss 
        epoch_size += bs_size
        epoch_train_loss += output_dict['loss']*bs_size

        # accurancy
        ground = batch['intent'].detach().cpu()
        predict = output_dict['pred_labels'].detach().cpu()
        epoch_train_correct += predict.eq(ground.view_as(predict)).sum().item()

        loss = output_dict['loss']

        loss.backward()
        optimizer.step()

    Train_loss = epoch_train_loss / epoch_size + ERP
    Train_acc = epoch_train_correct / epoch_size + ERP
    print(f'Train Loss: {Train_loss:3.4f}\t Acc: {Train_acc:3.4f}')
    return None


@torch.no_grad() 
def _validation(args, model, val_loader):
    model.eval()
    epoch_size = 0
    epoch_Val_loss = 0
    epoch_Val_correct = 0

    for batch in val_loader:
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)
        bs_size = batch['intent'].size(0)

        output_dict = model(batch)

         #Total Loss 
        epoch_size += bs_size
        epoch_Val_loss += output_dict['loss']*bs_size

        # accurancy
        ground = batch['intent'].detach().cpu()
        predict = output_dict['pred_labels'].detach().cpu()
        epoch_Val_correct += predict.eq(ground.view_as(predict)).sum().item()

    Val_loss = epoch_Val_loss / epoch_size + 1e-10
    Val_acc = epoch_Val_correct / epoch_size + 1e-10
    print(f'Val Loss: {Val_loss:3.4f}\t Acc: {Val_acc:3.4f}')
    return Val_acc


def _save_best(model, ckp_dir, epoch):
    ckp_path = ckp_dir / '{}-model.pth'.format(epoch + 1)
    best_ckp_path = ckp_dir / 'best-model.pth'.format(epoch + 1)
    torch.save(model.state_dict(), ckp_path)
    torch.save(model.state_dict(), best_ckp_path)
    print('Save model checkpoints into {}...'.format(ckp_path))


def main(args):
    torch.manual_seed(args.seed)
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    ckpt_dir = args.ckpt_dir / f"{args.name}_{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    data_loaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) for split, split_dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    max_accurancy = 0.0

    epoch_pbar = args.num_epoch
    for epoch in range(epoch_pbar):
        # TODO: Training loop - iterate over train dataloader and update model weights
        print(f"EPOCH: {epoch+1}")
        _train(args, model, data_loaders[TRAIN], optimizer)
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        val_acc = _validation(args, model, data_loaders[DEV])

        if val_acc > max_accurancy:
            max_accurancy = val_acc
            _save_best(model, ckpt_dir, epoch)
        


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=30)
    
    # only for track
    parser.add_argument('--seed', default=101, type=int, help="Model seed")
    parser.add_argument('--name', default='r10h41007', type=str, help='model name')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
