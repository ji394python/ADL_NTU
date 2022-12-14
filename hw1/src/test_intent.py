import json
import pickle
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO crecate DataLoader for test dataset
    loader_test = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes
    ).to(args.device)
    model.eval()
    
    ckpt = torch.load(args.ckpt_path,map_location=torch.device('cpu'))
    # TODO load weights into model
    model.load_state_dict(ckpt)

    # TODO predict dataset
    ids = []
    preds = []
    for bs in loader_test:
        bs['text'] = bs['text'].to(args.device)
        bs['intent'] = bs['intent'].to(args.device)
        out = model(bs)
        ids = ids + bs['id']
        preds = preds + out['pred_labels'].tolist()

    # TODO write prediction to file (args.pred_file)
    if args.pred_file.parent:
        args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.pred_file, 'w') as f:
        f.write('id,intent\n')
        for i, la in zip(ids, preds):
            f.write("%s,%s\n" %(i, dataset.idx2label(la)))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred_intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--att", action="store_true")
    parser.add_argument("--att_unit", type=int, default=128)
    parser.add_argument("--att_hops", type=int, default=16)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument('--seed', default=101, type=int, help="seed for model training")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)