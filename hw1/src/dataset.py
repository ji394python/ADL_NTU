from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len
import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        
        samples.sort(key=lambda x: len(x['text'].split()), reverse=True) # 比較多字的樣本排在前面?
        bs_text = []
        bs_len = []
        bs_id = []
        for i in samples:
            text= i['text'].split()
            bs_text.append(text)
            bs_len.append(min(len(text), self.max_len))
            bs_id.append(i['id'])
        bs_text = torch.tensor(self.vocab.encode_batch(bs_text, self.max_len))
        check_data = True if 'intent' in samples[0].keys() else False
        bs_intent = torch.tensor([self.label2idx(i['intent']) for i in samples]) if check_data else torch.zeros(len(bs_id), dtype=torch.long)


        return {'text':bs_text, 'len':bs_len, 'id':bs_id,'intent':bs_intent}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.count = torch.zeros(len(self.label_mapping))
        if 'tags' in self.data[0]:
            for d in self.data:
                for t in d['tags']:
                    self.count[self.label2idx(t)] += 1
        else:
            self.count += 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # implement collate_fn
        samples.sort(key=lambda x: len(x['tokens']), reverse=True)

        bs_tokens,bs_lens,bs_ids = [],[],[]
        for i in samples:
            tokens = i['tokens']
            bs_tokens.append(tokens)
            bs_lens.append(min(len(tokens), self.max_len))
            bs_ids.append(i['id'])
        bs_lens = torch.tensor(bs_lens)
        bs_tokens = torch.tensor(self.vocab.encode_batch(bs_tokens, self.max_len))
        bs_tags = torch.tensor(pad_to_len([[self.label2idx(t) for t in s['tags']] for s in samples], self.max_len, 0)) if 'tags' in samples[0].keys() else torch.tensor([[0] * self.max_len] * len(samples))
        bs_mk = bs_tokens.gt(0).float()
       
        batch = {}
        batch['tokens'] = [s['tokens'] for s in samples]
        batch['len'] = torch.tensor([min(len(s), self.max_len) for s in [s['tokens'] for s in samples]])
        batch['tokens'] = self.vocab.encode_batch(batch['tokens'], self.max_len)
        batch['tokens'] = torch.tensor(batch['tokens'])
        
        return {'tokens':bs_tokens,'len':bs_lens,'id':bs_ids,'tags':bs_tags,'mk':bs_mk}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


