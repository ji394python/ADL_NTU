from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn import functional as F


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_class = num_class

        self.encode = nn.LSTM(self.embed_dim, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidirectional, batch_first=True)
        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder_output_size, self.num_class)
        )
        
    @property
    def encoder_output_size(self) -> int:
        # TODO calculate the output dimension of rnn
        return  2 * self.hidden_size if self.bidirectional else self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        out = {}
        packed_x = nn.utils.rnn.pack_padded_sequence(self.embed(batch['text']), batch['len'], batch_first=True)
        self.encode.flatten_parameters()
        x, (h, _) = self.encode(packed_x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True) 
        
        ########### Test for concat method ##############
        h = torch.cat((h[-1], h[-2]), axis=-1) if self.bidirectional else h[-1]
        # h = torch.cat((h[-1], h.max(0)[0]), axis=-1) if self.bidirectional else h[-1]
        # h = torch.cat((h[-1], h.mean(0)), axis=-1) if self.bidirectional else h[-1]
        # h = torch.cat((h[-1], h.mean(0),h.max(0)[0]), axis=-1) if self.bidirectional else h[-1]
        # h = torch.cat((h[-1],h[-2] ,h.mean(0), h.max(0)[0]), axis=-1) if self.bidirectional else h[-1]

        pred = [self.linear(h)]

        out['pred_logits'] = pred
        out['pred_labels'] = pred[-1].max(1, keepdim=True)[1].reshape(-1)

        out['loss'] = F.cross_entropy(pred[-1], batch['intent'].long())


        return out


class SeqTagging(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        layers_cnn_nums: int,
        hidden_size: int,
        layers_bilstm_nums: int,
        dropout: float,
        bidirectional: bool,
        num_class: int
    ) -> None:
        super(SeqTagging, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # model architecture
        self.embed_dim = embeddings.size(1)
        self.layers_cnn_nums = layers_cnn_nums
        self.hidden_size = hidden_size
        self.layers_bilstm_nums = layers_bilstm_nums
        self.bidirectional = bidirectional
        self.num_class = num_class
        
        cnn = []
        for i in range(layers_cnn_nums):
            conv_layer = nn.Sequential(
                nn.Conv1d(self.embed_dim, self.embed_dim, 5, 1, 2),
                nn.ReLU(),
                nn.Dropout()
            )
            cnn.append(conv_layer)
        self.conv1d = nn.ModuleList(cnn)
        
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, self.layers_bilstm_nums, dropout=dropout, bidirectional=self.bidirectional, batch_first=True)

        
        self.outputTransform = outputTrans(self.encoder_output_size, self.num_class, dropout)

    @property
    def encoder_output_size(self) -> int:
        # calculate the output dimension of rnn
        return 2 * self.hidden_size if self.bidirectional else self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        out = {}
        x = self.embed(batch['tokens']) # [batch_size, max_len, embed_dim]

        # Conv1d
        x = x.permute(0, 2, 1) # [batch_size, embed_dim, max_len]
        for conv in self.conv1d:
            x = conv(x) 
        x = x.permute(0, 2, 1) # [batch_size, max_len, embed_dim]

        # LSTM 
        if self.layers_bilstm_nums > 0:
            packed_x = nn.utils.rnn.pack_padded_sequence(x, batch['len'], batch_first=True)
            self.lstm.flatten_parameters()
            x, _ = self.lstm(packed_x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True) # [batch_size, max_len, hid_dim])

        batch['mk'] = batch['mk'][:, :x.size(1)]
        batch['tags'] = batch['tags'][:, :x.size(1)]
        
        out['loss'] = self.outputTransform.loss(x, batch['tags'], batch['mk'])
        out['max_score'], out['pred_labels'] = self.outputTransform(x, batch['mk'])

        return out

class outputTrans(nn.Module):
    def __init__(self, in_dim, num_class, dropout):
        super(outputTrans, self).__init__()
        self.num_class = num_class + 2
        self.start_idx = self.num_class - 2
        self.stop_idx = self.num_class - 1

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, self.num_class)
        )

        self.transitions = nn.Parameter(torch.randn(self.num_class, self.num_class), requires_grad=True)

        self.transitions.data[self.start_idx, :] = -1e5
        self.transitions.data[:, self.stop_idx] = -1e5

    def forward(self, x, mk):
        x = self.fc(x)
        return self.transDecode(x, mk)
    
    def loss(self, x, tags, mk):
        x = self.fc(x)
        dim1, dim2, _ = x.shape
        
        # first part 
        forwar_s = torch.full_like(x[:, 0, :], -1e5)
        forwar_s[:, self.start_idx] = 0
        for t in range(dim2):
            newer = forwar_s.unsqueeze(1) + self.transitions.unsqueeze(0) + x[:, t].unsqueeze(-1)
            score_t = newer.max(-1)[0] + torch.log(torch.sum(torch.exp(newer - newer.max(-1)[0].unsqueeze(-1)), -1))
            mk_t = mk[:, t].unsqueeze(-1) # [B, 1]
            forwar_s = score_t * mk_t + forwar_s * (1 - mk_t)
        outda = forwar_s + self.transitions[self.stop_idx]
        outdb = outda.max(-1)[0]
        f_scr =  outdb + torch.log(torch.sum(torch.exp(outda - outdb.unsqueeze(-1)), -1))

        # second part
        idx2 = tags.unsqueeze(-1)
        tags = torch.cat([torch.full((dim1, 1), self.start_idx, dtype=torch.long).type_as(tags), tags], 1) 
        src = self.transitions[self.stop_idx, tags.gather(dim=1, index=mk.sum(-1).long() .unsqueeze(-1)).squeeze(-1)] 
        g_scr = ((x.gather(dim=2, index=idx2).squeeze(-1)  + self.transitions[tags[:, 1:], tags[:, :-1]] ) * mk).sum(-1) + src

        loss = (f_scr - g_scr).mean()

        return loss


    @torch.no_grad()
    def transDecode(self, x, mk):
        B, L, C = x.shape
        tolerance = torch.zeros_like(x).long() 

        max_score = torch.full_like(x[:, 0, :], -1e5)
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mk_t = mk[:, t].unsqueeze(-1) 
            emit_score_t = x[:, t] 

            score_t = max_score.unsqueeze(1) + self.transitions
            score_t, tolerance[:, t, :] = score_t.max(-1)
            score_t += emit_score_t

            max_score = score_t * mk_t + max_score * (1 - mk_t)

        max_score += self.transitions[self.stop_idx]
        path_score, best_tags = max_score.max(-1)
        
        best_paths = []
        for b in range(B):
            best_tag_id = best_tags[b].item()
            best_path = [best_tag_id]

            for bptrs_t in tolerance[b, :mk.sum(-1).long()[b]].flip(0):
                best_tag_id = bptrs_t[best_tag_id]
                best_path += [best_tag_id]

            best_path.pop()
            best_path.reverse()
            best_paths.append(best_path + [0] * (L - len(best_path)))
        
        best_paths = torch.tensor(best_paths)

        return max_score, best_paths
