from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DebertaV2Config, DebertaV2Model



class TemporalGruAggregator(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout=dropout,
            batch_first = True
        )
    def forward(self, x, mask):
        x = x * (mask[:, :, None])
        x, _ = self.gru(x)

        output_idx = mask.sum(axis=1) - 1
        x = x[range(len(mask)), output_idx]
        return x


def build_relative_position(query_size, key_size):
    """
    Build relative position according to the query and key
    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)
    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position
    Return:
        `torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """
    q_ids = torch.arange(0, query_size)
    k_ids = torch.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


class TemporalDebertaAggregator(nn.Module):
    def __init__(self, hidden_size=128,
    num_layers=2,
    num_head=4,
    cls_pos=-1,
    dropout=0.3,
     max_len=512):
        super().__init__()
        self.cls_pos = cls_pos
        self.cls = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        config = DebertaV2Config(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_size*4,
            num_attention_heads=num_head,
            hidden_dropout_prob=dropout,
            relative_attention=True,
            max_relative_positions=max_len+1,
        )
        self.encoder = DebertaV2Model(config).encoder

    def forward(self, x, mask, last_attn):
        bs = len(x)
        # add special token, out_size = (bs, nrows+1, hidden)
        cls = self.cls.view(1, -1).repeat(bs, 1, 1)
        x = torch.cat([cls, x], dim=1)
        mask = torch.cat([torch.ones(bs).bool().to(x.get_device())[:, None] , mask], dim=1)
        x = self.encoder(x, mask, output_hidden_states=False).last_hidden_state
        x = x[:, 0]
        return x

class TemporalDebertaAggregatorV2(nn.Module):
    def __init__(self, hidden_size=128,
    num_layers=2,
    num_head=4,
    cls_pos=-1,
    dropout=0.3,
     max_len=512):
        super().__init__()
        self.cls_pos = cls_pos
        self.max_len = max_len
        self.cls_token = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        config = DebertaV2Config(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_size*4,
            num_attention_heads=num_head,
            hidden_dropout_prob=dropout,
            relative_attention=True,
            max_relative_positions=max_len,
        )
        self.encoder = DebertaV2Model(config).encoder

    def forward(self, x, mask, last_attn):
        """cls token position: at sequence last (first padding position)

        Args:
            x (torch.tensor): shape (bs, max_len, emd_dim)
            mask (torch.tensor): shape (bs, max_len)
            last_attn (torch.tensor): shape (bs, )

        Returns:
            _type_: shape (bs, emb_dim) # [CLS] embeddings of a batch 
        """
        bs = len(x)
        device = x.device
        cls_position = torch.zeros(bs).long().to(device)
        for i in range(bs):
            first_pad = torch.min(last_attn[i] + 1, torch.tensor(self.max_len-1).to(device))
            x[i, first_pad] = self.cls_token
            mask[i, first_pad] = 1
            cls_position[i] = first_pad
        x = self.encoder(x, mask, output_hidden_states=False).last_hidden_state
        cls_position = cls_position.squeeze()
        # https://stackoverflow.com/questions/71950956/select-tensor-slice-along-a-dimension-based-on-index
        x = x[torch.arange(bs), cls_position] # getting cls token embeddings
        return x
#%%
if __name__ == "__main__":
    import torch
    from temporal_aggregator import TemporalDebertaAggregator as td;
    model = td(hidden_size=12, max_len=30)
    rdt = torch.randn(4, 30, 12)
    mask = torch.ones(4, 30).bool()
    out = model(rdt, mask)
    print(rdt)

    print(out.shape) # (4,12)

