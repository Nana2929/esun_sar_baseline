#%%
from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DebertaV2Config, DebertaV2Model




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
        # adding special token, out_size = (bs, max_len+1, hidden)
        ## get batch size
        bs = len(x)

        # constructing the cls token forms considering the batch
        cls_token = self.cls_token.view(1, -1).repeat(bs, 1, 1)
        ## reconstruct the inputs
        # x = torch.cat([cls, x], dim=1)

        device = x.device
        cls_position = torch.zeros(bs).long().to(device)
        # print('!', mask.shape, last_attn.shape, cls_position.shape)
        for i in range(bs):
            # print(last_attn.shape)
            # print(last_attn[i])
            # print(self.max_len)
            first_pad = torch.min(last_attn[i] + 1, torch.tensor(self.max_len-1).to(device))
            # may masking out the last data but
            # I quit
            print('first pad:', first_pad)
            # put cls token in the first pad position
            x[i, first_pad] = self.cls_token
            mask[i, first_pad] = 1
            cls_position[i] = first_pad
        x = self.encoder(x, mask, output_hidden_states=False).last_hidden_state
        # x = x[:, 0]
        cls_position = cls_position.squeeze()
        # https://stackoverflow.com/questions/71950956/select-tensor-slice-along-a-dimension-based-on-index
        # print("cls_position", cls_position.shape)
        # print("output shape:", x.shape)
        # print("batch_size", bs)
        # print(cls_position)
        # print(x)

        x = x[torch.arange(bs), cls_position] # getting cls token embeddings
        return x
#%%

# import torch
# bi = torch.tensor([ 0,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  6,  6,  6,
#          6,  6,  6,  7,  7,  8,  8,  9,  9,  9, 10, 10, 10, 10, 12, 12, 13, 13,
#         13, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 17, 19, 20, 22, 22,
#         22, 22, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 26, 26, 26, 26, 27,
#         27, 27, 27, 27, 27, 27, 29, 30, 31, 31, 31, 33, 33, 33, 33, 33, 35, 35,
#         35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37,
#         37, 37, 37, 37, 37, 38, 38, 39, 39, 42, 42, 42, 42, 44, 44, 45, 45, 45,
#         48, 48, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
#         51, 51, 51, 51, 51, 51, 51, 53, 53, 53, 53, 53, 53, 56, 56, 56, 56, 56,
#         56, 56, 56, 56, 57, 57, 58, 58, 58, 58, 58, 61, 61, 61, 61, 63, 63, 63,
#         63])

# x = torch.tensor([[[0,1,2],
#                   [3,4,5]],
#                   [[6,7,8],
#                   [9,10,11]]])

# a = torch.tensor([0,1])
# a = a.squeeze()
# res = x[torch.arange(2), a]

# si = torch.tensor([318,   0,   0,   1,   2,   3,   4,   5,   6,   7,  13,  14,   0,  53,
#         104,   0,  62, 126, 155, 192, 256,   0, 232,  89, 276,   0, 100, 183,
#          65, 190, 320, 411, 292, 479,  32, 211, 387,   0,  30,  75, 123,   0,
#           2,  12,  16,  18,  20,  22,  25,   0, 236,   0,   0,   3,   4,   6,
#           0,   1,   2,   3,   4,   7,   8,   9,  10,  11,  25,   0,   1,  21,
#          36,   0,  87, 136, 205, 262, 350, 435,   0,   0,   0,  30,  77,  17,
#         125, 212, 293, 410,   0,   1,   2,   3,   4,   5,   6,   7,   8,   0,
#          43,  66,  93, 127, 167,   0,  24,  36,  48,  65,  77,  92, 109, 130,
#         160,   0, 121,  20, 442,  54, 148, 215, 365,   0,  51, 136, 314, 453,
#           0, 156,   0,  11,  30,  53,  72,  94,   0,  20,  35,  44,  49,  60,
#          76,  85, 141, 242,  32,  94, 215, 287, 367, 435, 501,  37, 119, 214,
#         316, 398, 483,  83, 123, 232, 242, 244, 248, 261, 270, 420, 151, 434,
#          40, 169, 206, 334, 475,   0, 113, 233, 364,  14, 113, 219, 358],
#        ) #%%



# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # hidden_size = 1
# # batch_size = 4
# # max_len = 20
# # last_attn = {0:5, 1:13, 2:20, 3:12}
# # x = torch.randn(batch_size, max_len, hidden_size)
# # cls = nn.Parameter(torch.full((hidden_size,), fill_value=200.0), requires_grad=True)
# # for i in range(batch_size):
# #     first_pad = min(last_attn[i] + 1, max_len-1)
# #     x[i, first_pad] = cls
# #     mask[i, first_pad] = 1
# # print("after...")
# # print(x)

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

