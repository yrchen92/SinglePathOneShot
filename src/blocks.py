import math
import torch
import torch.nn as nn

from pytorch_transformers.modeling_bert import BertLayer, BertLayerNorm

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Shufflenet(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.base_mid_channel = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize,
                      stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs, affine=False),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(
                    inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


class Shuffle_Xception(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, stride):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]

        self.base_mid_channel = mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        outputs = oup - inp

        branch_main = [
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp, affine=False),
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3,
                      1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3,
                      1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, affine=False),
            # pw
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs, affine=False),
            nn.ReLU(inplace=True),
        ]

        self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, inputs, mask=None):
        inputs = gelu(inputs)
        inputs = self.linear(inputs)
        return inputs

class SimConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, dilation=1):
        super(SimConv, self).__init__()
        # self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernal_size, 1, bias=True,
                          padding= dilation * (kernal_size - 1) // 2, dilation=dilation)
        # self.bn = nn.BatchNorm1d(out_channels, affine=False, track_running_stats=False)
        # self.ln = torch.nn.LayerNorm(out_channels)
        
    def forward(self, inputs, mask=None):
        inputs = gelu(inputs)
        inputs = torch.transpose(inputs, 1, 2)  # to (N, C, L)
        # inputs = self.relu(inputs)
        inputs = self.conv(inputs)
        inputs = torch.transpose(inputs, 1, 2)  # to (N, L, C)
        # inputs = self.ln(inputs)
        return inputs

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=1):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        # self.relu = nn.ReLU()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = gelu(hidden_states)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # print(attention_scores.size())
        # print(extended_attention_mask.size())
        attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = context_layer

        return outputs

class SkipOp(nn.Module):
    def __init__(self):
        super(SkipOp, self).__init__()
        # self.relu = nn.ReLU()
        
    def forward(self, inputs, mask=None):
        # inputs = self.relu(inputs)
        return inputs

class RnnOp(nn.Module):
    def __init__(self, hidden_size):
        super(RnnOp, self).__init__()
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, inputs, mask=None):
        # Sort by length (keep idx)
        length = torch.sum(mask, -1)
        length, idx_sort = torch.sort(length, descending=True)
        _, idx_unsort = torch.sort(idx_sort)
        inputs = inputs.index_select(0, idx_sort)
        mask = mask.index_select(0, idx_sort)

         # self.gru.flatten_parameters()
        hidden_states = nn.utils.rnn.pack_padded_sequence(inputs, length.cpu(), batch_first=True)
        hidden_states = self.rnn(hidden_states)[0]
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(hidden_states,
                                            batch_first=True, total_length=mask.size(1))
        output_dim = int(hidden_states.size(-1) / 2)
        hidden_states_f = hidden_states[:, :, :output_dim]
        hidden_states_b = hidden_states[:, :, output_dim:]
        hidden_states = (hidden_states_f + hidden_states_b) / 2.0
        hidden_states = self.dropout(hidden_states)

        # Un-sort by length
        outputs = hidden_states.index_select(0, idx_unsort)
        return outputs

class ZeroOp(nn.Module):
    def __init__(self):
        super(ZeroOp, self).__init__()

    def forward(self, inputs, mask=None):
        return torch.zeros_like(inputs)

class MaxPooling(nn.Module):
    def __init__(self, kernal_size=3, stride=1):
        super(MaxPooling, self).__init__()
        # self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool1d(kernal_size, stride=stride, padding=(kernal_size - 1) // 2)
        
    def forward(self, inputs, mask=None):
        inputs = gelu(inputs)
        inputs = torch.transpose(inputs, 1, 2)  # to (N, C, L)
        # inputs = self.relu(inputs)
        inputs = self.maxpooling(inputs)
        inputs = torch.transpose(inputs, 1, 2)  # to (N, L, C)
        return inputs

class AvgPooling(nn.Module):
    def __init__(self, kernal_size=3, stride=1):
        super(AvgPooling, self).__init__()
        # self.relu = nn.ReLU()
        self.maxpooling = nn.AvgPool1d(kernal_size, stride=stride, padding=(kernal_size - 1) // 2)
        
    def forward(self, inputs, mask=None):
        inputs = gelu(inputs)
        inputs = torch.transpose(inputs, 1, 2)  # to (N, C, L)
        # inputs = self.relu(inputs)
        inputs = self.maxpooling(inputs)
        inputs = torch.transpose(inputs, 1, 2)  # to (N, L, C)
        return inputs



def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]
