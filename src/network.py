import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from blocks import *


class ShuffleNetV2_OneShot(nn.Module):

    def __init__(self, n_vocab, input_size=128, n_class=2, n_layer=4, n_cell=4, dp=0.5, emb=None):
        super(ShuffleNetV2_OneShot, self).__init__()
        self.input_size = input_size
        self.n_layer = n_layer
        self.n_cell = n_cell
        # building first layer
        self.emb = emb
        if emb is None:
            self.word_embeddings = nn.Embedding(n_vocab, input_size, padding_idx=0)
            self.position_embeddings = nn.Embedding(512, input_size)
            self.token_type_embeddings = nn.Embedding(2, input_size)
            emb_dim = input_size
        else:
            self.word_embeddings = torch.nn.Embedding.from_pretrained(emb[0], freeze=False)
            self.position_embeddings = torch.nn.Embedding.from_pretrained(emb[1], freeze=False)
            self.token_type_embeddings = torch.nn.Embedding.from_pretrained(emb[2], freeze=False)
            emb_dim = emb[0].size(-1)
        self.embedding_transform = torch.nn.Linear(emb_dim, input_size)
        self.embedding_layernorm = torch.nn.LayerNorm(input_size, eps=1e-12)
        self.embedding_dropout = torch.nn.Dropout(dp)
        self.features = torch.nn.ModuleList()
        self.features_drop = torch.nn.ModuleList()
        self.features_norm = torch.nn.ModuleList()
        archIndex = 0
        # layer
        for layer_i in range(self.n_layer):
            self.features.append(torch.nn.ModuleList())
            self.features_drop.append(torch.nn.Dropout(dp))
            self.features_norm.append(torch.nn.LayerNorm(input_size, eps=1e-12))
            layers = self.features[layer_i]
            for cell_i in range(self.n_cell):
                layers.append(torch.nn.ModuleList())
                ops = layers[cell_i]
                for op_i in range(cell_i + 1):
                    archIndex += 1
                    ops.append(torch.nn.ModuleList())
                    for blockIndex in range(10):
                        if blockIndex == 0:
                            # print('zero')
                            ops[-1].append(ZeroOp())
                        elif blockIndex == 1:
                            # print('skip')
                            ops[-1].append(SkipOp())    
                        elif blockIndex == 2:
                            # print('Conv3')
                            ops[-1].append(SVDSepConv(input_size, input_size, 3))
                        elif blockIndex == 3:
                            # print('Conv5')
                            ops[-1].append(SVDSepConv(input_size, input_size, 5))
                        elif blockIndex == 4:
                            # print('Conv7')
                            ops[-1].append(SVDSepConv(input_size, input_size, 7))
                        elif blockIndex == 5:
                            # print('linear')
                            ops[-1].append(SVDLinear(input_size, input_size))
                        elif blockIndex == 6:
                            # print('SelfAttention 1')
                            ops[-1].append(SVDSelfAttention(input_size, 1))
                        elif blockIndex == 7:
                            # print('SelfAttention 4')        
                            ops[-1].append(SVDSelfAttention(input_size, 4))
                        elif blockIndex == 8:
                            # print('maxpooling')
                            ops[-1].append(MaxPooling())
                        elif blockIndex == 9:
                            # print('avgpooling')
                            ops[-1].append(AvgPooling())
                        elif blockIndex == 10:
                            # print('RnnOp')
                            ops[-1].append(LSTMOp(input_size))
                        else:
                            raise NotImplementedError
        self.archLen = archIndex
        self.cellLen = self.archLen / self.n_layer
        self.g_pooling = GeneralizedPooler(input_size, input_size)
        self.final_dropout = torch.nn.Dropout(dp)
        self.fc1 = nn.Linear(input_size, 128)
        self.classifier = nn.Sequential(
            nn.Linear(128, n_class))

        self.bert_transform_layers = torch.nn.ModuleList()
        for _ in range(13):
            self.bert_transform_layers.append(nn.Linear(input_size, 768))

        self._initialize_weights()
    
    def get_arc_parameters(self, architecture=None):
        num_params = 0
        if architecture is None:
            for param in self.parameters():
                if param.requires_grad:
                    num_params += param.numel()
        else:
            assert self.archLen == len(architecture), 'arclen:{}, arch:{}'.format(self.archLen, len(architecture))
            arc_i = 0
            for n, param in self.named_parameters():
                if 'features' not in n and 'transform_layers' not in n:
                    num_params += param.numel()
            for layer_i in range(self.n_layer):
                layers = self.features[layer_i]
                for cell_i in range(self.n_cell):
                    ops = layers[cell_i]
                    for op_i in range(cell_i + 1):
                        op = ops[op_i][architecture[arc_i]]
                        for opp in op.parameters():
                            if opp.requires_grad:
                                num_params += opp.numel()
                        arc_i += 1
        return num_params / 1e6

    def print_arc(self, architecture=None):
        assert self.archLen == len(architecture), 'arclen:{}, arch:{}'.format(self.archLen, len(architecture))
        arc_i = 0
        arc_str = 'current model:\n{}'.format(str(architecture))
        for layer_i in range(self.n_layer):
            layers = self.features[layer_i]
            for cell_i in range(self.n_cell):
                ops = layers[cell_i]
                for op_i in range(cell_i+1):
                    op = ops[op_i][architecture[arc_i]]
                    for n, _ in op.named_parameters():
                        arc_str += '\n{}'.format(n)
                    arc_i += 1
        return arc_str

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, architecture=None, layer_i=-2, bert_inputs=None):
        if layer_i == -3:
            return self.forward_bert_transform(bert_inputs)
        elif layer_i == -2:
            return self.forward_net(input_ids, token_type_ids, attention_mask, labels, position_ids, architecture)
        elif layer_i == -1:
            return self.forward_embedding(input_ids, token_type_ids, attention_mask, position_ids)
        else:
            return self.forward_block(bert_inputs, attention_mask, architecture, layer_i)
    
    def forward_bert_transform(self, inputs):
        bert_outputs = []
        for i, input_id in enumerate(inputs):
            bert_outputs.append(self.bert_transform_layers[i](input_id))
        return bert_outputs

    def forward_net(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, architecture=None):

        len_input = torch.sum(attention_mask, dim=-1)
        len_input_max = torch.max(len_input).item()
        input_ids = input_ids[:,:len_input_max]
        token_type_ids = token_type_ids[:,:len_input_max]
        attention_mask = attention_mask[:,:len_input_max]

        assert self.archLen == len(architecture), 'arclen:{}, arch:{}'.format(self.archLen, len(architecture))

        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids) + self.token_type_embeddings(token_type_ids)
        x = self.embedding_transform(x)
        x = self.embedding_dropout(x)
        x = self.embedding_layernorm(x)
        
        arc_i = 0
        all_hidden_states = ()
        for layer_i in range(self.n_layer):
            all_hidden_states = all_hidden_states + (x,)
            layers = self.features[layer_i]
            layers_outs = [x]
            for cell_i in range(self.n_cell):
                ops = layers[cell_i]
                ops_outs = []
                for op_i in range(cell_i + 1):
                    op = ops[op_i][architecture[arc_i]]
                    ops_outs.append(op(layers_outs[op_i], attention_mask))
                    arc_i += 1
                layers_outs.append(sum(ops_outs))
            x = layers_outs[-1]
            x = self.features_drop[layer_i](x)
            x = self.features_norm[layer_i](x)

        all_hidden_states = all_hidden_states + (x,)

        x = self.g_pooling(x, attention_mask)
        x = self.final_dropout(x)
        x = self.fc1(x)
        x = self.classifier(x)
        bert_outputs = self.forward_bert_transform(all_hidden_states)
        outputs = (x,) + all_hidden_states

        return outputs, bert_outputs

    def forward_embedding(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None):
        len_input = torch.sum(attention_mask, dim=-1)
        len_input_max = torch.max(len_input).item()
        input_ids = input_ids[:,:len_input_max]
        token_type_ids = token_type_ids[:,:len_input_max]
        attention_mask = attention_mask[:,:len_input_max]

        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids) + self.token_type_embeddings(token_type_ids)
        x = self.embedding_transform(x)
        x = self.embedding_dropout(x)
        x = self.embedding_layernorm(x)

        return x

    def forward_block(self, input_ids, attention_mask, architecture, layer_i):
        x = input_ids
        mask = attention_mask
        arc_i = int(layer_i * self.cellLen)
        attention_mask = mask[:,:x.size(1)]
        layers = self.features[layer_i]
        layers_outs = [x]
        for cell_i in range(self.n_cell):
            ops = layers[cell_i]
            ops_outs = []
            for op_i in range(cell_i + 1):
                op = ops[op_i][architecture[arc_i]]
                ops_outs.append(op(layers_outs[op_i], attention_mask))
                arc_i += 1
            layers_outs.append(sum(ops_outs))
        x = layers_outs[-1]
        x = self.features_drop[layer_i](x)
        x = self.features_norm[layer_i](x)
        return x
            
    def global_max_pool(self, x, mask):
        # mask = torch.eq(mask.float(), 0.0).long()
        # mask = torch.unsqueeze(mask, dim=2).repeat(1, 1, x.size(2))
        # mask *= -(2 ** 8)
        # x += mask
        x = torch.max(x, dim=1)[0]
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.001)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.running_mean is not None:
                    nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BertLayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding) and self.emb is None:
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])

class GeneralizedPooler(nn.Module):
    def __init__(self, input_size, intermedia_size, num_heads = 1):
        super(GeneralizedPooler, self).__init__()
        self.nin = input_size
        self.nout = intermedia_size
        self.num_heads = num_heads

        self.w1 = nn.Parameter(torch.Tensor(self.nin, self.nout*self.num_heads))
        self.w2 = nn.Parameter(torch.Tensor(self.nout, self.nin*self.num_heads))
        self.bias1 = nn.Parameter(torch.Tensor(self.nout*self.num_heads,))
        self.bias2 = nn.Parameter(torch.Tensor(self.nin*self.num_heads,))
        # self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias1, -bound, bound)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w2)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias2, -bound, bound)

    def forward(self, hidden_states, hidden_mask):
        batch_size = hidden_states.size(0)
        extended_hidden_mask = hidden_mask.unsqueeze(2)
        extended_hidden_mask = (1.0 - extended_hidden_mask) * (-1e9)

        att_matrix = torch.bmm(hidden_states, self.w1.unsqueeze(0).expand(batch_size, *self.w1.size())) + self.bias1
        att_matrix = gelu(att_matrix) #[batch, len, nout*num_head]
        w2_heads = self.w2.unsqueeze(0).expand(batch_size, *self.w2.size()) # (batch_size, nout, nin*numheads)

        # Split and concat
        att_matrix = torch.cat(torch.chunk(att_matrix, self.num_heads, dim=2), dim=0)  # (head*batch_size, len, nout)
        w2_heads = torch.cat(torch.chunk(w2_heads, self.num_heads, dim=2), dim=0)  #(head*batch_size, nout, nin)

        att_matrix = torch.bmm(att_matrix, w2_heads) #(head*batch_size, len, nin)

        # Restore shape
        att_matrix = torch.cat(torch.chunk(att_matrix, self.num_heads, dim=0), dim=2) + self.bias2 # (batch_size, len, nin*head)

        # softmax
        att_matrix = att_matrix + extended_hidden_mask
        att_matrix = F.softmax(att_matrix, dim=1)  # [batch, length, nin*head]

        hidden_states = hidden_states.repeat(1, 1, self.num_heads) * att_matrix
        hidden_states = torch.sum(hidden_states, dim=1) # [batch, dim]
        # hidden_states = self.tanh(hidden_states)
        # hidden_states = self.layer_norm(hidden_states)

        return hidden_states

if __name__ == "__main__":
    # architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    # scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    # scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
    channels_scales = []
    for i in range(len(scale_ids)):
        channels_scales.append(scale_list[scale_ids[i]])
    model = ShuffleNetV2_OneShot()
    # print(model)

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())
