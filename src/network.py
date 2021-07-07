import torch
import torch.nn as nn
from blocks import *


class ShuffleNetV2_OneShot(nn.Module):

    def __init__(self, n_vocab, input_size=128, n_class=1000, emb=None):
        super(ShuffleNetV2_OneShot, self).__init__()
        self.input_size = input_size

        self.stage_repeats = [6, 6, 6, 6]

        # building first layer
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
        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            for i in range(numrepeat):
                archIndex += 1
                self.features.append(torch.nn.ModuleList())
                for blockIndex in range(9):
                    if blockIndex == 0:
                        # print('Skip')
                        self.features[-1].append(SkipOp())
                    elif blockIndex == 1:
                        # print('Conv3')
                        self.features[-1].append(SimConv(input_size, input_size, 3))
                    elif blockIndex == 2:
                        # print('Conv5')
                        self.features[-1].append(SimConv(input_size, input_size, 5))
                    elif blockIndex == 3:
                        # print('Conv7')
                        self.features[-1].append(SimConv(input_size, input_size, 7))
                    elif blockIndex == 4:
                        # print('Conv3')
                        self.features[-1].append(SimConv(input_size, input_size, 3, dilation=2))
                    elif blockIndex == 5:
                        # print('Conv5')
                        self.features[-1].append(SimConv(input_size, input_size, 5, dilation=2))
                    elif blockIndex == 6:
                        # print('Conv7')
                        self.features[-1].append(SimConv(input_size, input_size, 7, dilation=2))
                    elif blockIndex == 7:
                        # print('Conv7')
                        self.features[-1].append(MaxPooling())
                    elif blockIndex == 8:
                        # print('Conv7')
                        self.features[-1].append(AvgPooling())
                    else:
                        raise NotImplementedError

        self.archLen = archIndex
        # self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, n_class, bias=False))
        self._initialize_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, architecture=None):

        len_input = torch.sum(attention_mask, dim=-1)
        len_input_max = torch.max(len_input).item()
        input_ids = input_ids[:,:len_input_max]
        token_type_ids = token_type_ids[:,:len_input_max]
        attention_mask = attention_mask[:,:len_input_max]

        assert self.archLen == len(architecture) * len(self.stage_repeats)

        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids) + self.token_type_embeddings(token_type_ids)
        x = self.embedding_transform(x)

        for archs, arch_id in zip(self.features, architecture):
            x = archs[arch_id](x) + x

        x = self.global_max_pool(x, attention_mask)
        # x = x.view(-1, self.input_size)
        x = self.classifier(x)
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
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
