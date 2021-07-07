import torch
import torch.nn as nn

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


class SimConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, dilation=1):
        super(SimConv, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernal_size, 1, bias=True,
                          padding= dilation * (kernal_size - 1) // 2, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels, affine=False)
        
    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)  # to (N, C, L)
        inputs = self.relu(inputs)
        inputs = self.conv(inputs)
        inputs = self.bn(inputs)
        inputs = torch.transpose(inputs, 1, 2)  # to (N, L, C)
        return inputs

class SkipOp(nn.Module):
    def __init__(self):
        super(SkipOp, self).__init__()
        
    def forward(self, inputs):
        return inputs

class MaxPooling(nn.Module):
    def __init__(self, kernal_size=3, stride=1):
        super(MaxPooling, self).__init__()
        self.maxpooling = nn.MaxPool1d(kernal_size, stride=stride, padding=(kernal_size - 1) // 2)
        
    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)  # to (N, C, L)
        inputs = self.maxpooling(inputs)
        inputs = torch.transpose(inputs, 1, 2)  # to (N, L, C)
        return inputs

class AvgPooling(nn.Module):
    def __init__(self, kernal_size=3, stride=1):
        super(AvgPooling, self).__init__()
        self.maxpooling = nn.AvgPool1d(kernal_size, stride=stride, padding=(kernal_size - 1) // 2)
        
    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)  # to (N, C, L)
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
