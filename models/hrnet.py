import os

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def bn_func(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


def make_layer(block, in_planes, planes, num_blocks):
    if in_planes != planes * block.expansion:
        downsample = nn.Sequential(
            bn_func(in_planes, planes * block.expansion),
            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM)
        )
    else:
        downsample=None

    layers = []
    layers.append(block(in_planes, planes, stride=1, downsample=downsample))
    in_planes = planes * block.expansion
    for _ in range(num_blocks-1):
        layers.append(block(in_planes, planes))
    
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = bn_func(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.conv2 = conv3x3(planes, planesstride=stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.conv3 = bn_func(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HRModule(nn.Module):
    def __init__(
        self, 
        num_branches, 
        block, 
        num_blocks,
        num_channels, 
        fuse_method, 
        multi_scale_output=True
    ):
        super(HRModule, self).__init__()

        self.num_branches = num_branches
        self.num_channels = num_channels
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output

        self.branches = nn.ModuleList(
            [
                make_layer(block, num_channels[i], num_channels[i], num_blocks[i])
                for i in range(num_branches)
            ]
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j < i:
                    # keep downsampling until you reach desired resolution
                    ds_blocks = []
                    for _ in range(i-j-1):
                        ds_blocks.append(
                            nn.Sequential(
                                conv3x3(self.num_channels[j], self.num_channels[j], stride=2),
                                nn.BatchNorm2d(self.num_channels[j], momentum=BN_MOMENTUM),
                                nn.ReLU(True)
                            )
                        ) 
                    ds_blocks.append(
                        nn.Sequential(
                            conv3x3(self.num_channels[j], self.num_channels[i], stride=2),
                            nn.BatchNorm2d(self.num_channels[i], momentum=BN_MOMENTUM)
                        )
                    )
                    fuse_layer.append(nn.Sequential(*ds_blocks))
                elif j > i:
                    # upsamples since we are at a lower resolution
                    fuse_layer.append(
                        nn.Sequential(
                            bn_func(self.num_channels[j], self.num_channels[i]),
                            nn.BatchNorm2d(self.num_channels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                else:
                    # skip connection since it's the same resolution
                    fuse_layer.append(None)
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_channels(self):
        return self.num_channels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # propogate each resolution through their respective branches
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            # change topmost resolution to ith branch resolution
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])

            # change jth branch resolution to ith branch resolution
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNet(nn.Module):
    def __init__(self, params):
        super(HRNet, self).__init__()

        # stem net
        num_init_feat = 64
        self.conv1 = conv3x3(3, num_init_feat, stride=2)
        self.bn1 = nn.BatchNorm2d(num_init_feat, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(num_init_feat, num_init_feat, stride=2)
        self.bn2 = nn.BatchNorm2d(num_init_feat, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        stage1_cfg = params["STAGE1"]
        block = blocks_dict[stage1_cfg["BLOCK"]]
        num_channels = stage1_cfg['NUM_CHANNELS'][0]
        num_blocks = stage1_cfg["NUM_BLOCKS"][0]
        self.stage1 = make_layer(block, num_init_feat, num_channels, num_blocks)
        pre_stage_channels = [num_channels * block.expansion]

        # stages 2-4
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.num_branches_lst = []
        self.num_stages = params["NUM_STAGES"]
        for i in range(self.num_stages - 1):
            cfg = params["STAGE" + str(i + 2)]
            block = blocks_dict[cfg["BLOCK"]]
            num_channels = cfg["NUM_CHANNELS"]
            num_channels = [
                num_channels[i] * block.expansion for i in range(len(num_channels))
            ]
            self.transitions.append(
                self._make_transition_layer(pre_stage_channels, num_channels)
            )
            stage, pre_stage_channels = self._make_stage(cfg)

            self.num_branches_lst.append(cfg["NUM_BRANCHES"])
            self.stages.append(stage)

        # heatmap output
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=params["NUM_JOINTS"]+1,
            kernel_size=params["FINAL_CONV_KERNEL"],
            stride=1,
            padding=1 if params["FINAL_CONV_KERNEL"] == 3 else 0
        )

        self.pretrained_layers = params['PRETRAINED_LAYERS']

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        # transition layer makes new branches for the next stage
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        # propagate current branches
        transition_layers = nn.ModuleList()
        for i in range(num_branches_pre):
            if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                transition_layers.append(
                    nn.Sequential(
                        conv3x3(num_channels_pre_layer[i], num_channels_cur_layer[i]),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                transition_layers.append(None)

        # downsamples lowest resolution to make new branches
        for i in range(num_branches_pre, num_branches_cur):
            ds_blocks = []
            for j in range(i - num_branches_pre + 1):
                inchannels = num_channels_pre_layer[-1]
                outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                ds_blocks.append(
                    nn.Sequential(
                        conv3x3(inchannels, outchannels, stride=2),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    )
                )
            transition_layers.append(nn.Sequential(*ds_blocks))

        return transition_layers

    def _make_stage(self, layer_config, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_channels = modules[-1].get_num_channels()

        return nn.Sequential(*modules), num_channels

    def forward(self, x):
        # x = F.interpolate(x, size=384, mode="bicubic")

        # stem net
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # first stage
        y_list = [self.stage1(x)]

        # stages 2-4
        for stage, transition, num_branches in zip(
            self.stages, self.transitions, self.num_branches_lst
        ):
            x_list = []
            for i in range(num_branches):
                if transition[i] is not None:
                    # use y_list[-1] not y_list[i] because you only ever need
                    # the last branch to make new branches or to change num_channels
                    x_list.append(transition[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = stage(x_list)

        x = self.final_layer(y_list[0])
        
        return x

    def init_weights(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            raise ValueError('{} is not exist!'.format(pretrained))