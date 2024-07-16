import collections

import torch
#from einops.layer.torch import Rearrange

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MixerBlock(torch.nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        self.token_mix = torch.nn.Sequential(
            #torch.nn.LayerNorm(dim),
            torch.nn.LayerNorm(num_patch), #dim
            #Rearrange('b n d -> b d n'),
            #FeedForward(num_patch, token_dim, dropout),
            torch.nn.Linear(num_patch, token_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(token_dim, num_patch),
            torch.nn.Dropout(dropout),
            #Rearrange('b d n -> b n d')
        )
        self.channel_mix = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        #x = x + self.token_mix(x)
        #x = x + self.channel_mix(x)
        x + self.token_mix(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(torch.nn.Module):
    def __init__(self,
                 in_channels=3,
                 dim=512,
                 patch_size=4,
                 image_size=32,
                 depth=8,
                 token_dim=256,
                 channel_dim=2048,
                 num_classes=10):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        # self.to_patch_embedding = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels, dim, patch_size, patch_size),
        #     Rearrange('b c h w -> b (h w) c'),
        # )
        self.layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels, dim, patch_size, patch_size),
        ])
        self.layers.append(torch.nn.Identity())

        # self.layers = torch.nn.ModuleList([
        #     torch.nn.Conv2d(in_channels, dim, patch_size, patch_size),
        # ])
        # self.mixer_blocks = torch.nn.ModuleList([])
        # for _ in range(depth):
        #     self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        # self.layer_norm = torch.nn.LayerNorm(dim)
        # self.mlp_head = torch.nn.Sequential(
        #     torch.nn.Linear(dim, num_classes)
        # )
        for _ in range(depth):
            self.layers.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        self.layers.append(torch.nn.LayerNorm(dim))
        self.layers.append(torch.nn.Linear(dim, num_classes))

    def forward(self, x, start=0, end=None):
        # x = self.to_patch_embedding(x)
        # for mixer_block in self.mixer_blocks:
        #     x = mixer_block(x)
        # x = self.layer_norm(x)
        # x = x.mean(dim=1)
        # return self.mlp_head(x)
        if end is None:
            end = len(self.layers) - 1

        for i, layer in enumerate(self.layers):
            if i == 1 and (start <= i <= end):
                x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
            elif start <= i < end:
                x = layer(x)
            elif i == end:
                x = x.mean(dim=1)
                x = layer(x)

        return x