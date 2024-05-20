import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialPyramidPooling(nn.Module):
    def _init_(self, levels=[1, 2, 4]):
        super(SpatialPyramidPooling, self)._init_()
        self.levels = levels

    def forward(self, x):
        num, channels, height, width = x.size()
        pools = []

        for level in self.levels:
            kernel_size = (height // level, width // level)
            stride = kernel_size
            padding = (height % level // 2, width % level // 2)
            
            # Ensure the padding is within bounds
            padding = (min(padding[0], kernel_size[0] - 1), min(padding[1], kernel_size[1] - 1))

            pool = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
            pools.append(pool.view(num, -1))

        return torch.cat(pools, dim=1)
    