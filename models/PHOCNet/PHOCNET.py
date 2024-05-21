import torch.nn as nn
import torch.nn.functional as F
from models.PHOCNet.spatial_pyramid_pooling import SpatialPyramidPooling

# PHOCnet model architecture

class PHOCNet(nn.Module):

    def __init__(self, n_out, input_channels = 3, pooling_levels = 3, pool_type = 'max_pool'):
        super(PHOCNet, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.pooling_layer_fn = SpatialPyramidPooling(levels = pooling_levels) #, pool_type=pool_type     pool_type is not used in the forward function
        pooling_output_size = self.pooling_layer_fn.pooling_output_size
        
        self.fc1 = nn.Linear(pooling_output_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_out)

    def forward(self, x):
        out = self.conv_block1(x)
        out = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        out = self.conv_block2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        out = self.conv_block3(out)
        out = self.conv_block4(out)

        out = self.pooling_layer_fn(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = F.dropout(out, p = 0.5, training = self.training)
        out = self.fc2(out)
        out = F.relu(out)
        out = F.dropout(out, p = 0.5, training = self.training)
        out = self.fc3(out)

        # Sigmoid para salida multilabel
        out = F.sigmoid(x)
        return out

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)