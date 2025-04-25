import torch.nn as nn

# VGG16 Configuration
vgg16_config = [
    64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
    'M', 512, 512, 512, 'M'
]

def get_vgg_layers(config, batch_norm=False):
    """
    Returns a Sequential model for VGG layers based on the provided configuration.
    - config: A list defining the architecture (e.g., [64, 'M', 128, 'M', ...]).
    - batch_norm: Whether to apply batch normalization after each convolution layer.
    """
    layers = []
    in_channels = 3  # For RGB images

    for c in config:
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, features, output_dim, dropout=0.5):
        """
        Initializes the VGG model.
        - features: The convolutional layers created using get_vgg_layers.
        - output_dim: The number of output classes.
        - dropout: Dropout probability for the fully connected layers.
        """
        super().__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(7)

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)  # Flatten the tensor
        x = self.classifier(h)
        return x, h
