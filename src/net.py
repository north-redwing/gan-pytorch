import torch.nn as nn
from utils import init_weights


class Generator(nn.Module):
    def __init__(self, z_dim, image_channels, image_size):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.image_channels = image_channels
        self.image_size = image_size

        self.fc = nn.Sequential(
            # input is Z, going into a fully connected layer
            nn.Linear(self.z_dim, 1024),  # (-1, 1024)
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(
                1024,
                128 * (self.image_size // 4) * (self.image_size // 4)
            ),  # (-1, 128*7*7)
            nn.BatchNorm1d(
                128 * (self.image_size // 4) * (self.image_size // 4)),
            nn.ReLU()
        )
        # reshape -> (-1, 128, 7, 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (-1, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.image_channels, 4, 2, 1),
            # (-1, 3, 28, 28)
            nn.Tanh()
        )
        init_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
        x = self.deconv(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, image_channels, image_size):
        super(Discriminator, self).__init__()
        self.image_channels = image_channels
        self.image_size = image_size

        self.conv = nn.Sequential(
            # input a real image, going to a convolution
            nn.Conv2d(self.image_channels, 64, 4, 2, 1),  # (-1, 64, 14 ,14)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # (-1, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        # reshape -> (-1, 128*7*7)
        self.fc = nn.Sequential(
            nn.Linear(
                128 * (self.image_size // 4) * (self.image_size // 4),
                1024
            ),  # (-1, 1024)
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),  # (-1, 1)
            nn.Sigmoid()
        )
        init_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(
            -1,
            128 * (self.image_size // 4) * (self.image_size // 4)
        )
        x = self.fc(x)

        return x
