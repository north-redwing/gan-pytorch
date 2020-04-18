import torch.nn as nn
from torchsummary import summary
from utils import init_weights


class Generator(nn.Module):
    def __init__(self, z_dim=100, image_channels=1, image_size=28):
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
        self._print_info()

    def forward(self, input):
        input = input.reshape(-1, self.z_dim)
        out = self.fc(input)
        out = out.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
        out = self.deconv(out)

        return out

    def _print_info(self):
        print('\n[Generator summary]')
        summary(self, (self.z_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, image_channels=1, image_size=28):
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
        self._print_info()

    def forward(self, input):
        out = self.conv(input)
        feature = out
        feature = feature.view(feature.size()[0], -1)
        out = out.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
        out = self.fc(out)

        return out, feature

    def _print_info(self):
        print('\n[Discriminator summary]')
        summary(self, (3, self.image_size, self.image_size))


if __name__ == '__main__':
    G = Generator()
    D = Discriminator()
