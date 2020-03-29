import csv
import os
import warnings
import numpy as np
import datetime
import matplotlib.pyplot as plt
from time import time
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from net import Generator, Discriminator
from utils import get_dataloader, get_args


class GAN(object):
    def __init__(self):
        warnings.filterwarnings('ignore')
        self.start_time = time()

        self.args = get_args()
        if self.args.checkpoint_dir_name:
            dir_name = self.args.checkpoint_dir_name
        else:
            dir_name = datetime.datetime.now().strftime('%y%m%d%H%M%S')
        path_to_dir = Path(__file__).resolve().parents[1]
        path_to_dir = os.path.join(path_to_dir, *['log', dir_name])
        os.makedirs(path_to_dir, exist_ok=True)

        # tensorboard
        path_to_tensorboard = os.path.join(path_to_dir, 'tensorboard')
        os.makedirs(path_to_tensorboard, exist_ok=True)
        self.writer = SummaryWriter(path_to_tensorboard)

        # model saving
        os.makedirs(os.path.join(path_to_dir, 'model'), exist_ok=True)
        path_to_model = os.path.join(path_to_dir, *['model', 'model.tar'])

        # csv
        os.makedirs(os.path.join(path_to_dir, 'csv'), exist_ok=True)
        self.path_to_results_csv = os.path.join(
            path_to_dir,
            *['csv', 'results.csv']
        )
        path_to_args_csv = os.path.join(path_to_dir, *['csv', 'args.csv'])
        if not self.args.checkpoint_dir_name:
            with open(path_to_args_csv, 'a') as f:
                args_dict = vars(self.args)
                param_writer = csv.DictWriter(f, list(args_dict.keys()))
                param_writer.writeheader()
                param_writer.writerow(args_dict)

        # logging by hyperdash
        if not self.args.no_hyperdash:
            from hyperdash import Experiment
            self.exp = Experiment(
                'Generation task on ' + self.args.dataset + ' dataset with GAN'
            )
            for key in vars(self.args).keys():
                exec("self.args.%s = self.exp.param('%s', self.args.%s)" % (
                    key, key, key))
        else:
            self.exp = None

        path_to_dataset = os.path.join(
            Path(__file__).resolve().parents[2],
            'datasets'
        )
        os.makedirs(path_to_dataset, exist_ok=True)

        self.dataloader = get_dataloader(
            path_to_dataset,
            self.args.dataset,
            self.args.image_size,
            self.args.batch_size
        )
        sample_data = self.dataloader.__iter__().__next__()[0]
        image_channels = sample_data.shape[1]

        self.sample_z = torch.rand((self.args.batch_size, self.args.z_dim))

        print('\nGenerator --->')
        self.Generator = Generator(
            self.args.z_dim,
            image_channels,
            self.args.image_size
        )
        self.Generator_optimizer = optim.Adam(
            self.Generator.parameters(),
            lr=self.args.lr_Generator,
            betas=(self.args.beta1, self.args.beta2)
        )
        print(self.Generator)
        self.writer.add_graph(self.Generator, self.sample_z)
        self.Generator.to(self.args.device)

        print('\nDiscriminator --->')
        self.Discriminator = Discriminator(image_channels, self.args.image_size)
        self.Discriminator_optimizer = optim.Adam(
            self.Discriminator.parameters(),
            lr=self.args.lr_Discriminator,
            betas=(self.args.beta1, self.args.beta2)
        )
        print(self.Discriminator)
        self.writer.add_graph(self.Discriminator, sample_data)
        self.Discriminator.to(self.args.device)

        self.BCELoss = nn.BCELoss()

        self.sample_z = self.sample_z.to(self.args.device)

    def train(self):
        self.train_hist = {}
        self.train_hist['Generator_loss'] = []
        self.train_hist['Discriminator_loss'] = []

        self.y_real = torch.ones(self.args.batch_size, 1).to(self.args.device)
        self.y_fake = torch.zeros(self.args.batch_size, 1).to(self.args.device)

        self.Discriminator.train()

        global_step = 0
        #  -----training -----
        for epoch in range(1, self.args.n_epoch + 1):
            self.Generator.train()
            for idx, (x, _) in enumerate(self.dataloader):
                if idx == self.dataloader.dataset.__len__() // self.args.batch_size:
                    break

                z = torch.rand((self.args.batch_size, self.args.z_dim))
                z = z.to(self.args.device)
                x = x.to(self.args.device)

                # ----- update Discriminator -----
                self.Discriminator_optimizer.zero_grad()
                # real
                Discriminator_real = self.Discriminator(x)
                Discriminator_real_loss = self.BCELoss(
                    Discriminator_real,
                    self.y_real
                )
                # fake
                Discriminator_fake = self.Discriminator(self.Generator(z))
                Discriminator_fake_loss = self.BCELoss(
                    Discriminator_fake,
                    self.y_fake
                )

                Discriminator_loss = Discriminator_real_loss + Discriminator_fake_loss
                self.train_hist['Discriminator_loss'].append(
                    Discriminator_loss.item()
                )

                Discriminator_loss.backward()
                self.Discriminator_optimizer.step()

                # ----- update Generator -----
                self.Generator_optimizer.zero_grad()
                Discriminator_fake = self.Discriminator(self.Generator(z))
                Generator_loss = self.BCELoss(
                    Discriminator_fake,
                    self.y_real
                )
                self.train_hist['Generator_loss'].append(Generator_loss.item())
                Generator_loss.backward()
                self.Generator_optimizer.step()

                # ----- logging by tensorboard, csv and hyperdash
                # tensorboard
                self.writer.add_scalar(
                    'loss/Generator_loss',
                    Generator_loss.item(),
                    global_step
                )
                self.writer.add_scalar(
                    'loss/Discriminator_loss',
                    Discriminator_loss.item(),
                    global_step
                )
                # csv
                with open(self.path_to_results_csv, 'a') as f:
                    result_writer = csv.DictWriter(f,
                                                   list(self.train_hist.keys()))
                    if epoch == 1 and idx == 0: result_writer.writeheader()
                    result_writer.writerow(self.train_hist)
                # hyperdash
                if self.exp:
                    self.exp.metric(
                        'Generator loss',
                        Generator_loss.item()
                    )
                    self.exp.metric(
                        'Discriminator loss',
                        Discriminator_loss.item()
                    )

                if (idx % 10) == 0:
                    self._plot_sample(global_step)
                global_step += 1

        elapsed_time = time() - self.start_time
        print('\nTraining Finish, elapsed time ---> %f' % (elapsed_time))
        if self.exp:
            self.exp.end()
        self.writer.close()

    def _plot_sample(self, global_step):
        with torch.no_grad():
            total_n_sample = min(self.args.n_sample, self.args.batch_size)
            image_frame_dim = int(np.floor(np.sqrt(total_n_sample)))
            samples = self.Generator(self.sample_z)
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
            samples = (samples + 1) / 2
            fig = plt.figure(figsize=(24, 15))
            for i in range(image_frame_dim * image_frame_dim):
                ax = fig.add_subplot(
                    image_frame_dim,
                    image_frame_dim * 2,
                    (int(i / image_frame_dim) + 1) * image_frame_dim + i + 1,
                    xticks=[],
                    yticks=[]
                )
                if samples[i].shape[2] == 3:
                    ax.imshow(samples[i])
                else:
                    ax.imshow(samples[i][:, :, 0], cmap='gray')
            self.writer.add_figure(
                'sample images generated by GAN',
                fig,
                global_step
            )
