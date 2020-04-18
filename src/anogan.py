import os
import csv
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from gan import GAN
from utils import get_dataloader


class AnoGAN(GAN):
    def __init__(self):
        super(AnoGAN, self).__init__()
        # csv
        self.path_to_results_z_csv = os.path.join(
            self.path_to_dir,
            *['csv', 'results_z.csv']
        )

    def _anomaly_score(self, x, fake_img, Lambda=0.1):
        residual_loss = torch.abs(x - fake_img)
        residual_loss = residual_loss.view(residual_loss.size()[0], -1)
        residual_loss = torch.sum(residual_loss, dim=1)

        _, real_feature = self.Discriminator(x)
        _, fake_feature = self.Discriminator(fake_img)

        discrimination_loss = torch.abs(real_feature - fake_feature)
        discrimination_loss = discrimination_loss.view(
            discrimination_loss.size()[0], -1)
        discrimination_loss = torch.sum(discrimination_loss, dim=1)

        batch_loss = (1 - Lambda) * residual_loss + Lambda * discrimination_loss

        total_loss = torch.sum(batch_loss)

        return total_loss, batch_loss, residual_loss

    def anomaly_detect(self, normal_dataset='mnist',
                       anomaly_dataset='fashion-mnist'):
        normal_dataloader = get_dataloader(dataset=normal_dataset, batch_size=5)
        anomaly_dataloader = get_dataloader(dataset=anomaly_dataset,
                                            batch_size=5)
        normal_images = normal_dataloader.__iter__().__next__()[0]
        anomaly_images = anomaly_dataloader.__iter__().__next__()[0]
        images = torch.cat([normal_images, anomaly_images])
        images = images[torch.randperm(images.size()[0])]

        # optimize latent variables z
        z = torch.randn(10, self.args.z_dim).to(self.args.device)
        z.requires_grad = True
        z_optimizer = optim.Adam([z], lr=1e-3)
        train_hist = {}
        train_hist['loss_z_optimizing'] = 0.0

        for epoch in range(self.args.n_epoch_z_optim):
            fake_images = self.Generator(z)
            loss, _, _ = \
                self._anomaly_score(images, fake_images, Lambda=self.args.lam)
            z_optimizer.zero_grad()
            loss.backward()
            z_optimizer.step()
            train_hist['loss_z_optimizing'] = loss.item() / 5

            # ----- logging -----
            # tensorboard
            self.writer.add_scalar('loss/z_optimizing_loss', loss.item(), epoch)
            # hyperdash
            if self.exp:
                self.exp.metric('z optimizing loss', loss.item())
            # csv
            with open(self.path_to_results_z_csv, 'a') as f:
                result_writer = csv.DictWriter(f, list(train_hist.keys()))
                if epoch == 0: result_writer.writeheader()
                result_writer.writerow(train_hist)

        fake_images = self.Generator(z)
        total_loss, batch_loss, residual_loss = \
            self._anomaly_score(images, fake_images, Lambda=self.args.lam)

        fig = plt.figure(figsize=(18, 6))
        for i, (image, fake_image) in enumerate(zip(images, fake_images)):
            image = image.cpu().detach().numpy()
            image = (image + 1.0) / 2.0
            image = image.transpose(1, 2, 0)
            if 0 <= i <= 4:
                ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
            elif 5 <= i <= 9:
                ax = fig.add_subplot(4, 5, i + 6, xticks=[], yticks=[])
            ax.set_title('Score: {0:.1f}'.format(batch_loss[i]),
                         color=("black" if batch_loss[i] <= 400 else "red"))
            ax.imshow(image)

            fake_image = fake_image.cpu().detach().numpy()
            fake_image = (fake_image + 1.0) / 2.0
            fake_image = fake_image.transpose(1, 2, 0)
            if 0 <= i <= 4:
                ax = fig.add_subplot(4, 5, 5 + i + 1, xticks=[], yticks=[])
            elif 5 <= i <= 9:
                ax = fig.add_subplot(4, 5, 5 + i + 6, xticks=[], yticks=[])
            ax.imshow(fake_image)
        plt.show()
        print(self.writer)
        self.writer.add_figure('anomaly_detection', fig)


if __name__ == '__main__':
    anogan = AnoGAN()
    anogan.anomaly_detect()
