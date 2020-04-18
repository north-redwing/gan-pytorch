from gan import GAN
from anogan import AnoGAN
from utils import get_args


def main():
    args = get_args()

    # Training Generator/Discriminator
    if args.model == 'GAN':
        model = GAN()
    # elif args.model == 'LSGAN':
    #     model = LSGAN()
    # elif args.model == 'WGAN':
    #     model = WGAN()
    # elif args.model == 'WGAN_GP':
    #     model = WGAN_GP()
    # elif args.model == 'DRAGAN':
    #     model = DRAGAN()
    # elif args.model == 'EBGAN':
    #     model = EBGAN()
    # elif args.model == 'BEGAN':
    #     model = BEGAN()
    # elif args.model == 'SNGAN':
    #     model = SNGAN()
    elif args.model == 'AnoGAN':
        model = AnoGAN()
    model.train()

    # Anomaly Detection
    if args.model == 'AnoGAN':
        model.anomaly_detect()


if __name__ == '__main__':
    main()