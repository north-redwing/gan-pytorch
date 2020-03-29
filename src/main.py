from gan import GAN
from utils import get_args


def main():
    args = get_args()
    if args.model == 'GAN':
        model = GAN()
    model.train()


if __name__ == '__main__':
    main()