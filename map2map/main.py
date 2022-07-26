from .args import get_args
from . import train
from . import test
from . import train_swag
from . import test_swag

def main():

    args = get_args()

    if args.mode == 'train':
        if not args.swag:
            train.node_worker(args)
        else:
            train_swag.node_worker(args)
    elif args.mode == 'test':
        if not args.swag:
            test.test(args)
        else:
            test_swag.sample_swag(args)

if __name__ == '__main__':
    main()
