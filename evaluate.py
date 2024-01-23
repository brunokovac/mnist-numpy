import sys

from data.dataset import Dataset, DataLoader
import mnist
from model.models import MnistModelSmall, MnistModelBig
from training.evaluator import ModelEvaluator
import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ckpt_dir", type=str, default="ckpts/")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    model = MnistModelSmall()
    model.load_weights(args.ckpt_dir)

    test_dataset = Dataset(mnist.test_images(), mnist.test_labels())
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False,
                                  normalize=True)

    evaluator = ModelEvaluator(model)
    evaluator.evaluate(test_data_loader)
    print(evaluator.get_accuracy())


if __name__ == '__main__':
    main(sys.argv[1:])
