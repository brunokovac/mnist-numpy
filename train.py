import os
import argparse
import sys
from data.dataset import Dataset, DataLoader
import mnist
from model.models import MnistModelSmall, MnistModelBig
from model.layers import SoftmaxWithCrossEntropyLoss
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator
from training.augmentations import JitteringAugmentation, ShiftingAugmentation
from training.learning_rate import ConstantLearningRate, PolynomialLearningRate


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--val_pct", type=float, default=0.2)

    parser.add_argument("--learning_rate_schedule", type=str,
                        default="constant",
                        choices=["constant", "polynomial"])

    parser.add_argument("--learning_rate", type=float,
                        default=2e-3)

    parser.add_argument("--start_learning_rate", type=float,
                        default=2e-3)
    parser.add_argument("--end_learning_rate", type=float,
                        default=2e-4)
    parser.add_argument("--power", type=float, default=1.5)

    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--use_augmentations", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int,
                        default=50)

    parser.add_argument("--ckpt_dir", type=str, default="ckpts/")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    assert args.train_pct + args.val_pct == 1.0

    train_dataset = Dataset(mnist.train_images(), mnist.train_labels())
    train_dataset, val_dataset = train_dataset.split(
        [args.train_pct, args.val_pct])


    if args.use_augmentations:
        train_augmentations = [
            JitteringAugmentation(max_value=0.5),
            ShiftingAugmentation(max_shift=5)
        ]
    else:
        train_augmentations = []

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True, normalize=True,
                                   augmentations=train_augmentations)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, drop_last=False, normalize=True)

    model = MnistModelSmall()
    loss_fn = SoftmaxWithCrossEntropyLoss(num_classes=10, name="ce_loss")

    evaluator = ModelEvaluator(model)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.learning_rate_schedule == "constant":
        learning_rate_scheduler = ConstantLearningRate(args.learning_rate)
    elif args.learning_rate_schedule == "polynomial":
        learning_rate_scheduler = PolynomialLearningRate(
            start_learning_rate=args.start_learning_rate,
            end_learning_rate=args.end_learning_rate,
            power=args.power,
            max_steps=args.num_steps)
    else:
        learning_rate_scheduler = ConstantLearningRate(args.learning_rate)

    trainer = ModelTrainer(model, loss_fn,
                           log_every_n_steps=args.log_every_n_steps)
    trainer.train(train_data_loader, val_data_loader, evaluator, args.num_steps,
                  learning_rate_scheduler, args.ckpt_dir)

    #test
    test_dataset = Dataset(mnist.test_images(), mnist.test_labels())
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False,
                                  normalize=True)

    evaluator.evaluate(test_data_loader)
    print(evaluator.get_accuracy())


if __name__ == '__main__':
    main(sys.argv[1:])
