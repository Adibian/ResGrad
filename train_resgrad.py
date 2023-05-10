import argparse

from resgrad.train import resgrad_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_epoch", type=int, default=0)
    args = parser.parse_args()
    resgrad_train(args)