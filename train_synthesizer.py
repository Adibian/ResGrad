from synthesizer.train import train_model

import yaml
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("-p", "--preprocess_config", type=str, default='synthesizer/config/persian/preprocess.yaml', required=False, help="path to preprocess.yaml")
    parser.add_argument("-m", "--model_config", type=str, default='synthesizer/config/persian/model.yaml', required=False, help="path to model.yaml")
    parser.add_argument("-t", "--train_config", type=str, default='synthesizer/config/persian/train.yaml', required=False, help="path to train.yaml")
    args = parser.parse_args()
    
    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    train_model(args, configs)