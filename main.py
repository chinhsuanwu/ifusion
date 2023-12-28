import argparse

from ifusion import finetune, inference, optimize_pose

from util.util import load_config, parse_model, set_random_seed


def main(config):
    model = parse_model(config.model)

    print("[INFO] Pose optimization")
    optimize_pose(model, **config.pose)

    print("[INFO] Sparse-view finetuning")
    finetune(model, **config.finetune)

    print("[INFO] Inference")
    inference(model, **config.inference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yaml")
    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)

    set_random_seed(config.seed)
    main(config)
