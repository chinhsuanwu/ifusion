import argparse
import itertools
import multiprocessing
import os


from dataset.base import load_frames
from ifusion import finetune, inference, optimize_pose

from util.util import load_config, parse_model, set_random_seed, str2list, split_list


def set_default_latlon(config):
    id = str2list(config.data.id)[0]
    latlon = load_frames(
        config.data.image_dir, config.data.gt_transform_fp, verbose=False
    )[2]
    default_latlon = latlon[id].tolist()
    config.pose.default_theta = default_latlon[0]
    config.pose.default_azimuth = default_latlon[1]
    config.pose.default_radius = default_latlon[2]


def gen_pose_all(model, config, scenes, ids):
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Optimizing pose {scene}:{id}")
            config.data.scene = scene
            config.data.id = id
            set_default_latlon(config)
            optimize_pose(model, **config.pose)


def gen_nvs_all(model, config, scenes, ids):
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Fine-tuning {scene}:{id}")
            config.data.scene = scene
            config.data.id = id
            finetune(model, **config.finetune)
            inference(model, **config.inference)


def main(config, mode, gpu_ids):
    def worker(config, mode, scenes, ids, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        model = parse_model(config.model)
        if mode[0]:
            gen_pose_all(model, config, scenes, ids)
        if mode[1]:
            gen_nvs_all(model, config, scenes, ids=["0,1"])
    
    perm = list(itertools.permutations(range(5), 2))
    ids = [",".join(map(str, p)) for p in perm]
    gpu_ids = str2list(gpu_ids)
    scenes = sorted(os.listdir(f"{config.data.root_dir}/{config.data.name}"))

    # split scenes and multi-process
    scenes = split_list(scenes, len(gpu_ids))
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = multiprocessing.Process(target=worker, args=(config, mode, scenes[i], ids, gpu_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yaml")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--nvs", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)

    set_random_seed(config.seed)
    main(config, [args.pose, args.nvs], args.gpu_ids)
