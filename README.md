# iFusion

### [Project Page](https://chinhsuanwu.github.io/ifusion) | [Paper](https://arxiv.org/abs/2312.17250)

This is the official implementation of [iFusion](), a framework that extends existing single-view reconstruction to pose-free sparse-view reconstruction by repurposing [Zero123](https://github.com/cvlab-columbia/zero123) for camera pose estimation.

<img src="https://github.com/chinhsuanwu/ifusion/assets/67839539/d90bb4a3-f6a6-4121-995f-833c3350c302" width=600><br>

[iFusion: Inverting Diffusion for Pose-Free Reconstruction from Sparse Views]() <br>[Chin-Hsuan Wu](https://chinhsuanwu.github.io),
[Yen-Chun Chen](https://www.microsoft.com/en-us/research/people/yenche/),
[Bolivar Solarte](https://enriquesolarte.github.io/),
[Lu Yuan](https://www.microsoft.com/en-us/research/people/luyuan/),
[Min Sun](https://aliensunmin.github.io/)<br>

## Installation
```bash
git clone https://github.com/chinhsuanwu/ifusion.git
cd ifusion

# conda environment.yaml is also available
pip install -r requirements.txt
```
Download Zero123-XL to `ldm/ckpt`
```bash
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt -P ldm/ckpt
```

## Quick Start

Run demo by specifying the image directory containing 2+ images
```bash
python demo.py data.image_dir=asset/sorter
```
The output includes a NeRF-style `transform.json` file (from camera pose estimation), `lora.ckpt` (from fine-tuning), and `demo.png` (from fine-tuning, as shown below), all located in the given directory.

![](https://github.com/chinhsuanwu/ifusion/assets/67839539/a5ac8b90-af95-4bd2-9a6a-077808a5fcaa)

One can also run a quick ablation without including our method, i.e., the original single-view Zero123, for comparison
```bash
python demo.py data.image_dir=asset/sorter \
               data.demo_fp=asset/sorter/demo_single_view.png \
               inference.use_single_view=true
```
![](https://github.com/chinhsuanwu/ifusion/assets/67839539/b683f37d-fb4a-44ff-a7f2-e74c760d208b)


For 3D reconstruction, please check out [ifusion-threestudio](https://github.com/chinhsuanwu/ifusion-threestudio).

## Evaluation
```bash
# download the renderings for GSO and OO3D
bash download_data.sh

# camera pose estimation
python main.py --pose \
               --gpu_ids=0,1,2,3 \
               data.root_dir=rendering \
               data.name=GSO \
               data.exp_root_dir=exp

# novel view synthesis
python main.py --nvs \
               --gpu_ids=0,1,2,3 \
               data.root_dir=rendering \
               data.name=GSO \
               data.exp_root_dir=exp

# evaluation
python eval.py --pose
python eval.py --nvs
```
Please refer to `config/main.yaml` for detailed hyper-parameters and arguments.

## Citation

```bibtex
@article{wu2023ifusion,
  author = {Wu, Chin-Hsuan and Chen, Yen-Chun, Solarte, Bolivar and Yuan, Lu and Sun, Min},
  title = {iFusion: Inverting Diffusion for Pose-Free Reconstruction from Sparse Views},
  journal = {arXiv preprint arXiv:2312.17250},
  year = {2023}
}
```

## Acknowledgements
This repo is a wild mixture of [zero123](https://github.com/cvlab-columbia/zero123), [threestudio](https://github.com/threestudio-project/threestudio), and [lora](https://github.com/cloneofsimo/lora). Kudos to the authors for their amazing work!
