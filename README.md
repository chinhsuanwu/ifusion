# iFusion

### [Project Page](https://chinhsuanwu.github.io/ifusion) | [Paper]()

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
pip install -r requirements.txt
```

Download Zero123-XL and place it under `ldm/ckpt`
```bash
cd ldm/ckpt && wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```

## Usage

Create an image directory that contains at least 2 images. Images can be preprocessed through
```bash
rembg i input.png output.png
```
Run demo by specifing the image directory
```bash
python main.py data.image_dir=asset/sorter
```
You should see a NeRF-style `transform.json` in the same folder as the output of the pose optimization and `lora.ckpt` after the sparse-view fine-tuning stage. Qualitative visualization of novel view synthesis is shown at `demo.png` as follows. Please refer to `config/main.yaml` for detailed hyper-parameters.

![](https://github.com/chinhsuanwu/ifusion/assets/67839539/a5ac8b90-af95-4bd2-9a6a-077808a5fcaa)

For 3D reconstruction, please check out [ifusion-threestudio](https://github.com/chinhsuanwu/ifusion-threestudio).

## Acknowledgements
This repo is a wild mixture of [zero123](https://github.com/cvlab-columbia/zero123), [threestudio](https://github.com/threestudio-project/threestudio), and [lora](https://github.com/cloneofsimo/lora). Kudos to the authors for their amazing work!
