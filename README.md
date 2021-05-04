# Learning Skeletal Articulations with Neural Blend Shapes

![Python](https://img.shields.io/badge/Python->=3.8-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.8.0-Red?logo=pytorch)
![Blender](https://img.shields.io/badge/Blender-%3E=2.8-Orange?logo=blender)

This repository provides an end-to-end library for automatic character rigging and blend shapes generation . It is based on our work [Learning Skeletal Articulations with Neural Blend Shapes](https://peizhuoli.github.io/neural-blend-shapes/index.html) that is published in SIGGRAPH 2021.

<img src="https://peizhuoli.github.io/neural-blend-shapes/images/video_teaser.gif" slign="center">

## Prerequisites

Our code has been tested on Ubuntu 18.04. Befor starting, please configure the Anaconda environment by

~~~bash
conda env create -f environment.yaml
conda activate neural-blend-shapes
~~~

Or you may install the following packages (and their dependencies) manually:

- pytorch 1.8
- tensorboard
- tqdm
- chumpy
- opencv-python

## Quick Start

We provide a pretrained model that is dedicated for biped character. Download and extracat the pretrained model from [Google Drive](https://drive.google.com/file/d/1S_JQY2N4qx1V6micWiIiNkHercs557rG/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1y8iBqf1QfxcPWO0AWd2aVw) (9ras) and put the `pre_trained` folder under the project directory. Run

~~~bash
python demo.py --pose_file=./eval_constant/sequences/greeting.npy --obj_path=./eval_constant/meshes/maynard.obj
~~~

The nice greeting animation showed above will be saved in `demo/obj` as obj files. Besides, the generated skeleton will be saved as `demo/skeleton.bvh` and the skinning weight matrix will be saved as `demo/weight.npy`.

If you are also interested in traditional linear blend skinning(LBS) technique results generated with our rig, you can also specify `--envelope_only=1` to evaluate our model with only envelope branch.

We also provided several other meshes and animation sequences, feel free to try their combinations!

### Test on Customized Meshes

You may also try to run our model with your own meshes. Please make sure your mesh is triangulated and has a consistent upright and front facing orientation. Our model requires the input meshes are spatially aligned, so please also specify `--normalize=1`. Alternatively, you can try to scale and translate your mesh to align the provided `eval_constant/meshes/smpl_std.obj` and specify `--normalize=0`.

### Evaluation

To reconstruct the quantitative result with the pretrained model, you need to download the test dataset from [Google Drive](https://drive.google.com/file/d/1RwdnnFYT30L8CkUb1E36uQwLNZd1EmvP/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1c5QCQE3RXzqZo6PeYjhtqQ) (8b0f) and put the two extracted folders under `./dataset` and run

~~~bash
python evaluation.py
~~~



## Blender Visualization

We provide a simple wrapper of blender's python API (>=2.80) for rendering 3D mesh animations and visualize skinning weight. The following code has been tested on Ubuntu 18.04 and macOS Big Sur with Blender 2.92.

Note that due to the limitation of Blender, you cannot run Eevee render engine with a headless machine. 

We also provide several parameters to control the behavior of the scripts. , please refer to the code for more details. To pass parameters to python script in blender, please do following:

~~~bash
blender [blend file path(optional)] -P [python script path] [-b] -- --arg1 [ARG1] --arg2 [ARG2]
~~~



### Animation

We provide a simple light and camera setting in `eval_constant/simple_scene.blend`. We use `ffmpeg` to convert images into video. Pealse make sure you have installed it before running. You may need to adjust it before using. To render the obj files genrated above, run

~~~bash
cd blender_script
blender ../eval_constant/simple_scene.blend -P render_mesh.py -b
~~~

The rendered per-frame image will be saved in `demo/images` and composited video will be saved as `demo/video.mov`. 

### Skinning Weight

Visualize the skinning weight is a good sanity check to see whether the model works as expected. We provide a script using Blender's built-in ShaderNodeVertexColor to visualize the skinning weight. Simply run

~~~bash
cd blender_script
blender -P vertex_color.py
~~~

You will see something similar to this if the model works as expected:

<img src="https://peizhuoli.github.io/neural-blend-shapes/images/skinning_vis.png" slign="center" width="30%">

Mean while, you can import the generated skeleton (in `demo/skeleton.bvh`) to Blender. For skeleton rendering, please refer to [deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing).

## Acknowledgements

The code in `meshcnn` is adapted from [MeshCNN](https://github.com/ranahanocka/MeshCNN) by [@ranahanocka](https://github.com/ranahanocka/).

The code in `models/skeleton.py` is adapted from [deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing) by [@kfiraberman](https://github.com/kfiraberman), [@PeizhuoLi](https://github.com/PeizhuoLi) and [@HalfSummer11](https://github.com/HalfSummer11).

The code in `dataset/smpl_layer` is adapted from [smpl_pytorch](https://github.com/gulvarol/smplpytorch) by [@gulvarol](https://github.com/gulvarol).

Part of the test models are taken from and [SMPL](https://smpl.is.tue.mpg.de/en), [MultiGarmentNetwork](https://github.com/bharat-b7/MultiGarmentNetwork) and [Adobe Mixamo](https://www.mixamo.com).

## Citation

If you use this code for your research, please cite our papers:

~~~bibtex
@article{li2021learning,
  author = {Li, Peizhuo and Aberman, Kfir and Hanocka, Rana and Liu, Libin and Sorkine-Hornung, Olga and Chen, Baoquan},
  title = {Learning Skeletal Articulations with Neural Blend Shapes},
  journal = {ACM Transactions on Graphics (TOG)},
  volume = {40},
  number = {4},
  pages = {1},
  year = {2021},
  publisher = {ACM}
}
~~~



Note: This repository is still under construction. We are planning to release the code and dataset for training soon.