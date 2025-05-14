# Planning with Diffusion &nbsp;&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing)


Training and visualizing of diffusion models from [Planning with Diffusion for Flexible Behavior Synthesis](https://diffusion-planning.github.io/).
This branch has the Maze2D experiments and will be merged into main shortly.

<p align="center">
    <img src="https://diffusion-planning.github.io/images/diffuser-card.png" width="60%" title="Diffuser model">
</p>

## Quickstart

Load a pretrained diffusion model and sample from it in your browser with [scripts/diffuser-sample.ipynb](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing).


## Installation

```
conda env create -f environment.yml
conda activate diffusion
pip install -e .
```
### mujoco
install the mujoco from and use pip install
```
https://github.com/google-deepmind/mujoco/releases/tag/2.1.0 # Download to Ubuntu download directory
mkdir -p ~/.mujoco
cp ~/Downloads/mujoco210-linux-x86_64.tar.gz ~/.mujoco/
cd ~/.mujoco
tar -xvf mujoco210-linux-x86_64.tar.gz
pip install mujoco-py
```
do not for got to setup the env
```
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin
source ~/.bashrc
```
Trouble shooting if Cython error please check the version
```
pip uninstall Cython
pip install Cython==0.29.36
```
then rebuild mujoco-py
```
pip uninstall mujoco-py
pip install mujoco-py==2.1.2.14 --no-cache-dir
```
## Usage

Train a diffusion model with:
```
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1
```

The default hyperparameters are listed in [`config/maze2d.py`](config/maze2d.py).
You can override any of them with runtime flags, eg `--batch_size 64`.

Plan using the diffusion model with:
```
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1
```
## Config
```
for change the default training parameters and config, see config/maze2d.py
```

## Docker

1. Build the container:
```
docker build -f azure/Dockerfile . -t diffuser
```

2. Test the container:
```
docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$HOME/.d4rl,target=/root/.d4rl \
    diffuser \
    bash -c \
    "export PYTHONPATH=$PYTHONPATH:/home/code && \
    python /home/code/scripts/train.py --dataset hopper-medium-expert-v2 --logbase logs/docker"
```


## Running on Azure

#### Setup

1. Launching jobs on Azure requires one more python dependency:
```
pip install git+https://github.com/JannerM/doodad.git@janner
```

2. Tag the image built in [the previous section](#Docker) and push it to Docker Hub:
```
export DOCKER_USERNAME=$(docker info | sed '/Username:/!d;s/.* //')
docker tag diffuser ${DOCKER_USERNAME}/diffuser:latest
docker image push ${DOCKER_USERNAME}/diffuser
```

3. Update [`azure/config.py`](azure/config.py), either by modifying the file directly or setting the relevant [environment variables](azure/config.py#L47-L52). To set the `AZURE_STORAGE_CONNECTION` variable, navigate to the `Access keys` section of your storage account. Click `Show keys` and copy the `Connection string`.

4. Download [`azcopy`](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10): `./azure/download.sh`

#### Usage

Launch training jobs with `python azure/launch.py`. The launch script takes no command-line arguments; instead, it launches a job for every combination of hyperparameters in [`params_to_sweep`](azure/launch_train.py#L36-L38).


#### Viewing results

To rsync the results from the Azure storage container, run `./azure/sync.sh`.

To mount the storage container:
1. Create a blobfuse config with `./azure/make_fuse_config.sh`
2. Run `./azure/mount.sh` to mount the storage container to `~/azure_mount`

To unmount the container, run `sudo umount -f ~/azure_mount; rm -r ~/azure_mount`


## Reference
```
@inproceedings{janner2022diffuser,
  title = {Planning with Diffusion for Flexible Behavior Synthesis},
  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}
```


## Acknowledgements

The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.
The organization of this repo and remote launcher is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo.
