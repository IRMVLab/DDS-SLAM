## Installation

```
# Create conda environment
conda create -n ddsslam python=3.7
conda activate ddsslam

# Install the pytorch first (Please check the cuda version)
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -r requirements.txt

# Build extension (marching cubes from neuralRGBD)
cd external/NumpyMarchingCubes
python setup.py install
```

## Run

## Semantic-Super Dataset

The data can be found on  [Data](https://drive.google.com/file/d/1ZxWw2kNmgeMhBXAGyovL2icXHzn2OCVV/view?usp=sharing). Then utilize the pre-trained depth estimation models provided by [Semantic-SuPer](https://github.com/ucsdarclab/Python-SuPer)  to obtain deep priors.


Next, run DDS-SLAM

```
python ddsslam.py --config ./configs/Super/trail3.yaml 
```

### StereoMIS Dataset

The data can be found [here](https://zenodo.org/records/7727692). Then utilize the pre-trained depth estimation models provided by [robust-pose-estimator](https://github.com/aimi-lab/robust-pose-estimator) to obtain deep priors.


Next, run DDS-SLAM

```
python ddsslam.py --config ./configs/StereoMIS/p2_1.yaml 
```

## Related Repositories

Our codebase is based on [Co-SLAM](https://github.com/HengyiWang/Co-SLAM). We are grateful to the authors for sharing their codebase with the public. Your significant contributions have been instrumental in making our work a reality!

## Citation

If you find our code or paper useful, please cite

```
@inproceedings{shan2024dds,
  title={DDS-SLAM: Dense Semantic Neural SLAM for Deformable Endoscopic Scenes},
  author={Shan, Jiwei and Li, Yirui and Yang, Lujia and Feng, Qiyu and Han, Lijun and Wang, Hesheng},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={10837--10842},
  year={2024},
  organization={IEEE}
}
```
