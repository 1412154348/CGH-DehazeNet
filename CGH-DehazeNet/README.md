

## Getting Started

### Environment

Create a new conda environment and install dependencies:

```
conda create -n pytorch_1_10 python=3.8
conda activate pytorch_1_10
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```
When evaluating on OTS with jpeg images as input, please make sure that the version of pillow is 8.3.2, this is to ensure that the same decoding algorithm is used for jpeg images during evaluation and training.
### Data Preparation

1. Download the dataset: [[RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)] and [[HAZE4K](https://github.com/liuye123321/DMT-Net)].
2. Make sure the file structure is consistent with the following:

```
dataset/
├── HAZE4K
│   ├── test
│   |   ├── clear
│   |   │   ├── 1.png
│   |   │   └── 2.png
│   |   │   └── ...
│   |   └── hazy
│   |       ├── 1_0.89_1.56.png
│   |       └── 2_0.93_1.66.png
│   |       └── ...
│   └── train
│       ├── clear
│       │   ├── 1.png
│       │   └── 2.png
│       │   └── ...
│       └── hazy
│           ├── 1_0.68_0.66.png
│           └── 2_0.59_1.95.png
│           └── ...
├── ITS
│   ├── test
│   |   ├── clear
│   |   │   ├── 1400.png
│   |   │   └── 1401.png
│   |   │   └── ...
│   |   └── hazy
│   |       ├── 1400_1.png
│   |       └── ...
│   |       └── 1400_10.png
│   |       └── 1401_1.png
│   |       └── ...
│   └── train
│       ├── clear
│       │   ├── 1.png
│       │   └── 2.png
│       │   └── ...
│       └── hazy
│           ├── 1_1_0.90179.png
│           └── ...
│           └── 1_10_0.98796.png
│           └── 2_1_0.99082.png
│           └── ...
└── OTS
    ├── test
    |   ├── clear
    |   │   ├── 0001.png
    |   │   └── 0002.png
    |   │   └── ...
    |   └── hazy
    |       ├── 0001_0.8_0.2.jpg
    |       └── 0002_0.8_0.08.jpg
    |       └── ...
    └── train
        ├── clear
        │   ├── 0005.jpg
        │   └── 0008.jpg
        |	└── ...
        └── hazy
            ├── 0005_0.8_0.04.jpg
            └── 0005_1_0.2.jpg
            └── ...
            └── 0008_0.8_0.04.jpg
            └── ...
```

### Training

 Run the following script to train DEA-Net-CR from scratch:

```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 300 --iters_per_epoch 5000 --finer_eval_step 1400000 --w_loss_L1 1.0 --w_loss_CR 0.1 --start_lr 0.0001 --end_lr 0.000001 --exp_dir ../experiment/ --model_name DEA-Net-CR --dataset ITS
```

### Evaluation
 Run the following script to evaluation the pre-trained model and save the inference results:
```
cd code/
python3 eval.py --dataset yourdataset --model_name CGH-DehazeNet --pre_trained_model your-weight.pth --save_infer_results



```

