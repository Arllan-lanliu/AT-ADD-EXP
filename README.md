<h1 align="center">AT-ADD Challenge: ThreeTO</h1>

This repository provides the EXP of conventional and self-supervised learning (SSL)-based countermeasures for the **AT-ADD: All-Type Audio Deepfake Detection Challenge**.  

---

## 1. Data Preparation

**Dataset url** : https://huggingface.co/datasets/xieyuankun/AT-ADD-Track2


Please download the AT-ADD dataset and organize it as follows:

```
├── T2/
│   ├── train/
│   │   └── *.wav
│   ├── dev/
│   │   └── *.wav
│   ├── eval/
│   │   └── *.wav
│   └── labels/
│       ├── train.csv
│       ├── dev.csv
```

---

## 2. SSL model download
```
>> cd yourmodeldir/
>> pip install huggingface-hub
>> export HF_ENDPOINT=https://hf-mirror.com
>> hf download facebook/wav2vec2-xls-r-300m --local-dir ./wav2vec2-xls-r-300m
>> hf download microsoft/wavlm-large --local-dir ./wavlm-large
>> hf download m-a-p/MERT-v1-330M --local-dir ./MERT-v1-330M
>> hf download nsivaku/nithin_checkpoints --local-dir ./BEATs_iter3 
>> hf download laion/larger_clap_music_and_speech --local-dir ./larger_clap_music_and_speech 
```

---
## 3. Environment Setup
```
>> conda create -n atadd_t1_3.10 python=3.10.13
>> conda activate atadd_t1_3.10
>> python -m pip uninstall -y torch torchvision torchaudio triton
>> python -m pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
>> python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
>> pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
>> pip install -r requirements.txt

pip install timm==1.0.3
git clone https://github.com/Adamdad/rational_kat_cu.git
cd rational_kat_cu
pip install -e .
```

---
## 4. Training and Evaluation
```
>> git clone git@github.com:Arllan-lanliu/AT-ADD-EXP.git
>> cd ./AT-ADD-EXP
>> conda activate atadd3.10
>> chmod +x ./run.sh
>> ./run.sh
```


