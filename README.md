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
>> conda create -n atadd3.10 python=3.10.13
>> conda activate atadd3.10
>> pip install -r requirements.txt
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


