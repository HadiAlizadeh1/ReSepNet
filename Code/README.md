# In the Name of Allah

## ReSepNet: A Unified-Light Model for Recursive Speech Separation with Unknown Speaker Count

This repository contains the implementation for our **Interspeech 2025** paper:
**“ReSepNet: A Unified-Light Model for Recursive Speech Separation with Unknown Speaker Count.”**

📄 [Read the Paper (ISCA Archive)](https://www.isca-archive.org/interspeech_2025/alizadeh25_interspeech.pdf)

---

## Plan

* [x] Create Environment
* [x] Data Pre-processing
* [x] Training
* [x] Evaluation
* [x] Separate

---

## Create Environment

Install **conda** and run the following commands:

```bash
conda create -n ReSepNet python=3.9.12 -y
conda activate ReSepNet
cd ReSepNet
pip install -r requirements.txt
```

---

## Data Pre-processing

To create the **WSJ_2–5mix** dataset, we use [pywsj0-mix](https://github.com/mpariente/pywsj0-mix).

After creating the dataset, chunk the **training data** into 4-second segments:

```bash
python chunking.py  --base_dir "./wav8k/min/tr" --target_length 4 --sample_rate 8000
```

## Data Preparation

Before training, generate the data paths and metadata JSON files.

### For 2-Speaker Mixtures

```bash
python preprocess.py --in-dir /data/min --out-dir data --sample-rate 8000
```

### For 3-Speaker Mixtures

```bash
python preprocess3.py --in-dir /data/min --out-dir data --sample-rate 8000
```

> **Notes:**
>
> * `--in-dir`: Path to the root directory containing your original audio files.
> * `--out-dir`: Destination directory where processed data and metadata JSON files will be saved.
> * `--sample-rate`: Target sampling rate.
>
> Make sure your dataset is organized and accessible before running the preprocessing script.


## Training

To start training:
```bash
python train2.py --train_dir2 2spk/tr --valid_dir2 2spk/cv --train_dir3 3spk/tr --valid_dir3 3spk/cv 
```

---

## Evaluation

Evaluate your trained model on the test set. This will calculate the SI-SNR, SDR, and counting accuracy for 2-speaker mixtures

```bash
python evaluate.py --data_dir 'data/tt' --model_path 'exp/temp/temp_best.pth.tar'
```

---

## Separate

To separate mixed audio samples for 1 iteration:

```bash
python separate.py --mix_json '{your_directory}/tt/mix.json' --mix_dir test.wav
```

---

## Reference

* [Conv-TasNet](https://github.com/kaituoxu/Conv-TasNet)
* [DPTNet](https://github.com/ujscjj/DPTNet)

