# VISinger2

> 🍴 Forked from [zhangyongmao/VISinger2](https://github.com/zhangyongmao/VISinger2)

> 📌 This clone repo is created as GitHub does not support Git LFS on public fork

**Fixed PyTorch 2.0.1 compatibility issues, now VISinger 2 can be run in cuda12.**

### Updates

- Add FastAPI server for serving inference
- Added Cantonese singing data for training
- Added Cantonese pre-trained model

## Pre-requisites

1. Install python requirements: pip install -r requirements.txt
2. Download the [Opencpop Dataset](https://wenet.org.cn/opencpop/).
3. prepare data like data/opencpop (wavs, train.txt, test.txt, train.list, test.list)
4. To generate train.list, use `awk -F'|' '{print $1}' train.txt > train.list`
5. To generate test.list, use `awk -F'|' '{print $1}' test.txt > test.list`
6. modify the egs/visinger2/config.json (data/data_dir, train/save_dir)

## extract pitch and mel

```
cd egs/visinger2
bash bash/preprocess.sh config.json
```

## Training

```
cd egs/visinger2
bash bash/train.sh 0
```

We trained the model for 500k steps with batch size of 16.

## Inference

modify the model_dir, input_dir, output_dir in inference.sh

```
cd egs/visinger2
bash bash/inference.sh
```

Some audio samples can be found in [demo website](https://zhangyongmao.github.io/VISinger2/) and [bilibili](https://www.bilibili.com/video/BV1wX4y167rb/?share_source=copy_web&vd_source=4e678224f5616d7af7dfaf2401b5d574).

The pre-trained model trained using opencpop is [here](https://drive.google.com/file/d/1MgXLQuquPT2qu1__JNF010-tg48N0hZn/view?usp=share_link), the config.json is [here](https://drive.google.com/file/d/10GI9OUtE4fQ8om8MvycDYQpcP6lgHLNZ/view?usp=share_link), and the result of the test set synthesized by this pre-trained model is [here](https://drive.google.com/file/d/1JTMhtkexo5z3q0bpLoqh4EJmx1HjZyMr/view?usp=share_link).
