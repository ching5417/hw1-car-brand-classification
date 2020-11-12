# hw1-car-brand-classification
Code for NCTU Selected Topics in Visual Recognition using Deep Learning Homework 1

## Hardware
The following specs were used to create the original solution.
- Ubuntu 20.04 LTS
- Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz (8 Cores)
- 1x NVIDIA GeForce GTX 1080

## Reproducing Submission
To reproduct the training and testing process, do the following steps:
1. [Installation](#installation)
2. [Training](#training)
6. [Make Submission](#make-submission)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
pip install -r requirements.txt
```

## Dataset Preparation
All required files are already in data directory.

## Training

### Train models
To train models, run following commands.
```
$ python train.py \
  [--data_dir] \
  [--model_name={resnet,wide_resnet,inception}] \
  [--batch_size] \
  [--num_epochs] \
  [--learning_rate]
```
After training, the model will save to `model/` folder.
```
model
+- savetimestamp_model.pt (ex. 20201030_2334_resnet50.pt)
```
[Here](https://drive.google.com/drive/folders/15Yg0yYuCBFOXIPGJL9tNPQ_DDJu4H8m9?usp=sharing) download the model (`20201102_0700_inception.pt` is the best one)

## Make Submission
Following command will ensemble of all models and make submissions.
```
$ python test.py \
  --data_dir={testfolder} \
  --PATH={modelsavepath} \
  --model_name={resnet,wide_resnet,inception}
```
After testing, it will output a `output.csv` in `output/` folder.

I also put my testing result in `1output/` folder. (`output_inception.csv` and `output_ensemble.csv`)
