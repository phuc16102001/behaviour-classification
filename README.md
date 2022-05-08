# Behaviour classification

## Introduction
A real-time system to classify single-object behaviour. Using MediaPipe Pose and Tensorflow with LSTM model to not only inference on trained model, but also to train.

## Architecture

In each frame of picture, I used the `Media Pipe Pose` library from Google to detect poses' landmark. Subsequently, these landmarks were converted as a vector. Each landmark not only contains two coordinates (x, y) but also includes (z, visibility). At the generating data stage, 600 video frames were captured after 10 second count down. I used a Recurrent neural network (RNN) model which contains LSTM layers and trained it by feeding `N_TIME` window frames continuously. Furthermore, the model also attached some drop-out layers to reduce overfitting.

## Folder structure
- The `data` folder includes the generated data from `src/gen_data.py`
- The `src` folder contains source code:
  - `config.py`: The configuration of number of epochs (`N_EPOCH`), batch size (`N_BATCH`) and window size of LSTM layer (`N_TIME`)
  - `gen_data.py`: Used to generate data for different classes
  - `train.py`: The file should be run after generating all the behaviour from `gen_data.py` to create the model
  - `inference.py`: Thereafter, the model is used by this file to inference
- The `models` folder contains the model `best.h5` which were trained by `train.py`

## Contribution

This project was done by [phuc16102001](https://github.com/phuc16102001/)
You can reference if you needed, but **do not copy** without permission

## Reference

This project was built by referencing and improving the project from [Mi AI](https://www.miai.vn/2022/02/14/nhan-dien-hanh-vi-con-nguoi-bang-mediapipe-pose-va-lstm-model-mi-ai/)
