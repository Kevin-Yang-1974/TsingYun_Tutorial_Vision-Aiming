"""MNIST digit model scaffold for Task 2.

Detector code should call the inference function in this module. Training code
lives in train.py so detector.py stays focused on board detection, corner
geometry, and PnP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from train import MNISTClassifier

RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mnist_classifier.npz"
model=MNISTClassifier()
model.load_state_dict(torch.load(DEFAULT_MODEL_PATH))
model.eval()

def preprocess_mnist_crop(board_crop: ImageLike) -> np.ndarray:
    board_crop=np.asarray(board_crop,dtype=np.uint8)
    gray=cv2.cvtColor(board_crop,cv2.COLOR_RGB2GRAY)
    _,v=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    v_resized=cv2.resize(v,(28,28))[np.newaxis,np.newaxis,:,:]
    v_normalized=v_resized/255.0
    return torch.tensor(v_normalized, dtype=torch.float32)

def load_mnist_model(model_path: Path = DEFAULT_MODEL_PATH) -> object:
    model=MNISTClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_mnist_digit(model: object, model_input: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        outputs=model(model_input)
        probabilities=F.softmax(outputs,dim=1)
        digit=torch.argmax(probabilities,dim=1).item()
        confidence=probabilities[0,digit].item()
    return digit, confidence


def classify_mnist_digit(board_crop: ImageLike, model_path: Path = DEFAULT_MODEL_PATH) -> tuple[int, float]:
    model_input = preprocess_mnist_crop(board_crop)
    # model = load_mnist_model(model_path)
    digit, confidence = predict_mnist_digit(model, model_input)
    return digit, confidence
