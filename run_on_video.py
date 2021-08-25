import sys
import os
import argparse

import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Test of video')
parser.add_argument('--video', help='Path to video')
parser.add_argument('--model', help='Path to model')

args = parser.parse_args()

OUTPUT_DIR = 'output'


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(args.video):
    sys.exit('Video does not exist')

checkpoint = torch.load(args.model)
model = checkpoint['model']

model.to(device)
model.eval()

video = cv2.VideoCapture(args.video)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

base_file_name = os.path.splitext(os.path.basename(args.video))[0]

out = None

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input = (rgb_frame / 255).astype(np.float32)
    input = np.expand_dims(np.moveaxis(input, 2, 0), 0)
    input = torch.tensor(input).to(device)

    with torch.no_grad():
        pred = model(input)

    depth = pred.detach().cpu().numpy()[0][0]
    depth /= depth.max()
    depth = (depth * 255).astype(np.uint8)
    # cv2.imshow('frame', frame)
    # cv2.imshow('depth', depth)
    # cv2.waitKey(0)

    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
    result = np.concatenate((frame, depth), axis=1)
    cv2.imshow('result', result)
    cv2.waitKey(1)

    if out is None:
        out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f'{base_file_name}.avi'), fourcc, 30, result.shape[1::-1])
    out.write(result)
