import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar
import math
import time
import os
import scipy.stats as st
from sklearn.linear_model import LinearRegression


# Get flag option value
video_path = './output/corr_C0183.MP4'
output_path = './output/blended_C0183.MP4'

# Set video variables
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), 'Unable to load video. check your video path.'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
prop_fps = int(cap.get(cv2.CAP_PROP_FPS))
prop_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
prop_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
prop_channer = 3
prop_size = (prop_height, prop_width, prop_channer)
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # number of total frame
out = cv2.VideoWriter(output_path, fourcc, prop_fps, (prop_width, prop_height))

# Information
print('INPUT PATH = {}\nOUTPUT PATH = {}\nPROPS FPS = {}\nPROPS SIZE = ({}, {})\nTOTAL FRAME = {}\n'.format(
    video_path, output_path, prop_fps, prop_height, prop_width, total_frame))

kernel_size = 5
kernel = cv2.getGaussianKernel(kernel_size, sigma=1)


def main():
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frame-1)
    ret, last_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = [first_frame] * int(kernel_size / 2)

    for _ in progressbar(range(total_frame)):
        while len(frames) < kernel_size:
            ret, frame = cap.read()
            if not ret:
                frame = last_frame[:]
                frames.append(last_frame)
            else:
                frames.append(frame)
        blended_frame = np.zeros_like(frame)
        for i, w in enumerate(kernel):
            blended_frame = cv2.addWeighted(blended_frame, 1, frames[i], w, 0)
        out.write(blended_frame)
        del frames[0]


if __name__ == '__main__':
    main()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Finish')
