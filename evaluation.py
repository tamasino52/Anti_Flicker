import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import math
import time
import os
from sklearn.linear_model import LinearRegression


# Get flag option value
video_path = './video/C0183.MP4'
output_path = './output/blended_C0183.MP4'

# Set video variables
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), 'Unable to load video. check your video path.'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
prop_fps = int(cap.get(cv2.CAP_PROP_FPS))
prop_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
prop_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
prop_channer = 3
prop_size = (prop_height, prop_width, prop_channer)
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # number of total frame

# Information
print('INPUT PATH = {}\nPROPS FPS = {}\nPROPS SIZE = ({}, {})\nTOTAL FRAME = {}\n'.format(
    video_path, prop_fps, prop_height, prop_width, total_frame))


def main():
    measure = 50
    #original_intensity = np.array([])
    #corrected_intensity = np.array([])
    cap = cv2.VideoCapture('./output/blended_C0183.MP4')
    for idx in range(measure):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite('./figure/final_{}.jpg'.format(idx), frame)
        #original_intensity = np.append(original_intensity, getIntensity(frame))

    cap.open(output_path)
    for idx in range(measure):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite('./figure/restored_{}.jpg'.format(idx), frame)
        corrected_intensity = np.append(corrected_intensity, getIntensity(frame))

    # intensity plot
    x_max, y_max = [], []
    x_min, y_min = [], []
    for idx in range(measure)[::8]:
        y_max.append(np.max(original_intensity[idx:idx+8]))
        x_max.append(idx + np.argmax(original_intensity[idx:idx + 8]))
        y_min.append(np.min(original_intensity[idx:idx+8]))
        x_min.append(idx + np.argmin(original_intensity[idx:idx + 8]))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.ylim(110, 180)
    plt.plot(range(len(original_intensity)), original_intensity, '.-', label='Original frames')
    plt.plot(x_max, y_max, 'r.-', label='Maximum points')
    plt.plot(x_min, y_min, 'b.-', label='Minimum points')
    plt.legend(loc='upper left')
    plt.xlabel('Frame number')
    plt.ylabel('Intensity')
    plt.xlim(0, measure)


    x_max, y_max = [], []
    x_min, y_min = [], []
    for idx in range(measure)[::8]:
        y_max.append(np.max(corrected_intensity[idx:idx+8]))
        x_max.append(idx + np.argmax(corrected_intensity[idx:idx + 8]))
        y_min.append(np.min(corrected_intensity[idx:idx+8]))
        x_min.append(idx + np.argmin(corrected_intensity[idx:idx + 8]))

    plt.subplot(1, 2, 2)
    plt.plot(range(len(corrected_intensity)), corrected_intensity, 'g.-', label='Restored frames')
    plt.plot(x_max, y_max, 'r.-', label='Maximum points')
    plt.plot(x_min, y_min, 'b.-', label='Minimum points')
    plt.legend(loc='upper left')
    plt.xlabel('Frame number')
    plt.ylabel('Intensity')
    plt.xlim(0, measure)
    plt.show()

    x_max, y_max = [], []
    x_min, y_min = [], []
    for idx in range(measure)[::8]:
        y_max.append(np.max(original_intensity[idx:idx+8]))
        x_max.append(idx + np.argmax(original_intensity[idx:idx + 8]))
        y_min.append(np.min(original_intensity[idx:idx+8]))
        x_min.append(idx + np.argmin(original_intensity[idx:idx + 8]))

    x_max, y_max = np.array(x_max), np.array(y_max)
    x_min, y_min = np.array(x_min), np.array(y_min)
    print(((y_max - y_min)/(y_max + y_min)).mean())

    x_max, y_max = [], []
    x_min, y_min = [], []
    for idx in range(measure)[::8]:
        y_max.append(np.max(corrected_intensity[idx:idx+8]))
        x_max.append(idx + np.argmax(corrected_intensity[idx:idx + 8]))
        y_min.append(np.min(corrected_intensity[idx:idx+8]))
        x_min.append(idx + np.argmin(corrected_intensity[idx:idx + 8]))

    x_max, y_max = np.array(x_max), np.array(y_max)
    x_min, y_min = np.array(x_min), np.array(y_min)
    print(((y_max - y_min)/(y_max + y_min)).mean())


# Return average pixel value of single frame
def getIntensity(image: np.ndarray) -> float:
    return np.sum(image) / np.size(image)


if __name__ == '__main__':
    final = cv2.imread('./figure/final_15.jpg')
    middle = cv2.imread('./figure/middle_15.jpg')
    origin = cv2.imread('./figure/origin_15.jpg')

    cv2.imshow('final', final[450:650, 1250:1550])
    cv2.imwrite('./figure/final.jpg', final[450:650, 1250:1550])
    cv2.waitKey(0)

    cv2.imwrite('./figure/middle.jpg', middle[450:650, 1250:1550])
    cv2.waitKey(0)

    cv2.imwrite('./figure/origin.jpg', origin[450:650, 1250:1550])
    cv2.waitKey(0)
    #main()
    #cap.release()
    #cv2.destroyAllWindows()
    print('Finish')
