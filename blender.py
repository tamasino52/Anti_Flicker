import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar
import math
import time

from sklearn.linear_model import LinearRegression

# Flags setting
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", default="video/C0183.MP4", help="Process a path of video.")
parser.add_argument("--no_display", default=False, type=bool, help="Enable to disable the visual display.")
parser.add_argument("-o", "--output", default='output', help="Process a path of output video")

args = parser.parse_known_args()

# Flag option value initialization
video_path = args[0].video
measure = float('inf')


def main():
    # Get intensity cycle of video
    cycle = getPriorCycle(True)
    print('(INFO) VIDEO CYCLE : {}'.format(cycle))

    # Get Linear Regression filter
    #line_filter = getLineFilter(cycle, True)
    removeFlicker(cycle, True)


# Chck Video validity
def video_process(path):
    def wrapper(func):
        def decorator(*args, **kwargs):
            global measure
            cap = cv2.VideoCapture(path)
            assert cap.isOpened(), 'Unable to load video. check your video path.'
            print('(INFO) Iterating Video Source ...'
                  '\nPATH: {}'
                  '\nWIDTH: {}'
                  '\nHEIGHT : {}'
                  '\nCOUNT : {}\n'.format(path,
                                          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                          int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            if measure > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                measure = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            time.sleep(1)
            result = func(*args, cap=cap, **kwargs)
            cap.release()
            return result
        return decorator
    return wrapper


# Show video frames removed flicker effect
@ video_process(video_path)
def removeFlicker(cycle: int, show_frame: bool = False, cap: cv2.VideoCapture = None):
    def predict_regression(X, regression):
        coef, intercept = regression[0], regression[1]
        y = X.astype(np.float64)
        y = y * coef + intercept
        y[y > 255] = 255
        y[y < 0] = 0
        return y.astype(np.uint8)

    assert type(cycle) is int, 'TYPE ERROR : CYCLE IS NOT INTEGER'
    assert cycle > 0, 'INVALID PARAMETER ERROR : CYCLE VALUE IS INVALID'

    corr_intensity = []
    orig_intensity = []
    ref_frame = None

    for _ in progressbar(range(measure), bar='smooth'):
        ret, frame = cap.read()
        if not ret:
            break

        offset = int((cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) % cycle)
        if offset == 0:
            ref_frame = np.zeros_like(frame).astype(np.float_)
            frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)

            for _ in range(cycle):
                ref_frame += frame
                ret, frame = cap.read()
                if not ret:
                    break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ref_frame /= cycle
            ref_frame = ref_frame.astype(np.uint8)

        corr_frame = np.zeros_like(frame)
        for color in range(3):
            line_filter = LinearRegression(fit_intercept=True)
            line_filter.fit(frame[:, :, color].reshape(-1, 1), ref_frame[:, :, color].reshape(-1, 1))
            corr_frame[:, :, color] = predict_regression(frame[:, :, color], (line_filter.coef_[0][0], line_filter.intercept_[0]))

        orig_intensity.append(getIntensity(frame))
        corr_intensity.append(getIntensity(corr_frame))

        if show_frame:
            cv2.imshow('Anti-Filcker Result', np.hstack([corr_frame, frame]))
            cv2.waitKey(1)

    if show_frame:
        # intensity plot
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.plot(range(len(corr_intensity)), corr_intensity, '.')
        plt.plot(range(len(corr_intensity)), orig_intensity, '.')
        plt.xlabel('Frame number')
        plt.ylabel('Average intensity')
        plt.show()

# Make blended frames per cycle offset first, then train Linear Regression model
@ video_process(video_path)
def getLineFilter(cycle: int, show_plot: bool = False, cap: cv2.VideoCapture = None):
    assert type(cycle) is int, 'TYPE ERROR : CYCLE IS NOT INTEGER'
    assert cycle > 0, 'INVALID PARAMETER ERROR : CYCLE VALUE IS INVALID'
    assert measure > 10, 'INVALID PARAMETER ERROR : MEASURE VALUE IS INVALID'

    meanCycleImg = [0] * cycle
    numOverlap = [0] * cycle

    for idx in progressbar(range(measure), bar='smooth'):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, dsize=(0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_LINEAR)

        if idx < cycle:
            meanCycleImg[idx] = frame.astype(np.float64)
        else:
            numOverlap[idx % cycle] += 1
            meanCycleImg[idx % cycle] = np.add(meanCycleImg[idx % cycle], frame.astype(np.float64))

    # make reference image
    for idx in range(cycle):
        meanCycleImg[idx] = np.divide(meanCycleImg[idx], numOverlap[idx]).astype(np.uint8)

    for idx in range(cycle):
        if idx is 0:
            refCycleImg = meanCycleImg[idx].astype(np.float64)
        else:
            refCycleImg = np.add(meanCycleImg[idx].astype(np.float64), refCycleImg)
    refCycleImg = np.divide(refCycleImg, cycle).astype(np.uint8)

    ref_b, ref_g, ref_r = cv2.split(refCycleImg)
    flat_ref_b, flat_ref_g, flat_ref_r = ref_b.flatten(), ref_g.flatten(), ref_r.flatten()

    line_filter = [0] * cycle
    if show_plot:
        plt.figure()
    for idx in progressbar(range(cycle), bar='smooth'):
        b, g, r = cv2.split(meanCycleImg[idx])
        flat_b, flat_g, flat_r = b.flatten(), g.flatten(), r.flatten()

        line_filter_r = LinearRegression(fit_intercept=True)
        line_filter_g = LinearRegression(fit_intercept=True)
        line_filter_b = LinearRegression(fit_intercept=True)

        line_filter_r.fit(flat_r.reshape(-1, 1), flat_ref_r.reshape(-1, 1))
        line_filter_g.fit(flat_g.reshape(-1, 1), flat_ref_g.reshape(-1, 1))
        line_filter_b.fit(flat_b.reshape(-1, 1), flat_ref_b.reshape(-1, 1))

        line_filter[idx] = []
        line_filter[idx].append((line_filter_r.coef_[0][0], line_filter_r.intercept_[0]))
        line_filter[idx].append((line_filter_g.coef_[0][0], line_filter_g.intercept_[0]))
        line_filter[idx].append((line_filter_b.coef_[0][0], line_filter_b.intercept_[0]))

        if show_plot:
            # R color plot
            plt.subplot(3, cycle, idx + 1)
            plt.subplots_adjust(wspace=0.5, hspace=0.3)
            plt.xlim(0, 255)
            plt.ylim(0, 255)
            plt.title('Offset {}'.format(idx))
            plt.ylabel('Reference')
            plt.scatter(flat_r, flat_ref_r, c='red', marker='.', alpha=0.1, linewidths=1)

            predict_r = line_filter_r.predict(flat_r.reshape(-1, 1)).flatten()
            plt.plot(flat_r, predict_r, color='yellow', linewidth=1)

            # G color plot
            plt.subplot(3, cycle, cycle + idx + 1)
            plt.subplots_adjust(wspace=0.5, hspace=0.3)
            plt.xlim(0, 255)
            plt.ylim(0, 255)
            plt.ylabel('Reference')
            plt.scatter(flat_g, flat_ref_g, c='green', marker='.', alpha=0.1, linewidths=1)

            predict_g = line_filter_g.predict(flat_g.reshape(-1, 1)).flatten()
            plt.plot(flat_g, predict_g, color='yellow', linewidth=1)

            # B color plot
            plt.subplot(3, cycle, cycle * 2 + idx + 1)
            plt.subplots_adjust(wspace=0.5, hspace=0.3)
            plt.xlim(0, 255)
            plt.ylim(0, 255)
            plt.ylabel('Reference')
            plt.scatter(flat_b, flat_ref_b, c='blue', marker='.', alpha=0.1, linewidths=1)

            predict_b = line_filter_b.predict(flat_b.reshape(-1, 1)).flatten()
            plt.plot(flat_b, predict_b, color='yellow', linewidth=1)

    if show_plot:
        plt.show()

    return line_filter


# Return average pixel value
def getIntensity(image: np.ndarray) -> float:
    return np.sum(image) / np.size(image)


# After FFT, return video cycle
@ video_process(video_path)
def getPriorCycle(show_plot: bool = False, cap: cv2.VideoCapture = None) -> float:

    video_intensity = []
    for _ in progressbar(range(measure), bar='smooth'):
        ret, frame = cap.read()
        if not ret:
            break

        # Get average video pixel value
        video_intensity.append(getIntensity(frame))

    # Get frequency from FFT
    frequency = np.fft.fft(video_intensity)
    frequency = np.abs(np.fft.fftshift(frequency))

    # Draw plot about intensity per frame
    if show_plot:
        # intensity plot
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        plt.plot(range(measure), video_intensity, '.')
        plt.xlabel('Frame number')
        plt.ylabel('Average intensity')

        # frequency plot
        plt.subplot(2, 1, 2)
        plt.plot(range(int(-measure / 2), int(measure / 2)), frequency, '-')
        plt.xlabel('Frequency')
        plt.show()

    # Get prior frequency from max frequency (except 0)
    prior_frequency = np.argmax(frequency[int(measure / 2) + 1:])

    # Get frame cycle
    cycle = math.floor(measure / prior_frequency)

    # Cycle intensity float
    if show_plot:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        offset = [math.fmod(x, cycle) for x in range(len(video_intensity))]
        ax.scatter(range(len(video_intensity)), video_intensity, c=offset, cmap='hsv')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Frame Intensity')
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        offset = [math.fmod(x, cycle) for x in range(len(video_intensity))]
        ax.scatter(range(len(video_intensity)), video_intensity, offset, c=offset, cmap='hsv')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Frame Intensity')
        ax.set_zlabel('Offset')
        plt.show()

    # Return prior cycle
    return cycle


if __name__ == '__main__':
    main()
    print('Finish')