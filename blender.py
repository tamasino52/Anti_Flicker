import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import progressbar
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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channer = 3

    bar = progressbar.ProgressBar(maxval=measure).start()
    while True:
        cycle_frames = []
        ref_frame = np.zeros((height, width, channer), dtype=np.float_)
        ret = False
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)

        for _ in range(cycle):
            ret, frame = cap.read()
            if not ret:
                break
            cycle_frames.append(frame)

        for frame in cycle_frames:
            ref_frame += frame
        ref_frame /= len(cycle_frames)
        ref_frame = ref_frame.astype(np.uint8)

        corr_frames = []
        for offset in range(len(cycle_frames)):
            corr_frame = np.zeros((height, width, channer), dtype=np.uint8)
            for color in range(channer):
                line_filter = LinearRegression(fit_intercept=True)
                line_filter.fit(cycle_frames[offset][:, :, color].reshape(-1, 1), ref_frame[:, :, color].reshape(-1, 1))
                corr_frame[:, :, color] = predict_regression(cycle_frames[offset][:, :, color],
                                                             (line_filter.coef_[0][0], line_filter.intercept_[0]))
            corr_frames.append(corr_frame)

        for offset in range(len(cycle_frames)):
            orig_intensity.append(getIntensity(cycle_frames[offset]))
            corr_intensity.append(getIntensity(corr_frames[offset]))
            if show_frame:
                cv2.imshow('Anti-Filcker Result', np.hstack([corr_frames[offset], cycle_frames[offset]]))
                cv2.waitKey(1)
            bar.update(int(frame_idx + offset))

        if not ret:
            bar.finish()
            break

    if show_frame:
        # intensity plot
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.plot(range(len(corr_intensity)), corr_intensity, '.')
        plt.plot(range(len(corr_intensity)), orig_intensity, '.')
        plt.xlabel('Frame number')
        plt.ylabel('Average intensity')
        plt.show()


# Return average pixel value
def getIntensity(image: np.ndarray) -> float:
    return np.sum(image) / np.size(image)


# After FFT, return video cycle
@ video_process(video_path)
def getPriorCycle(show_plot: bool = False, cap: cv2.VideoCapture = None) -> float:

    video_intensity = []
    for _ in progressbar.progressbar(range(measure)):
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

    # Return prior cycle
    return cycle


if __name__ == '__main__':
    main()
    print('Finish')
