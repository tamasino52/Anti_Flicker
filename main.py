import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import math
import time
import os
from sklearn.linear_model import LinearRegression

# Flag option setting
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", default="./video/C0183.MP4", help="Process a path of video.")
parser.add_argument("--no_display", default=False, type=bool, help="Enable to disable the visual display.")
parser.add_argument("-o", "--output", default='./output', help="Process a path of output video")
args = parser.parse_known_args()

# Get flag option value
video_path = args[0].video
output_path = os.path.join(args[0].output, os.path.split(video_path)[1])

# Set video variables
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), 'Unable to load video. check your video path.'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
prop_fps = int(cap.get(cv2.CAP_PROP_FPS))
prop_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
prop_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
prop_channer = 3
prop_size = (prop_height, prop_width, prop_channer)
out = cv2.VideoWriter(output_path, fourcc, prop_fps, (prop_width, prop_height))
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # number of total frame
bar = progressbar.ProgressBar(maxval=total_frame)

# Information
print('INPUT PATH = {}\nPROPS FPS = {}\nPROPS SIZE = ({}, {})\nTOTAL FRAME = {}\n'.format(
    video_path, prop_fps, prop_height, prop_width, total_frame))


def main():
    # Get intensity cycle of video
    cycle = getPriorCycle(show_plot=True)
    print('VIDEO CYCLE = {}\n'.format(cycle))

    # Get Linear Regression filter
    removeFlicker(cycle, show_plot=True)


# Open video
def open_video(func):
    def decorator(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return decorator


# Show video frames removed flicker effect
def removeFlicker(cycle: int, show_plot: bool = False):
    assert cycle > 0, 'INVALID PARAMETER ERROR : CYCLE VALUE IS INVALID'

    def predict_regression(X, regression):
        coef, intercept = regression[0], regression[1]
        y = X.astype(np.float64)
        y = y * coef + intercept
        y[y > 255] = 255
        y[y < 0] = 0
        return y.astype(np.uint8)

    corr_intensity = []  # corrected video intensity
    orig_intensity = []  # original video intensity

    # Video loop
    print('Removing all flicker ...')
    cap.open(video_path)
    bar.start()
    while True:
        # Variables initialization
        cycle_frames = []  # original frame list in same cycle
        corr_frames = []  # corrected frame list in same cycle
        ref_frame = np.zeros(prop_size, dtype=np.float_)  # mean of frames in same cycle
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)  # frame number
        ret = False  # frame validity

        # Scan frame cycle
        for _ in range(cycle):
            ret, frame = cap.read()
            if not ret:
                break
            cycle_frames.append(frame)

        # Make reference frame
        for frame in cycle_frames:
            ref_frame += frame
        ref_frame /= len(cycle_frames)
        ref_frame = ref_frame.astype(np.uint8)

        # Flicker removal using linear regression
        for offset in range(len(cycle_frames)):
            corr_frame = np.zeros((prop_height, prop_width, prop_channer), dtype=np.uint8)
            for color in range(prop_channer):
                line_filter = LinearRegression(fit_intercept=True)
                line_filter.fit(cycle_frames[offset][:, :, color].reshape(-1, 1), ref_frame[:, :, color].reshape(-1, 1))
                corr_frame[:, :, color] = predict_regression(cycle_frames[offset][:, :, color],
                                                             (line_filter.coef_[0][0], line_filter.intercept_[0]))
            corr_frames.append(corr_frame)

        # Show result and save intensity
        for offset in range(len(cycle_frames)):
            orig_intensity.append(getIntensity(cycle_frames[offset]))
            corr_intensity.append(getIntensity(corr_frames[offset]))
            if show_plot:
                cv2.imshow('Anti-Filcker Result', np.hstack([corr_frames[offset], cycle_frames[offset]]))
                cv2.waitKey(1)
            bar.update(int(frame_idx + offset))

            # Write corrected video
            out.write(corr_frames[offset])

        if not ret:
            bar.finish()
            break

    # Make plot
    if show_plot:
        # intensity plot
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.plot(range(len(corr_intensity)), corr_intensity, '.')
        plt.plot(range(len(corr_intensity)), orig_intensity, '.')
        plt.xlabel('Frame number')
        plt.ylabel('Average intensity')
        plt.show()


# Return average pixel value of single frame
def getIntensity(image: np.ndarray) -> float:
    return np.sum(image) / np.size(image)


# Return flicker cycle of video
def getPriorCycle(show_plot: bool = False) -> float:
    video_intensity = []  # frame intensity list

    # Read all frame to get intensity
    print('Scanning all frame Intensity ...')
    cap.open(video_path)
    bar.start()
    for idx in range(total_frame):
        bar.update(idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Get average video pixel value
        video_intensity.append(getIntensity(frame))
    bar.finish()

    # Get frequency from FFT
    frequency = np.fft.fft(video_intensity)
    frequency = np.abs(np.fft.fftshift(frequency))

    # Draw plot about intensity per frame
    if show_plot:
        # Intensity plot
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(range(total_frame), video_intensity, '.')
        plt.xlabel('Frame number')
        plt.ylabel('Average intensity')

        # Frequency plot
        plt.subplot(2, 1, 2)
        plt.plot(range(int(-total_frame / 2), int(total_frame / 2)), frequency, '-')
        plt.title('FFT result')
        plt.xlabel('Frequency')
        plt.show()

    # Get prior frequency from max frequency (but, x > 0)
    prior_frequency = np.argmax(frequency[int(total_frame / 2) + 1:])

    # Get frame cycle from frequency
    cycle = math.floor(total_frame / prior_frequency)

    # Intensity classification
    if show_plot:
        plt.figure()
        offset = [math.fmod(x, cycle) for x in range(len(video_intensity))]
        plt.scatter(range(len(video_intensity)), video_intensity, c=offset, marker='.', cmap='hsv')
        plt.colorbar(label='offset')
        plt.title('Intensity classification by Offset')
        plt.xlabel('Frame Number')
        plt.ylabel('Intensity')
        plt.show()

    return cycle


if __name__ == '__main__':
    main()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Finish')
