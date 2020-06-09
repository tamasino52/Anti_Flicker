import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import progressbar
from progressbar import progressbar as pbar
import math
import time
import os
from sklearn.linear_model import LinearRegression

# Flag option setting
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", default="./video/C0181.MP4", help="path of original video.")
parser.add_argument("-i", "--interim", default='./interim', help="path of interim video")
parser.add_argument("-o", "--output", default='./output', help="path of output video")
parser.add_argument("--show_plot", required=False, help="Enable to display plot.")
args = parser.parse_known_args()

# Path initilization
video_path = args[0].video
output_path = os.path.join(args[0].output, os.path.split(video_path)[1])
interim_path = os.path.join(args[0].interim, os.path.split(video_path)[1])
show_plot = args[0].show_plot

def main():

    if not os.path.exists(interim_path):
        # Prior flicker cycle extraction
        cycle = getPriorCycle(video_path, show_plot=show_plot)
        print('FLICKER CYCLE = {}\n'.format(cycle))

        # Luminance Equalization
        equalLuminance(video_path, interim_path, cycle, show_plot=show_plot)

    if os.path.exists(interim_path):
        # Frame blending
        blendFrames(interim_path, output_path, kernel_size=7)


def blendFrames(input_path: str, output_path: str, kernel_size: int):
    # Video setting
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), 'Unable to load video. check your video path.'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    prop_fps = int(cap.get(cv2.CAP_PROP_FPS))
    prop_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    prop_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, prop_fps, (prop_width, prop_height))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # number of total frame

    # Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma=1)

    # frame padding (0 and EOF)
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frame-1)
    ret, last_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = [first_frame] * int(kernel_size / 2)

    print('Frame Blending ...')
    for _ in pbar(range(total_frame)):
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


# Luminance equalization of video
def equalLuminance(input_path: str, output_path: str, cycle: int, show_plot: bool = False):
    assert cycle > 0, 'INVALID PARAMETER ERROR : CYCLE VALUE IS INVALID'

    def predict_regression(X, regression):
        coef, intercept = regression[0], regression[1]
        y = X.astype(np.float64)
        y = y * coef + intercept
        y[y > 255] = 255
        y[y < 0] = 0
        return y.astype(np.uint8)

    # Video setting
    cap = cv2.VideoCapture(input_path)
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

    # Initialize loop
    corr_intensity = []  # corrected video intensity
    orig_intensity = []  # original video intensity
    print('Luminance Equalizing ...')
    bar.start()

    while True:
        # Variables initialization
        cycle_frames = []  # original frame list in same cycle
        corr_frames = []  # corrected frame list in same cycle
        ref_frame = np.zeros(prop_size, dtype=np.float_)  # mean of frames in same cycle
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)  # frame number

        # Scan one cycle
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

        # Linear regression
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
        plt.scatter(range(len(corr_intensity)), corr_intensity, marker='.', label='Corrected video')
        plt.scatter(range(len(corr_intensity)), orig_intensity, marker='.', label='Original video')
        plt.legend(loc='upper left')
        plt.xlabel('Frame number')
        plt.ylabel('Intensity')
        plt.show()

    # Destroyer
    cap.release()
    out.release()


# Return average pixel value of single frame
def getIntensity(image: np.ndarray) -> float:
    return np.sum(image) / np.size(image)


# Return flicker cycle of video
def getPriorCycle(input_path: str, show_plot: bool = False) -> float:
    # Set video variables
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), 'Unable to load video. check your video path.'
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # number of total frame
    bar = progressbar.ProgressBar(maxval=total_frame)
    video_intensity = []  # frame intensity list

    # Read all frame to get intensity
    print('Scanning all frame Intensity ...')
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
    print('Finish')
