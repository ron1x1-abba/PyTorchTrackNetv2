import numpy as np
import pandas as pd
import cv2

from pathlib import Path
from typing import Union


def generate_heat_map(width, height, x, y, sigma, mag):
    if x < 0 or y < 0:
        return np.zeros((height, width))
    cx, cy = np.meshgrid(np.linspace(1, width, width), np.linspace(1, height, height))
    heatmap = (cy - (y + 1)) ** 2 + (cx - (x + 1)) ** 2
    heatmap[heatmap <= sigma ** 2] = 1
    heatmap[heatmap > sigma ** 2] = 0
    return heatmap * mag


def generate_same(
    cap: cv2.VideoCapture,
    nums,
    x,
    y,
    tgt,
    width,
    height,
    consecutive_frames,
    sigma,
    mag
):
    imgs = []
    targets = []

    cur_frame = -1
    cur_label = 0
    w_ratio, h_ratio = None, None

    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # Reached the end of video
            break
        if w_ratio is None:
            w_ratio = frame.shape[1] / width
            h_ratio = frame.shape[0] / height
        cur_frame += 1

        if cur_frame - len(frames) != nums[cur_label]:
            frames = []
            continue

        if nums[cur_label] + (consecutive_frames - 1) != nums[cur_label + (consecutive_frames - 1)]:
            cur_label += 1
            frames = []
            continue

        frames.append(np.transpose(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), [2, 0, 1]))

        if len(frames) == 1:
            for _ in range(consecutive_frames - 1):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(np.transpose(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA),
                                           [2, 0, 1]))
                cur_frame += 1
            if not ret:
                break

        imgs.append(np.concatenate(frames, axis=0))  # shape (3 * cons_frames, H, W)

        tmp = []

        for j in range(consecutive_frames):
            if tgt[cur_label + j] == 0:
                tmp.append(generate_heat_map(width, height, -1, -1, sigma, mag)[None, :, :])
            else:
                tmp.append(generate_heat_map(width, height,
                                             int(x[cur_label + j] / w_ratio),
                                             int(y[cur_label + j] / h_ratio),
                                             sigma, mag)[None, :, :])

        targets.append(np.concatenate(tmp, axis=0))  # shape (cons_frames, H, W)
        frames = frames[1:]
        cur_label += 1

    return imgs, targets


def generate_data(video_path: Union[str, Path], label_path: [str, Path],
                  sigma: float, mag: float,
                  width: int, height: int, consecutive_frames: int = 3,
                  same_in_out: bool = True):
    """
    Generate heatmaps for video frames by given .csv markup.
    Images returned in BGR format.
    :param video_path: path to video file (.mp4).
    :param label_path: path to .csv markup.
    :param sigma: radius of ball in heatmap.
    :param mag: heatmap magnitude. (i.e instead of ones on ball you will get mag values)
    :param width: width of images which will be used in your model.
    :param height: height of images which will be used in your model.
    :param consecutive_frames: num of consecutive frames to detect ball.
    :param same_in_out: whether to have for each consecutive frames same amount of labels or not.
    :return: imgs -- list of np.array's of shape (consecutive_frames * 3, H, W), targets -- list of np.array's of shape
    (consecutive_frames, H, W).
    """

    if consecutive_frames < 1:
        raise ValueError(f"Amount of consecutive frames must be >= 1, but got {consecutive_frames}!")

    labels = pd.read_csv(str(label_path))

    x = labels.x.values
    y = labels.y.values
    tgt = labels.visible.values
    nums = labels.frame_num.values

    cap = cv2.VideoCapture(str(video_path))

    imgs, targets = generate_same(cap, nums, x, y, tgt, width, height, consecutive_frames, sigma, mag)
    if not same_in_out:
        targets = [x[[-1], :, :] for x in targets]

    cap.release()
    cv2.destroyAllWindows()
    return imgs, targets
