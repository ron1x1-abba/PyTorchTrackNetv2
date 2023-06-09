import numpy as np
import pandas as pd
import cv2

from pathlib import Path
from typing import Union
from tqdm.auto import tqdm


def generate_heat_map(width, height, x, y, sigma, mag):
    """
    Generate heatmap by a markup.
    :param width: width of heatmap.
    :param height: height of heatmap.
    :param x: x-pixel of ball markup.
    :param y: y-pixel of ball markup.
    :param sigma: radius of ball in heatmap.
    :param mag: heatmap magnitude. (i.e instead of ones on ball you will get mag values)
    :return: (H, W)
    """
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
):
    imgs = []
    targets = []

    cur_frame = -1
    cur_label = 0
    w_ratio, h_ratio = 1 / width, 1 / height

    frames = []

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), leave=False) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:  # Reached the end of video
                break

            cur_frame += 1
            pbar.update(1)

            if cur_frame - len(frames) != nums[cur_label]:
                if nums[cur_label] < cur_frame - len(frames):
                    cur_label += 1
                frames = []
                continue

            if cur_label + (consecutive_frames - 1) >= len(nums):
                break

            if nums[cur_label] + (consecutive_frames - 1) != nums[cur_label + (consecutive_frames - 1)]:
                cur_label += 1
                frames = []
                continue

            frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))

            if len(frames) == 1:
                for _ in range(consecutive_frames - 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
                    cur_frame += 1
                    pbar.update(1)
                if not ret:
                    break

            imgs.append(frames)  # shape (3 * cons_frames, H, W)

            tmp = []

            for j in range(consecutive_frames):
                tmp.append((-1, -1, 0) if tgt[cur_label + j] == 0 else
                           (int(x[cur_label + j] / w_ratio), int(y[cur_label + j] / h_ratio), 1))

            targets.append(tmp)
            frames = frames[1:]
            cur_label += 1

    return imgs, targets


def generate_data(video_path: Union[str, Path], label_path: [str, Path],
                  width: int, height: int, consecutive_frames: int = 3,
                  same_in_out: bool = True):
    """
    Generate heatmaps for video frames by given .csv markup.
    Images returned in BGR format.
    :param video_path: path to video file (.mp4).
    :param label_path: path to .csv markup.
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

    imgs, targets = generate_same(cap, nums, x, y, tgt, width, height, consecutive_frames)
    if not same_in_out:
        targets = [x[[-1], :, :] for x in targets]

    cap.release()
    cv2.destroyAllWindows()
    return imgs, targets
