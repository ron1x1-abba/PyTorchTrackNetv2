import argparse
import glob
import os
from tqdm import tqdm
import numpy as np
import cv2

from tracknetv2.dataset import generate_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="videos", help="Path to directory with all video data.")
    parser.add_argument("--label_path", type=str, default="markups", help="Path to directory with all markup data.")
    parser.add_argument("--output_path", type=str, default="train_data", help="Path to directory where to save \
    processed data.")
    parser.add_argument("--width", type=int, default=1280, help="Width of images which will be used in model.")
    parser.add_argument("--height", type=int, default=720, help="Height of images which will be used in model.")
    parser.add_argument("--consecutive_frames", type=int, default=3, help="Num of consecutive frames which will be\
     used by model.")
    parser.add_argument("--same_in_out", action='store_true', help="Whether to have for each consecutive frames same\
     amount of labels or not. If not used will generate only 1 heatmap as target for each n consecutive frames.")
    return parser.parse_args()


def create_dataset(args):
    videos = glob.glob(args.video_path + "/*.mp4")

    final_imgs = []
    final_heatmaps = []

    print("Start processing data...")
    for video in tqdm(videos):
        csv = os.path.join(args.label_path, video.split('/')[-1].replace('.mp4', '.csv'))
        imgs, tgts = generate_data(
            video_path=video,
            label_path=csv,
            width=args.width,
            height=args.height,
            consecutive_frames=args.consecutive_frames,
            same_in_out=args.same_in_out
        )
        final_imgs += imgs
        final_heatmaps += tgts

    assert len(final_imgs) == len(final_heatmaps)

    save_path = args.output_path
    os.makedirs(save_path, exist_ok=True)
    print(f"Start saving examples to {save_path}")
    for i, (imgs, heatmap) in tqdm(enumerate(zip(final_imgs, final_heatmaps)), total=len(final_imgs)):
        path = os.path.join(save_path, str(i))
        os.makedirs(path, exist_ok=True)
        for j, img in enumerate(imgs):
            cv2.imwrite(os.path.join(path, f"{j}.jpg"), img)
        with open(os.path.join(path, "heatmap.txt"), "w+") as f:
            f.write('\t'.join([','.join([str(y) for y in x]) for x in heatmap]))

    print("Finish processing data!")
    print(f"Count {len(final_heatmaps)} valid examples.")


if __name__ == "__main__":
    args = parse_args()
    create_dataset(args)
