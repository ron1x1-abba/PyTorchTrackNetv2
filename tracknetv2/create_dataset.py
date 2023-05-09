import argparse
import glob
import os
from tqdm import tqdm
import numpy as np

from tracknetv2.dataset import generate_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="videos", help="Path to directory with all video data.")
    parser.add_argument("--label_path", type=str, default="markups", help="Path to directory with all markup data.")
    parser.add_argument("--output_path", type=str, default="train_data", help="Path to directory where to save \
    processed data.")
    parser.add_argument("--width", type=int, default=1280, help="Width of images which will be used in model.")
    parser.add_argument("--height", type=int, default=720, help="Height of images which will be used in model.")
    parser.add_argument("--magnitude", type=float, default=2.0, help="Magnitude for heatmap.")
    parser.add_argument("--sigma", type=float, default=2.5, help="Radius of ball on heatmap.")
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
            sigma=args.sigma,
            mag=args.magnitude,
            width=args.width,
            height=args.height,
            consecutive_frames=args.consecutive_frames,
            same_in_out=args.same_in_out
        )
        final_imgs += imgs
        final_heatmaps += tgts

    save_path = args.output_path
    os.makedirs(save_path, exist_ok=True)
    print(f"Start saving examples to {save_path}")
    for i, img in enumerate(final_imgs):
        with open(os.path.join(save_path, f"img_{i}.npy"), 'wb') as f:
            np.save(f, img)
    for i, heatmap in enumerate(final_heatmaps):
        with open(os.path.join(save_path, f"heatmap_{i}.npy"), 'wb') as f:
            np.save(f, heatmap)

    print("Finish processing data!")
    print(f"Count {len(final_heatmaps) / (args.consecutive_frames if args.same_in_out else 1)} valid examples.")


if __name__ == "__main__":
    args = parse_args()
    create_dataset(args)
