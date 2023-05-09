# PyTorchTrackNetv2
An unofficial implementation of [TrackNetv2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2) in PyTorch.  

## Installation
```
git clone https://github.com/ron1x1-abba/PyTorchTrackNetv2.git
cd PyTorchtrackNetv2
pip install -r requirements.txt
python setup.py install
```

## Label Data
Keybindings:
- <kbd>l</kbd> : next frame
- <kbd>h</kbd> : previous frame
- <kbd>v</kbd> : visible mode
- <kbd>o</kbd> : occluded mode
- <kbd>m</kbd> : motion mode
- <kbd>f</kbd> : forward/pause
- <kbd>x</kbd> : delete annotation
- <kbd>q</kbd> : finish and save annotations
```
python labelling_tool.py <path_to_video>.mp4
```

## Generate Train/Val Data
```
python create_dataset.py --video_path=<directory_with_video> --label_path=<directory_with_csv_markup> --same_in_out
```
### Available Options
| Name | Type | Default | Description                                                                                                                                                 |
|------|------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | video_path | str  | videos | Path to video files in .mp4 format                                                                                                                          |
 | label_path | str  | markups | Path to .csv files with markup                                                                                                                              |
 | output_path | str  | train_data | Path where to save train data.                                                                                                                              |
 | width | int  | 1280 | Width of images which will be used in model.                                                                                                                |
 | height | int | 720 | Height of images which will be used in model.                                                                                                               |
 | magnitude | float | 2.0 | Magnitude for heatmap.                                                                                                                                      |
 | sigma | float | 2.5 | Radius of ball on heatmap.                                                                                                                                  |
 | same_in_out | bool | False | Whether to have for each consecutive frames same amount of labels or not. If not used will generate only 1 heatmap as target for each n consecutive frames. |
 | consecutive_frames | int | 3 | Num of consecutive frames which will be used by model. |

Args without default values are 'store-action' args.

## Train model

```
python train.py --train_data=<directory_with_train_data> --val_data=<directory_with_val_data> \
 --save_path=<path_to_save_weights> --logdir=<path_to_save_logs>
```
### Available options
| Name | Type | Default | Description                        |
|------|------|---------|------------------------------------|
 | train_data | str | train | Path to directory with train data. |
 | val_data | str | val | Path to directory with val data.   |
 | save_path | str | weights | Path to directory where to save model weights. |
 | logdir | str | mylogs | Path to directory with training logs. |
 | train_config | str | configs/train_config.json | Path to .json config with training info. |
 | model_config | str | configs/model_config.json | Path to .json config with model info. |

All available values for configs presented in files configs/*.json . Except for "optimizer_config".
You can use in it any key/value pair compatible with pytorch optimizers.

## Prediction
