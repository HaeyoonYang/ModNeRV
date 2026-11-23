# ModNeRV
Official implementation for [Efficient Implicit Neural Representations for Videos with Feature Modulation (ModNeRV)](https://ieeexplore.ieee.org/document/11084530)  (ICIP 2025)


## Prepare the dataset
This implementation requires first converting the videos into PNGs. For example, you can use [FFMpeg](https://www.ffmpeg.org/):
```
mkdir {video_name}
ffmpeg -video_size 1920x1080 -pixel_format yuv420p -i video.yuv {video_name}/%4d.png
```

The UVG dataset can be downloaded [here](https://ultravideo.fi/dataset.html).


## Training
See ```run.sh``` for example

```
python train.py --opt_path {} --dataset {} --epochs {}
```
Specify path for opt, dataset name, and number of epochs.
Change the path to data in the opt file.

```cfgs/uvg-SND-*.yaml``` files are for 'ShakeNDry' dataset.


## Evaluation
```
python eval.py --test_folder {} --visualize
```
Specify path for evaluation.
If you want to save visualized images, add ```--visualize```.


## Compression
You can further fine-tune the model before video compression with ```compress_train.py```.

```compress_train.py```: the model is fine-tuned after pruning and quantization.
Specify pruning & quantization options in ```compress_train.py``` and run it.


```compress_eval.py```: you can compress either the overfitted model or the fine-funed model.
Specify ```--test_folder``` and ```--quant_bit```.
