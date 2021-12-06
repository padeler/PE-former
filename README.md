# POTR: Pose estimation transformer
Vision transformer architectures perform very well for image classification tasks. Efforts to solve more challenging vision tasks with transformers rely on convolutional backbones for feature extraction.

POTR is a pure transformer architecture (no CNN backbone) for 2D body pose estimation. It uses an encoder-decoder architecture with a vision transformer as an encoder and a transformer decoder (derived from DETR).

You can use the code in this repository to train and evaluate different POTR configurations on the COCO dataset.

## Model

POTR is based on building blocks derived from recent SOTA models. As shown in the figure there are two major components: A Visual Transformer encoder, and a Transformer decoder.

![model](/figures/model_draft1.jpg)

The input image is initially converted into tokens following the ViT paradigm. A position embedding is used to help retain the patch-location information. The tokens and the position embedding are used as input to transformer encoder. The transformed tokens are used as the memory input of the transformer decoder.
The inputs of the decoder are _M_ learned queries.
For each query the network will produce a joint prediction.
The output tokens from the transformer decoder are passed through two heads (FFNs). 
 - The first is a classification head used to predict the joint type (i.e class) of each query.
 - The second is a regression head that predicts the normalized coordinates (in the range [0,1]) of the joint in the input image.

Predictions that do not correspond to joints are mapped to a "no object" class.


## Acknowledgements
The code in this repository is based on the following:

- [End-to-End Object Detection with Transformers (DETR)](https://github.com/facebookresearch/detr)
- [Cross-Covariance Image Transformer (XCiT)](https://github.com/facebookresearch/xcit)
- [DeiT: Data-efficient Image Transformers](https://github.com/facebookresearch/deit)
- [Simple Baselines for Human Pose Estimation and Tracking](https://github.com/microsoft/human-pose-estimation.pytorch)
- [Pytoch image models (TIMM)](https://github.com/rwightman/pytorch-image-models)

Thank you!

## Preparing

Create a python venv and install all the dependencies:

```bash
python -m venv pyenv
source pyenv/bin/activate
pip install -r requirements.txt
```

## Training 
Here are some CLI examples using the ```lit_main.py```
script.

Training POTR with a deit_small encoder, patch size of 16x16 pixels and input resolution 192x256:

```bash
python lit_main.py --vit_arch deit_deit_small --patch_size 16 --batch_size 42 --input_size 192 256 --hidden_dim 384 --vit_dim 384 --gpus 1 --num_workers 24
```

POTR with Xcit_small_p16 encoder:

```bash
 python lit_main.py --vit_arch xcit_small_12_p16 --batch_size 42 --input_size 288 384 --hidden_dim 384 --vit_dim 384 --gpus 1 --num_workers 24   --vit_weights https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth

```

POTR with the ViT as Backbone (VAB) configuration:

```bash
 python lit_main.py --vit_as_backbone --vit_arch resnet50 --batch_size 42 --input_size 192 256 --hidden_dim 384 --vit_dim 384 --gpus 1 --position_embedding learned_nocls --num_workers 16 --num_queries 100 --dim_feedforward 1536 --accumulate_grad_batches 1
```

Baseline that uses a resnet50 (pretrained with dino) as an encoder:

```bash
 python lit_main.py --vit_arch resnet50 --patch_size 16 --batch_size 42 --input_size 192 256 --hidden_dim 384 --vit_dim 384 --gpus 1 --num_workers 24 --vit_weights https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth --position_embedding learned_nocls
 ```

Check the ```lit_main.py``` cli arguments for a complete list.
```bash
python lit_main.py --help
```

## Evaluation

Evaluate a trained model using the ```evaluate.py``` script.

For example to evaluate POTR with an xcit_small_12_p8 encoder:

```bash
python evaluate.py --vit_arch xcit_small_12_p8 --patch_size 8 --batch_size 42 --input_size 192 256 --hidden_dim 384 --vit_dim 384  --position_embedding enc_xcit --num_workers 16 --num_queries 100 --dim_feedforward 1536 --init_weights paper_experiments/xcit_small12_p8_dino_192_256_paper/checkpoints/checkpoint-epoch\=065-AP\=0.736.ckpt --use_det_bbox
```

Evaluate POTR with a deit_small encoder:

```bash
 python evaluate.py --vit_arch deit_deit_small --patch_size 16 --batch_size 42 --input_size 192 256 --hidden_dim 384 --vit_dim 384 --num_workers 24 --init_weights lightning_logs/version_0/checkpoints/checkpoint-epoch\=074-AP\=0.622.ckpt  --use_det_bbox
```

Set the argument of --init_weights to your model's checkpoint.


## Model Zoo

|name                | input   | params| AP   | AR   | url     |
| ---                | ---     | ---   | ---  | ---  | ---     |
| POTR-Deit-dino-p8  | 192x256 | 36.4M | 70.6 | 78.1 | [model](https://drive.google.com/file/d/1WaycMXmGxqNJngPnItDux83e1wGPxGsG/view?usp=sharing) |
| POTR-Xcit-p16      | 288x384 | 40.6M | 70.2 | 77.4 | [model](https://drive.google.com/file/d/1UZtYrdNdzCBSERExkzgt_f_q-Asyq7FM/view?usp=sharing) |
| POTR-Xcit-dino-p16 | 288x384 | 40.6M | 70.7 | 77.9 | [model](https://drive.google.com/file/d/1ArLTovtHIPJWIits3D2TRI2iZwta4DLZ/view?usp=sharing) | 
| POTR-Xcit-dino-p8  | 192x256 | 40.5M | 71.6 | 78.7 | [model](https://drive.google.com/file/d/1f4wdR-YDXXz3UrxrscsKsxg0EIR43NUR/view?usp=sharing) | 
| POTR-Xcit-dino-p8  | 288x384 | 40.5M | 72.6 | 79.4 | [model](https://drive.google.com/file/d/1NQCyhJr-uagO4Y0fiqKv4pm9LzxPb-HD/view?usp=sharing) 
   


Check the [experiments](/experiments) folder for configuration files and evaluation results.

All trained models and tensorboard training logs can be downloaded from this [drive folder](https://drive.google.com/drive/folders/1LFszPEva0QmWTPqqSbxMmB_GqNhjm5kq?usp=sharing).





# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
