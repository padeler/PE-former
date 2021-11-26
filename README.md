# POTR: Pose estimation transformer
Vision transformer architectures have been demonstrated to work 
for image classification tasks. Efforts to solve more challenging vision tasks with transformers rely on convolutional backbones for feature extraction.

POTR is a pure transformer architecture (no CNN backbone) for 2D body pose estimation.

You can use the code in this repository to train and evaluate different POTR configurations on the COCO dataset.

## Acknowledgements
The code in this repository is based on the following:

- [End-to-End Object Detection with Transformers (DETR)](https://github.com/facebookresearch/detr)
Xcit
- [Cross-Covariance Image Transformer (XCiT)](https://github.com/facebookresearch/xcit)

- [DeiT: Data-efficient Image Transformers](https://github.com/facebookresearch/deit)

- [Simple Baselines for Human Pose Estimation and Tracking](https://github.com/microsoft/human-pose-estimation.pytorch)

- [Pytoch image models (TIMM)](https://github.com/rwightman/pytorch-image-models)


Thank you!

## Preparing

Create a python venv and install all the dependenciesQ

```bash
python -m venv pyenv
source pyenv/bin/activate
pip install -r requirements.txt
```

## Training 

Training POTR with a __deit_small__ encoder, patch size of __16x16__ pixels and input resolution __192x256__:

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

Check the ```lit_main.py``` cli arguments for a complete list.
```bash
python lit_main.py --help
```

## Evaluation

Evaluate a trained model using the ```evaluate.py``` script.

For example to evaluate POTR trained with an xcit_small_12_p8 encoder:

```
python evaluate.py --vit_arch xcit_small_12_p8 --patch_size 8 --batch_size 42 --input_size 192 256 --hidden_dim 384 --vit_dim 384  --position_embedding enc_xcit --num_workers 16 --num_queries 100 --dim_feedforward 1536 --init_weights paper_experiments/xcit_small12_p8_dino_192_256_paper/checkpoints/checkpoint-epoch\=065-AP\=0.736.ckpt --use_det_bbox
```

Set the argument of --init_weights to your model's checkpoint.


## Experiments








