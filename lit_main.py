
import argparse

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from datasets.coco_person import build as build_coco_person
from datasets import get_coco_api_from_dataset


import util.misc as utils
from models import lit_vitdetr

# PPP Fixes issue with "too many files open" during training.
# ulimit -n 1000000


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=42, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--vit_arch', default="dino_deit_small", type=str)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--debug', action='store_true', help="for faster debugging")

    # Model parameters
    parser.add_argument('--init_weights', type=str, default=None,
                        help="Path to the pretrained model.")

    parser.add_argument('--position_embedding', default='enc_xcit',
                        type=str, choices=('enc_sine', 'enc_learned', 'enc_xcit',
                                           'learned_cls', 'learned_nocls', 'none'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--activation', default='gelu', type=str, choices=('relu', 'gelu', "glu"),
                        help="Activation function used for the transformer decoder")

    parser.add_argument('--vit_as_backbone', action='store_true', help="Use VIT as the backbone of DETR, instead of the encoder part in vitdetr")
    parser.add_argument('--input_size', nargs="+", default=[224, 224], type=int,
                        help="Input image size. Default is %(default)s")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings for the DETR transformer")
    # PPP When VIT is used as a backbone this argument only affects the backbone.
    # The DETR transformer still has the same hidden_dims 
    # (controlled by the transformer.d_model value)
    # When using vitdetr (no backbone) vit_dim must be equal to hidden_dim
    parser.add_argument('--vit_dim', default=384, type=int,
                        help="Output token dimension of the VIT")
    parser.add_argument('--vit_weights', type=str, default=None,
                        help="Path to the weights for vit (must match the vit_arch, input_size and patch_size).")
    parser.add_argument('--vit_dropout', default=0., type=float,
                        help="Dropout applied in the vit backbone")

    # * Transformer
    parser.add_argument('--dec_arch', default="detr", type=str, choices=('xcit', 'detr'))
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1536, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0., type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--with_lpi', action='store_true',
                        help="For the xcit decoder. Use lpi in decoder blocks")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)

    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default="/home/padeler/work/datasets/coco_2017", type=str)

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_det_bbox', action='store_true', help='For keypoints detection, use person detected \
                        bboxes (from json file) for evaluation')
    parser.add_argument('--scale_factor', default=0.3, type=float, help="Augmentation scaling parameter \
                                                            (default from simple baselines is %(default)s)")

    parser.add_argument('--num_workers', default=16, type=int)

    return parser


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    # fix the seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Make input size a tuple of width,height
    input_size = args.input_size
    if len(input_size) == 1:
        input_size = 2 * input_size
    else:
        input_size = input_size[:2]
    args.input_size = tuple(input_size)

    dataset_val = build_coco_person(image_set='val', args=args)
    if not args.debug and not args.eval:
        dataset_train = build_coco_person(image_set='train', args=args)
    else:
        dataset_train = dataset_val

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    pin_memory = False
    if args.gpus is not None:
        pin_memory = True
    
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn2,
                                   num_workers=args.num_workers, pin_memory=pin_memory)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn2,
                                 num_workers=args.num_workers,
                                 pin_memory=pin_memory)

    base_ds = get_coco_api_from_dataset(dataset_val)

    lit_model = lit_vitdetr.LitVitDetr(args, base_ds)

    if args.init_weights is not None:
        checkpoint = torch.load(args.init_weights, map_location='cpu')
        res = lit_model.load_state_dict(checkpoint['state_dict'])
        print("Loaded weights from args.init_weights. Result: ", res)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='AP',
                                                       filename="checkpoint-{epoch:03d}-{AP:0.3f}",
                                                       save_top_k=3,
                                                       mode='max')

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], max_epochs=args.epochs)

    if not args.eval:
        trainer.fit(lit_model, data_loader_train, data_loader_val)
    else:
        lit_model.summarize()
        trainer.test(lit_model, data_loader_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VITDETR training and evaluation script', parents=[get_args_parser()])
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
