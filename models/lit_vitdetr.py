import torch
import pytorch_lightning as pl
from datasets.coco_eval import CocoEvaluator

from models import potr
from models import detr


class LitVitDetr(pl.LightningModule):

    def __init__(self, args, base_ds):
        super().__init__()
        self.args = args
        self.base_ds = base_ds
        self.save_hyperparameters(args)

        if args.vit_as_backbone:  # replace the DETR backbone with a VIT
            self.vitdetr, self.criterion, self.postprocessors = detr.build(args)
        else:  # remove the backbone, use VIT as the encoder of the transformer
            self.vitdetr, self.criterion, self.postprocessors = potr.build(args)

        assert 'keypoints' in self.postprocessors, "LitVitDetr module only supports keypoints (pose estimation)"

        self.max_norm = args.clip_max_norm

    def forward(self, x):
        return self.vitdetr(x)

    def configure_optimizers(self):
        if self.args.vit_as_backbone:
            bb = "backbone"
        else:
            bb = "encoder"

        param_dicts = [
            {"params": [p for n, p in self.vitdetr.named_parameters() if bb not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.vitdetr.named_parameters() if bb in n and p.requires_grad],
                "lr": self.args.lr_backbone,
            },
        ]

        optimizer = torch.optim.AdamW(param_dicts, lr=self.args.lr,
                                      weight_decay=self.args.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_drop)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        samples, targets = batch

        outputs = self.vitdetr(samples)

        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        self.log("loss", losses)

        return losses

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.vitdetr(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        self.log("val/val_loss", losses)

        res = self.postprocessors['keypoints'](outputs, targets)

        # if coco_evaluator is not None:
        #     coco_evaluator.update_keypoints(res)
        self.log("loss", losses)

        return {"detections": res, "loss": losses}

    def validation_epoch_end(self, outputs) -> None:
        iou_types = ('keypoints',)
        coco_evaluator = CocoEvaluator(self.base_ds, iou_types)

        for out in outputs:
            coco_evaluator.update_keypoints(out["detections"])

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        # Log to Tensorboard
        stats = coco_evaluator.coco_eval['keypoints'].stats.tolist()
        self.log("AP", stats[0])  # for the checkpoint callback monitor.
        self.log("val/AP", stats[0])
        self.log("val/AP.5", stats[1])
        self.log("val/AP.75", stats[2])
        self.log("val/AP.med", stats[3])
        self.log("val/AP.lar", stats[4])
        self.log("val/AR", stats[5])
        self.log("val/AR.5", stats[6])
        self.log("val/AR.75", stats[7])
        self.log("val/AR.med", stats[8])
        self.log("val/AR.lar", stats[9])
        
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
    
    def on_after_backward(self) -> None:
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.vitdetr.parameters(), self.max_norm)