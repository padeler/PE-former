"""
POTR model and criterion classes.
Code based on the DETR class by FAIR.
"""
from numpy.core.fromnumeric import argsort
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, accuracy, get_world_size,
                       is_dist_avail_and_initialized)

from .matcher import build_matcher
from .transformer_vit import build_transformer_vit

from util.sb_transforms import transform_preds
import numpy as np
from collections import defaultdict


class VITDETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        num_points = 2  # 2 for keypoints, 4 for bounding boxes
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, num_points, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.aux_loss = aux_loss

    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if type(samples) is NestedTensor:
            samples = samples.tensors
        # features, pos = self.backbone(samples)
        # src, mask = features[-1].decompose()
        # assert mask is not None

        hs = self.transformer(samples, self.query_embed.weight)[0]
        
        # TODO: For inference only the last transformer block is used (hs[-1])
        # For calculating the loss all blocks are needed.
        # It will save some flops to only pass the MLPs from the last block for inference

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the keypoints: the L1 regression loss.
           Targets dicts must contain the key "keypoints" containing a tensor of dim [nb_target_boxes, 2]
           The target keypoints are expected in format (x, y), normalized by the image size.
        """

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'keypoints': self.loss_keypoints,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    def __init__(self):
        super().__init__()

    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # NOTE: orig_size and size are stored as [width, height]
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # and from relative [0, 1] to absolute coordinates (model input dims, i.e 224, 224)
        keypoints = out_bbox * target_sizes[:, None, :]
        BS, Q, _ = keypoints.shape
        L = out_logits.shape[-1] - 1  # number of labels is num_classes-1

        # skeleton score is the sum of the probs for each
        # joint (excluding the no-object)
        scores = scores.cpu().numpy()
        # score = scores.sum(-1) / Q

        # convert to original image coordinates
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).cpu().numpy()
        target_sizes = target_sizes.cpu().numpy()
        centers = torch.stack([t["center"] for t in targets], dim=0).squeeze(1).cpu().numpy()
        scales = torch.stack([t["scale"] for t in targets], dim=0).squeeze(1).cpu().numpy()
        image_ids = torch.stack([t["image_id"] for t in targets], dim=0).squeeze(1).cpu().numpy()
        target_scores = torch.stack([t['score'] for t in targets], dim=0).squeeze(1).cpu().numpy()
        labels = labels.cpu().numpy()

        areas = np.prod(scales*200, 1)

        coords = keypoints.cpu().numpy()

        # Keypoints shape is BS, NUM_QUERIES, 2, where NUM_QUERIES is 17 for COCO person joints
        target_coords = np.ones((BS, Q, 3), dtype=np.float32)
        for i, (pred, center, scale, output_size) in enumerate(zip(coords, centers, scales, target_sizes)):
            target_coords[i, :, :2] = transform_preds(pred, center, scale, output_size)

        ordered_kp = [order_keypoints(k, l, sc, L) for k, l, sc in zip(target_coords, labels, scores)]

        results = [{'image_id': id, 'category_id': 1, 'score': okp[:, 3].sum() ,
                    "score_norm": okp[:, 2].sum(), "bbox_score": bb_sc, 'area': a, 
                    'keypoints': okp[:, :3].flatten().tolist()}
                   for id, okp, a, bb_sc in zip(image_ids, ordered_kp, areas, target_scores)]

        return results


def rescore_and_oks_nms(kpts_list, oks_thre=0.9):
    '''
    oks_nms adapted from simplebaselines repo
    '''
    
    # rescoring    
    for r in kpts_list:
        # from Simple baselines: Multiply  the normalized bbox score with the heatmap score. 
        n = r['score_norm']
        if n>0:
            r['score'] = r['bbox_score'] * r['score'] / n
        else:
            r['score'] = 0

     # image x person x (keypoints)
    kpts = defaultdict(list)
    for kpt in kpts_list:
        kpts[kpt['image_id']].append(kpt)

    # oks nms
    oks_nmsed_kpts = []
    for img in kpts.keys():
        img_kpts = kpts[img]
        keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))],
                        oks_thre)
        if len(keep) == 0:
            oks_nmsed_kpts.extend(img_kpts)
        else:
            oks_nmsed_kpts.extend([img_kpts[_keep] for _keep in keep])
    
    return oks_nmsed_kpts


##### Code from Simple baselines #####

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'] for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

###########################################

def order_keypoints(keypoints, labels, scores, num_labels):
    '''
    Keypoints predictions are not in order.
    Rearrange them using labels an pick the highest scorring ones.
    TODO: Re-write this in declarative form
    '''
    L = num_labels  # number of labels (without the detr no-op label)
    Q = keypoints.shape[0]  # number of queries

    result = np.zeros((L, 4), dtype=np.float32)
    labelmap = np.hstack((np.arange(Q).reshape(-1, 1), labels[..., None], scores[..., None]))
    # print("LABELMAP\n", labelmap)

    slm = labelmap[labelmap[:, 2].argsort()[::-1]]
    # print("SORTED\n", slm)
    for k, l, s in slm:
        li = int(l)
        ki = int(k)
        if result[li, 2] == 0:  # if not set yet, pick the highest scoring
            result[li, :3] = keypoints[ki, :3]
            result[li, 3] = s

    # print("RESULT\n", result)

    return result


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    # Ensure we are working with coco person keypoints.
    assert args.dataset_file == 'coco', \
        "Keypoints detector only supports coco keypoints for now."
    assert args.num_queries >= 17, \
        "Keypoints detector is expecting single person input with 17 coco joints. \
        Transformer decoder queries must be at least 17. Use --num_queries 17"

    num_classes = 17  # keypoints max_idx is 16. num_classes = max_idx+1

    transformer = build_transformer_vit(args)

    model = VITDETR(
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    
    # This is to be compatible with the matcher code.
    # TODO Remove on cleanup from old detr code.
    args.set_cost_giou = 0.0

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    # weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'keypoints', 'cardinality']

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)

    postprocessors = {'keypoints': PostProcess()}

    return model, criterion, postprocessors
