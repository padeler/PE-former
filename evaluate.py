import os
from lit_main import get_args_parser
import torch
from torch.utils.data import DataLoader

from models import lit_vitdetr
from datasets import get_coco_api_from_dataset
from datasets.coco_person import build as build_coco_person
from util import misc as utils
import numpy as np

import torchvision.transforms.functional as F

from PIL import Image, ImageDraw
from tqdm import tqdm

from datasets.coco_eval import CocoEvaluator

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def sum2str(stat, ap=1, iouThr=None, areaRng='all', maxDets=100):
    p = [0.5, 0.95]
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}\n'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p[0], p[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    return iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, stat)


def create_report(stats):
    report = []
    report.append(sum2str(stats[0], 1, maxDets=20))
    report.append(sum2str(stats[1], 1, maxDets=20, iouThr=.5))
    report.append(sum2str(stats[2], 1, maxDets=20, iouThr=.75))
    report.append(sum2str(stats[3], 1, maxDets=20, areaRng='medium'))
    report.append(sum2str(stats[4], 1, maxDets=20, areaRng='large'))
    report.append(sum2str(stats[5], 0, maxDets=20))
    report.append(sum2str(stats[6], 0, maxDets=20, iouThr=.5))
    report.append(sum2str(stats[7], 0, maxDets=20, iouThr=.75))
    report.append(sum2str(stats[8], 0, maxDets=20, areaRng='medium'))
    report.append(sum2str(stats[9], 0, maxDets=20, areaRng='large'))
    return report


def merge_flip(keypoints, keypoints_flipped):
    
    keypoints = np.array(keypoints).reshape(17, -1)
    keypoints_flipped = np.array(keypoints_flipped).reshape(17, -1)
    
    res = np.zeros_like(keypoints)
    # print("KEYPOINTS\n", keypoints)
    for r, k, kf in zip(res, keypoints, keypoints_flipped):
        if k[2] != 0 and kf[2] != 0:  # average
            r[:] = (k[:] + kf[:]) / 2.0
        elif kf[2] != 0:  # use the flipped detection
            r[:] = kf[:]
        else:  # use the original
            r[:] = k[:]

    return res.flatten().tolist()


def save_res(res, targets, dataset_val, count):

    for i, pred in enumerate(res):
        filename = dataset_val.image_path_from_index(pred['image_id'])
        # print(count, "Loading Image: ", filename)
        canvas = Image.open(filename)
        draw = ImageDraw.Draw(canvas)

        keypoints = np.array(pred['keypoints']).reshape(17, -1).astype(np.int32)
        # labels = pred['labels']
        # scores = pred['scores']

        # labelmap = np.hstack((np.arange(17).reshape(-1, 1), labels[..., None], scores[..., None]))
        # print("LABELS and scores\n", labelmap)

        # pk = np.ones_like(keypoints)
        # for p, l in enumerate(labels):
        #     pk[l] = keypoints[p]
        # keypoints = pk
        # print(f"PRED[{i}]\n", keypoints)
        # order_keypoints(keypoints, labels, scores)

        for kp in keypoints:
            # print("Drawing kp", kp)
            x, y = kp[:2]
            b = 4
            draw.ellipse((x - b, y - b, x + b, y + b), fill='blue')

        t = targets[i]['gt_joints'].cpu().numpy()
        t[:, 2] = 1
        # print(f"TARGETS[{i}]\n", t)
        for kp in t:
            x, y = kp[:2]
            b = 2
            draw.ellipse((x - b, y - b, x + b, y + b), fill='red')

        canvas.save(f"tmp/{count}.jpg")
    

def main():
    args = get_args_parser().parse_args()

    # Make input size a tuple of width,height
    input_size = args.input_size
    if len(input_size) == 1:
        input_size = 2 * input_size
    else:
        input_size = input_size[:2]
    args.input_size = tuple(input_size)

    dataset_val = build_coco_person(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                             drop_last=False, collate_fn=utils.collate_fn2,
                             num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    device = torch.device("cuda")
    model = lit_vitdetr.LitVitDetr(args, base_ds)
    model.to(device)

    postprocessors = model.postprocessors

    assert 'keypoints' in postprocessors.keys(), "Only keypoints visualization is supported"

    assert args.init_weights, "Provide model weights with --init_weights <file>"
    if args.init_weights:
        checkpoint = torch.load(args.init_weights, map_location='cpu')
        checkpoint_model = checkpoint['state_dict']
        # from util.misc import _reshape_pos_embed
        # XXX Only for resizing VIT models (not XCiT)
        # _reshape_pos_embed(checkpoint_model, 'vitdetr.transformer.encoder.pos_embed', 
        #                    model.vitdetr.transformer.encoder.patch_embed.num_patches, model.vitdetr.transformer.encoder.pos_embed.shape)
        # XXX Fox Xcit Resizing we resize the pos_embed used only by the decoder
        # FIXME This works only when Patch size is square (w==h)
        # _reshape_pos_embed(checkpoint_model, 'vitdetr.transformer.pos_embed',
        #                    model.vitdetr.transformer.encoder.patch_embed.num_patches, model.vitdetr.transformer.pos_embed.shape)

        res = model.load_state_dict(checkpoint_model)
        print("Loaded model weights: ", res)

    model.eval()
    count = 0
    results = []
    
    coco_evaluator = CocoEvaluator(base_ds, ["keypoints"])

    max_batches = -1  # limit batches to test (set to <=0 to disable)
    do_flip_test = True  # flip test following simple-baselines protocol
    do_save_res = False
    print("LEN DATASET", len(dataset_val), "Flip Test is", do_flip_test, ". Max Batches ", max_batches)

    with torch.no_grad():
        for samples, targets in tqdm(data_loader):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)

            # loss_dict = criterion(outputs, targets)
            # weight_dict = criterion.weight_dict
            # print("LOSS ", loss_dict)
            res = postprocessors['keypoints'](outputs, targets)
            
            if do_flip_test:
                outputs_flipped = model(F.hflip(samples))

                # flip back
                # Pred_boxes shape is BS, Q, 2
                out_bbox_flipped = outputs_flipped['pred_boxes']
                out_bbox_flipped[:, :, 0] = 1. - out_bbox_flipped[:, :, 0]

                res_flipped = postprocessors['keypoints'](outputs_flipped, targets)

                # flip left-right labels
                for d in res_flipped:
                    kp = np.array(d['keypoints']).reshape(17, -1)
                    for pair in dataset_val.flip_pairs:
                        kp[pair] = kp[pair[::-1]]
                    d['keypoints'] = kp.flatten().tolist()

                # res = res_flipped
                # Merge with res
                for r, rf in zip(res, res_flipped):
                    assert r['image_id'] == rf['image_id'], f"Images do not have the same id {r['image_id']} and {rf['image_id']}"
                    r['keypoints'] = merge_flip(r['keypoints'], rf['keypoints'])

            if do_save_res:
                save_res(res, targets, dataset_val, count)

            if coco_evaluator is not None:
                coco_evaluator.update_keypoints(res)

            results.extend(res)
            count += 1
            # x = input("Press Enter")
            # if x == 'q':
            #     print("User quit.")
            #     break
            # cv2.imshow("Viz", img)
            # k = cv2.waitKey(0) & 0xFF
            # print(f"BATCH {count}")
            if max_batches > 0 and count > max_batches:
                break

    # print("RESULTS:\n", results)

    if coco_evaluator is not None:
        # coco_evaluator.update_keypoints(results)
        coco_evaluator.synchronize_between_processes()
        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    print("====CLASSIC COCO=====")
    pickle_results_path = "res/visulize_results.pickle"
    print("Storing to ", pickle_results_path)
    import pickle
    with open(pickle_results_path, "wb") as f:
        pickle.dump(results, f)
    print("Done")

    # in case of bbox_dets apply rescoring and nms (per simple baselines)
    if args.use_det_bbox:
        from models.potr import rescore_and_oks_nms
        # Use NMS from simple-baselines to further merge the results.
        results = rescore_and_oks_nms(results)

        # # filter results with area smaller than thres (for coco eval it is 32**2)
        for r in results:
            kp = np.array(r['keypoints']).reshape(17, 3)
            kpv = kp[:, 2] > 0
            x0 = np.min(kp[kpv, 0])
            y0 = np.min(kp[kpv, 1])
            x1 = np.max(kp[kpv, 0])
            y1 = np.max(kp[kpv, 1])
            w = x1 - x0
            h = y1 - y0
            area = w * h
            if area < 32**2:
                r['score'] = 0

    coco = COCO(args.coco_path + "/annotations/person_keypoints_val2017.json")
    cocoDt = coco.loadRes(results)

    imgIds = list(np.unique([k['image_id'] for k in results]))
    print("TOTAL PERSON INSTANCES", len(results))
    # print("UNIQUE IMAGES:", len(imgIds), imgIds)
    cocoEval = COCOeval(coco, cocoDt, "keypoints")
    # cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # save report
    rep = create_report(cocoEval.stats)
    path = os.path.dirname(args.init_weights)
    report_path = os.path.join(path, "eval_results.txt")

    with open(report_path, 'w') as f:
        f.writelines(rep)


if __name__ == "__main__":
    main()
