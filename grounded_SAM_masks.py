import os
from PIL import Image
import cv2
import torch
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from groundingdino.util.inference import Model
import supervision as sv
import pickle

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM segment everything masks extracting params")
    
    parser.add_argument("--source_path", '-s', type=str, required=True)
    parser.add_argument('--images', type=str, default='images')
    parser.add_argument("--sam_checkpoint_path", default='third_party/segment-anything/weights/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--groundingdino_checkpoint_path", default='third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth', type=str)
    parser.add_argument("--groundingdino_config_path", default='third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', type=str)
    parser.add_argument("--downsample", default=1, type=int)
    parser.add_argument("--downsample_type", default='image', type=str, choices=['image', 'mask'], help="Downsample then segment, or segment then downsample.")
    parser.add_argument('--classes', nargs='+', type=str, default=['chair', 'table', 'plant'])
    parser.add_argument('--box_threshold', type=float, default=0.35)
    parser.add_argument('--text_threshold', type=float, default=0.35)
    parser.add_argument('--nms_threshold', type=float, default=0.8)

    args = parser.parse_args()
    
    print("Initializing Grounded SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    sam_predictor = SamPredictor(sam)
    grounding_dino_model = Model(model_config_path=args.groundingdino_config_path, model_checkpoint_path=args.groundingdino_checkpoint_path)

    downsample_manually = False
    if args.downsample == 1 or args.downsample_type == 'mask':
        images_path = os.path.join(args.source_path, args.images)
    else:
        images_path = os.path.join(args.source_path, f'{args.images}_{str(args.downsample)}')
        if not os.path.exists(images_path):
            images_path = os.path.join(args.source_path, args.images)
            downsample_manually = True
            print("No downsampled images, do it manually.")

    assert os.path.exists(images_path) and "Please specify a valid image root"
    masks_path = os.path.join(args.source_path, 'sam_masks')
    os.makedirs(masks_path, exist_ok=True)
    detections_path = os.path.join(args.source_path, 'detections')
    os.makedirs(detections_path, exist_ok=True)
    rgb_masks_path = os.path.join(args.source_path, 'rgb_masks')
    os.makedirs(rgb_masks_path, exist_ok=True)
    
    print("Extracting Grounded SAM masks...")
    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    for image_name in tqdm(sorted(os.listdir(images_path))):
        image = cv2.imread(os.path.join(images_path, image_name))
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if downsample_manually:
            image = cv2.resize(image,dsize=(image.shape[1] // args.downsample, image.shape[0] // args.downsample),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=args.classes,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
        )
        # NMS post process
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            args.nms_threshold
        ).numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        with open(os.path.join(detections_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.pkl'), "wb") as file:
            pickle.dump(detections, file)

        mask_list=[]
        background = torch.ones(image.shape[:2], dtype=torch.bool)
        if args.downsample_type == 'mask':
            background = torch.ones((image.shape[0] // args.downsample, image.shape[1] // args.downsample), dtype=torch.bool)
        for mask in detections.mask:
            mask_score = torch.from_numpy(mask).float()

            if args.downsample_type == 'mask':
                mask_score = torch.nn.functional.interpolate(mask_score[None, None, ...], size=(image.shape[0] // args.downsample, image.shape[1] // args.downsample) , mode='bilinear', align_corners=False).squeeze()
                mask_score[mask_score >= 0.5] = 1
                mask_score[mask_score != 1] = 0
            mask_score = mask_score.bool()
            background = background & ~mask_score
            mask_list.append(mask_score)
        # mask_list.append(background)
        if len(mask_list)!=0:
            masks = torch.stack(mask_list, dim=0)
            torch.save(masks.permute(0, 2, 1).flip(2), os.path.join(masks_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.pt')) # bool[masks, h, w]


        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [
            f"{args.classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(rgb_masks_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.jpg'), cv2.rotate(annotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE))