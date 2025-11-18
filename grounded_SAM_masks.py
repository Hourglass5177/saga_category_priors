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
import hashlib

def words_to_tensors(word_list, dim=32, device='cpu'):
    """使用正弦余弦函数生成确定性向量"""
    vectors = torch.zeros((len(word_list), dim), device=device)
    
    for i, word in enumerate(word_list):
        hash_val = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        
        frequencies = np.random.randn(dim // 2) * 10
        phases = np.random.rand(dim // 2) * 2 * np.pi
        
        for j in range(dim // 2):
            vectors[i, 2*j] = torch.tensor(np.sin(frequencies[j] + phases[j]))
            vectors[i, 2*j + 1] = torch.tensor(np.cos(frequencies[j] + phases[j]))
    
    # 添加归一化步骤
    norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
    normalized_vectors = vectors / norms
    
    return normalized_vectors

if __name__ == '__main__':

    parser = ArgumentParser(description="SAM masks extracting params")
    
    parser.add_argument("--images_path", '-s', type=str, required=True)
    parser.add_argument("--masks_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--progress_path", type=str, required=True)
    parser.add_argument("--annotated_images_path", '-a', type=str, default=None)
    parser.add_argument('--images', type=str, default='images')
    parser.add_argument("--sam_checkpoint_path", default='weights/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--groundingdino_checkpoint_path", default='weights/groundingdino_swint_ogc.pth', type=str)
    parser.add_argument("--groundingdino_config_path", default='weights/GroundingDINO_SwinT_OGC.py', type=str)
    parser.add_argument("--downsample", default=1, type=int)
    parser.add_argument("--downsample_type", default='image', type=str, choices=['image', 'mask'], help="Downsample then segment, or segment then downsample.")
    parser.add_argument('--classes', nargs='+', type=str, default=['chair', 'table', 'plant', 'flower', 'foliage', 'tv', 'painting', 'sofa', 'cabinet', 'bed', 'wall', 'floor', 'ceiling', 'person'])
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
        images_path = args.images_path
    else:
        images_path = args.images_path
        downsample_manually = True
        print("No downsampled images, do it manually.")

    assert os.path.exists(images_path) and "Please specify a valid image root"
    masks_path = args.masks_path
    os.makedirs(masks_path, exist_ok=True)
    labels_path = args.labels_path
    os.makedirs(labels_path, exist_ok=True)
    if args.annotated_images_path:
        os.makedirs(args.annotated_images_path, exist_ok=True)
    # detections_path = os.path.join(args.output_path, 'detections')
    # os.makedirs(detections_path, exist_ok=True)
    # rgb_masks_path = os.path.join(args.output_path, 'rgb_masks')
    # os.makedirs(rgb_masks_path, exist_ok=True)
    
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
    def rotate_detections_90_ccw(detections, image_width, image_height):
        # 提取原始检测框坐标
        x1, y1, x2, y2 = detections.xyxy.T

        # 计算旋转后的坐标
        new_x1 = y1
        new_y1 = image_width - x2
        new_x2 = y2
        new_y2 = image_width - x1

        # 更新检测框坐标
        rotated_boxes = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
        detections.xyxy = rotated_boxes

        return detections
    torch.save(words_to_tensors(args.classes), os.path.join(labels_path, 'features.pt'))
    length = len(sorted([e for e in os.listdir(images_path) if e.endswith('.jpg')]))
    for i, image_name in tqdm(list(enumerate(sorted([e for e in os.listdir(images_path) if e.endswith('.jpg')])))):
        with open(args.progress_path, 'w') as f:
            f.write(str((i+1)*100//length))
        image = cv2.imread(os.path.join(images_path, image_name))
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if downsample_manually:
            rotated_image = cv2.resize(rotated_image,dsize=(rotated_image.shape[1] // args.downsample, rotated_image.shape[0] // args.downsample),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        detections = grounding_dino_model.predict_with_classes(
            image=rotated_image,
            classes=args.classes,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
        )
        # detections.metadata.update({
        #     'label_to_class': dict(enumerate(args.classes)),
        #     'downsample': args.downsample,
        #     'image_height': image.shape[0],
        #     'image_width': image.shape[1]})
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
            image=cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        # with open(os.path.join(detections_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.pkl'), "wb") as file:
        #     pickle.dump(detections, file)

        mask_list=[]
        # background = torch.ones(image.shape[:2], dtype=torch.bool)
        # if args.downsample_type == 'mask':
        #     background = torch.ones((image.shape[0] // args.downsample, image.shape[1] // args.downsample), dtype=torch.bool)
        for mask in detections.mask:
            mask_score = torch.from_numpy(mask).float()

            if args.downsample_type == 'mask':
                mask_score = torch.nn.functional.interpolate(mask_score[None, None, ...], size=(rotated_image.shape[0] // args.downsample, rotated_image.shape[1] // args.downsample) , mode='bilinear', align_corners=False).squeeze()
                mask_score[mask_score >= 0.5] = 1
                mask_score[mask_score != 1] = 0
            mask_score = mask_score.bool()
            # background = background & ~mask_score
            mask_list.append(mask_score)
        # mask_list.append(background)
        if len(mask_list)!=0:
            masks = torch.stack(mask_list, dim=0)
            torch.save(masks.permute(0, 2, 1).flip(1), os.path.join(masks_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.pt')) # bool[masks, h, w]
            torch.save(torch.from_numpy(detections.class_id), os.path.join(labels_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.pt')) # int[masks]

        if args.annotated_images_path:
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()
            label_annotator = sv.LabelAnnotator()
            labels = [
                f"{args.classes[class_id]} {confidence:0.2f}" 
                for _, _, confidence, class_id, _, _ 
                in detections]
            annotated_image = mask_annotator.annotate(scene=rotated_image.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            cv2.imwrite(os.path.join(args.annotated_images_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.jpg'), cv2.rotate(annotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE))