import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
from torchvision.utils import save_image
import numpy as np
import shutil
import torch.nn.functional as F
from PIL import Image

def pth_to_json():
    import json
    label_to_class = {0:'plant', 1:'chair', 2:'table', 3:'object'}
    point_ins_labels_path = 'temp/tys/point_ins_labels.pth'
    point_labels_path = 'temp/tys/point_labels.pth'
    point_ins_labels = torch.load(point_ins_labels_path, weights_only=True)
    point_labels = torch.load(point_labels_path, weights_only=True)
    output = dict()
    output['point_instance_labels'] = point_ins_labels.tolist()
    output['instances'] = [{'class': label_to_class[l]} for l in point_labels.tolist()]
    with open('temp/tys/output.json','w') as f:
        json.dump(output,f)

def sam_masks_rgb():
    spath = 'data/temp/tys/sam_masks'
    dpath = 'data/temp/tys/sam_masks_rgb'
    os.makedirs(dpath, exist_ok=True)
    for masks_name in os.listdir(spath):
        print(os.path.join(spath, masks_name))
        masks=torch.load(os.path.join(spath, masks_name))
        n, h, w = masks.shape
        colormap = torch.rand((n+1, 3)).cuda()
        colormap[-1]=0
        masks = sorted(masks, key=lambda m: m.sum())
        segmentmap=torch.full(masks[0].shape, -1).cuda()
        for idx, mask in enumerate(masks):
            segmentmap[mask]=idx
        save_image(colormap[segmentmap].permute(2,0,1).cpu(), os.path.join(dpath, masks_name.split('.')[0]+'.jpg'))

def pick_image():
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
    # sparse_path = '/mnt/d/BaiduNetdiskDownload/nanfeng/分块sfm数据/chunks/layer_3/chunk_003_001/sparse/0/'
    sparse_path = 'data/temp/nanfeng/sparse/0/'
    image_path = '/mnt/d/BaiduNetdiskDownload/nanfeng/图像集/images/'
    dst_path = 'data/temp/nanfeng/images'
    images = read_extrinsics_binary(os.path.join(sparse_path, 'images.bin'))
    cameras = read_intrinsics_binary(os.path.join(sparse_path, 'cameras.bin'))
    for image in images.values():
        shutil.copy(os.path.join(image_path, image.name), dst_path)

def test_clip():
    from PIL import Image
    import cv2
    device = 'cuda'
    def get_entity(image):
        # from segment_anything import (SamAutomaticMaskGenerator, SamPredictor, sam_model_registry)
        # sam = sam_model_registry['vit_h'](checkpoint='./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth').to(device)
        # mask_generator = SamAutomaticMaskGenerator(
        #     model=sam,
        #     points_per_side=32,
        #     pred_iou_thresh=0.88,
        #     box_nms_thresh=0.7,
        #     stability_score_thresh=0.95,
        #     crop_n_layers=0,
        #     crop_n_points_downscale_factor=1,
        #     min_mask_region_area=100,
        # )
        from transformers import SamProcessor, SamModel
        from transformers import pipeline
        model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        generator =  pipeline("mask-generation", model='facebook/sam-vit-huge', device = device, points_per_batch = 2)
        with torch.no_grad():
            outputs = generator(image, points_per_batch = 2)
        inputs = processor(image, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        records = mask_generator.generate(np.array(image))
        def get_bbox(mask: np.ndarray):
            # 查找掩码中的 True 元素的索引
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            # 如果没有 True 元素，则返回全零的边界框
            if not np.any(rows) or not np.any(cols):
                return (0, 0, 0, 0)
            
            # 获取边界框的上下左右边界
            x_min, x_max = np.where(rows)[0][[0, -1]] # h
            y_min, y_max = np.where(cols)[0][[0, -1]] # w
            
            # 返回边界框
            return (x_min, y_min, x_max + 1 - x_min, y_max + 1 - y_min) # x, y, h, w
        def get_entity_image(image: np.ndarray, mask: np.ndarray)->np.ndarray:
            if mask.sum()==0:
                return np.zeros((224,224,3), dtype=np.uint8)
            image = image.copy()
            # crop by bbox
            x,y,h,w = get_bbox(mask)
            image[~mask] = np.zeros(3, dtype=np.uint8) #分割区域外为白色
            image = image[x:x+h, y:y+w, ...] #将img按分割区域bbox裁剪
            # pad to square
            l = max(h,w)
            paded_img = np.zeros((l, l, 3), dtype=np.uint8)
            if h > w:
                paded_img[:,(h-w)//2:(h-w)//2 + w, :] = image
            else:
                paded_img[(w-h)//2:(w-h)//2 + h, :, :] = image
            paded_img = cv2.resize(paded_img, (224,224))
            return paded_img
        masks = [record['segmentation'] for record in records]
        entity = [get_entity_image(np.array(image), record['segmentation']) for record in records]
        return torch.from_numpy(np.stack(entity)), torch.from_numpy(np.stack(masks))

    def get_semantics(entity):
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        inputs = clip_processor(images=entity, return_tensors='pt')
        inputs = inputs.to(clip_model.device)
        semantics = clip_model.get_image_features(**inputs)
        semantics = F.normalize(semantics,dim=-1).detach().cpu()
        return semantics
    
    def get_relevancy_map(entity, masks, semantics, ptexts, ntexts):
        from transformers import CLIPModel, CLIPProcessor
        def get_semantic_map(masks: torch.Tensor, semantics):
            semantic_map = torch.full(masks.shape[1:3], -1, dtype=torch.int64)
            semantics = torch.concat((semantics, torch.zeros((1, semantics.shape[-1]))), dim=0)
            for index, mask in enumerate(masks):
                semantic_map[mask] = index
            return semantics[semantic_map]
        def get_relevancy(raw_semantic_map: torch.Tensor, pembed: torch.Tensor, nembed: torch.Tensor):
            s = raw_semantic_map.shape[:-1]
            c = raw_semantic_map.shape[-1]
            raw_semantics = raw_semantic_map.flatten(0, -2)
            psim=pembed@raw_semantics.T # (p, i)
            nsim=nembed@raw_semantics.T # (n, i)
            nsim=nsim.unsqueeze(0).repeat_interleave(pembed.shape[0],dim=0) # (p, n ,i)
            psim=psim.unsqueeze(1).repeat_interleave(nembed.shape[0],dim=1) # (p, n, i)
            sim=torch.stack((psim,nsim), dim=-1) # (p, n, i, 2)
            sim=torch.softmax(10*sim, dim=-1) # (p, n, i, 2)
            sim, indice = sim[...,0].min(dim=1) # (p, i)
            return sim.unflatten(1, s)

        semantic_map = get_semantic_map(masks, semantics)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        pembed = clip_processor(text=ptexts, return_tensors='pt', padding=True)
        pembed = pembed.to(clip_model.device)
        pembed = clip_model.get_text_features(**pembed)
        pembed = F.normalize(pembed, dim=-1).detach().cpu()

        nembed = clip_processor(text=ntexts, return_tensors='pt', padding=True)
        nembed = nembed.to(clip_model.device)
        nembed = clip_model.get_text_features(**nembed)
        nembed = F.normalize(nembed, dim=-1).detach().cpu()

        relevancy_map = get_relevancy(semantic_map, pembed, nembed)
        return relevancy_map


    image = Image.open('data/temp/nanfeng/images/00000000001-00000001113-A01113.jpg')
    # entity, masks = get_entity(image)
    # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #     semantics = get_semantics(entity)
    # torch.save(entity, 'temp/entity.pth')
    # torch.save(masks, 'temp/masks.pth')
    # torch.save(semantics, 'temp/semantics.pth')

    entity = torch.load('temp/entity.pth')
    masks = torch.load('temp/masks.pth')
    semantics = torch.load('temp/semantics.pth')

    ptexts = ['house', 'building', 'pavilion', 'tree', 'vegetable field', 'car', 'pool', 'water', 'grass']
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        relevancy_map = get_relevancy_map(entity, masks, semantics, ptexts, ["object", "things", "stuff", "texture"])
    torch.save(relevancy_map, 'temp/relevancy_map.pth')

    relevancy_map = torch.load('temp/relevancy_map.pth')
    for ptext, map in zip(ptexts, relevancy_map):
        img=torch.from_numpy(np.array(image)).permute(2,0,1)
        img = (img*(map>0.5)).permute(1,2,0)
        Image.fromarray(img.numpy()).save(f'temp/{ptext}.jpg')

def test_dinov2():
    device = 'cuda'
    # inp = torch.randn(1, 3, 10, 12)
    # w = torch.randn(2, 3, 4, 5)
    # inp_unf = torch.nn.functional.unfold(inp, (4, 5))  # shape of inp_unf is (1,3*4*5,7*8)

    # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)  # shape of out_unf is (1,2,56)
    # #  以上代码相当于 inp_unf(1, 60, 56) .t() * w(2 , 3 * 4 * 5).t() → out_unf (1, 56, 2 ) → out_unf (1, 2, 56)

    # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))  # out.size() = (1,2,7,8)

    # print((torch.nn.functional.conv2d(inp, w) - out).abs().max())  # tensor(1.9073e-06)
    # F.conv2d()
    from transformers import AutoImageProcessor, AutoModel

    image = Image.open('data/temp/nanfeng/images/00000000001-00000001113-A01113.jpg')

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs.to(model.device))
    last_hidden_states = outputs[0]

    print(outputs['last_hidden_state'].shape)

def main():
    pth_to_json()
    # sam_masks_rgb()
    # pick_image()
    # test_clip()
    # test_dinov2()

if __name__ =='__main__':
    main()