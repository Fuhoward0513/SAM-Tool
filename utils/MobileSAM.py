import cv2
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class Mobile_SAM:
    def __init__(self, ckpt, model, device):
        self.sam_checkpoint = ckpt
        self.model_type = model
        self.device = device
        self.masks = None
        
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint) # load model
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)   # define predictor
        
    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(self, coords, labels, ax, marker_size=200):
        try:
            pos_points = coords[labels==1]
            neg_points = coords[labels==0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        except:
            print('No input_points or input_labels')
        
    def show_box(self, box, ax):
        try:
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        except:
            print('No box')

    def warmup(self):   # warmup gpu
        dummy_input = torch.randn(3, 1024, 1024).to(self.device)
        self.predictor.set_image(dummy_input)
        _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes = None,
            multimask_output=False)
    
    def segment_prompt(self, img, input_points=None, input_labels=None, input_boxes=None):
        
        boxes_torch, points_torch, labels_torch, transformed_boxes = None, None, None, None
    
        t0 = time.time()
        self.predictor.set_image(img)
        
        if(len(input_points)==0 and len(input_boxes)==0):
            print('No prompt!')
            return
        
        if(len(input_points)!=0 and len(input_boxes)!=0):   # check number of points == number of boxes
            assert input_points.shape[0] == input_boxes.shape[0]
        
        if(len(input_boxes)!=0):    # ndarray -> tensor
            boxes_torch = torch.tensor(input_boxes).to(device=self.predictor.device)  
            transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_torch, img.shape[:2])
        else:
            transformed_boxes = None
        
        if(len(input_points)!=0):   # ndarray -> tensor
            points_torch = self.predictor.transform.apply_coords(input_points, img.shape[:2])
            points_torch = torch.as_tensor(points_torch, dtype=torch.float, device=self.predictor.device)
            labels_torch = torch.as_tensor(input_labels, dtype=torch.int, device=self.predictor.device)
        else:
            points_torch = None
            labels_torch = None
            
        masks, scores, logits = self.predictor.predict_torch(
            point_coords=points_torch,
            point_labels=labels_torch,
            boxes = transformed_boxes,
            multimask_output=False
        )
        
        print('runtime:',(time.time()-t0))
        
        plt.figure()
        plt.imshow(img)
        
        print(f'mask shape: {masks.shape}')
        
        for mask in masks:
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        
        if(boxes_torch is not None):
            for box in boxes_torch:
                self.show_box(box.cpu().numpy(), plt.gca())
        
        if(points_torch is not None):
            self.show_points(input_points, input_labels, plt.gca())
        
        plt.axis('off')
        plt.show()
        
        masks = tuple(mask.cpu().numpy() for mask in masks)
        masks = np.asarray(masks)
        
        self.masks = masks
        # return masks