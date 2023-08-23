from FastSAM.fastsam import FastSAM, FastSAMPrompt
import os
import time
import numpy as np

class MyFastSAM:
    def __init__(self, ckpt, device):
        self.checkpoint = ckpt
        self.device = device
        self.masks = None
        self.model = FastSAM(self.checkpoint)
    
    def segment(self, img_pth, mode, input=None, label=None):
        
        t0 = time.time()
        everything_results = self.model(img_pth, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(img_pth, everything_results, device=self.device)
        
        ann = None
        if(mode=='everything'):
            ann = prompt_process.everything_prompt()
        elif(mode=='bbox' and input is not None):
            ann = prompt_process.box_prompt(bboxes=input)
        elif(mode=='point' and input is not None):
            ann = prompt_process.point_prompt(points=input, pointlabel=label)
        elif(mode=='text' and input is not None):
            ann = prompt_process.text_prompt(text=input)
        else:
            print('wrong mode')
            return 0
        print('runtime:', (time.time()-t0))
        self.masks = np.array(ann)
        print(np.array(self.masks).shape)
        
        
        id = self.get_img_id(img_pth)
        prompt_process.plot(annotations=ann, output_path=f'./output/{id}.png',)
    
    def get_img_id(self, img_path):
        image_name = os.path.basename(img_path)
        root, extension = os.path.splitext(image_name)
        return root