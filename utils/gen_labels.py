import json
import numpy as np
import os

class JSON_FILE:
    def __init__(self):
        self.json_format = {
            "images": {},
            "annotations": []
        }
        self.id = None
        
    def add_label(self, img_path, masks_lst, class_labels, bbox_lst=None):
        """
        Convert a segmentation mask to JSON format for a single annotation.
        
        Parameters:
            masks_lst List[ndarray]: The segmentation mask as a NumPy array with integer class labels. (N x B x C x H x w)
            class_labels List[str]: The class label for this annotation (e.g., "person," "car," etc.). (N)
            bbox_lst List[ndarray]: The bbox for the segmentation. (N x B x 4)
            image_filename (str): The filename of the corresponding image.  
        
        Returns:
            dict: The annotation data in JSON format.
        """
        
        # check dim.
        assert(masks_lst is not None)
        assert(len(masks_lst)==len(class_labels))
        if(len(bbox_lst)!=0):
            assert(masks_lst[0].shape[0]==bbox_lst[0].shape[0])
        
        # image information
        self.id = self.get_img_id(img_path)
        image = {
            "id": self.id,
            "file_path": img_path,
            "height": masks_lst[0].shape[-2],
            "width": masks_lst[0].shape[-1]
        }
        self.json_format["images"] = image
        
        # annotation information
        for i, masks in enumerate(masks_lst):
            for j, mask in enumerate(masks):  
                height, width = mask.shape[-2], mask.shape[-1]
                annotation = {
                    "id": self.id,
                    "class": class_labels[i],
                    "segmentation": [],
                    "bbox": []
                }
                
                if(len(bbox_lst)!=0):
                    annotation["bbox"] = bbox_lst[i][j].tolist()   # add bounding box labeling
                
                for y in range(height): # add segmentation labeling
                    for x in range(width):
                        if(mask.ndim==2):
                            if mask[y, x] > 0:  # Assuming class labels are positive integers
                                annotation["segmentation"].append([x, y])
                        elif(mask.ndim==3):
                            if mask[0, y, x] > 0:  # Assuming class labels are positive integers
                                annotation["segmentation"].append([x, y])
                        else:
                            print('wrong mask dimension')
                            return
                        
                self.json_format["annotations"].append(annotation)
        
        return self.json_format
    
    def write(self, path):
        json_str = json.dumps(self.json_format)
        
        with open(path + f"/{self.id}.json", "w") as f:
            f.write(json_str)
            
        print(f'Successfully write {self.id}.json!')
    
    def get_img_id(self, img_path):
        image_name = os.path.basename(img_path)
        root, extension = os.path.splitext(image_name)
        return root