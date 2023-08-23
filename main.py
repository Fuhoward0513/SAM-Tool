from utils.fastsam_GUI import*
from utils.sam_GUI import*
from utils._fastsam import*
from utils.SAM import*
from utils.MobileSAM import*
import torch
import argparse
import yaml

if __name__ == "__main__":
    
    # load configuration file
    global cfg, args
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        help='Configuration file to us'
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)
    print(cfg)
    
    # load device
    device = cfg['device']
    print(f"device: {device}")
    
    # load model
    if(cfg['model']=='SAM'):
        sam = SAM(cfg['ckpt'], cfg['encoder'], device)
    elif(cfg['model']=='mobileSAM'):
        sam = Mobile_SAM(cfg['ckpt'], cfg['encoder'], device)
    elif(cfg['model']=='fastSAM'):
        sam = MyFastSAM(cfg['ckpt'], device)
    
    # activate GUI
    root = tk.Tk()
    root.title("Image Annotation Tool")
    app = None
    
    if(cfg['model']=='fastSAM'):
        app = fastsam_ImageAnnotationTool(root, sam)
    elif(cfg['model']=='mobileSAM' or cfg['model']=='SAM'):
        app = sam_ImageAnnotationTool(root, sam)
    root.mainloop()