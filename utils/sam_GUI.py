import tkinter as tk
import cv2
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
from .gen_labels import*

class sam_ImageAnnotationTool:
    def __init__(self, root, sam_model):
        self.root = root
        self.width = 640
        self.height = 400
        self.canvas = tk.Canvas(self.root, width=self.width,  height=self.height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.sam_model = sam_model
        self.sam_model.warmup()
        
        # load image
        self.image = None
        self.image_path = None
        self.scale_x = None
        self.scale_y = None
        self.load_image_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_image_button.pack(side=tk.LEFT)

        # Bottons: 1. draw points 2. draw bboxes 3. turn to segmentation labeling 4. Make segmentation to label(.json)
        self.draw_bbox_botton = tk.Button(self.root, text="Draw Bbox", command=self.label_bbox)
        self.draw_bbox_botton.pack(side=tk.LEFT)
        
        self.draw_point_button = tk.Button(self.root, text="Draw Points", command=self.label_point)
        self.draw_point_button.pack(side=tk.LEFT)
        
        self.segment_button = tk.Button(self.root, text="Gen. Seg.", command=self.make_seg)
        self.segment_button.pack(side=tk.LEFT)
        
        self.gen_labels_button = tk.Button(self.root, text="Get Label", command=self.gen_labels)
        self.gen_labels_button.pack(side=tk.RIGHT)
        
        # Text
        self.label = tk.Label(self.root, text="Class Name")
        self.class_name = tk.Entry(self.root)
        self.label.pack(side=tk.LEFT)
        self.class_name.pack(side=tk.LEFT)
        
        # Draw points
        self.points_list = []
        self.labels_list = []
        self.labels = []
        self.points = []
        self.current_point = None
        
        # Draw bounding boxes
        self.bbox_list = []
        self.current_bbox = None
        self.current_draw_bbox = None
        self.current_draw_class = None
        
        # masks
        self.masks_list = []
        self.class_list = []
        self.class_bbox_list = []

    def reset_image(self):
        self.image = None
        self.image_path = None
        self.scale_x = None
        self.scale_y = None
    
    def reset_prompts(self):
        self.points_list = []
        self.labels_list = []
        self.labels = []
        self.points = []
        self.bbox_list = []
        self.current_bbox = None
        self.current_draw_bbox = None
        self.current_draw_box_class = None
        self.current_draw_point_class = None
    
    def load_image(self):
        self.reset_image()
        self.reset_prompts()
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if self.image_path:
            self.image = Image.open(self.image_path)
            self.scale_x = self.image.size[0]/self.width
            self.scale_y = self.image.size[1]/self.height
            self.image = self.image.resize((self.width, self.height))
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def label_bbox(self):
        self.canvas.bind("<ButtonPress-1>", self.start_bbox)
        self.canvas.bind("<B1-Motion>", self.draw_bbox)
        self.canvas.bind("<ButtonRelease-1>",self.save_bbox)
    
    def start_bbox(self, event):
        self.current_draw_bbox = None
        self.current_bbox = [event.x, event.y, event.x, event.y]

    def draw_bbox(self, event):
        self.canvas.delete(self.current_draw_bbox)
        self.canvas.delete(self.current_draw_box_class)
        self.current_bbox[2] = event.x
        self.current_bbox[3] = event.y
        self.current_draw_bbox = self.canvas.create_rectangle(self.current_bbox, outline='green', width=2, tags=self.class_name.get())
        self.current_draw_box_class = self.canvas.create_text(self.current_bbox[0], self.current_bbox[1]-10,
                                                               text=self.class_name.get(), anchor=tk.W, fill="white")  # Create text next to the rectangle

    def save_bbox(self, event):
        # Add code here to save the bounding box coordinates and associated labels to a file
        x1 = self.current_bbox[0] * self.scale_x
        y1 = self.current_bbox[1] * self.scale_y
        x2 = self.current_bbox[2] * self.scale_x
        y2 = self.current_bbox[3] * self.scale_y
        if(x1==x2 and y1==y2):
            print("Not a box")
            return
        print('save box:', [x1, y1, x2, y2])
        
        self.bbox_list.append([x1, y1, x2, y2])
    
    def label_point(self):
        self.canvas.bind("<ButtonPress-1>", self.draw_valid_point)
        self.canvas.bind("<ButtonPress-3>", self.draw_remove_point)
        self.canvas.bind("<ButtonPress-2>",self.save_points)
        self.canvas.bind("<ButtonRelease-1>", self.unbind)
        self.canvas.bind("<B1-Motion>", self.unbind)
    
    def unbind(self, event):
        pass
        
    def draw_valid_point(self, event):
        self.current_point = [event.x, event.y]
        x1, y1 = (event.x - 3), (event.y - 3)
        x2, y2 = (event.x + 3), (event.y + 3)
        self.canvas.create_oval(x1, y1, x2, y2, fill='green')
        self.current_draw_point_class = self.canvas.create_text(self.current_point[0], self.current_point[1]-10,
                                                        text=self.class_name.get(), anchor=tk.W, fill="white") 
        self.push_point(event, 1)
        
    def draw_remove_point(self, event):
        self.current_point = [event.x, event.y]
        x1, y1 = (event.x - 3), (event.y - 3)
        x2, y2 = (event.x + 3), (event.y + 3)
        self.canvas.create_oval(x1, y1, x2, y2, fill='red')
        self.current_draw_point_class = self.canvas.create_text(self.current_point[0], self.current_point[1]-10,
                                                        text="not "+self.class_name.get(), anchor=tk.W, fill="white")
        self.push_point(event, 0)
    
    def push_point(self, event, flag):
        x = event.x * self.scale_x
        y = event.y * self.scale_y
        self.points.append([x, y])
        self.labels.append(flag)
        
    def save_points(self, event):
        self.points_list.append(self.points)
        self.labels_list.append(self.labels)
        self.points = []
        self.labels = []
        
        print("point list: ", self.points_list)
        print("point labels: ", self.labels_list)
    
    def make_seg(self):
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.sam_model.segment_prompt(img, self.image_path, # Find segmentation using SAM
                                      input_points=np.array(self.points_list), 
                                      input_labels=np.array(self.labels_list),
                                      input_boxes=np.array(self.bbox_list))
        
        self.masks_list.append(self.sam_model.masks)    # Add to List
        self.class_list.append(self.class_name.get())
        if(len(self.bbox_list)!=0):
            self.class_bbox_list.append(np.array(self.bbox_list))
        
        self.class_name.delete(0, tk.END)   # reset params
        self.reset_prompts()
    
    def gen_labels(self):
        json_file = JSON_FILE()
        json_file.add_label(self.image_path, self.masks_list, self.class_list, self.class_bbox_list)
        json_file.write("./labels")
        
        