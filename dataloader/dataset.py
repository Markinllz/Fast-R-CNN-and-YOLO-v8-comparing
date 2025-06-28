import torch
from torch.utils.data import Dataset
import os
import cv2
from pathlib import Path
import numpy as np


class EyeCataractDataset(Dataset):
    def __init__(self, images_dir, labels_dir, model_type : str, img_size = 640):
        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.model_type = model_type
        self.img_size = img_size
        images = os.listdir(images_dir)
        self.images_paths = [os.path.join(images_dir, image) for image in images if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
        labels = os.listdir(labels_dir)
        self.labels_paths = [os.path.join(labels_dir, label) for label in labels if label.lower().endswith(('.txt'))]
    

    def __len__(self):
        return len(self.images_paths)
    



    def __getitem__(self, index):
        img_path = self.images_paths[index]

        image_filename = os.path.basename(img_path)
        basename = os.path.splitext(image_filename)[0] + '.txt'

        label_path = os.path.join(self.labels_dir,basename)


        label = self.parse_label(label_path)

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_size_h , orig_size_w = img.shape[:2] 

        img = cv2.resize(img, (self.img_size, self.img_size))

        img = img.astype(np.float32) / 255
        
        
        img = img.transpose(2,0,1)


        img = torch.from_numpy(img)

        return {
        'image': img,
        'labels': [label],
        'orig_size': torch.tensor([orig_size_h, orig_size_w]),
        'image_path': img_path,
        'label_path': label_path
    }


    def parse_label(self, label_path):
        with open(label_path, 'r') as f:
            content = f.read()


        data = [float(x) for x in content.split()]
        class_id = int(data[0])
        x_center = data[1]
        y_center = data[2]
        width = data[3]
        height = data[4]

        return [class_id,x_center,y_center,width,height]
        









