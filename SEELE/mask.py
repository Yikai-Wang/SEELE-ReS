import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from matplotlib.patches import Polygon
from matplotlib.path import Path

class CustomCOCODataset(Dataset):
    def __init__(self, annFile, img_dir, transform=None, option='removal'):
        """
        Args:
            annFile (str): Path to the COCO annotation file.
            img_dir (str): Directory containing COCO images.
            transform (callable, optional): Optional transform to be applied on a sample.
            option (str): Either 'removal' or 'completion', determines the mask operation.
        """
        self.coco = COCO(annFile)
        self.img_dir = img_dir
        self.option = option
        self.transform = transform
        self.image_ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Get annotations for the image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # Randomly select one annotation (object)
        ann = random.choice(anns)
        # ann = anns[0]
        mask = self.coco.annToMask(ann)

        # Apply removal or completion transformation
        if self.option == 'removal':
            target_mask = self.apply_removal(image.size, mask)
        elif self.option == 'completion':
            target_mask = self.apply_completion(mask)

        if self.transform:
            image = self.transform(image)

        return image, mask, target_mask

    def apply_removal(self, image_size, mask):
        """Randomly move the mask to another region and discard overlapping part."""
        width, height = image_size
        # print(height, width)

        # loop until a valid mask is generated
        while True:
            try:
                # Randomly shift the mask
                shift_x = random.randint(-width//2, width//2)
                shift_y = random.randint(-height//2, height//2)
                # Ensure mask is in uint8 format and scaled to [0, 255] for OpenCV operations
                mask = (mask * 255).astype(np.uint8)
                x, y, w, h = cv2.boundingRect(mask)
                cropped_mask = mask[y:y+h, x:x+w]
                new_x, new_y = x + shift_x, y + shift_y
                # Calculate the region where the subject will be placed
                target_x_start = max(new_x, 0)
                target_y_start = max(new_y, 0)
                target_x_end = min(new_x + w, width)
                target_y_end = min(new_y + h, height)
                # Calculate cropping needed if the subject moves outside the image
                crop_x_start = max(0, -new_x)
                crop_y_start = max(0, -new_y)
                crop_x_end = min(w, width - new_x)
                crop_y_end = min(h, height - new_y)
                cropped_mask = cropped_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                # Create a blank canvas for placing the moved subject and moved mask
                mask_canvas = np.zeros((height, width), dtype=np.uint8)
                # Place the mask at the adjusted position
                mask_canvas[target_y_start:target_y_end, target_x_start:target_x_end] = cropped_mask
                # Compute unfilled region mask
                # Start with the original mask of the subject
                # Subtract the moved mask area to leave only the empty areas
                unfilled_mask = cv2.bitwise_and(mask_canvas, cv2.bitwise_not(mask))
                break
            except:
                continue

        return unfilled_mask

    def apply_completion(self, mask):
        """Randomly select part of the mask and enlarge it by several pixels."""
        mask_height, mask_width = mask.shape
        shapes = ['circle', 'ellipse', 'polygon']
        shape = random.choice(shapes)
        # shape = 'polygon'

        x, y, w, h = cv2.boundingRect(mask)
        if shape == "circle":
            # Random center and radius for the circle
            center_x = np.random.randint(x, x+w)
            center_y = np.random.randint(y, y+h)
            radius = np.random.randint(max(w, h) // 10, max(w, h) // 2)

            # Create a mask for the circle shape
            Y, X = np.ogrid[:mask_height, :mask_width]
            circle_mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
            result_mask = mask * circle_mask

        elif shape == "ellipse":
            # Random center, axes, and angle for the ellipse
            center_x = np.random.randint(x, x+w)
            center_y = np.random.randint(y, y+h)
            axis_x = np.random.randint(w // 4, w // 2)
            axis_y = np.random.randint(h // 4, h // 2)
            angle = np.random.randint(0, 360)

            # Create the ellipse mask
            Y, X = np.ogrid[:mask_height, :mask_width]
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            ellipse_mask = ((cos_angle * (X - center_x) + sin_angle * (Y - center_y)) ** 2) / axis_x ** 2 + \
                        ((-sin_angle * (X - center_x) + cos_angle * (Y - center_y)) ** 2) / axis_y ** 2 <= 1
            result_mask = mask * ellipse_mask

        elif shape == "polygon":
            # Random number of points for the polygon
            num_sides = np.random.randint(3, 10)

            # Random points for the polygon
            polygon_points = np.array([[
                np.random.randint(x, x+w),
                np.random.randint(y, y+h)
            ] for _ in range(num_sides)])

            # Create a mask for the polygon using matplotlib's path
            polygon = Polygon(polygon_points, closed=True, fill=False)
            path = Path(polygon.get_xy())

            # Get the coordinates inside the polygon
            grid_x, grid_y = np.meshgrid(np.arange(mask_width), np.arange(mask_height))
            points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
            mask_points = path.contains_points(points).reshape(mask_height, mask_width)

            result_mask = mask * mask_points

        mask = result_mask
        enlarged_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        return enlarged_mask

# Usage example:
annFile = '/Users/yikai/Downloads/annotations/instances_val2014.json'  # COCO annotations file
img_dir = '/Users/yikai/Downloads/val2014/'  # COCO images directory

# Create dataset for 'removal' or 'completion' option
dataset = CustomCOCODataset(annFile=annFile, img_dir=img_dir, transform=None, option='completion')

# Example usage of the dataset
image, mask, target_mask = dataset[2]  # Get first image and mask
plt.figure()
plt.imshow(image)  # Display image
plt.imshow(mask, alpha=0.5)  # Display mask
plt.savefig('original.png')
plt.figure()
plt.imshow(image)  # Display image
plt.imshow(target_mask, alpha=0.5)  # Display target mask
plt.savefig('moved_target.png')
