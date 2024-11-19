import os
from copy import deepcopy

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


torch.set_grad_enabled(False)

def exif_transpose(img):
    if not img:
        return img 
    exif_orientation_tag = 274

    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        if orientation == 1:
            pass 
        elif orientation == 2:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            img = img.rotate(180)
        elif orientation == 4:
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    return img

class ReS(Dataset):

    def __init__(self, root_dir, img_size=512, load_square=False):
        super().__init__()
        num_pairs = 100 
        self.img_size = img_size
        self.load_square = load_square
        self.tasks = []
        for i in range(1, num_pairs+1):
            img_1 = os.path.join(root_dir, f'p{i}_1.jpg')
            img_2 = os.path.join(root_dir, f'p{i}_2.jpg')
            mask_1 = os.path.join(root_dir, f'p{i}_1_mask.png')
            mask_2 = os.path.join(root_dir, f'p{i}_2_mask.png')
            # if one has occluded part, we only use it as the source image
            amodal_1 = os.path.join(root_dir, f'p{i}_1_amodal.png')
            amodal_2 = os.path.join(root_dir, f'p{i}_2_amodal.png')
            if os.path.exists(amodal_1):
                data = self.set_task(img_1, mask_1, img_2, mask_2, amodal_1)
                self.tasks.append(data)
            elif os.path.exists(amodal_2):
                data = self.set_task(img_2, mask_2, img_1, mask_1, amodal_2)
                self.tasks.append(data)
            else:
                data = self.set_task(img_1, mask_1, img_2, mask_2)
                self.tasks.append(data)
                data = self.set_task(img_2, mask_2, img_1, mask_1)
                self.tasks.append(data)
        
        print('Dataset size: {} Num of tasks:{}'.format(num_pairs, len(self.tasks)))
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, index):
        data = self.tasks[index]

        input_image = data['input']
        mask = data['mask']
        output_image = data['output']
        amodal = data['amodal']
        move_x, move_y = data['direction']

        mask = mask / 255.
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        new_mask = np.zeros_like(mask)
        if amodal is not None:
            amodal = amodal / 255.
            if len(amodal.shape) == 3:
                amodal = amodal[:, :, 0]
            new_amodal = np.zeros_like(mask)
            idxs = np.where(amodal > 0)
        else:
            idxs = np.where(mask > 0)
        y_min, y_max = np.min(idxs[0]), np.max(idxs[0])
        x_min, x_max = np.min(idxs[1]), np.max(idxs[1])
        y, x = input_image.shape[:2]

        y_min, y_min2 = self.clip_bbox_(y_min, move_y, 0, y - 1)
        y_max, y_max2 = self.clip_bbox_(y_max, move_y, 0, y - 1)
        x_min, x_min2 = self.clip_bbox_(x_min, move_x, 0, x - 1)
        x_max, x_max2 = self.clip_bbox_(x_max, move_x, 0, x - 1)
        new_mask[y_min2:y_max2, x_min2:x_max2] = mask[y_min:y_max, x_min:x_max]
        if amodal is not None:
            new_amodal[y_min2:y_max2, x_min2:x_max2] = amodal[y_min:y_max, x_min:x_max]
            new_amodal = np.clip(new_amodal - amodal - new_mask, 0, 1)

        # move based on mask
        new_input_image = deepcopy(input_image)
        new_input_image[y_min2:y_max2, x_min2:x_max2] = input_image[y_min:y_max, x_min:x_max]
        input_image = input_image * (1 - new_mask[:, :, None]) + new_input_image * new_mask[:, :, None]
        new_mask2 = new_mask.copy()
        new_mask2[new_mask2 > 0] = 1
        mask = np.clip(mask - new_mask2, 0, 1)
        mask[mask > 0] = 1

        mask[mask>0] = 255
        input_image = Image.fromarray(input_image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        output_image = Image.fromarray(output_image.astype(np.uint8))
        if amodal is not None:
            new_amodal[new_amodal>0] = 255
            new_amodal = Image.fromarray(new_amodal.astype(np.uint8))
        else:
            new_amodal = None
        
        # original dataset
        example = {'image': input_image, 'mask':mask, 'gt':output_image, 'amodal':new_amodal, 'size':data['size']}

        # inputs for SD
        input_image, mask, output_image = np.array(input_image),np.array(mask), np.array(output_image)
        example = {
            'image': torch.from_numpy(input_image) / 127.5 - 1,
            'mask': torch.from_numpy(mask[:, :, None]),
            'gt': torch.from_numpy(output_image) / 127.5 - 1,
        }

        # only use this if you want to combine subject removal and completion in a single run
        if amodal is not None:
            example['mask'] += torch.from_numpy(new_amodal[:, :, None])
            example['mask'] = torch.clamp(example['mask'], 0, 1)
        
        example['masked_image'] = example['image'] * (example['mask']<0.5)
        return example
    
    def clip_bbox_(self, side1, move, min_limit=0, max_limit=511):
        side2 = side1.copy()
        side2 += move
        if side2 > max_limit:
            side1 -= (side2 - max_limit)
            side2 -= (side2 - max_limit)
        if side2 < min_limit:
            side1 += (min_limit - side2)
            side2 += (min_limit - side2)
        return side1, side2
    
    def load_img(self, image):
        return np.array(exif_transpose(Image.open(image)).convert('RGB'))
    
    def resize(self, image, h, w, mode):
        if mode == 'img':
            return np.array(Image.fromarray(image).resize((h, w), Image.BICUBIC))
        elif mode == 'mask':
            return np.array(Image.fromarray(image).resize((h, w), Image.NEAREST))
        else:
            raise NotImplementedError
    
    def get_center(self, mask):
        mask = mask.astype(np.uint8)
        mask[mask>127.5] = 255
        mask[mask<=127.5] = 0
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        max_idx = np.argmax(areas)
        M = cv2.moments(contours[max_idx])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return cx, cy
    
    def dilate(self, mask):
        mask = cv2.dilate(mask, kernel=np.ones((2, 2), np.uint8), iterations=1)
        return mask

    def set_task(self, input, mask, output, output_mask, amodal=None):
        input_img = self.load_img(input)
        output_img = self.load_img(output)
        mask = self.load_img(mask)
        output_mask = self.load_img(output_mask)
        if amodal is not None:
            amodal = self.load_img(amodal)

        h, w = Image.fromarray(input_img).size
        factor = self.img_size / (min(h, w))
        new_h, new_w = int(h * factor), int(w * factor)
        if self.load_square:
            h, w = self.img_size, self.img_size
        else:
            h, w = new_h, new_w

        input_img = self.resize(input_img, h, w, 'img')
        output_img = self.resize(output_img, h, w, 'img')
        mask = self.resize(mask, h, w, 'mask')
        output_mask = self.resize(output_mask, h, w, 'mask')
        mask = self.dilate(mask)
        output_mask = self.dilate(output_mask)
        if amodal is not None:
            amodal = self.resize(amodal, h, w, 'mask')
            amodal = self.dilate(amodal)
        input_x, input_y = self.get_center(amodal) if amodal is not None else self.get_center(mask)
        output_x, output_y = self.get_center(output_mask)
        direction = np.array([output_x - input_x, output_y - input_y])
        return {'input': input_img, 'output': output_img, 'mask': mask, 'direction': direction, 'amodal': amodal, 'size':[new_h, new_w]}