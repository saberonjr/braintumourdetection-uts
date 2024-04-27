import os
import cv2
import json
import numpy as np
from pycocotools.coco import COCO
from clearml import Dataset, Task
from detectron2.data.transforms import Transform, AugInput, apply_transform_gens, Resize, RandomFlip, RandomRotation
from config_setup import setup_cfg  # Assuming this function is implemented correctly

# Define transformation classes
#from detectron2.data.transforms import Transform
class GaussianBlurTransform(Transform):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def apply_image(self, img):
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
    
    def apply_coords(self, coords):
        # Since blurring does not affect coordinates, just return them unchanged
        return coords

class AddGaussianNoiseTransform(Transform):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def apply_image(self, img):
        noise = np.random.normal(0, self.sigma, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def apply_coords(self, coords):
        # Since adding noise does not affect coordinates, just return them unchanged
        return coords

# Preprocess images and upload to ClearML
def preprocess_and_upload(dataset_id, register_name, cfg):
    coco_path = 'C:/Users/Leon-PC/Downloads/Brain Tumor Detection Dataset/Dataset/valid/_annotations.coco.json'
    coco = COCO(coco_path)
    processed_images_dir = os.path.join('C:/Users/Leon-PC/Downloads/Preprocessed data/Preprocessed_valid', register_name)
    os.makedirs(processed_images_dir, exist_ok=True)

    for img_info in coco.dataset['images']:
        image_path = os.path.join('C:/Users/Leon-PC/Downloads/Brain Tumor Detection Dataset/Dataset/valid', img_info['file_name'])
        image = cv2.imread(image_path)
        image = process_image(image, cfg)  # Assume process_image applies necessary transforms
        processed_image_path = os.path.join(processed_images_dir, f"{img_info['id']:012d}.jpg")
        cv2.imwrite(processed_image_path, image)

    # Save the merged annotations to a JSON file
    annotations_path = os.path.join(processed_images_dir, '_annotations.coco.json')
    with open(annotations_path, 'w') as f:
        json.dump(coco.dataset, f)    

    # Initialize ClearML task only now and upload processed data
    task = Task.init(project_name="Brain Tumor Detection Project", task_name="Upload PreProcessed valid Data", task_type=Task.TaskTypes.data_processing)
    processed_dataset = Dataset.create(dataset_name="PreProcessed Valid", dataset_project="Brain Tumor Detection Project", parent_datasets= [dataset_id])
    processed_dataset.add_files(processed_images_dir)
    processed_dataset.upload()
    processed_dataset.finalize()
    task.close()

def process_image(image, cfg):
    aug_input = AugInput(image)
    aug_list = [
        Resize(shape=(640, 640)),
        RandomFlip(prob=0.5),
        RandomRotation(angle=[-4, 4], expand=False, sample_style='range'),
        GaussianBlurTransform(5),
        AddGaussianNoiseTransform(25)
    ]
    aug_input, _ = apply_transform_gens(aug_list, aug_input)
    return aug_input.image

def main():
    cfg = setup_cfg()
    preprocess_and_upload('8c7de6d5a12842059073eb3e99e8a0ab', 'Preprocessed valid', cfg)

if __name__ == '__main__':
    main()
