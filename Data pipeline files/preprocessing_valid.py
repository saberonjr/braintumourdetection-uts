import os
import cv2
import json
import numpy as np
from detectron2.data import DatasetMapper
from detectron2.data.transforms import Transform, apply_transform_gens, AugInput, Resize, RandomFlip, RandomRotation
from pycocotools.coco import COCO
from clearml import Dataset, Task
from config_setup import setup_cfg  # This must be correctly configured to return a Detectron2 configuration

# Custom Transform Classes
class GaussianBlurTransform(Transform):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def apply_image(self, img):
        """Applies Gaussian Blur to the image."""
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)

    def apply_coords(self, coords):
        """Coordinates are not altered by Gaussian blur."""
        return coords

class AddGaussianNoiseTransform(Transform):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def apply_image(self, img):
        """Applies Gaussian noise to the image."""
        noise = np.random.normal(0, self.sigma, img.shape)
        noisy_image = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def apply_coords(self, coords):
        """Coordinates are not altered by adding noise."""
        return coords

# Custom Dataset Mapper including preprocessing
class CustomDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.augmentations = [
            Resize(shape=(640, 640)),
            RandomFlip(prob=0.5),
            RandomRotation(angle=[-4, 4], expand=False, sample_style="range"),
            GaussianBlurTransform(5),  # Using a fixed kernel size for simplicity
            AddGaussianNoiseTransform(25)  # Using a fixed sigma for simplicity
        ]

    def __call__(self, dataset_dict):
        aug_input = AugInput(dataset_dict["image"])
        aug_input, _ = apply_transform_gens(self.augmentations, aug_input)
        dataset_dict["image"] = aug_input.image
        return dataset_dict

def preprocess_and_upload(dataset_id, register_name, cfg):
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = dataset.get_local_copy()

    coco = COCO(os.path.join(dataset_path, "_annotations.coco.json"))
    processed_annotations = {
        "images": [],
        "annotations": [],
        "categories": coco.dataset["categories"]
    }

    processed_dataset_path = os.path.join('C:/Users/Leon-PC/Downloads/Brain Tumor Detection Dataset/Dataset', dataset.name)
    # Clear existing files to avoid duplication
    if os.path.exists(processed_dataset_path):
        for file in os.listdir(processed_dataset_path):
            os.remove(os.path.join(processed_dataset_path, file))
    else:
        os.makedirs(processed_dataset_path, exist_ok=True)

    annotation_id = 1
    for img_info in coco.dataset['images']:
        print(f"Processing image ID: {img_info['id']}")  # Logging each processed image
        image_path = os.path.join(dataset_path, img_info['file_name'])
        image = cv2.imread(image_path)
        dataset_dict = {
            "image": image,
            "image_id": img_info['id'],
            "height": image.shape[0],
            "width": image.shape[1],
            "annotations": coco.loadAnns(coco.getAnnIds(imgIds=img_info['id']))
        }
        processed_image = CustomDatasetMapper(cfg, is_train=True)(dataset_dict)["image"]
        processed_image_path = os.path.join(processed_dataset_path, f"{img_info['id']:012d}.jpg")
        cv2.imwrite(processed_image_path, processed_image)

        processed_annotations["images"].append({
            "file_name": f"{img_info['id']:012d}.jpg",
            "height": image.shape[0],
            "width": image.shape[1],
            "id": img_info['id']
        })

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_info['id'])):
            ann["image_id"] = img_info['id']
            ann["id"] = annotation_id
            annotation_id += 1
            processed_annotations["annotations"].append(ann)

    with open(os.path.join(processed_dataset_path, "_annotations.coco.json"), "w") as f:
        json.dump(processed_annotations, f)

    processed_dataset = Dataset.create(dataset_name=f"PreProcessed {dataset.name}",
                                       parent_datasets=[dataset.id],
                                       dataset_project=dataset.project)
    processed_dataset.add_files(processed_dataset_path)
    processed_dataset.upload()
    processed_dataset.finalize()

def main():
    cfg = setup_cfg()
    task = Task.init(project_name="Brain Tumor Detection Project", task_name="Data Preprocessing - Validation Data", task_type=Task.TaskTypes.data_processing)
    task.connect(cfg)
    preprocess_and_upload('8c7de6d5a12842059073eb3e99e8a0ab', 'my_preprocessed_valid', cfg)
    task.close()

if __name__ == '__main__':
    main()
