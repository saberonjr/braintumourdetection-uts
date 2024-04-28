import os
import cv2
import json
import numpy as np
import pandas as pd
import detectron2
import pickle
import yaml
import matplotlib.pyplot as plt
import torch
import clearml
import optuna
from pycocotools.coco import COCO
from clearml import Dataset, Task, TaskTypes
from clearml.automation.controller import PipelineDecorator
from detectron2.data import DatasetMapper, DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.transforms import Transform, apply_transform_gens, AugInput, Resize, RandomFlip, RandomRotation
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import Checkpointer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
#from registering_preprocessed_datasets import register_dataset
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn.metrics import accuracy_score


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(name="UploadRawTrainData", return_values=["train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one(dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset, Task

    task = Task.current_task()
    dir = os.path.join(dataset_root,"train")
    annotations_file = os.path.join(dir, "_annotations.coco.json")
    dataset = Dataset.create(
        dataset_name=f"{dataset_name}RawTrainData", dataset_project=dataset_project
    )
    dataset.add_files(path=dir, wildcard="*.jpg")
    if os.path.exists(annotations_file):
        dataset.add_files(annotations_file)

    dataset.upload()
    dataset.finalize()
    task.set_parameters({"train_dataset_id:": dataset.id})
    print(f"Train dataset uploaded with ID: {dataset.id}")
    return dataset.id 
    
@PipelineDecorator.component(parents=['UploadRawTrainData'],  name="UploadRawTestData", return_values=["test_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two(dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset, Task
    task = Task.current_task()
    dir = os.path.join(dataset_root,"test")
    annotations_file = os.path.join(dir, "_annotations.coco.json")

    dataset = Dataset.create(
       dataset_name=f"{dataset_name}RawTestData", dataset_project=dataset_project
    )

    dataset.add_files(path=dir, wildcard="*.jpg")

    if os.path.exists(annotations_file):
       dataset.add_files(annotations_file)

    dataset.upload()
    dataset.finalize()
    task.set_parameters({"test_dataset_id:": dataset.id})
    print(f"Test dataset uploaded with ID: {dataset.id}")
    return dataset.id 
    
@PipelineDecorator.component(parents=['UploadRawTestData'],name="UploadRawValidData", return_values=["valid_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_three(dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset, Task
    task = Task.current_task()

    dir = os.path.join(dataset_root,"valid")
    annotations_file = os.path.join(dir, "_annotations.coco.json")

    dataset = Dataset.create(
        dataset_name=f"{dataset_name}RawValidData", dataset_project=dataset_project
    )

    dataset.add_files(path=dir, wildcard="*.jpg")

    if os.path.exists(annotations_file):
        dataset.add_files(annotations_file)

    dataset.upload()
    dataset.finalize()
    task.set_parameters({"train_dataset_id:": dataset.id})
    print(f"Valid dataset uploaded with ID: {dataset.id}")
    return dataset.id 
 
@PipelineDecorator.component(parents=['UploadRawValidData'],name="PreprocessAndUploadTrainData", return_values=["preprocessed_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def preprocess_and_upload_train_data(raw_train_dataset_id, dataset_root, dataset_name, processed_dataset_root, cfg):
    #from image_transforms import GaussianBlurTransform, AddGaussianNoiseTransform
    from clearml import Task
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
        
    coco_path = f'{dataset_root}/train/_annotations.coco.json'
    coco = COCO(coco_path)
    processed_images_dir = os.path.join(f'{processed_dataset_root}', 'train')
    os.makedirs(processed_images_dir, exist_ok=True)

    for img_info in coco.dataset['images']:
        image_path = os.path.join(f'{dataset_root}/train', img_info['file_name'])
        image = cv2.imread(image_path)
        #=====
        #image = process_image(image, cfg)  # Assume process_image applies necessary transforms        
        aug_input = AugInput(image)
        aug_list = [
            Resize(shape=(640, 640)),
            RandomFlip(prob=0.5),
            RandomRotation(angle=[-4, 4], expand=False, sample_style='range'),
            GaussianBlurTransform(5),
            AddGaussianNoiseTransform(25)
        ]
        aug_input, _ = apply_transform_gens(aug_list, aug_input)
        image = aug_input.image
        #=====
        processed_image_path = os.path.join(processed_images_dir, f"{img_info['id']:012d}.jpg")
        cv2.imwrite(processed_image_path, image)

    # Save the merged annotations to a JSON file
    annotations_path = os.path.join(processed_images_dir, '_annotations.coco.json')
    with open(annotations_path, 'w') as f:
        json.dump(coco.dataset, f)    

    # Initialize ClearML task only now and upload processed data
    processed_dataset = Dataset.create(dataset_name="BrainScanPreprocessedTrainDataset", dataset_project="BrainScan", parent_datasets= [raw_train_dataset_id])
    processed_dataset.add_files(processed_images_dir)
    processed_dataset.upload()
    processed_dataset.finalize()    
    task = Task.current_task()
    task.set_parameters({"preprocessed_train_dataset_id:": processed_dataset.id})
    return processed_dataset.id

@PipelineDecorator.component(parents=['PreprocessAndUploadTrainData'],name="PreprocessAndUploadTestData", return_values=["preprocessed_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def preprocess_and_upload_test_data(raw_test_dataset_id, dataset_root, dataset_name, processed_dataset_root, cfg):
    #from image_transforms import GaussianBlurTransform, AddGaussianNoiseTransform
    from clearml import Task
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
    coco_path = f'{dataset_root}/test/_annotations.coco.json'
    coco = COCO(coco_path)
    processed_images_dir = os.path.join(f'{processed_dataset_root}', 'test')
    os.makedirs(processed_images_dir, exist_ok=True)

    for img_info in coco.dataset['images']:
        image_path = os.path.join(f'{dataset_root}/test', img_info['file_name'])
        image = cv2.imread(image_path)
        #=====
        #image = process_image(image, cfg)  # Assume process_image applies necessary transforms        
        aug_input = AugInput(image)
        aug_list = [
            Resize(shape=(640, 640)),
            RandomFlip(prob=0.5),
            RandomRotation(angle=[-4, 4], expand=False, sample_style='range'),
            GaussianBlurTransform(5),
            AddGaussianNoiseTransform(25)
        ]
        aug_input, _ = apply_transform_gens(aug_list, aug_input)
        image = aug_input.image
        #=====
        processed_image_path = os.path.join(processed_images_dir, f"{img_info['id']:012d}.jpg")
        cv2.imwrite(processed_image_path, image)

    # Save the merged annotations to a JSON file
    annotations_path = os.path.join(processed_images_dir, '_annotations.coco.json')
    with open(annotations_path, 'w') as f:
        json.dump(coco.dataset, f)    

    # Initialize ClearML task only now and upload processed data
    processed_dataset = Dataset.create(dataset_name="BrainScanPreprocessedTestDataset", dataset_project="BrainScan", parent_datasets= [raw_test_dataset_id])
    processed_dataset.add_files(processed_images_dir)
    processed_dataset.upload()
    processed_dataset.finalize()    
    task = Task.current_task()
    task.set_parameters({"preprocessed_test_dataset_id:": processed_dataset.id})            
    return processed_dataset.id


@PipelineDecorator.component(parents=['PreprocessAndUploadTestData'],name="PreprocessAndUploadValidData", return_values=["preprocessed_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def preprocess_and_upload_valid_data(raw_valid_dataset_id, dataset_root, dataset_name, processed_dataset_root, cfg):
    #from image_transforms import GaussianBlurTransform, AddGaussianNoiseTransform
    from clearml import Task
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
        
    coco_path = f'{dataset_root}/valid/_annotations.coco.json'
    coco = COCO(coco_path)
    processed_images_dir = os.path.join(f'{processed_dataset_root}', 'valid')
    os.makedirs(processed_images_dir, exist_ok=True)

    for img_info in coco.dataset['images']:
        image_path = os.path.join(f'{dataset_root}/valid', img_info['file_name'])
        image = cv2.imread(image_path)
        #=====
        #image = process_image(image, cfg)  # Assume process_image applies necessary transforms        
        aug_input = AugInput(image)
        aug_list = [
            Resize(shape=(640, 640)),
            RandomFlip(prob=0.5),
            RandomRotation(angle=[-4, 4], expand=False, sample_style='range'),
            GaussianBlurTransform(5),
            AddGaussianNoiseTransform(25)
        ]
        aug_input, _ = apply_transform_gens(aug_list, aug_input)
        image = aug_input.image
        #=====
        processed_image_path = os.path.join(processed_images_dir, f"{img_info['id']:012d}.jpg")
        cv2.imwrite(processed_image_path, image)

    # Save the merged annotations to a JSON file
    annotations_path = os.path.join(processed_images_dir, '_annotations.coco.json')
    with open(annotations_path, 'w') as f:
        json.dump(coco.dataset, f)    

    # Initialize ClearML task only now and upload processed data
    processed_dataset = Dataset.create(dataset_name="BrainScanPreprocessedValidDataset", dataset_project="BrainScan", parent_datasets= [raw_valid_dataset_id])
    processed_dataset.add_files(processed_images_dir)
    processed_dataset.upload()
    processed_dataset.finalize()    
    task = Task.current_task()
    task.set_parameters({"preprocessed_valid_dataset_id:": processed_dataset.id})
    return processed_dataset.id

#end Preprocess Raw Data

# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(name="BrainScanDataPipeline", project="Strykers", target_project="Strykers", pipeline_execution_queue="default", default_queue="default") #, version="0.0.6")
def executing_data_pipeline(dataset_project, dataset_name, dataset_root, processed_dataset_name, processed_dataset_root, output_root, cfg):

    print("::=======================================::")
    print("Step1: Launch UploadTrainRawDataset Task")
    print("::=======================================::")
    raw_train_dataset_id = step_one(dataset_project, dataset_name, dataset_root)

    print("::=======================================::")
    print("Step 2: Launch PreprocessRawDataset Task")
    print("::=======================================::")
    raw_test_dataset_id = step_two(dataset_project, dataset_name, dataset_root)
    
    print("::=======================================::")
    print("Step 3: Launch TrainModel Task")
    print("::=======================================::")
    raw_valid_dataset_id = step_three(dataset_project, dataset_name, dataset_root)

    print("::=======================================::")
    print("Step 4: Launch PreprocessAndUploadTrainDataset Task")
    print("::=======================================::")
    processed_train_dataset_id = preprocess_and_upload_train_data(raw_train_dataset_id, dataset_root, dataset_name, processed_dataset_root, cfg)

    print("::=======================================::")
    print("Step 5: Launch PreprocessAndUploadTestDataset Task")
    print("::=======================================::")
    processed_test_dataset_id = preprocess_and_upload_test_data(raw_test_dataset_id, dataset_root, dataset_name, processed_dataset_root, cfg)

    print("::=======================================::")
    print("Step 6 Launch PreprocessAndUploadValidDataset Task")
    print("::=======================================::")
    processed_valid_dataset_id = preprocess_and_upload_valid_data(raw_valid_dataset_id, dataset_root, dataset_name, processed_dataset_root, cfg)

    

if __name__ == "__main__":
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    PipelineDecorator.set_default_execution_queue('default')
    # Run the pipeline steps as subprocesses on the current machine, great for local executions
    # (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)
    PipelineDecorator.run_locally()
    #PipelineDecorator.debug_pipeline()
    # Start the pipeline execution logic.
    def setup_cfg():
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("BrainScanPreprocessedTrainDataset",)
        cfg.DATASETS.TEST = ("BrainScanPreprocessedTestDataset",)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.0025
        cfg.SOLVER.MAX_ITER = 1000
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adjust as per your dataset
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        return cfg

    cfg = setup_cfg()

    executing_data_pipeline(
        dataset_project="BrainScan",
        dataset_name="BrainScan",
        dataset_root="/Users/roysaberon/Developer/GitHub/braintumourdetection/Dataset",
        processed_dataset_root="/Users/roysaberon/Downloads/BrainScan/preprocesseddata",
        processed_dataset_name="BrainScan",
        output_root="/Users/roysaberon/Downloads/BrainScan/output",
        cfg=cfg                      
    )

    print("process completed")