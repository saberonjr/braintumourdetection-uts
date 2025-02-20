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


@PipelineDecorator.component(name="OptimizeHyperparameters", return_values=['task_id'], cache=True, task_type=TaskTypes.training)#, execution_queue="default")
def optimize_hyperparameters(train_dataset_id, valid_dataset_id, test_dataaset_id, output_root):
    import os
    import json
    import optuna
    import yaml
    from clearml import Dataset, Task    
    from detectron2.structures import BoxMode
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.evaluation import COCOEvaluator

    def get_dataset_dicts(dataset_id):
        dataset = Dataset.get(dataset_id=dataset_id)
        dataset_path = dataset.get_local_copy()

        # Path to the COCO-format JSON file
        json_file = os.path.join(dataset_path, '_annotations.coco.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []
        for img in imgs_anns['images']:
            record = {}
            file_name = f"{img['id']:012}.jpg"  # Adjust the file name format here
            record["file_name"] = os.path.join(dataset_path, file_name)
            record["image_id"] = img['id']
            record["height"] = img['height']
            record["width"] = img['width']
            annos = [anno for anno in imgs_anns['annotations'] if anno['image_id'] == img['id']]
            objs = []
            for anno in annos:
                obj = {
                    "bbox": anno['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,  # Make sure to use the correct bbox mode
                    "category_id": anno['category_id'] - 1,  # Adjust the category_id here
                    "segmentation": anno.get("segmentation", []),
                    "iscrowd": anno.get("iscrowd", 0)
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    def register_dataset(dataset_name, dataset_id):
        def loader():
            return get_dataset_dicts(dataset_id)
        DatasetCatalog.register(dataset_name, loader)
        MetadataCatalog.get(dataset_name).set(thing_classes=["Tumour", "Non-Tumour"])  # Update if more classes

    def setup_cfg(lr, ims_per_batch, max_iter, momentum, weight_decay, output_root):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ('BrainScanPreprocessedTrainDataset',)
        cfg.DATASETS.TEST = ('BrainScanPreprocessedTestDataset',)
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.MAX_ITER = max_iter
        #cfg.SOLVER.MOMENTUM = momentum
        #cfg.SOLVER.WEIGHT_DECAY = weight_decay
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.OUTPUT_DIR = output_root
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.MODEL.DEVICE = 'cpu'  # Use CPU for training
        #cfg.MODEL.DEVICE = 'cuda'
        #cfg.DATASETS.ANNOTATIONS = True  # Include annotations in training data
        return cfg

    def objective(trial):
        task = Task.create(project_name="BrainScan", task_name=f"HPOTrial{trial.number}", task_type=Task.TaskTypes.training)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        ims_per_batch = 2 #trial.suggest_categorical('ims_per_batch', [2, 4, 8, 16])
        max_iter = trial.suggest_int('max_iter', 1000, 3500)
        momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
        cfg = setup_cfg(lr, ims_per_batch, max_iter, momentum, weight_decay, output_root)
        task.connect(cfg)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        evaluator = COCOEvaluator(dataset_name="BrainScanPreprocessedValidDataset", output_dir=output_root)
        trainer.test(cfg, trainer.model, evaluators=[evaluator])
        val_loss = trainer.test(cfg, trainer.model)
        if isinstance(val_loss, list):
            return val_loss[0]
        elif isinstance(val_loss, float):
            return val_loss
        else:
            return 0.0  # Return a default value if no valid value can be calculated
        task.close()

    task = Task.init(project_name='BrainScan', task_name='OptimizeHyperparameters', task_type=Task.TaskTypes.optimizer)
    
    datasets = {
        "BrainScanPreprocessedTrainDataset": train_dataset_id,
        "BrainScanPreprocessedValidDataset": valid_dataset_id,
        "BrainScanPreprocessedTestDataset": test_dataaset_id
    }

    for name, id in datasets.items():
        register_dataset(name, id)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    best_hyperparams_path = f'{output_root}/best_hyperparams.yaml'
    with open(best_hyperparams_path, 'w') as file:
        yaml.dump(study.best_trial.params, file, default_flow_style=False)
    all_trials_path = f'{output_root}/all_trials.yaml'
    with open(all_trials_path, 'w') as file:
        yaml.dump(study.trials_dataframe().to_dict(), file, default_flow_style=False)
    task.upload_artifact('best_hyperparameters', best_hyperparams_path)
    task.upload_artifact('study_summary', all_trials_path)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best hyperparameters: {study.best_trial.params}")

    return task.id


@PipelineDecorator.component(parents = ['OptimizeHyperparameters'],name="TrainModel", return_values=['task_id'], cache=True, task_type=TaskTypes.training)#, execution_queue="default")
def train_model(optimize_hyperparameter_task_id, train_dataset_id, valid_dataset_id, test_dataset_id, output_root):
    import os
    import json
    import yaml
    from clearml import Dataset, Task
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.engine import DefaultTrainer, DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.structures import BoxMode
    #from registering_preprocessed_datasets import register_dataset
    import pickle
    from detectron2.checkpoint import Checkpointer

    def get_dataset_dicts(dataset_id):
        dataset = Dataset.get(dataset_id=dataset_id)
        dataset_path = dataset.get_local_copy()

        # Path to the COCO-format JSON file
        json_file = os.path.join(dataset_path, '_annotations.coco.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []
        for img in imgs_anns['images']:
            record = {}
            file_name = f"{img['id']:012}.jpg"  # Adjust the file name format here
            record["file_name"] = os.path.join(dataset_path, file_name)
            record["image_id"] = img['id']
            record["height"] = img['height']
            record["width"] = img['width']
            annos = [anno for anno in imgs_anns['annotations'] if anno['image_id'] == img['id']]
            objs = []
            for anno in annos:
                obj = {
                    "bbox": anno['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,  # Make sure to use the correct bbox mode
                    "category_id": anno['category_id'] - 1,  # Adjust the category_id here
                    "segmentation": anno.get("segmentation", []),
                    "iscrowd": anno.get("iscrowd", 0)
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts
    
    def register_dataset(dataset_name, dataset_id):
        def loader():
            return get_dataset_dicts(dataset_id)
        DatasetCatalog.register(dataset_name, loader)
        MetadataCatalog.get(dataset_name).set(thing_classes=["Tumour", "Non-Tumour"])  # Update if more classes

    task = Task.current_task()
        
    SEED = 99
    THRESHOLD = 0.6
    #EPOCHS = 3300
    #NUM_CLASSES = 3
    BASE_LR = 0.0001

    # Register the dataset
    register_dataset("BrainScanPreprocessedTrainDataset", train_dataset_id)
    register_dataset("BrainScanPreprocessedValidDataset", valid_dataset_id)
    register_dataset("BrainScanPreprocessedTestDataset", test_dataset_id)
   
    # Load the best hyperparameters from the HPO script
    with open(f'{output_root}/best_hyperparams.yaml', 'r') as file:
        best_hyperparams = yaml.safe_load(file)
    
    # Set up the configuration for the model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ('BrainScanPreprocessedTrainDataset',)
    cfg.DATASETS.TEST = ('BrainScanPreprocessedTestDataset',)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2 #best_hyperparams['ims_per_batch']
    cfg.SOLVER.BASE_LR = BASE_LR #best_hyperparams['lr']
    cfg.SOLVER.MAX_ITER = best_hyperparams['max_iter']
    cfg.SOLVER.MOMENTUM = best_hyperparams['momentum']
    cfg.SOLVER.WEIGHT_DECAY = best_hyperparams['weight_decay']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.OUTPUT_DIR = output_root
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.DEVICE = 'cpu'  # Use CPU for training
    #cfg.MODEL.DEVICE = 'cuda'
    #cfg.DATASETS.ANNOTATIONS = True  # Include annotations in training data
    
    # Train the model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Save the trained model
    model_path = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    checkpointer = Checkpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save("model_final")

    # Save the config
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD
    #predictor = DefaultPredictor(cfg)
    
    with open(f'{output_root}/cfg.pkl', "wb") as f:
        pickle.dump(cfg, f)

    task.upload_artifact('trained_model', model_path)
    return task.id

# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
# Specifying `return_values` makes sure the function step can return an object to the pipeline logic
# In this case, the returned object will be stored as an artifact named "accuracy"
@PipelineDecorator.component(parents=['TrainModel'],name="EvaluateModel",return_values=["task_id"], cache=True, task_type=TaskTypes.qc)#, execution_queue="default")
def evaluate_model(train_model_task_id, test_dataset_id, valid_dataset_id, output_root):
    from sklearn.linear_model import LogisticRegression  # noqa
    from sklearn.metrics import accuracy_score

    import os
    import json
    import yaml
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2    
    import torch
    from detectron2.structures import BoxMode
    from clearml import Task
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
    from detectron2.engine import DefaultPredictor        
    from detectron2.utils.visualizer import Visualizer, ColorMode
    #from registering_preprocessed_datasets import register_dataset
    
    def get_dataset_dicts(dataset_id):
        dataset = Dataset.get(dataset_id=dataset_id)
        dataset_path = dataset.get_local_copy()

        # Path to the COCO-format JSON file
        json_file = os.path.join(dataset_path, '_annotations.coco.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []
        for img in imgs_anns['images']:
            record = {}
            file_name = f"{img['id']:012}.jpg"  # Adjust the file name format here
            record["file_name"] = os.path.join(dataset_path, file_name)
            record["image_id"] = img['id']
            record["height"] = img['height']
            record["width"] = img['width']
            annos = [anno for anno in imgs_anns['annotations'] if anno['image_id'] == img['id']]
            objs = []
            for anno in annos:
                obj = {
                    "bbox": anno['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,  # Make sure to use the correct bbox mode
                    "category_id": anno['category_id'] - 1,  # Adjust the category_id here
                    "segmentation": anno.get("segmentation", []),
                    "iscrowd": anno.get("iscrowd", 0)
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts
    
    def register_dataset(dataset_name, dataset_id):
        def loader():
            return get_dataset_dicts(dataset_id)
        DatasetCatalog.register(dataset_name, loader)
        MetadataCatalog.get(dataset_name).set(thing_classes=["Tumour", "Non-Tumour"])  # Update if more classes


    SEED = 99
    THRESHOLD = 0.6    
    
    register_dataset("BrainScanPreprocessedValidDataset", valid_dataset_id)
    register_dataset("BrainScanPreprocessedTestDataset", test_dataset_id)

    # Load the saved config
    with open(f'{output_root}/cfg.pkl', "rb") as f:
        cfg = pickle.load(f)

    # Set the device to GPU
    #cfg.MODEL.DEVICE = 'cuda'

    # Create a predictor
    predictor = DefaultPredictor(cfg)

    # Load the test dataset
    test_dataset = DatasetCatalog.get("BrainScanPreprocessedTestDataset")

    # Create a COCO evaluator
    output_dir = f'{output_root}'
    evaluator = COCOEvaluator("BrainScanPreprocessedValidDataset", False, output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, "BrainScanPreprocessedValidDataset", num_workers=4)

    # Evaluate the model
    inference_on_dataset(predictor.model, test_loader, evaluator)

    # Upload the evaluation results to ClearML
    task = Task.current_task()
    task.upload_artifact("evaluation_results", os.path.join(output_dir, "coco_instances_results.json"))

    # Visualize the predictions
    my_dataset_test_metadata = MetadataCatalog.get("BrainScanPreprocessedValidDataset")
    dataset_dicts = DatasetCatalog.get("BrainScanPreprocessedValidDataset")

    def create_predictions(dataset_dict, dataset_metadata, seed, image_scale=0.8):
        # ... (rest of the function remains the same)
        np.random.seed(seed=seed)
        images = np.random.permutation(dataset_dict)[:3]

        fig, axs = plt.subplots(3,2, figsize = (20,20), dpi = 120)

        for i in range(3):
            im = images[i]
            img_link = im['file_name']
            img_id = im['image_id']
            img = cv2.imread(img_link)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            visualizer1 = Visualizer(img, metadata= dataset_metadata, scale=image_scale)

            vis_original = visualizer1.draw_dataset_dict(im)
            original_bbox = vis_original.get_image()

            visualizer2 = Visualizer(img[:, :, ::-1], metadata= dataset_metadata, scale=image_scale, instance_mode=ColorMode.IMAGE_BW)
            #visualizer2 = Visualizer(img, metadata= dataset_metadata, scale=image_scale, instance_mode=ColorMode.IMAGE_BW)
            # Move the image tensor to the GPU
            #img = torch.from_numpy(img).to('cuda')
            #img = img.to('cuda')

            # Move the predictor model to the GPU
            #predictor.model.to('cuda')
            #outputs = predictor(img.cpu().numpy())
            outputs = predictor(img)
            out = visualizer2.draw_instance_predictions(outputs["instances"].to("cpu"))
            out_img = cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB)
            final_bbox = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

            axs[i][0].set_title('Predicted bbox (id: ' + str(img_id) +')', fontsize = 20, color = 'red')
            axs[i][0].axis('off')
            axs[i][0].imshow(original_bbox)

            axs[i][1].set_title('Original (id: ' + str(img_id) +')', fontsize = 20)
            axs[i][1].axis('off')
            axs[i][1].imshow(final_bbox[:, :, ::-1])

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_{seed}_original.png'))

        task = Task.current_task()

        # Create a report
        report = {
            'title': 'Model Evaluation Predictions',
            'text': 'These are the predictions',
            'images': [
                {
                    'title': 'Prediction 1',
                    'image': os.path.join(output_dir, 'prediction_1.png')
                },
                {
                    'title': 'Prediction 2',
                    'image': os.path.join(output_dir, 'prediction_2.png')
                }
            ]
        }

       # Create an HTML report
        html_report = '<h1>{title}</h1><p>{text}</p>'.format(**report)
        for image in report['images']:
            html_report += '<h2>{title}</h2><img src="{image}">'.format(**image)

        # Upload the report as an HTML file
        with open(f'{output_root}/report.html', 'w') as f:
            f.write(html_report)
        task.upload_artifact('report.html', html_report)
        # Upload the report
        #task.upload_report(report)

    create_predictions(dataset_dicts, my_dataset_test_metadata, seed=421, image_scale=1)
    create_predictions(dataset_dicts, my_dataset_test_metadata, seed=83, image_scale=1)

    print("Evaluating model")

    return task.id

@PipelineDecorator.component(parents=['EvaluateModel'],name="TestModel", return_values=['task_id'], cache=True, task_type=TaskTypes.qc)#, execution_queue="default")
def test_model(evaluate_model_task_id, test_dataset_id, output_root):
    from clearml import Task
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import ColorMode
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.utils.visualizer import Visualizer
    from detectron2.structures import BoxMode
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import os

    def get_dataset_dicts(dataset_id):
        dataset = Dataset.get(dataset_id=dataset_id)
        dataset_path = dataset.get_local_copy()

        # Path to the COCO-format JSON file
        json_file = os.path.join(dataset_path, '_annotations.coco.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []
        for img in imgs_anns['images']:
            record = {}
            file_name = f"{img['id']:012}.jpg"  # Adjust the file name format here
            record["file_name"] = os.path.join(dataset_path, file_name)
            record["image_id"] = img['id']
            record["height"] = img['height']
            record["width"] = img['width']
            annos = [anno for anno in imgs_anns['annotations'] if anno['image_id'] == img['id']]
            objs = []
            for anno in annos:
                obj = {
                    "bbox": anno['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,  # Make sure to use the correct bbox mode
                    "category_id": anno['category_id'] - 1,  # Adjust the category_id here
                    "segmentation": anno.get("segmentation", []),
                    "iscrowd": anno.get("iscrowd", 0)
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts
    
    def register_dataset(dataset_name, dataset_id):
        def loader():
            return get_dataset_dicts(dataset_id)
        DatasetCatalog.register(dataset_name, loader)
        MetadataCatalog.get(dataset_name).set(thing_classes=["Tumour", "Non-Tumour"])  # Update if more classes


    task = Task.init(project_name='BrainScan', task_name='TestModel', task_type=Task.TaskTypes.testing)

    register_dataset("BrainScanPreprocessedTestDataset",  test_dataset_id)

    # Load the saved config
    with open(f'{output_root}/cfg.pkl', "rb") as f:
        cfg = pickle.load(f)

    # Create a predictor
    predictor = DefaultPredictor(cfg)

    output_dir=output_root
    # Create a COCO evaluator
    evaluator = COCOEvaluator("BrainScanPreprocessedTestDataset", False, output_dir=output_dir)

    # Build the test loader
    test_loader = build_detection_test_loader(cfg, "BrainScanPreprocessedTestDataset")

    # Evaluate the model
    inference_on_dataset(predictor.model, test_loader, evaluator)

    # Upload the evaluation results to ClearML
    task.upload_artifact("evaluation_results", os.path.join(output_dir, "coco_instances_results.json"))

    # Visualize the predictions
    my_dataset_test_metadata = MetadataCatalog.get("BrainScanPreprocessedTestDataset")
    dataset_dicts = DatasetCatalog.get("BrainScanPreprocessedTestDataset")

    def create_predictions(dataset_dict, dataset_metadata, seed, image_scale=0.8):
        np.random.seed(seed=seed)
        images = np.random.permutation(dataset_dict)[:3]

        fig, axs = plt.subplots(3,2, figsize = (20,20), dpi = 120)

        for i in range(3):
            im = images[i]
            img_link = im['file_name']
            img_id = im['image_id']
            img = cv2.imread(img_link)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            visualizer1 = Visualizer(img, metadata= dataset_metadata, scale=image_scale)

            vis_original = visualizer1.draw_dataset_dict(im)
            original_bbox = vis_original.get_image()

            visualizer2 = Visualizer(img[:, :, ::-1], metadata= dataset_metadata, scale=image_scale, instance_mode=ColorMode.IMAGE_BW)
    
            outputs = predictor(img)
            out = visualizer2.draw_instance_predictions(outputs["instances"].to("cpu"))
            out_img = cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB)
            final_bbox = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

            axs[i][0].set_title('Original bbox (id: ' + str(img_id) +')', fontsize = 20)
            axs[i][0].axis('off')
            axs[i][0].imshow(original_bbox)

            axs[i][1].set_title('Predicted bbox (id: ' + str(img_id) +')', fontsize = 20, color = 'red')
            axs[i][1].axis('off')
            axs[i][1].imshow(final_bbox[:, :, ::-1])

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_{seed}_original.png'))

        # Create a report
        report = {
            'title': 'Model Evaluation Predictions',
            'text': 'These are the predictions',
            'images': [
                {
                    'title': 'Prediction 1',
                    'image': os.path.join(output_dir, 'prediction_1.png')
                },
                {
                    'title': 'Prediction 2',
                    'image': os.path.join(output_dir, 'prediction_2.png')
                }
            ]
        }

       # Create an HTML report
        html_report = '<h1>{title}</h1><p>{text}</p>'.format(**report)
        for image in report['images']:
            html_report += '<h2>{title}</h2><img src="{image}">'.format(**image)

        # Upload the report as an HTML file
        with open(f'{output_root}/report.html', 'w') as f:
            f.write(html_report)
        task.upload_artifact('report.html', html_report)
    
    create_predictions(dataset_dicts, my_dataset_test_metadata, seed=154, image_scale=1)
    create_predictions(dataset_dicts, my_dataset_test_metadata, seed=51, image_scale=1)
    return task.id
  
@PipelineDecorator.component(parents=['TestModel'],name="UploadModel", cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def upload_model(model_id, env_path, REPO_URL, DEVELOPMENT_BRANCH, project_name):

    import argparse
    import datetime
    import os
    import shutil
    from clearml import Model, Task
    from dotenv import load_dotenv
    from git import GitCommandError, Repo
    import argparse
    import datetime
    import os
    import shutil

   
    def configure_ssh_key(DEPLOY_KEY_PATH):
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Configure Git to use the SSH deploy key for operations."""
        os.environ["GIT_SSH_COMMAND"] = f"ssh -i {DEPLOY_KEY_PATH} -o IdentitiesOnly=yes"


    def clone_repo(REPO_URL, branch, DEPLOY_KEY_PATH) -> tuple[Repo, str]:
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Clone the repository."""
        configure_ssh_key(DEPLOY_KEY_PATH)
        repo_path = REPO_URL.split("/")[-1].split(".git")[0]
        try:
            repo: Repo = Repo.clone_from(
                REPO_URL, repo_path, branch=branch, single_branch=True
            )
            print(repo_path)
            return repo, repo_path
        except GitCommandError as e:
            print(f"Failed to clone repository: {e}")
            exit(1)


    def ensure_archive_dir(repo: Repo):
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Ensures the archive directory exists within weights."""
        archive_path = os.path.join(repo.working_tree_dir, "weights", "archive")
        os.makedirs(archive_path, exist_ok=True)


    def archive_existing_model(repo: Repo) -> str:
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Archives existing model weights."""

        weights_path = os.path.join(repo.working_tree_dir, "weights")
        model_file = os.path.join(weights_path, "model.h5")
        if os.path.exists(model_file):
            today = datetime.date.today().strftime("%Y%m%d")
            archived_model_file = os.path.join(weights_path, "archive", f"model-{today}.h5")
            os.rename(model_file, archived_model_file)
            return archived_model_file  # Return the path of the archived file


    def update_weights(repo: Repo, model_path):
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Updates the model weights in the repository."""
        weights_path = os.path.join(repo.working_tree_dir, "weights")
        ensure_archive_dir(repo)
        archived_model_file = archive_existing_model(repo)
        target_model_path = os.path.join(weights_path, "model.h5")
        shutil.move(model_path, target_model_path)  # Use shutil.move for cross-device move
        # Add the newly archived model file to the Git index
        repo.index.add([archived_model_file])
        # Also add the new model file to the Git index
        repo.index.add([target_model_path])


    def commit_and_push(repo: Repo, model_id, DEVELOPMENT_BRANCH):
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Commits and pushes changes to the remote repository."""
        commit_message = f"Update model weights: {model_id}"
        tag_name = f"{model_id}-{datetime.datetime.now().strftime('%Y%m%d')}"
        try:
            repo.index.commit(commit_message)
            repo.create_tag(tag_name, message="Model update")
            repo.git.push("origin", DEVELOPMENT_BRANCH)
            repo.git.push("origin", "--tags")
        except GitCommandError as e:
            print(f"Failed to commit and push changes: {e}")
            exit(1)


    def cleanup_repo(repo_path):
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model, Task
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Safely remove the cloned repository directory."""
        shutil.rmtree(repo_path, ignore_errors=True)


    task = Task.init(
        project_name=project_name,
        task_name="UploadModel",
        task_type=Task.TaskTypes.custom,
    )
    
    #task.execute_remotely(queue_name="queue_name", exit_process=True)
    
    """Fetches the trained model using its ID and updates it in the repository."""
    load_dotenv(dotenv_path=env_path)
    DEPLOY_KEY_PATH = os.getenv("DEPLOY_KEY_PATH")

    # Prepare repository and SSH key
    repo, repo_path = clone_repo(REPO_URL, DEVELOPMENT_BRANCH, DEPLOY_KEY_PATH)
    try:
        # Fetch the trained model
        model = Model(model_id=model_id)
        model_path = model.get_local_copy()

        # Update weights and push changes
        update_weights(repo, model_path)
        commit_and_push(repo, model_id, DEVELOPMENT_BRANCH)
    finally:
        cleanup_repo(repo_path)  # Ensure cleanup happens even if an error occurs

@PipelineDecorator.pipeline(name="BrainScanModelPipeline", project="Strykers", target_project="Strykers", pipeline_execution_queue="default", default_queue="default") #, version="0.0.6")
def executing_model_pipeline(dataset_project, dataset_name, dataset_root, processed_dataset_name, processed_dataset_root, output_root, cfg):

    #task = Task.init(project_name=dataset_project, task_name="PreprocessAndUploadTrainData")
    #params = task.get_parameters()
    processed_train_dataset_id = "c3249b39ff7243fabff953851ccb297d" # params["preprocessed_train_dataset_id"]
    #task = Task.init(project_name=dataset_project, task_name="PreprocessAndUploadValidData")
    #params = task.get_parameters()
    processed_valid_dataset_id = "93a57f9fab4842f796c91be9ad18c129" #params["preprocessed_valid_dataset_id"]   
    #task = Task.init(project_name=dataset_project, task_name="PreprocessAndUploadTestData")
    #params = task.get_parameters()
    processed_test_dataset_id = "08ed95e86a5a4de686ce75834f4172ca"# params["preprocessed_test_dataset_id"]

    print("::=======================================::")
    print("Step 1. Launch OptimizeHyperparameters Task")
    print("::=======================================::")
    #optimize_hyperparameter_task_id = optimize_hyperparameters(processed_train_dataset_id, processed_valid_dataset_id, processed_test_dataset_id, output_root)
    optimize_hyperparameter_task_id = "654e2f17cc7e46bcb73e833ae6992c93"

    print("::=======================================::")
    print("Step 2. Launch TrainModel Task")
    print("::=======================================::")
    train_model_task_id = train_model(optimize_hyperparameter_task_id, processed_train_dataset_id, processed_valid_dataset_id, processed_test_dataset_id, output_root)

    print("::=======================================::")
    print("Step 3. Launch EvaluateModel Task")
    print("::=======================================::")
    evaluate_model_task_id = evaluate_model(train_model_task_id, processed_test_dataset_id, processed_valid_dataset_id, output_root)

    print("::=======================================::")
    print("Step 4. Launch TestModel Task")
    print("::=======================================::")
    test_model_task_id = test_model(evaluate_model_task_id, processed_test_dataset_id, output_root)

    print("::=======================================::")
    print("Step 5. Launch UploadModel Task")
    print("::=======================================::")
    #upload_model(test_model_task_id, model_id, env_path, REPO_URL, DEVELOPMENT_BRANCH, project_name)


if __name__ == "__main__":
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    PipelineDecorator.set_default_execution_queue('default')
    # Run the pipeline steps as subprocesses on the current machine, great for local executions
    # (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)
    #PipelineDecorator.run_locally()
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

    executing_model_pipeline(
        dataset_project="BrainScan",
        dataset_name="BrainScan",
        dataset_root="/Users/soterojrsaberon/Downloads/brainscan/Dataset",
        processed_dataset_root="/Users/soterojrsaberon/Downloads/brainscan/preprocesseddata",
        processed_dataset_name="BrainScan",
        output_root="/Users/soterojrsaberon/Downloads/brainscan/output",
        cfg=cfg                      
    )

    print("process completed")
