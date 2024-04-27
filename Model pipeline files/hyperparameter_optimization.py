import os
import json
from detectron2.structures import BoxMode
from clearml import Dataset, Task
from detectron2.data import DatasetCatalog, MetadataCatalog
import optuna
import yaml
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
    MetadataCatalog.get(dataset_name).set(thing_classes=["Tumor", "Non-Tumor"])  # Update if more classes

def setup_cfg(lr, ims_per_batch, max_iter, momentum, weight_decay):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ('brain_tumor_train',)
    cfg.DATASETS.TEST = ('brain_tumor_valid',)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_iter
    #cfg.SOLVER.MOMENTUM = momentum
    #cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.OUTPUT_DIR = 'C:/Users/Leon-PC/Downloads/Preprocessed data'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    #cfg.MODEL.DEVICE = 'cpu'  # Use CPU for training
    cfg.MODEL.DEVICE = 'cuda'
    #cfg.DATASETS.ANNOTATIONS = True  # Include annotations in training data
    return cfg

def objective(trial):
    task = Task.create(project_name="Brain Tumor Detection Project", task_name=f"HPO Trial {trial.number}", task_type=Task.TaskTypes.training)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    ims_per_batch = 2 #trial.suggest_categorical('ims_per_batch', [2, 4, 8, 16])
    max_iter = trial.suggest_int('max_iter', 1000, 3500)
    momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    cfg = setup_cfg(lr, ims_per_batch, max_iter, momentum, weight_decay)
    task.connect(cfg)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    evaluator = COCOEvaluator(dataset_name="brain_tumor_valid", output_dir='C:/Users/Leon-PC/Downloads/Preprocessed data')
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    val_loss = trainer.test(cfg, trainer.model)
    if isinstance(val_loss, list):
        return val_loss[0]
    elif isinstance(val_loss, float):
        return val_loss
    else:
        return 0.0  # Return a default value if no valid value can be calculated
    task.close()

def main():
    task = Task.init(project_name='Brain Tumor Detection Project', task_name='HyperParameter Optimization', task_type=Task.TaskTypes.optimizer)
    
    datasets = {
        "brain_tumor_train": "4f4ee1971cfe4f8e95109414834d5607",
        "brain_tumor_valid": "5bc2bf9094e74e90be7561d5f8d7a591",
        "brain_tumor_test": "8c2eee1fdbdc4a529a42d3de0ac3d1fa"
    }

    for name, id in datasets.items():
        register_dataset(name, id)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    best_hyperparams_path = 'C:/Users/Leon-PC/Downloads/Preprocessed data/best_hyperparams.yaml'
    with open(best_hyperparams_path, 'w') as file:
        yaml.dump(study.best_trial.params, file, default_flow_style=False)
    all_trials_path = 'C:/Users/Leon-PC/Downloads/Preprocessed data/all_trials.yaml'
    with open(all_trials_path, 'w') as file:
        yaml.dump(study.trials_dataframe().to_dict(), file, default_flow_style=False)
    task.upload_artifact('best_hyperparameters', best_hyperparams_path)
    task.upload_artifact('study_summary', all_trials_path)
    task.close()
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best hyperparameters: {study.best_trial.params}")

if __name__ == "__main__":
    main()