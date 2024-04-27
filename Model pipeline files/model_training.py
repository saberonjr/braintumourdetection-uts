import os
import json
import yaml
from clearml import Dataset, Task
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from registering_preprocessed_datasets import register_dataset
import pickle
from detectron2.checkpoint import Checkpointer

with open('C:/Users/Leon-PC/Downloads/Preprocessed data/best_hyperparams.yaml', 'r') as file:
    best_hyperparams = yaml.safe_load(file)
    print("Best Hyperparameters:", best_hyperparams)

SEED = 99
THRESHOLD = 0.6
#EPOCHS = 3300
#NUM_CLASSES = 3
BASE_LR = 0.0001

def main():
    task = Task.init(project_name='Brain Tumor Detection Project', task_name='Model Training', task_type=Task.TaskTypes.training)
    
    # Register the dataset
    register_dataset("brain_tumor_train", "4f4ee1971cfe4f8e95109414834d5607")
    register_dataset("brain_tumor_valid", "5bc2bf9094e74e90be7561d5f8d7a591")
    # Load the best hyperparameters from the HPO script
    with open('C:/Users/Leon-PC/Downloads/Preprocessed data/best_hyperparams.yaml', 'r') as file:
        best_hyperparams = yaml.safe_load(file)
    
    # Set up the configuration for the model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ('brain_tumor_train',)
    cfg.DATASETS.TEST = ('brain_tumor_valid',)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2 #best_hyperparams['ims_per_batch']
    cfg.SOLVER.BASE_LR = BASE_LR #best_hyperparams['lr']
    cfg.SOLVER.MAX_ITER = best_hyperparams['max_iter']
    cfg.SOLVER.MOMENTUM = best_hyperparams['momentum']
    cfg.SOLVER.WEIGHT_DECAY = best_hyperparams['weight_decay']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.OUTPUT_DIR = 'C:/Users/Leon-PC/Downloads/Preprocessed data'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    #cfg.MODEL.DEVICE = 'cpu'  # Use CPU for training
    cfg.MODEL.DEVICE = 'cuda'
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
    with open("cfg.pkl", "wb") as f:
        pickle.dump(cfg, f)

    # Upload the trained model to ClearML
    task.upload_artifact('trained_model', model_path)
    #model_path = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    #trainer.save_model(model_path)
    #task.upload_artifact('trained_model', model_path)
    
    task.close()

if __name__ == "__main__":
    main()
