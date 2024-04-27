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
from pycocotools.coco import COCO
from clearml import Dataset, Task, TaskTypes
from clearml.automation.controller import PipelineDecorator
from detectron2.data import DatasetMapper, DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.transforms import Transform, apply_transform_gens, AugInput, Resize, RandomFlip, RandomRotation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import Checkpointer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from registering_preprocessed_datasets import register_dataset
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn.metrics import accuracy_score



def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_valid",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adjust as per your dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    return cfg


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(name="UploadRawTrainData", return_values=["train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one(dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset

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
    print(f"Train dataset uploaded with ID: {dataset.id}")
    return dataset.id 
    
@PipelineDecorator.component(name="UploadRawTestData", return_values=["test_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two(dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset

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

    print(f"Test dataset uploaded with ID: {dataset.id}")
    return dataset.id 
    
@PipelineDecorator.component(name="UploadRawValidData", return_values=["valid_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_three(dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset

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

    print(f"Valid dataset uploaded with ID: {dataset.id}")
    return dataset.id 
 
@PipelineDecorator.component(name="PreprocessAndUploadTrainData", return_values=["preprocessed_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def preprocess_and_upload_train_data(raw_train_dataset_id, dataset_root, dataset_name, processed_dataset_root, cfg):
    #from image_transforms import GaussianBlurTransform, AddGaussianNoiseTransform

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
            #GaussianBlurTransform(5),
            #AddGaussianNoiseTransform(25)
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

    return processed_dataset.id

@PipelineDecorator.component(name="PreprocessAndUploadTestData", return_values=["preprocessed_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def preprocess_and_upload_test_data(raw_test_dataset_id, dataset_root, dataset_name, processed_dataset_root, cfg):
    #from image_transforms import GaussianBlurTransform, AddGaussianNoiseTransform

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
            #GaussianBlurTransform(5),
            #AddGaussianNoiseTransform(25)
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

    return processed_dataset.id


@PipelineDecorator.component(name="PreprocessAndUploadValidData", return_values=["preprocessed_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def preprocess_and_upload_valid_data(raw_valid_dataset_id, dataset_root, dataset_name, processed_dataset_root, cfg):
    #from image_transforms import GaussianBlurTransform, AddGaussianNoiseTransform

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
            #GaussianBlurTransform(5),
            #AddGaussianNoiseTransform(25)
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

    return processed_dataset.id

#end Preprocess Raw Data

@PipelineDecorator.component(name="TrainModel", return_values=['model'], cache=True, task_type=TaskTypes.training)#, execution_queue="default")
def train_model(train_dataset_id, valid_dataset_id, output_root):
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

    task = Task.current_task()
        
    SEED = 99
    THRESHOLD = 0.6
    #EPOCHS = 3300
    #NUM_CLASSES = 3
    BASE_LR = 0.0001

    # Register the dataset
    register_dataset("BrainScanPreprocessedTrainDataset", train_dataset_id)
    register_dataset("BrainScanPreprocessedValidDataset", valid_dataset_id)
   
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

    task.upload_artifact('trained_model', model_path)
    

# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
# Specifying `return_values` makes sure the function step can return an object to the pipeline logic
# In this case, the returned object will be stored as an artifact named "accuracy"
@PipelineDecorator.component(name="EvaluateModel",return_values=["accuracy"], cache=True, task_type=TaskTypes.qc)#, execution_queue="default")
def evaluate_model(valid_dataset_id, output_root):
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
    from clearml import Task
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
    from detectron2.engine import DefaultPredictor        
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from registering_preprocessed_datasets import register_dataset
    

    SEED = 99
    THRESHOLD = 0.6    
    
    register_dataset("BrainScanPreprocessedValidDataset", valid_dataset_id)

    # Load the saved config
    with open("cfg.pkl", "rb") as f:
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
    task = Task.get_task()
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

        task = Task.get_task()

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
        with open('report.html', 'w') as f:
            f.write(html_report)
        task.upload_artifact('report.html', 'report.html')
        # Upload the report
        #task.upload_report(report)

    create_predictions(dataset_dicts, my_dataset_test_metadata, seed=421, image_scale=1)
    create_predictions(dataset_dicts, my_dataset_test_metadata, seed=83, image_scale=1)

    print("Evaluating model")

@PipelineDecorator.component(name="TestModel", cache=True, task_type=TaskTypes.qc)#, execution_queue="default")
def test_model():

    print("Testing model")




@PipelineDecorator.component(name="UploadModelToGit", cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def upload_model(model):

    print("Uploading model to github")




# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(name="BrainScanPipeline", project="Strykers", target_project="Strykers", pipeline_execution_queue="default", default_queue="default") #, version="0.0.6")
def executing_pipeline(dataset_project, dataset_name, dataset_root, processed_dataset_name, processed_dataset_root, output_root, cfg):
    #print("pipeline args:", pickle_url, mock_parameter)

    # Use the pipeline argument to start the pipeline and pass it ot the first step
    print("::=======================================::")
    print("Step1: Launch UploadTrainRawDataset Task")
    print("::=======================================::")
    raw_train_dataset_id = step_one(dataset_project, dataset_name, dataset_root)

    # Use the returned data from the first step (`step_one`), and pass it to the next step (`step_two`)
    # Notice! unless we actually access the `data_frame` object,
    # the pipeline logic does not actually load the artifact itself.
    # When actually passing the `data_frame` object into a new step,
    # It waits for the creating step/function (`step_one`) to complete the execution
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

    # Notice since we are "printing" the `model` object,
    # we actually deserialize the object from the third step, and thus wait for the third step to complete.
    #print("returned model: {}".format(model))
    print("::=======================================::")
    print("Step 7. Launch TrainModel Task")
    print("::=======================================::")
    train_model(processed_train_dataset_id, processed_valid_dataset_id, output_root)

    print("::=======================================::")
    print("Step 8. Launch EvaluateModel Task")
    print("::=======================================::")
    evaluate_model(processed_valid_dataset_id, output_root)

    print("::=======================================::")
    print("Step 9. Launch TestModel Task")
    print("::=======================================::")
    test_model()

    # Notice since we are "printing" the `accuracy` object,
    # we actually deserialize the object from the fourth step, and thus wait for the fourth step to complete.
    #print(f"Accuracy={accuracy}%")

    print("::=======================================::")
    print("Step 10. Launch UploadModel Task")
    print("::=======================================::")
    upload_model()


if __name__ == "__main__":
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    PipelineDecorator.set_default_execution_queue('default')
    # Run the pipeline steps as subprocesses on the current machine, great for local executions
    # (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)
    PipelineDecorator.run_locally()
    #PipelineDecorator.debug_pipeline()
    # Start the pipeline execution logic.

    cfg = setup_cfg()
    executing_pipeline(
        dataset_project="BrainScan",
        dataset_name="BrainScan",
        dataset_root="/Users/roysaberon/Developer/GitHub/braintumourdetection/Dataset",
        processed_dataset_root="/var/folders/h6/g7hdfnxn5mgb5_8z9fg0w1d00000gp/T/preprocesseddata",
        processed_dataset_name="BrainScan",
        output_root="/var/folders/h6/g7hdfnxn5mgb5_8z9fg0w1d00000gp/T/output",
        cfg=cfg                      
    )

    print("process completed")