import os
import json
import yaml
from clearml import Task
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from registering_preprocessed_datasets import register_dataset
import torch

SEED = 99
THRESHOLD = 0.6

def main():
    
    task = Task.init(project_name='Brain Tumor Detection Project', task_name='Model Evaluation', task_type=Task.TaskTypes.qc)
    
    register_dataset("brain_tumor_valid", "5bc2bf9094e74e90be7561d5f8d7a591")

    # Load the saved config
    with open("cfg.pkl", "rb") as f:
        cfg = pickle.load(f)

    # Set the device to GPU
    #cfg.MODEL.DEVICE = 'cuda'

    # Create a predictor
    predictor = DefaultPredictor(cfg)

    # Load the test dataset
    test_dataset = DatasetCatalog.get("brain_tumor_valid")

    # Create a COCO evaluator
    output_dir = 'C:/Users/Leon-PC/Downloads/Preprocessed data'
    evaluator = COCOEvaluator("brain_tumor_valid", False, output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, "brain_tumor_valid", num_workers=4)

    # Evaluate the model
    inference_on_dataset(predictor.model, test_loader, evaluator)

    # Upload the evaluation results to ClearML
    task = Task.get_task()
    task.upload_artifact("evaluation_results", os.path.join(output_dir, "coco_instances_results.json"))

    # Visualize the predictions
    my_dataset_test_metadata = MetadataCatalog.get("brain_tumor_valid")
    dataset_dicts = DatasetCatalog.get("brain_tumor_valid")

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

    task.close()

if __name__ == "__main__":
    main()