import os
import numpy as np
import pandas as pd
import clearml
from clearml import Dataset, Task, TaskTypes
from clearml.automation.controller import PipelineDecorator


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(name="UploadRawTrainData", return_values=["train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one(dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset

    #dir = os.path.join(dataset_root,"train", "images")
    #dataset = Dataset.create(
    #    dataset_name=f"{dataset_name}RawTrainData", dataset_project=dataset_project
    #)
    #dataset.add_files(path=dir, wildcard="*.jpg")
    #dir = os.path.join(dataset_root,"train", "labels")
    #dataset.add_files(path=dir, wildcard="*.txt")
    #dataset.upload()
    #dataset.finalize()
    
    #print(f"Train dataset uploaded with ID: {dataset.id}")
    #return dataset.id 
    return "UploadRawTrainDataID"

    
    
@PipelineDecorator.component(name="UploadRawValidData", return_values=["valid_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two(raw_data_id, dataset_project, dataset_name, dataset_root):
    #import os
    #from clearml import Dataset
    #dir = os.path.join(dataset_root,"valid", "images")
    #dataset = Dataset.create(
    #    dataset_name=f"{dataset_name}RawValidData", dataset_project=dataset_project
    #)
    #dataset.add_files(path=dir, wildcard="*.jpg")
    #dir = os.path.join(dataset_root,"train", "labels")
    #dataset.add_files(path=dir, wildcard="*.txt")
    #dataset.upload()
    #dataset.finalize()
    
    #print(f"Train dataset uploaded with ID: {dataset.id}")
    #return dataset.id 
    return "UploadRawValidID"
 

@PipelineDecorator.component(name="ProcessRawTrainData", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_three(
    raw_dataset_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    #def save_preprocessed_data(data, labels, data_filename, labels_filename):
    #    import argparse
    #    import os

    #    import numpy as np
    #    from clearml import Dataset

       np.save(data_filename, data)
        np.save(labels_filename, labels)
        

    raw_dataset = Dataset.get(dataset_id=raw_dataset_id)
    raw_data_path = raw_dataset.get_local_copy()

    # Load the numpy arrays from the raw dataset
    train_images = np.load(f"{raw_data_path}/train_images_10.npy")
    train_labels = np.load(f"{raw_data_path}/train_images_10.npy")
    test_images = np.load(f"{raw_data_path}/train_images_10.npy")
    test_labels = np.load(f"{raw_data_path}/train_images_10.npy")

    # Preprocess the images (normalize the pixel values)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Save the preprocessed arrays to files
    save_preprocessed_data(
        train_images,
        train_labels,
        "train_images_preprocessed.npy",
        "train_labels_preprocessed.npy",
    )
    save_preprocessed_data(
        test_images,
        test_labels,
        "test_images_preprocessed.npy",
        "test_labels_preprocessed.npy",
    )

    # Create a new ClearML dataset for the preprocessed data
    processed_dataset = Dataset.create(
        dataset_name=processed_dataset_name,
        dataset_project=processed_dataset_project,
        parent_datasets=[raw_dataset_id],
    )

    # Add the saved numpy files to the dataset
    processed_dataset.add_files("train_images_preprocessed.npy")
    processed_dataset.add_files("train_labels_preprocessed.npy")
    processed_dataset.add_files("test_images_preprocessed.npy")
    processed_dataset.add_files("test_labels_preprocessed.npy")

    # Upload the dataset to ClearML
    processed_dataset.upload()
    processed_dataset.finalize()

    # Clean up: Remove the numpy files after upload
    os.remove("train_images_preprocessed.npy")
    os.remove("train_labels_preprocessed.npy")
    os.remove("test_images_preprocessed.npy")
    os.remove("test_labels_preprocessed.npy")

    print(f"Preprocessed CIFAR-100 dataset uploaded with ID: {processed_dataset.id}")
    return processed_dataset.id


   
# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(name="BrainScan2DataPipeline", project="Strykers2", target_project="Strykers2", pipeline_execution_queue="uts-strykers-queue", default_queue="uts-strykers-queue") #, version="0.0.6")
def executing_data_pipeline(dataset_project, dataset_name, dataset_root, output_root):

    print("::=======================================::")
    print("Step1: Launch UploadTrainRawDataset Task")
    print("::=======================================::")
    raw_train_dataset_id = step_one(dataset_project, dataset_name, dataset_root)

    print("::=======================================::")
    print("Step 2: Launch UploadValidationRawDataset Task")
    print("::=======================================::")
    raw_validation_dataset_id = step_two(raw_train_dataset_id, dataset_name, dataset_root)



if __name__ == "__main__":
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    PipelineDecorator.set_default_execution_queue('uts-strykers-queue')
    # Run the pipeline steps as subprocesses on the current machine, great for local executions
    # (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)
    PipelineDecorator.run_locally()
    #PipelineDecorator.debug_pipeline()
    # Start the pipeline execution logic.
    

    executing_data_pipeline(
        dataset_project="BrainScan2",
        dataset_name="BrainScan2",
        dataset_root="/root/braintumourdetection/brainscan2/datasets/brain-tumor",
        output_root="/root/braintumourdetection/brainscan2/output"
    )

    print("process completed")


#from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data='brain-tumor.yaml', epochs=100, imgsz=640)