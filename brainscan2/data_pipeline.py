import os
import numpy as np
import pandas as pd
import clearml
from clearml import Dataset, Task, TaskTypes
from clearml.automation.controller import PipelineDecorator
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, GridSearch


@PipelineDecorator.component(name="StartTask", return_values=["start_task_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def start_data_pipeline(dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset, Task
    
    return "startdatapipeline"

# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(name="UploadRawTrainData", return_values=["train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one_a(start_task_id, dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset

    dataset = Dataset.get(
        dataset_id=None,  
        dataset_project=dataset_project + "Train",
        dataset_name=dataset_name + "Train",
        dataset_tags="trainrawdata",
        auto_create=True
    )

    if not os.path.isdir(dataset_root):
        raise ValueError(
            f"The specified path '{dataset_root}' is not a directory or does not exist."
        )
    
    train_raw_dataset_path = os.path.join(dataset_root, 'train')

    print(f"Uploading train raw dataset from '{train_raw_dataset_path}'.")
    dataset.add_files(train_raw_dataset_path)
    dataset.upload()
    dataset.finalize()
    
    return dataset.id

    
    
@PipelineDecorator.component(name="UploadRawValidData", return_values=["valid_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one_b(start_task_id, dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset
    
    dataset = Dataset.get(
        dataset_id=None,  
        dataset_project=dataset_project + "Valid",
        dataset_name=dataset_name + "Valid",
        dataset_tags="validrawdata",
        auto_create=True
    )

    if not os.path.isdir(dataset_root):
        raise ValueError(
            f"The specified path '{dataset_root}' is not a directory or does not exist."
        )
    
    valid_raw_dataset_path = os.path.join(dataset_root, 'valid')

    print(f"Uploading valid raw dataset from '{valid_raw_dataset_path}'.")
    dataset.add_files(valid_raw_dataset_path)
    dataset.upload()
    dataset.finalize()
    
    return dataset.id

@PipelineDecorator.component(name="UploadRawTestData", return_values=["test_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one_c(start_task_id, dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset
    
    dataset = Dataset.get(
        dataset_id=None,  
        dataset_project=dataset_project + "Test",
        dataset_name=dataset_name + "Test",
        dataset_tags="testrawdata",
        auto_create=True
    )

    if not os.path.isdir(dataset_root):
        raise ValueError(
            f"The specified path '{dataset_root}' is not a directory or does not exist."
        )
    
    test_raw_dataset_path = os.path.join(dataset_root, 'test')

    print(f"Uploading valid raw dataset from '{test_raw_dataset_path}'.")
    dataset.add_files(test_raw_dataset_path)
    dataset.upload()
    dataset.finalize()
    
    return dataset.id
 

@PipelineDecorator.component(name="ProcessTrainData", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two_a(
    raw_train_dataset_id, processed_dataset_project, processed_dataset_name, dataset_root, processed_dataset_root
):
    import os
    import numpy as np
    from clearml import Dataset
    import cv2
    import glob

    # Fetch the dataset from ClearML
    dataset = Dataset.get(raw_train_dataset_id) #dataset_name=processed_dataset_name, dataset_project=processed_dataset_project)
    local_dataset_path = dataset.get_local_copy()

    # Define paths for train, valid, and test datasets
    mages_path = os.path.join(local_dataset_path, "train/images")
    labels_path = os.path.join(local_dataset_path, "train/labels")
    
    print(f"Local Dataset Path: {local_dataset_path}")

    def load_labels(label_path):
        with open(label_path, 'r') as file:
            data = file.readlines()
        labels = [list(map(float, line.strip().split())) for line in data]
        return labels

    def load_dataset(images_path, labels_path):
        image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
        label_files = sorted(glob.glob(os.path.join(labels_path, "*.txt")))
        
        images = [cv2.imread(img) for img in image_files]
        labels = [load_labels(lbl) for lbl in label_files]
        
        return np.array(images), np.array(labels)

    # Load datasets
    train_images, train_labels = load_dataset(mages_path, labels_path)
    
    # Save datasets as NumPy arrays
    np.save(f"{processed_dataset_root}/train_images.npy", train_images)
    np.save(f"{processed_dataset_root}/train_labels.npy", train_labels)
    
    # Create a new ClearML dataset for the NumPy files
    new_dataset = Dataset.create(dataset_name=f"{processed_dataset_name}ProcessedTrainDataset" , dataset_project=processed_dataset_project)

    # Add the NumPy files to the new dataset
    new_dataset.add_files(f"{processed_dataset_root}/train_images.npy")
    new_dataset.add_files(f"{processed_dataset_root}/train_labels.npy")
   
    # Upload the new dataset to ClearML
    new_dataset.upload()

    # Finalize the dataset
    new_dataset.finalize()

    # Clean up: Remove the numpy files after upload
    os.remove(f"{processed_dataset_root}/train_images.npy")
    os.remove(f"{processed_dataset_root}/train_labels.npy")


    print("New dataset with NumPy arrays has been created and uploaded to ClearML.")

    return dataset.id

@PipelineDecorator.component(name="ProcessValidData", return_values=["processed_valid_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two_b(
    raw_valid_dataset_id, processed_dataset_project, processed_dataset_name, dataset_root, processed_dataset_root
):
    import os
    import numpy as np
    from clearml import Dataset
    import cv2
    import glob

    # Fetch the dataset from ClearML
    dataset = Dataset.get(raw_valid_dataset_id) #dataset_name=processed_dataset_name, dataset_project=processed_dataset_project)
    local_dataset_path = dataset.get_local_copy()

    # Define paths for train, valid, and test datasets
    images_path = os.path.join(local_dataset_path, "valid/images")
    labels_path = os.path.join(local_dataset_path, "valid/labels")
    
    print(f"Local Dataset Path: {local_dataset_path}")

    def load_labels(label_path):
        with open(label_path, 'r') as file:
            data = file.readlines()
        labels = [list(map(float, line.strip().split())) for line in data]
        return labels

    def load_dataset(images_path, labels_path):
        image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
        label_files = sorted(glob.glob(os.path.join(labels_path, "*.txt")))
        
        images = [cv2.imread(img) for img in image_files]
        labels = [load_labels(lbl) for lbl in label_files]
        
        return np.array(images), np.array(labels)

    # Load datasets
    valid_images, valid_labels = load_dataset(images_path, labels_path)
    
    # Save datasets as NumPy arrays
    np.save(f"{processed_dataset_root}/valid_images.npy", valid_images)
    np.save(f"{processed_dataset_root}/valid_labels.npy", valid_labels)
    
    # Create a new ClearML dataset for the NumPy files
    new_dataset = Dataset.create(dataset_name=f"{processed_dataset_name}ProcessedValidDataset" , dataset_project=processed_dataset_project)

    # Add the NumPy files to the new dataset
    new_dataset.add_files(f"{processed_dataset_root}/valid_images.npy")
    new_dataset.add_files(f"{processed_dataset_root}/valid_labels.npy")
   
    # Upload the new dataset to ClearML
    new_dataset.upload()

    # Finalize the dataset
    new_dataset.finalize()

    # Clean up: Remove the numpy files after upload
    os.remove(f"{processed_dataset_root}/valid_images.npy")
    os.remove(f"{processed_dataset_root}/valid_labels.npy")


    print("New dataset with NumPy arrays has been created and uploaded to ClearML.")

    return dataset.id


@PipelineDecorator.component(name="ProcessTestData", return_values=["processed_test_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two_c(
    raw_test_dataset_id, processed_dataset_project, processed_dataset_name, dataset_root, processed_dataset_root
):
    import os
    import numpy as np
    from clearml import Dataset
    import cv2
    import glob

    # Fetch the dataset from ClearML
    dataset = Dataset.get(raw_test_dataset_id) #dataset_name=processed_dataset_name, dataset_project=processed_dataset_project)
    local_dataset_path = dataset.get_local_copy()

    # Define paths for train, valid, and test datasets
    images_path = os.path.join(local_dataset_path, "valid/images")
    labels_path = os.path.join(local_dataset_path, "valid/labels")
    
    print(f"Local Dataset Path: {local_dataset_path}")

    def load_labels(label_path):
        with open(label_path, 'r') as file:
            data = file.readlines()
        labels = [list(map(float, line.strip().split())) for line in data]
        return labels

    def load_dataset(images_path, labels_path):
        image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
        label_files = sorted(glob.glob(os.path.join(labels_path, "*.txt")))
        
        images = [cv2.imread(img) for img in image_files]
        labels = [load_labels(lbl) for lbl in label_files]
        
        return np.array(images), np.array(labels)

    # Load datasets
    test_images, test_labels = load_dataset(images_path, labels_path)
    
    # Save datasets as NumPy arrays
    np.save(f"{processed_dataset_root}/test_images.npy", test_images)
    np.save(f"{processed_dataset_root}/test_labels.npy", test_labels)
    
    # Create a new ClearML dataset for the NumPy files
    new_dataset = Dataset.create(dataset_name=f"{processed_dataset_name}ProcessedTestDataset" , dataset_project=processed_dataset_project)

    # Add the NumPy files to the new dataset
    new_dataset.add_files(f"{processed_dataset_root}/test_images.npy")
    new_dataset.add_files(f"{processed_dataset_root}/test_labels.npy")
   
    # Upload the new dataset to ClearML
    new_dataset.upload()

    # Finalize the dataset
    new_dataset.finalize()

    # Clean up: Remove the numpy files after upload
    os.remove(f"{processed_dataset_root}/test_images.npy")
    os.remove(f"{processed_dataset_root}/test_labels.npy")


    print("New dataset with NumPy arrays has been created and uploaded to ClearML.")

    return dataset.id




@PipelineDecorator.component(name="MergeDataTasks", return_values=["start_model_pipeline_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_three_merge(
     process_train_dataset_id, process_valid_dataset_id, process_test_dataset_id, processed_dataset_project, processed_dataset_name
):
    
    import numpy as np
    from clearml import Dataset, Task

    task = Task.current_task()

    task.connect({
        'process_train_dataset_id': process_train_dataset_id,
        'process_valid_dataset_id': process_valid_dataset_id,
        'process_test_dataset_id': process_test_dataset_id
    })
    
    return task.id

   
# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(name="BrainScan2DataPipeline", project="BrainScan2", target_project="BrainScan2", pipeline_execution_queue="uts-strykers-queue", default_queue="uts-strykers-queue") #, version="0.0.6")
def executing_data_pipeline(dataset_project, dataset_name, dataset_root, processed_dataset_root, output_root, queue_name, results_dir):

    start_data_pipeline_id = start_data_pipeline(dataset_project, dataset_name, dataset_root)

    raw_train_dataset_id = step_one_a(start_data_pipeline_id, dataset_project, dataset_name, dataset_root)

    
    raw_validation_dataset_id = step_one_b(start_data_pipeline_id, dataset_project, dataset_name, dataset_root)
    
    
    raw_test_dataset_id = step_one_c(start_data_pipeline_id, dataset_project, dataset_name, dataset_root)
    
    
    process_train_dataset_id = step_two_a(raw_train_dataset_id, dataset_project, dataset_name, dataset_root, processed_dataset_root)
    
    
    process_valid_dataset_id = step_two_b(raw_validation_dataset_id, dataset_project, dataset_name, dataset_root, processed_dataset_root)
    
    
    process_test_dataset_id = step_two_c(raw_test_dataset_id, dataset_project, dataset_name, dataset_root, processed_dataset_root)




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
        #dataset_root="/root/braintumourdetection/brainscan2/datasets/brain-tumor",
        dataset_root="/Users/soterojrsaberon/UTS/braintumourdetection/brainscan2/datasets/brain-tumor",
        processed_dataset_root="/Users/soterojrsaberon/UTS/braintumourdetection/brainscan2/datasets/processed",
        output_root="/Users/soterojrsaberon/UTS/braintumourdetection/brainscan2/output",
        queue_name="uts-strykers-queue", 
        results_dir="/Users/soterojrsaberon/UTS/braintumourdetection/brainscan2/models"
    )

    print("process completed")


#from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data='brain-tumor.yaml', epochs=100, imgsz=640)