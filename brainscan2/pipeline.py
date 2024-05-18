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



@PipelineDecorator.component(name="TrainModel", return_values=["training_task_id"], cache=True, task_type=TaskTypes.training)#, execution_queue="default")
def step_four( start_model_pipeline_id, dataset_name, dataset_root, processed_dataset_root
):
    import argparse
    import os
    from ultralytics import YOLO
    import numpy as np
    from clearml import Dataset, Task, OutputModel
    import cv2

#def load_numpy_datasets(train_dataset_id, valid_dataset_id, test_dataset_id, processed_dataset_root):
#    # Fetch datasets
#    train_dataset = Dataset.get(dataset_id=train_dataset_id)
#    valid_dataset = Dataset.get(dataset_id=valid_dataset_id)
#    test_dataset = Dataset.get(dataset_id=test_dataset_id)
#
#    # Get the local paths for these datasets
#    train_dataset_path = train_dataset.get_local_copy()
#    valid_dataset_path = valid_dataset.get_local_copy()
#    test_dataset_path = test_dataset.get_local_copy()
#
#    print(train_dataset_path, valid_dataset_path, test_dataset_path)
#
#    # Load the NumPy arrays
#    train_images = np.load(os.path.join(train_dataset_path, "train_images.npy"))
#    train_labels = np.load(os.path.join(train_dataset_path, "train_labels.npy"))
#    valid_images = np.load(os.path.join(valid_dataset_path, "valid_images.npy"))
#    valid_labels = np.load(os.path.join(valid_dataset_path, "valid_labels.npy"))
#    test_images = np.load(os.path.join(test_dataset_path, "test_images.npy"))
#    test_labels = np.load(os.path.join(test_dataset_path, "test_labels.npy"))
#
#    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels
#
##Connect to the previous task and fetch the dataset IDs
    previous_task_id = start_model_pipeline_id
    previous_task = Task.get_task(task_id=previous_task_id)
    #
    ## Retrieve the dataset IDs
    process_train_dataset_id = previous_task.get_parameters()['General/process_train_dataset_id']
    process_valid_dataset_id = previous_task.get_parameters()['General/process_valid_dataset_id']
    process_test_dataset_id = previous_task.get_parameters()['General/process_test_dataset_id']
#
#print(f"Train Dataset ID: {process_train_dataset_id}")
#print(f"Valid Dataset ID: {process_valid_dataset_id}")
#print(f"Test Dataset ID: {process_test_dataset_id}")
#
## Load datasets
#train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_numpy_datasets(
#    process_train_dataset_id, process_valid_dataset_id, process_test_dataset_id, processed_dataset_root)
#
#os.makedirs(f'{processed_dataset_root}/train/images', exist_ok=True)
#os.makedirs(f'{processed_dataset_root}/train/labels', exist_ok=True)
#os.makedirs(f'{processed_dataset_root}/valid/images', exist_ok=True)
#os.makedirs(f'{processed_dataset_root}/valid/labels', exist_ok=True)
#os.makedirs(f'{processed_dataset_root}/test/images', exist_ok=True)
#os.makedirs(f'{processed_dataset_root}/test/labels', exist_ok=True)
#
#for i, (img, lbl) in enumerate(zip(train_images, train_labels)):
#    cv2.imwrite(f'{processed_dataset_root}/train/images/{i}.jpg', img)
#    np.savetxt(f'{processed_dataset_root}/train/labels/{i}.txt', lbl, fmt='%f')
#
#for i, (img, lbl) in enumerate(zip(valid_images, valid_labels)):
#    cv2.imwrite(f'{processed_dataset_root}/valid/images/{i}.jpg', img)
#    np.savetxt(f'{processed_dataset_root}/valid/labels/{i}.txt', lbl, fmt='%f')
#
#for i, (img, lbl) in enumerate(zip(test_images, test_labels)):
#    cv2.imwrite(f'{processed_dataset_root}/test/images/{i}.jpg', img)
#    np.savetxt(f'{processed_dataset_root}/test/labels/{i}.txt', lbl, fmt='%f')
#
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    #results = model.train(data='brainscan.yaml', epochs=10, imgsz=640)

    # Define the output directory for training results
    results_dir = '/Users/soterojrsaberon/UTS/braintumourdetection/brainscan2/models'

    task = Task.current_task()
    # Train the model
    results = model.train(data='brainscan.yaml', epochs=3, imgsz=640, project=results_dir, name='brain_tumor_model')

    # Save the trained model weights
    model_output_path = os.path.join(results_dir, 'brain_tumor_model', 'weights', 'best.pt')
    output_model = OutputModel(task=task, framework="PyTorch")
    output_model.update_weights(weights_filename=model_output_path, auto_delete_file=False)

    # Log the model ID
    model_id = output_model.id
    print(f"Trained model ID: {model_id}")

    task.connect({
        'process_train_dataset_id': process_train_dataset_id,
        'process_valid_dataset_id': process_valid_dataset_id,
        'process_test_dataset_id': process_test_dataset_id
    })

    return model_id


@PipelineDecorator.component(name="EvaluateModel", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.training)#, execution_queue="default")
def step_five(
    train_model_task_id, processed_dataset_project, processed_dataset_name
):
    import os
    from ultralytics import YOLO
    from clearml import Task, Dataset
    import numpy as np

    # Connect to the previous task and fetch the dataset IDs
    previous_task = Task.get_task(task_id=train_model_task_id)

    # Retrieve the dataset IDs
    process_test_dataset_id = previous_task.get_parameters()['General/process_test_dataset_id']

    # Load the test dataset
    test_dataset = Dataset.get(dataset_id=process_test_dataset_id)
    test_dataset_path = test_dataset.get_local_copy()

    # Load the test images and labels
    test_images = np.load(os.path.join(test_dataset_path, "test_images.npy"))
    test_labels = np.load(os.path.join(test_dataset_path, "test_labels.npy"))

    # Load the trained model
    task = Task.get_task(task_id=train_model_task_id)
    model_path = task.models['output_model'][0].get_local_copy()

    # Initialize the model
    model = YOLO(model_path)

    # Evaluate the model on the test dataset
    results = model.val(data='brainscan.yaml', imgsz=640)

    # Print evaluation metrics
    print(f"Precision: {results.metrics['precision']}")
    print(f"Recall: {results.metrics['recall']}")
    print(f"mAP@0.5: {results.metrics['map50']}")
    print(f"mAP@0.5:0.95: {results.metrics['map']}")
    
    return results.metrics
    

@PipelineDecorator.component(name="HPO", return_values=["top_experiment_id"], cache=True, task_type=TaskTypes.optimizer)#, execution_queue="default")
def step_six(
    base_task_id, queue_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "hpo_id"

def testme(base_task_id, queue_name):

    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task


    def job_complete_callback(
        job_id,  # type: str
        objective_value,  # type: float
        objective_iteration,  # type: int
        job_parameters,  # type: dict
        top_performance_job_id,  # type: str
        ):
        
        print(
            "Job completed!", job_id, objective_value, objective_iteration, job_parameters
        )
        if job_id == top_performance_job_id:
            print(
                "Objective reached {}".format(
                    objective_value
                )
            )

    # Define Hyperparameter Space
    param_ranges = [
        UniformIntegerParameterRange(
            "Args/epochs", min_value=5, max_value=10, step_size=5
        ),
        ### you could make anything like batch_size, number of nodes, loss function, a command line argument in base task and use it as a parameter to be optimised. ###
    ]

    # Setup HyperParameter Optimizer
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=param_ranges,
        objective_metric_title="epoch_accuracy",
        objective_metric_series="epoch_accuracy",
        objective_metric_sign="max",
        optimizer_class=GridSearch,
        execution_queue=queue_name,
        max_number_of_concurrent_tasks=2,
        optimization_time_limit=60.0,
        # Check the experiments every 6 seconds is way too often, we should probably set it to 5 min,
        # assuming a single experiment is usually hours...
        pool_period_min=0.1,
        compute_time_limit=120,
        total_max_jobs=20,
        min_iteration_per_job=15000,
        max_iteration_per_job=150000,
    )
    # report every 12 seconds, this is way too often, but we are testing here J
    optimizer.set_report_period(0.2)
    # start the optimization process, callback function to be called every time an experiment is completed
    # this function returns immediately
    optimizer.start(job_complete_callback=job_complete_callback)
    # set the time limit for the optimization process (2 hours)
    optimizer.set_time_limit(in_minutes=90.0)
    # wait until process is done (notice we are controlling the optimization process in the background)
    optimizer.wait()
    # optimization is completed, print the top performing experiments id
    top_exp = optimizer.get_top_experiments(top_k=3)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    optimizer.stop()

    print("Optimisation Done")
    return top_exp[0].id


@PipelineDecorator.component(name="TestModel", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.testing)#, execution_queue="default")
def step_seven(
    train_model_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "test_model_id"


@PipelineDecorator.component(name="PushModel", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.service)#, execution_queue="default")
def step_eight(
    train_model_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "push_model_id"



   
# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(name="BrainScan2DataPipeline", project="BrainScan2", target_project="BrainScan2", pipeline_execution_queue="uts-strykers-queue", default_queue="uts-strykers-queue") #, version="0.0.6")
def executing_data_pipeline(dataset_project, dataset_name, dataset_root, processed_dataset_root, output_root, queue_name):

    start_data_pipeline_id = start_data_pipeline(dataset_project, dataset_name, dataset_root)

    raw_train_dataset_id = step_one_a(start_data_pipeline_id, dataset_project, dataset_name, dataset_root)

    
    raw_validation_dataset_id = step_one_b(start_data_pipeline_id, dataset_project, dataset_name, dataset_root)
    
    
    raw_test_dataset_id = step_one_c(start_data_pipeline_id, dataset_project, dataset_name, dataset_root)
    
    
    process_train_dataset_id = step_two_a(raw_train_dataset_id, dataset_project, dataset_name, dataset_root, processed_dataset_root)
    
    
    process_valid_dataset_id = step_two_b(raw_validation_dataset_id, dataset_project, dataset_name, dataset_root, processed_dataset_root)
    
    
    process_test_dataset_id = step_two_c(raw_test_dataset_id, dataset_project, dataset_name, dataset_root, processed_dataset_root)

    start_model_pipeline_id = step_three_merge(process_train_dataset_id, process_valid_dataset_id, process_test_dataset_id, dataset_project, dataset_name)
    
    
    step_four_id = step_four(start_model_pipeline_id, dataset_name, dataset_root, processed_dataset_root)

    
    step_five_id = step_five(step_four_id, dataset_name, dataset_root)

    
    step_six_id = step_six(step_five_id, queue_name)


    step_seven_id = step_seven(step_six_id,dataset_project, dataset_name)

    
    step_eight_id = step_eight(step_seven_id,dataset_project, dataset_name)




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
        output_root="/Users/soterojrsaberon/UTS/braintumourdetection/brainscan2/datasets/brain-tumor/output",
        queue_name="uts-strykers-queue"
    )

    print("process completed")


#from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data='brain-tumor.yaml', epochs=100, imgsz=640)