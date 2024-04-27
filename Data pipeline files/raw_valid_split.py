import argparse
import os
from clearml import Dataset, Task

def upload_valid_split(dataset_project, dataset_name, dataset_root):
    task = Task.init(
        project_name=dataset_project,
        task_name="Valid Split Dataset Upload",
        task_type=Task.TaskTypes.data_processing,
    )
    #task.execute_remotely(queue_name="queue_name", exit_process=True)

    # Path to the train directory and its COCO annotation file
    valid_dir = os.path.join(dataset_root, "valid")
    annotations_file = os.path.join(valid_dir, "_annotations.coco.json")

    # Create a new ClearML dataset for the valid split
    valid_dataset = Dataset.create(
        dataset_name=dataset_name, dataset_project=dataset_project
    )
    
    # Add images to the dataset
    valid_dataset.add_files(path=valid_dir, wildcard="*.jpg")

    # Add the COCO annotations file if it exists
    if os.path.exists(annotations_file):
        valid_dataset.add_files(annotations_file)

    # Upload the dataset to ClearML
    valid_dataset.upload()
    valid_dataset.finalize()

    print(f"Valid split dataset uploaded with ID: {valid_dataset.id}")
    return valid_dataset.id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Valid Split of Brain Tumor Dataset to ClearML")
    parser.add_argument("--dataset_project", type=str, required=True, help="ClearML dataset project name")
    parser.add_argument("--dataset_name", type=str, required=True, help="ClearML dataset name for the valid split")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root path to the dataset directory")
    args = parser.parse_args()
    upload_valid_split(args.dataset_project, args.dataset_name, args.dataset_root)
