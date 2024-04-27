import argparse
import os
from clearml import Dataset, Task

def upload_train_split(dataset_project, dataset_name, dataset_root):
    task = Task.init(
        project_name=dataset_project,
        task_name="Train Split Dataset Upload",
        task_type=Task.TaskTypes.data_processing,
    )
    #task.execute_remotely(queue_name="queue_name", exit_process=True)

    # Path to the train directory and its COCO annotation file
    train_dir = os.path.join(dataset_root, "train")
    annotations_file = os.path.join(train_dir, "_annotations.coco.json")

    # Create a new ClearML dataset for the train split
    train_dataset = Dataset.create(
        dataset_name=dataset_name, dataset_project=dataset_project
    )
    
    # Add images to the dataset
    train_dataset.add_files(path=train_dir, wildcard="*.jpg")

    # Add the COCO annotations file if it exists
    if os.path.exists(annotations_file):
        train_dataset.add_files(annotations_file)

    # Upload the dataset to ClearML
    train_dataset.upload()
    train_dataset.finalize()

    print(f"Train split dataset uploaded with ID: {train_dataset.id}")
    return train_dataset.id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Train Split of Brain Tumor Dataset to ClearML")
    parser.add_argument("--dataset_project", type=str, required=True, help="ClearML dataset project name")
    parser.add_argument("--dataset_name", type=str, required=True, help="ClearML dataset name for the train split")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root path to the dataset directory")
    args = parser.parse_args()
    upload_train_split(args.dataset_project, args.dataset_name, args.dataset_root)
