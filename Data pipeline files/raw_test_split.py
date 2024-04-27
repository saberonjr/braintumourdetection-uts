import argparse
import os
from clearml import Dataset, Task

def upload_test_split(dataset_project, dataset_name, dataset_root):
    task = Task.init(
        project_name=dataset_project,
        task_name="Test Split Dataset Upload",
        task_type=Task.TaskTypes.data_processing,
    )
    #task.execute_remotely(queue_name="queue_name", exit_process=True)

    # Path to the test directory and its COCO annotation file
    test_dir = os.path.join(dataset_root, "test")
    annotations_file = os.path.join(test_dir, "_annotations.coco.json")

    # Create a new ClearML dataset for the valid split
    test_dataset = Dataset.create(
        dataset_name=dataset_name, dataset_project=dataset_project
    )
    
    # Add images to the dataset
    test_dataset.add_files(path=test_dir, wildcard="*.jpg")

    # Add the COCO annotations file if it exists
    if os.path.exists(annotations_file):
        test_dataset.add_files(annotations_file)

    # Upload the dataset to ClearML
    test_dataset.upload()
    test_dataset.finalize()

    print(f"Valid split dataset uploaded with ID: {test_dataset.id}")
    return test_dataset.id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Test Split of Brain Tumor Dataset to ClearML")
    parser.add_argument("--dataset_project", type=str, required=True, help="ClearML dataset project name")
    parser.add_argument("--dataset_name", type=str, required=True, help="ClearML dataset name for the test split")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root path to the dataset directory")
    args = parser.parse_args()
    upload_test_split(args.dataset_project, args.dataset_name, args.dataset_root)
