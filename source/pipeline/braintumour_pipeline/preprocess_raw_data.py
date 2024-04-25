
import argparse
import os

import numpy as np
from clearml import Dataset, Task


def save_preprocessed_data(data, labels, data_filename, labels_filename):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset

    np.save(data_filename, data)
    np.save(labels_filename, labels)


def preprocess_and_upload_brainscan_data(
    raw_dataset_id, processed_dataset_project, 
    processed_dataset_name, processed_dataset_temp_path
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    task = Task.init(
        project_name=processed_dataset_project,
        task_name="Dataset Preprocessing",
        task_type=Task.TaskTypes.data_processing,
    )
    task.execute_remotely(queue_name="default", exit_process=True)
    ###raw_dataset = Dataset.get(dataset_id=raw_dataset_id)
    ###raw_data_path = raw_dataset.get_local_copy()
##
    ##raw_data_path = "/Users/soterojrsaberon/Downloads/Dataset"
##
    ### Load the numpy arrays from the raw dataset
    ##train_images = np.load(f"{raw_data_path}/train_images.npy")
    ##train_labels = np.load(f"{raw_data_path}/train_annotations.npy", allow_pickle=True)
    ##test_images = np.load(f"{raw_data_path}/test_images.npy")
    ##test_labels = np.load(f"{raw_data_path}/test_annotations.npy", allow_pickle=True)
    ##valid_images = np.load(f"{raw_data_path}/valid_images.npy")
    ##valid_labels = np.load(f"{raw_data_path}/valid_annotations.npy", allow_pickle=True)
##
    ### Preprocess the images (normalize the pixel values)
    ##train_images, test_images, valid_images = train_images / 255.0, test_images / 255.0, valid_images / 255.0
##
    ### Save the preprocessed arrays to files
    ##save_preprocessed_data(
    ##    train_images,
    ##    train_labels,        
    ##    f'{processed_dataset_temp_path}/train_images_preprocessed.npy',
    ##    f'{processed_dataset_temp_path}/train_labels_preprocessed.npy',
    ##)
    ##save_preprocessed_data(
    ##    test_images,
    ##    test_labels,
    ##    f'{processed_dataset_temp_path}/test_images_preprocessed.npy',
    ##    f'{processed_dataset_temp_path}/test_labels_preprocessed.npy',
    ##)
    ##save_preprocessed_data(
    ##    valid_images,
    ##    valid_labels,
    ##    f'{processed_dataset_temp_path}/valid_images_preprocessed.npy',
    ##    f'{processed_dataset_temp_path}/valid_labels_preprocessed.npy',
    ##)
##
    ### Create a new ClearML dataset for the preprocessed data
    ##processed_dataset = Dataset.create(
    ##    dataset_name=processed_dataset_name,
    ##    dataset_project=processed_dataset_project,
    ##    parent_datasets=[raw_dataset_id],
    ##)
##
    ### Add the saved numpy files to the datast
    ##processed_dataset.add_files(f'{processed_dataset_temp_path}/train_images_preprocessed.npy')
    ##processed_dataset.add_files(f'{processed_dataset_temp_path}/train_labels_preprocessed.npy')
    ##processed_dataset.add_files(f'{processed_dataset_temp_path}/test_images_preprocessed.npy')
    ##processed_dataset.add_files(f'{processed_dataset_temp_path}/test_labels_preprocessed.npy')
    ##processed_dataset.add_files(f'{processed_dataset_temp_path}/valid_images_preprocessed.npy')
    ##processed_dataset.add_files(f'{processed_dataset_temp_path}/valid_labels_preprocessed.npy')
##
    ### Upload the dataset to ClearML
    ##processed_dataset.upload()
    ##processed_dataset.finalize()
##
    ### Clean up: Remove the numpy files after upload
    ##os.remove(f'{processed_dataset_temp_path}/train_images_preprocessed.npy')
    ##os.remove(f'{processed_dataset_temp_path}/train_labels_preprocessed.npy')
    ##os.remove(f'{processed_dataset_temp_path}/test_images_preprocessed.npy')
    ##os.remove(f'{processed_dataset_temp_path}/test_labels_preprocessed.npy')
    ##os.remove(f'{processed_dataset_temp_path}/valid_images_preprocessed.npy')
    ##os.remove(f'{processed_dataset_temp_path}/valid_labels_preprocessed.npy')
##
    ##print(f"Preprocessed BrainScan dataset uploaded with ID: {processed_dataset.id}")
    task.close()

    ##return processed_dataset.id
    return "da7430e10033427e86fbabb9e9cc4745"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and Upload BrainScan Data to ClearML"
    )
    parser.add_argument(
        "--raw_dataset_id",
        type=str,
        required=True,
        help="ID of the raw BrainScan dataset in ClearML",
    )
    parser.add_argument(
        "--processed_dataset_project",
        type=str,
        required=True,
        help="ClearML project name for the processed dataset",
    )
    parser.add_argument(
        "--processed_dataset_name",
        type=str,
        required=True,
        help="Name for the processed dataset in ClearML",
    )
    parser.add_argument(
        "--processed_dataset_temp_path",
        type=str,
        required=True,
        help="Dataset base path",
    ) 
    #args = parser.parse_args()
    #preprocess_and_upload_brainscan_data(
        #args.raw_dataset_id, args.processed_dataset_project, args.processed_dataset_name, args.processed_dataset_temp_path
    #)
    preprocess_and_upload_brainscan_data("1db93c08a49f4300b2c64ba9a38ba3ee",
                                         "BrainScan",
                                         "BrainScanPreprocessedDataset",
                                         "/Users/soterojrsaberon/Downloads/Dataset"
                                        )