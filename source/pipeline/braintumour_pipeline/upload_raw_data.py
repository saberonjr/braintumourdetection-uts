import argparse
import os

import os
import json
from PIL import Image

def save_numpy_arrays(data, labels, data_filename, labels_filename):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset
  
    np.save(data_filename, data)
    np.save(labels_filename, labels)

def load_dataset(dataset_base_path):
    datasets = {}
    for dataset_type in ['test', 'train', 'valid']:
        images = []
        annotations = []
        folder_path = os.path.join(dataset_base_path, dataset_type)
        
        # Load annotations
        annotation_file = os.path.join(folder_path, '_annotations.coco.json')
        if os.path.isfile(annotation_file):
            with open(annotation_file, 'r') as file:
                annotations = json.load(file)
        
        # Load images
        for file_name in os.listdir(folder_path):
            print("processing: ", file_name)
            if file_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(folder_path, file_name)
                images.append(Image.open(image_path))
        
        datasets[dataset_type] = {'images': images, 'annotations': annotations}
    
    return datasets

def upload_raw_dataset_as_numpy_to_clearml(dataset_project, dataset_name, dataset_base_path, dataset_temp_path):
    import numpy as np
    from clearml import Dataset

    import numpy as np
    from clearml import Dataset, Task
   
    task = Task.init(
        project_name=dataset_project,
        task_name="Dataset Upload",
        task_type=Task.TaskTypes.data_processing,
    )
    #task.execute_remotely(queue_name="uts-strykers-queue", exit_process=True)

    datasets = load_dataset(dataset_base_path)
    #datasets.head()
    dataset = Dataset.create(f'BrainScan Raw Dataset')

    for dataset_type, data in datasets.items():
        images = data['images']
        annotations = data['annotations']
        
        # Convert images to numpy arrays
        images_np = np.array([np.array(image) for image in images])
        
        # Convert annotations to numpy arrays
        annotations_np = np.array(annotations)
        
        # Save numpy arrays
        save_numpy_arrays(images_np, annotations_np, f'{dataset_temp_path}/{dataset_type}_images.npy', f'{dataset_temp_path}/{dataset_type}_annotations.npy')
        
        # Upload to ClearML
        
        dataset.add_files(f'{dataset_temp_path}/{dataset_type}_images.npy')
        dataset.add_files(f'{dataset_temp_path}/{dataset_type}_annotations.npy')
        
    dataset.upload()
    dataset.finalize()

        # Clean up: Remove the numpy files after upload
    os.remove(f'{dataset_temp_path}/test_images.npy')
    os.remove(f'{dataset_temp_path}/test_annotations.npy')
    os.remove(f'{dataset_temp_path}/train_images.npy')
    os.remove(f'{dataset_temp_path}/train_annotations.npy')
    os.remove(f'{dataset_temp_path}/valid_images.npy')
    os.remove(f'{dataset_temp_path}/valid_annotations.npy')
    
        
    print(f'Uploaded Raw Dataset to ClearML with ID: {dataset.id}')

    return dataset.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Brain Tumour Raw Data to ClearML")
    parser.add_argument(
        "--dataset_project",
        type=str,
        required=True,
        help="ClearML dataset project name",
        
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="ClearML dataset name for raw data",
    )
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        required=True,
        help="Dataset base path",
    )
    parser.add_argument(
        "--dataset_temp_path",
        type=str,
        required=True,
        help="Dataset base path",
    ) 
    #args = parser.parse_args()
    #upload_raw_dataset_as_numpy_to_clearml(args.dataset_project, args.dataset_name, args.dataset_base_path)
    upload_raw_dataset_as_numpy_to_clearml("BrainScan", 
                                           "BrainScan Raw Dataset", 
                                           "/Users/soterojrsaberon/Library/CloudStorage/GoogleDrive-sotero.j.saberon@student.uts.edu.au/My Drive/42174/brain-tumour-detector/Dataset",
                                           "/Users/soterojrsaberon/Downloads/Dataset")