import argparse

from clearml import PipelineController, Task

from evaluate_model import (
    evaluate_model, 
    #log_debug_images
)
from preprocess_raw_data import (
    preprocess_and_upload_brainscan_data,
    #save_preprocessed_data,
)
from train_model import train_model
from update_model import (
  #archive_existing_model,
  #cleanup_repo,
  #clone_repo,
  #commit_and_push,
  #configure_ssh_key,
  #ensure_archive_dir,
    update_model,
   # update_weights,
)
from update_model import (
    update_model,
   # update_weights
)

def create_brain_tumour_pipeline(
    epochs: int = 10,
    pipeline_name: str = "BrainScanPipeline",
    dataset_project: str = "BrainScan",
    raw_dataset_name: str = "BrainScanRawDataset",
    processed_dataset_name: str = "BrainScanPreprocessedDataset",
    env_path: str = "/Users/soterojrsaberon/GitHub/braintumourdetection-team/source/pipeline/.env",
    repo_url: str = "git@github.com:uts-strykers/braintumourdetection.git",
    development_branch: str = "development",
    dataset_base_path: str = "/this is the path/",
    dataset_temp_path: str = "/this is the path/",
    processed_dataset_temp_path: str = "/this is the path/"
):
    from clearml import PipelineController, Task

    from evaluate_model import evaluate_model # , log_debug_images
    from preprocess_raw_data import (
        save_preprocessed_data,
    )
    from train_model import train_model
    from update_model import (
       #archive_existing_model,
       #cleanup_repo,
       #clone_repo,
       #commit_and_push,
       #configure_ssh_key,
       #ensure_archive_dir,
        update_model,
        #update_weights,
    )
    from upload_raw_data import (
        #save_numpy_arrays,
        #upload_brain_tumour_data_as_numpy,
        upload_raw_dataset_as_numpy_to_clearml
    )

    # Initialize a new pipeline controller task
    pipeline = PipelineController(
        name=pipeline_name,
        project=dataset_project,
        version="1.0",
        add_pipeline_tags=True,
        auto_version_bump=True,
        target_project=dataset_project
    )

    # Add pipeline-level parameters with defaults from function arguments
    pipeline.add_parameter(name="epochs", default=epochs)
    pipeline.add_parameter(name="dataset_project", default=dataset_project)
    pipeline.add_parameter(name="raw_dataset_name", default=raw_dataset_name)
    pipeline.add_parameter(
        name="processed_dataset_name", default=processed_dataset_name
    )
    pipeline.add_parameter(name="env_path", default=env_path)
    pipeline.add_parameter(name="REPO_URL", default=repo_url)
    pipeline.add_parameter(name="DEVELOPMENT_BRANCH", default=development_branch)
    pipeline.add_parameter(name="dataset_base_path", default=dataset_base_path)
    pipeline.add_parameter(name="dataset_temp_path", default=dataset_temp_path) 
    pipeline.add_parameter(name="processed_dataset_temp_path", default=processed_dataset_temp_path)
    pipeline.set_default_execution_queue("default")

    # Step 1: Upload BrainScan Raw Data
    pipeline.add_function_step(
        name="upload_brain_tumour_raw_data",
        function=upload_raw_dataset_as_numpy_to_clearml,
        function_kwargs={
            "dataset_project": "${pipeline.dataset_project}",
            "dataset_name": "${pipeline.raw_dataset_name}",
            "dataset_base_path": "${pipeline.dataset_base_path}",
            "dataset_temp_path": "${pipeline.dataset_temp_path}"
        },
        task_type=Task.TaskTypes.data_processing,
        task_name="Upload Brain Tumour Raw Data",
        function_return=["raw_dataset_id"],
        #helper_functions=[save_numpy_arrays],
        cache_executed_step=False
    )
    
    # Step 2: Preprocess Brain Tumour Data
    pipeline.add_function_step(
        name="preprocess_brain_tumour_data",
        function=preprocess_and_upload_brainscan_data,
        function_kwargs={
            "raw_dataset_id": "${upload_brain_tumour_raw_data.raw_dataset_id}",
            "processed_dataset_project": "${pipeline.dataset_project}",
            "processed_dataset_name": "${pipeline.processed_dataset_name}",
            "processed_dataset_temp_path": "${pipeline.processed_dataset_temp_path}",
        },
        task_type=Task.TaskTypes.data_processing,
        task_name="Preprocess and Upload Brain Tumour",
        function_return=["processed_dataset_id"],
        helper_functions=[save_preprocessed_data],
        cache_executed_step=False,
        parents=["upload_brain_tumour_raw_data"]
    )

    # Step 3: Train Model
    pipeline.add_function_step(
        name="train_brain_tumour_model",
        function=train_model,
        function_kwargs={
            "processed_dataset_id": "${preprocess_brain_tumour_data.processed_dataset_id}",
            "epochs": "${pipeline.epochs}",
            "project_name": "${pipeline.dataset_project}",
        },
        task_type=Task.TaskTypes.training,
        task_name="Train Brain Tumour Model",
        function_return=["model_id"],
        cache_executed_step=False,
        parents=["preprocess_brain_tumour_data"]
    )

    # Step 4: Evaluate Model
    pipeline.add_function_step(
        name="evaluate_brain_tumour_model",
        function=evaluate_model,
        function_kwargs={
            "model_id": "${train_brain_tumour_model.model_id}",
            "processed_dataset_id": "${preprocess_brain_tumour_data.processed_dataset_id}",
            "project_name": "${pipeline.dataset_project}",
        },
        task_type=Task.TaskTypes.testing,
        task_name="Evaluate Brain Tumour Model",
        #helper_functions=[log_debug_images],
        cache_executed_step=False,
        parents=["train_brain_tumour_model"]
    )

    # Step 5: Update Model in GitHub Repository
    pipeline.add_function_step(
        name="update_model_in_github",
        function=update_model,
        function_kwargs={
            "model_id": "${train_brain_tumour_model.model_id}",
            "env_path": "${pipeline.env_path}",
            "REPO_URL": "${pipeline.REPO_URL}",
            "DEVELOPMENT_BRANCH": "${pipeline.DEVELOPMENT_BRANCH}",
            "project_name": "${pipeline.dataset_project}",
        },
        helper_functions=[
           #configure_ssh_key,
           #clone_repo,
           #ensure_archive_dir,
           #archive_existing_model,
           #update_weights,
           #commit_and_push,
           #cleanup_repo,
        ],
        task_type=Task.TaskTypes.custom,
        task_name="Export Model to GitHub Repository",
        cache_executed_step=False,
        parents=["evaluate_brain_tumour_model"]
    )

    # Start the pipeline
    pipeline.start( queue="uts-strykers-queue")
    print("Brain Tumour pipeline initiated. Check ClearML for progress.")


if __name__ == "__main__":
    from pipeline import create_brain_tumour_pipeline
    parser = argparse.ArgumentParser(description="Run Brain Tumour Processing and Training Pipeline")

    # Add arguments
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--pipeline_name', type=str, default="Brain Tumour Processing and Training Pipeline", help='Name of the pipeline')
    parser.add_argument('--dataset_project', type=str, default="Brain Tumour Project", help='Project name for datasets')
    parser.add_argument('--raw_dataset_name', type=str, default="Brain Tumour Raw", help='Name for the raw dataset')
    parser.add_argument('--processed_dataset_name', type=str, default="Brain Tumour Preprocessed", help='Name for the processed dataset')
    parser.add_argument('--env_path', type=str, default="/.env", help='Path to the environment variables file')
    parser.add_argument('--repo_url', type=str, default="git@github.com:uts-strykers/braintumourdetection.git", help='URL to the Git repository')
    parser.add_argument('--development_branch', type=str, default="development", help='Default branch for development')
    parser.add_argument('--dataset_base_path', type=str, default="/this is the path/", help='Base path for datasets')
    parser.add_argument('--dataset_temp_path', type=str, default="/this is the path/", help='Temporary path for datasets')  
    parser.add_argument('--processed_dataset_temp_path', type=str, default="/this is the path/", help='Temporary path for processed datasets')  

    # Parse the arguments
    args = parser.parse_args()
    # Call the function with the parsed arguments
    create_brain_tumour_pipeline(
        epochs=args.epochs,
        pipeline_name=args.pipeline_name,
        dataset_project=args.dataset_project,
        raw_dataset_name=args.raw_dataset_name,
        processed_dataset_name=args.processed_dataset_name,
        env_path=args.env_path,
        repo_url=args.repo_url,
        development_branch=args.development_branch,
        dataset_base_path=args.dataset_base_path,
        dataset_temp_path=args.dataset_temp_path,
        processed_dataset_temp_path=args.processed_dataset_temp_path
    )

