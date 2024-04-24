from clearml import PipelineController, Task

#from BRAIN_TUMOUR_DETECTOR.evaluate_model import evaluate_model, log_debug_images
#from first_mlops_pipeline.preprocess_upload_cifar10 import (
#    preprocess_and_upload_cifar10,
#    save_preprocessed_data,
#)
#from first_mlops_pipeline.train_model import train_model
#from brain_tumour_detector.update_model import (
#    archive_existing_model,
#    cleanup_repo,
#    clone_repo,
#    commit_and_push,
#    configure_ssh_key,
#    ensure_archive_dir,
#    update_model,
#    update_weights,
#)
from brainscan_pipeline.upload_raw_data import (
    save_numpy_arrays,
    load_dataset,
    upload_raw_dataset_as_numpy_to_clearml
)


def create_brainscan_pipeline(
    epochs: int = 10,
    pipeline_name: str = "BrainScan Training Pipeline",
    dataset_project: str = "BrainScan Project",
    raw_dataset_name: str = "BrainScan Raw",
    processed_dataset_name: str = "BrainScan Preprocessed",
    env_path: str = "/path/to/.env",
    repo_url: str = "git@github.com:saberonjr/First_MLOPS_Pipeline.git",
    development_branch: str = "development",
):
    from clearml import PipelineController, Task

    #from first_mlops_pipeline.evaluate_model import evaluate_model, log_debug_images
    #from first_mlops_pipeline.preprocess_upload_cifar10 import (
    #    preprocess_and_upload_cifar10,
    #    save_preprocessed_data,
    #)
    #from first_mlops_pipeline.train_model import train_model
    #from first_mlops_pipeline.update_model import (
    #    archive_existing_model,
    #    cleanup_repo,
    #    clone_repo,
    #    commit_and_push,
    #    configure_ssh_key,
    #    ensure_archive_dir,
    #    update_model,
    #    update_weights,
    #)
    from brainscan_pipeline.upload_raw_data import (
        save_numpy_arrays,
        load_dataset,
        upload_raw_dataset_as_numpy_to_clearml
    )

    # Initialize a new pipeline controller task
    pipeline = PipelineController(
        name=pipeline_name,
        project=dataset_project,
        version="1.0",
        add_pipeline_tags=True,
        auto_version_bump=True,
        target_project=dataset_project,
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

    # Step 1: Upload CIFAR-10 Raw Data
    pipeline.add_function_step(
        name="upload_raw_dataset_as_numpy_to_clearml",
        function=upload_raw_dataset_as_numpy_to_clearml,
        function_kwargs={
            "dataset_project": "${pipeline.dataset_project}",
            "dataset_name": "${pipeline.raw_dataset_name}",
        },
        task_type=Task.TaskTypes.data_processing,
        task_name="Upload BrainScan Raw Data",
        function_return=["raw_dataset_id"],
        helper_functions=[save_numpy_arrays, load_dataset],
        cache_executed_step=False,
    )

    # Step 2: Preprocess CIFAR-10 Data
    #pipeline.add_function_step(
    #    name="preprocess_cifar10_data",
    #    function=preprocess_and_upload_cifar10,
    #    function_kwargs={
    #        "raw_dataset_id": "${upload_cifar10_raw_data.raw_dataset_id}",
    #        "processed_dataset_project": "${pipeline.dataset_project}",
    #        "processed_dataset_name": "${pipeline.processed_dataset_name}",
    #    },
    #    task_type=Task.TaskTypes.data_processing,
    #    task_name="Preprocess and Upload CIFAR-10",
    #    function_return=["processed_dataset_id"],
    #    helper_functions=[save_preprocessed_data],
    #    cache_executed_step=False,
    #)
#
    ## Step 3: Train Model
    #pipeline.add_function_step(
    #    name="train_cifar10_model",
    #    function=train_model,
    #    function_kwargs={
    #        "processed_dataset_id": "${preprocess_cifar10_data.processed_dataset_id}",
    #        "epochs": "${pipeline.epochs}",
    #        "project_name": "${pipeline.dataset_project}",
    #    },
    #    task_type=Task.TaskTypes.training,
    #    task_name="Train CIFAR-10 Model",
    #    function_return=["model_id"],
    #    cache_executed_step=False,
    #)
#
    ## Step 4: Evaluate Model
    #pipeline.add_function_step(
    #    name="evaluate_cifar10_model",
    #    function=evaluate_model,
    #    function_kwargs={
    #        "model_id": "${train_cifar10_model.model_id}",
    #        "processed_dataset_id": "${preprocess_cifar10_data.processed_dataset_id}",
    #        "project_name": "${pipeline.dataset_project}",
    #    },
    #    task_type=Task.TaskTypes.testing,
    #    task_name="Evaluate CIFAR-10 Model",
    #    helper_functions=[log_debug_images],
    #    cache_executed_step=False,
    #)
#
    ## Step 5: Update Model in GitHub Repository
    #pipeline.add_function_step(
    #    name="update_model_in_github",
    #    function=update_model,
    #    function_kwargs={
    #        "model_id": "${train_cifar10_model.model_id}",
    #        "env_path": "${pipeline.env_path}",
    #        "REPO_URL": "${pipeline.REPO_URL}",
    #        "DEVELOPMENT_BRANCH": "${pipeline.DEVELOPMENT_BRANCH}",
    #        "project_name": "${pipeline.dataset_project}",
    #    },
    #    helper_functions=[
    #        configure_ssh_key,
    #        clone_repo,
    #        ensure_archive_dir,
    #        archive_existing_model,
    #        update_weights,
    #        commit_and_push,
    #        cleanup_repo,
    #    ],
    #    task_type=Task.TaskTypes.custom,
    #    task_name="Export Model to GitHub Repository",
    #    cache_executed_step=False,
    #)

    # Start the pipeline
    pipeline.start(queue="uts-strykers-queue")
    print("BrainScan pipeline initiated. Check ClearML for progress.")