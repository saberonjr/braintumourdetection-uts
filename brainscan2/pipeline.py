import os
import numpy as np
import pandas as pd
import clearml
from clearml import Dataset, Task, TaskTypes
from clearml.automation.controller import PipelineDecorator
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, GridSearch


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(name="UploadRawTrainData", return_values=["train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one(dataset_project, dataset_name, dataset_root):
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
def step_two_a(raw_train_dataset_id, dataset_project, dataset_name, dataset_root):
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
def step_two_b(raw_train_dataset_id, dataset_project, dataset_name, dataset_root):
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
def step_two_c(
    raw_train_dataset_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "ProcessTrainDatasetID"

@PipelineDecorator.component(name="ProcessValidData", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_three_a(
    process_valid_dataset_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "ProcessValidDatasetID"


@PipelineDecorator.component(name="ProcessTestData", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_three_b(
    process_test_dataset_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "ProcessTestDatasetID"


@PipelineDecorator.component(name="TrainModel", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.training)#, execution_queue="default")
def step_four(
    process_train_dataset_id, process_valid_dataset_id, process_test_dataset_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "trainmodelid"


@PipelineDecorator.component(name="EvaluateModel", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.training)#, execution_queue="default")
def step_five(
    train_model_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "evaluate_model_id"

@PipelineDecorator.component(name="HPO", return_values=["top_experiment_id"], cache=True, task_type=TaskTypes.optimizer)#, execution_queue="default")
def step_six(
    base_task_id, queue_name
):
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
def executing_data_pipeline(dataset_project, dataset_name, dataset_root, output_root):

    raw_train_dataset_id = step_one(dataset_project, dataset_name, dataset_root)

    
    raw_validation_dataset_id = step_two_a(raw_train_dataset_id, dataset_project, dataset_name, dataset_root)
    
    
    raw_test_dataset_id = step_two_b(raw_train_dataset_id, dataset_project, dataset_name, dataset_root)
    
    
    process_train_dataset_id = step_two_c(raw_train_dataset_id, dataset_project, dataset_name)
    
    
    process_valid_dataset_id = step_three_a(raw_validation_dataset_id, dataset_name)
    
    
    process_test_dataset_id = step_three_b(raw_test_dataset_id, dataset_name)
    
    
    step_four_id = step_four(process_train_dataset_id, process_valid_dataset_id, process_test_dataset_id, dataset_name, dataset_root)

    
    step_five_id = step_five(step_four_id, dataset_name, dataset_root)

    
    step_six_id = step_six(step_five_id, dataset_name, dataset_root)


    step_seven_id = step_seven(step_six_id, dataset_name, dataset_root)

    
    step_eight_id = step_eight(step_seven_id, dataset_name, dataset_root)




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
        output_root="/Users/soterojrsaberon/UTS/braintumourdetection/brainscan2/datasets/brain-tumor/output"
    )

    print("process completed")


#from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data='brain-tumor.yaml', epochs=100, imgsz=640)