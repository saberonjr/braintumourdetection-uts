import os
import numpy as np
import pandas as pd
import clearml
from clearml import Dataset, Task, TaskTypes
from clearml.automation.controller import PipelineDecorator


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(name="UploadRawTrainData", return_values=["train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one(dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset

    
    return "RawTrainDatasetID"

    
    
@PipelineDecorator.component(name="UploadRawValidData", return_values=["valid_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two_a(raw_train_dataset_id, dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset
    
    return "RawValidDatasetID"

@PipelineDecorator.component(name="UploadRawTestData", return_values=["valid_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two_b(raw_train_dataset_id, dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset
    
    return "RawTestDatasetID"
 

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


@PipelineDecorator.component(name="HPO", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_four(
    process_train_dataset_id, process_valid_dataset_id, process_test_dataset_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "hpoid"

@PipelineDecorator.component(name="TrainModel", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_five(
    hpo_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "trainmodelid"


@PipelineDecorator.component(name="EvaluateModel", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_six(
    train_model_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "evaluate_model_id"


@PipelineDecorator.component(name="TestModel", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_seven(
    train_model_id, processed_dataset_project, processed_dataset_name
):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task

    return "test_model_id"


@PipelineDecorator.component(name="PushModel", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
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
@PipelineDecorator.pipeline(name="BrainScan2DataPipeline", project="Strykers2", target_project="Strykers2", pipeline_execution_queue="uts-strykers-queue", default_queue="uts-strykers-queue") #, version="0.0.6")
def executing_data_pipeline(dataset_project, dataset_name, dataset_root, output_root):

    raw_train_dataset_id = step_one(dataset_project, dataset_name, dataset_root)

    raw_validation_dataset_id = step_two_a(raw_train_dataset_id, dataset_name, dataset_root)
    
    raw_test_dataset_id = step_two_b(raw_train_dataset_id, dataset_name, dataset_root)
    
    process_train_dataset_id = step_two_c(raw_train_dataset_id, dataset_name, dataset_root)
    
    process_valid_dataset_id = step_three_a(raw_validation_dataset_id, dataset_name, dataset_root)
    
    process_test_dataset_id = step_three_b(raw_test_dataset_id, dataset_name, dataset_root)
    
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
    #PipelineDecorator.run_locally()
    #PipelineDecorator.debug_pipeline()
    # Start the pipeline execution logic.
    

    executing_data_pipeline(
        dataset_project="BrainScan2",
        dataset_name="BrainScan2",
        dataset_root="/root/braintumourdetection/brainscan2/datasets/brain-tumor",
        output_root="/root/braintumourdetection/brainscan2/output"
    )

    print("process completed")


#from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data='brain-tumor.yaml', epochs=100, imgsz=640)