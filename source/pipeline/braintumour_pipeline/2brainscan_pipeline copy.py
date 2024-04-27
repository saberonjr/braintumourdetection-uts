from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes
import pandas as pd

# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(name="UploadRawData", return_values=["data_frame"], cache=False, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one(pickle_data_url: str, extra: int = 43):
    #print("step_one")
    # make sure we have scikit-learn for this step, we need it to use to unpickle the object
    import sklearn  # noqa
    import pickle
    import pandas as pd
    from clearml import StorageManager

    local_iris_pkl = StorageManager.get_local_copy(remote_url=pickle_data_url)
    with open(local_iris_pkl, "rb") as f:
        iris = pickle.load(f)
    data_frame = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    data_frame.columns += ["target"]
    data_frame["target"] = iris["target"]
    return data_frame


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step.
# Specifying `return_values` makes sure the function step can return an object to the pipeline logic
# In this case, the returned tuple will be stored as an artifact named "X_train, X_test, y_train, y_test"
@PipelineDecorator.component(name="PreprocessModel",
    return_values=["X_train", "X_test", "y_train", "y_test"], cache=False, task_type=TaskTypes.data_processing#, execution_queue="default"
)
def step_two(data_frame, test_size=0.2, random_state=42):
    #print("step_two")
    # make sure we have pandas for this step, we need it to use the data_frame
    import pandas as pd  # noqa
    from sklearn.model_selection import train_test_split

    y = data_frame["target"]
    X = data_frame[(c for c in data_frame.columns if c != "target")]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
# Specifying `return_values` makes sure the function step can return an object to the pipeline logic
# In this case, the returned object will be stored as an artifact named "model"
@PipelineDecorator.component(name="TrainModel",return_values=["model"], cache=False, task_type=TaskTypes.training)#, execution_queue="default")
def step_three(X_train, y_train):
    #print("step_three")
    # make sure we have pandas for this step, we need it to use the data_frame
    import pandas as pd  # noqa
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver="liblinear", multi_class="auto")
    model.fit(X_train, y_train)
    return model


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
# Specifying `return_values` makes sure the function step can return an object to the pipeline logic
# In this case, the returned object will be stored as an artifact named "accuracy"
@PipelineDecorator.component(name="EvaluateModel",return_values=["accuracy"], cache=False, task_type=TaskTypes.qc)#, execution_queue="default")
def step_four(model, X_data, Y_data):
    from sklearn.linear_model import LogisticRegression  # noqa
    from sklearn.metrics import accuracy_score

    Y_pred = model.predict(X_data)
    return accuracy_score(Y_data, Y_pred, normalize=True)

@PipelineDecorator.component(name="TestModel", cache=False, task_type=TaskTypes.qc)#, execution_queue="default")
def step_five(model):

    print("Testing model")
    print(model)



@PipelineDecorator.component(name="UploadModelToGit", cache=False, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_six(model):

    print("Uploading model to github")
    print(model)



# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(name="BrainScanPipeline", project="Strykers", target_project="Strykers", pipeline_execution_queue="default", default_queue="default") #, version="0.0.6")
def executing_pipeline(pickle_url, mock_parameter="mock"):
    print("pipeline args:", pickle_url, mock_parameter)

    # Use the pipeline argument to start the pipeline and pass it ot the first step
    print("::=======================================::")
    print("Step1: Launch UploadRawDataset Task")
    print("::=======================================::")
    data_frame = step_one(pickle_url)

    # Use the returned data from the first step (`step_one`), and pass it to the next step (`step_two`)
    # Notice! unless we actually access the `data_frame` object,
    # the pipeline logic does not actually load the artifact itself.
    # When actually passing the `data_frame` object into a new step,
    # It waits for the creating step/function (`step_one`) to complete the execution
    print("::=======================================::")
    print("Step 2: Launch PreprocessRawDataset Task")
    print("::=======================================::")
    X_train, X_test, y_train, y_test = step_two(data_frame)
    
    print("::=======================================::")
    print("Step 3: Launch TrainModel Task")
    print("::=======================================::")
    model = step_three(X_train, y_train)

    # Notice since we are "printing" the `model` object,
    # we actually deserialize the object from the third step, and thus wait for the third step to complete.
    print("returned model: {}".format(model))

    print("::=======================================::")
    print("Step 4. Launch EvaluateModel Task")
    print("::=======================================::")
    accuracy = 100 * step_four(model, X_data=X_test, Y_data=y_test)

    print("::=======================================::")
    print("Step 5. Launch TestModel Task")
    print("::=======================================::")
    step_five(model)

    # Notice since we are "printing" the `accuracy` object,
    # we actually deserialize the object from the fourth step, and thus wait for the fourth step to complete.
    print(f"Accuracy={accuracy}%")

    print("::=======================================::")
    print("Step 6. Launch UploadModel Task")
    print("::=======================================::")
    step_six(model)


if __name__ == "__main__":
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    PipelineDecorator.set_default_execution_queue('default')
    # Run the pipeline steps as subprocesses on the current machine, great for local executions
    # (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)
    PipelineDecorator.run_locally()
    #PipelineDecorator.debug_pipeline()
    # Start the pipeline execution logic.
    executing_pipeline(
        pickle_url="https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl",
    )

    print("process completed")