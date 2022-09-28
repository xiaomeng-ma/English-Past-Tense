## Code Description

### Data Processing
The standard data_processing.py uses 90-10 train-dev split and adds labels (regularity, verb class, both). The resampled_data_processing.py is used for the resampling methods in the paper where we manipulate the data distribution for each training epoch.

### Transformer Models
The vanilla model used in the paper is Model.py. The model with pointer-generator mechanism is Modey_copy.py.

### Training and Evaluation
The lable_train-test.py and copy_train-test.py are used train and generated test results for the vanilla model and copy model respectively. 

The resample_train-test.py is used to train and generated test results for the resampling models.

The inf_label_train-test.py and inf_copy_train-test.py are used to do inferening on label to generate test results.


