# How Do We Get There

Hi! This is the companion repository for:
**Xiaomeng Ma** and **Lingyu Gao**. [How do we get there? Evaluating transformer neural networks as cognitive models for English past tense inflection.](https://xiaomeng-ma.github.io/English_Past_Tense_AACL.pdf) to appear at AACL-IJNLP 2022.

If you have any questions, please feel free to reach out to the Xiaomeng Ma at xm2158@tc.columbia.edu. 

## Experiments

(run the following commands in the Models folder, or modify the path of each .py file.)

#### Vanilla models

`python label_train-test.py -seed XX -data_path_train Data/Training/Train_type_reg.csv -label_spec XX -EPOCHS XX 
`

| Parameter        |                                                                                                          |
|------------------|----------------------------------------------------------------------------------------------------------|
| -data_path_train | Data/Training/Train_type_reg.csv .../Train_type_irr.csv .../Train_token_both.csv. ../Train_token_reg.csv |
| -label_spec      | no (no label) reg (regularity label) vc (verb class label) both (both reg and vc label)                  |


Used in paper:

Seeds: 42, 88, 266, 144, 24

Epochs: 30

Batch_size: default 32, (128 for Train_token_both and 64 for Train_token_irr due to RAM limit)

#### Copy models

`python copy_train-test.py -seed XX -data_path_train Data/Training/Train_type_reg.csv -label_spec XX -EPOCHS XX 
`

#### Resample Methods

`python resample_label_train-test.py -seed XX -data_path_train Data/Training/Train_type_reg.csv -label_spec XX -EPOCHS XX 
`

Used in paper:

Epochs: 100

Batch_size: 8

#### Inferencing Label for new test output

(modify the model path to import the trained models)

`python inf_label_train-test.py -seed XX -data_path_train Data/Training/Train_type_reg.csv -label_spec XX -EPOCHS XX -model_path (TRAINED_VANINLLA_MODEL_PATH)
`

`python inf_copy_train-test.py -seed XX -data_path_train Data/Training/Train_type_reg.csv -label_spec XX -EPOCHS XX -model_path (TRAINED_COPY_MODEL_PATH)
`

