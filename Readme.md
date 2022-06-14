Need to be run

## Label: 'Both'

!python English-Past-Tense/label_train-test.py -seed 266 -data_path_train Data/Training/Train_token_reg.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 20

!python English-Past-Tense/label_train-test.py -seed 24 -data_path_train Data/Training/Train_token_reg.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 20

!python English-Past-Tense/label_train-test.py -seed 144 -data_path_train Data/Training/Train_token_reg.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 20

### 跑以下 train_token_irr

### !python English-Past-Tense/label_train-test.py -seed 42 -data_path_train Data/Training/Train_token_irr.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 20

### !python English-Past-Tense/label_train-test.py -seed 88 -data_path_train Data/Training/Train_token_irr.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 20

### !python English-Past-Tense/label_train-test.py -seed 266 -data_path_train Data/Training/Train_token_irr.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 20

### !python English-Past-Tense/label_train-test.py -seed 24 -data_path_train Data/Training/Train_token_irr.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 20

### !python English-Past-Tense/label_train-test.py -seed 144 -data_path_train Data/Training/Train_token_irr.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 20

### 跑以下 Token_Parent

### !python English-Past-Tense/label_train-test.py -seed 42 -data_path_train Data/Training/Token_Parent.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 30

### !python English-Past-Tense/label_train-test.py -seed 88 -data_path_train Data/Training/Token_Parent.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 30

### !python English-Past-Tense/label_train-test.py -seed 266 -data_path_train Data/Training/Token_Parent.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 30

### !python English-Past-Tense/label_train-test.py -seed 24 -data_path_train Data/Training/Token_Parent.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 30

### !python English-Past-Tense/label_train-test.py -seed 144 -data_path_train Data/Training/Token_Parent.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'both' -EPOCHS 30

## Label: 'reg'

## 以下都跑

!python English-Past-Tense/label_train-test.py -seed 24 -data_path_train Data/Training/Train_token_irr.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'reg' -EPOCHS 20

!python English-Past-Tense/label_train-test.py -seed 144 -data_path_train Data/Training/Train_token_irr.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'reg' -EPOCHS 20

!python English-Past-Tense/label_train-test.py -seed 42 -data_path_train Data/Training/Token_Parent.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'reg' -EPOCHS 30

!python English-Past-Tense/label_train-test.py -seed 88 -data_path_train Data/Training/Token_Parent.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'reg' -EPOCHS 30

!python English-Past-Tense/label_train-test.py -seed 266 -data_path_train Data/Training/Token_Parent.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'reg' -EPOCHS 30

!python English-Past-Tense/label_train-test.py -seed 24 -data_path_train Data/Training/Token_Parent.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'reg' -EPOCHS 30

!python English-Past-Tense/label_train-test.py -seed 144 -data_path_train Data/Training/Token_Parent.csv -data_path_test Data/Test/Test.csv -data_path_nonce Data/Test/Nonce.csv -label_spec 'reg' -EPOCHS 30

