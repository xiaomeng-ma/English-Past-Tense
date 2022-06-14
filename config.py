import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description = "English Past Tense")
    ##seed
    parser.add_argument('-seed', type = int, default = 1, help = "random seeds")
    ##data path
    parser.add_argument('-data_path_train', type = str, default = 'Input_csv/Training/Train_type_reg.csv', help = 'training data path')
    parser.add_argument('-data_path_test', type = str, default='Input_csv/Test.csv', help = 'test data path')
    parser.add_argument('-data_path_nonce', type = str, default='Input_csv/Nonce.csv', help = 'nonce data path')
    ##data label
<<<<<<< HEAD
=======
    parser.add_argument('-label_pos', type = str, default = 'input', choices = ['input', 'output'], help = 'to add labels on the input or output')
>>>>>>> 614678e8b1d01d40024a25077b3a15d71995f9f5
    parser.add_argument('-label_spec', type = str, default = 'reg', choices = ['reg', 'vc', 'both', 'no'], help = 'reg: regularity, vc: verb class, both: both, no: no label')
    ##EPOCHS
    parser.add_argument('-EPOCHS',type = int, default = 30, metavar ='N',help= 'number of EPOCHS')
    ##top k
    parser.add_argument('-vk', type = int, default=5, help ='set k value')
<<<<<<< HEAD
    ##embedding
    parser.add_argument('-embedding', type= str, default = 'Embedding/type_vectors')
=======
    
>>>>>>> 614678e8b1d01d40024a25077b3a15d71995f9f5
    #model setting
    parser.add_argument('-model_path', type = str, default = 'Checkpoints/', help = 'data path')
    parser.add_argument('-num_heads', type = int, default = 4, metavar = 'N', help = 'number of heads')
    parser.add_argument('-d_model', type=int, default = 128, metavar = 'N', help = 'the number of expected features in the encoder/decoder inputs')
    parser.add_argument('-dff', type=int, default = 256, metavar = 'N', help = 'the dimension of the feedforward network model')
    
    ##learning
    parser.add_argument('-nlayers', type = int, default = 2, help='number of layers')
    parser.add_argument('-dropout', type = float, default = 0.1, help='dropout applied to layers (0 = no dropout)')
    
    return parser.parse_args()

def process_args(args):
<<<<<<< HEAD
    suffix_dir = str(args.seed) + '_' + str(args.vk) + '_' + args.label_spec + '_' + args.data_path_train.split('/')[-1]
=======
    suffix_dir = str(args.seed) + '_' + str(args.vk) + '_' + args.label_pos + '_' + args.label_spec + '_' + args.data_path_train.split('/')[-1]
>>>>>>> 614678e8b1d01d40024a25077b3a15d71995f9f5
    args.model_path = os.path.join(args.model_path, suffix_dir)
    print(args.model_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)
        
    return args