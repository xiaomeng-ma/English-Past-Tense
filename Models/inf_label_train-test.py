from tensorflow.python.client import device_lib
import os
import sys
import time
import logging
import data_process
import config
import pandas as pd
import torch
import torch.nn as nn
import math
from scipy.stats import entropy

from Model import *

loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    
class BeamSearch(nn.Module):
    def __init__(self, args, pad_id, vocab_size):
        super().__init__()
        self.vk = 1
        self.pad_id = pad_id
        self.vocab_size = vocab_size
    def step (self, step, lprobs, scores):
        bsz, beam_size, _ = lprobs.size()
        lprobs[:, :, self.pad_id] = -math.inf
        if step == 0:
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            assert scores is not None
            lprobs = lprobs[:, -1, :]
            scores = scores.repeat_interleave(self.vocab_size, dim = 1)
            lprobs = lprobs + scores
        scores_buf, indices_buf = torch.topk(lprobs.view(bsz, -1),  k = self.vk)
        beams_buf = tf.cast(indices_buf // self.vocab_size, tf.int32)
        indices_buf = indices_buf.fmod(self.vocab_size)
        return scores_buf, indices_buf, beams_buf
        
class Generate(object):
    def __init__(self, args, tokenizer, metric):
        self.args = args
        self.tokenizer = tokenizer
        self.pad_id = 0
        self.metric = metric
        self.vocab_size = vocab_num
        self.model = Transformer(
            num_layers = args.nlayers,
            d_model = args.d_model,
            num_heads = args.num_heads,
            dff = args.dff,
            input_vocab_size = vocab_num,
            target_vocab_size = vocab_num,
            rate = args.dropout)
        if self.metric == 'irr':
            self.load_model_irr()
        else:
            self.load_model()
        self.search = BeamSearch(args, self.pad_id, self.vocab_size)
        
    def load_model_irr(self):
        optimizer = tf.keras.optimizers.Adam(CustomSchedule(self.args.d_model, self.args.warmup), 
        beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        checkpoint_path = self.args.model_path
        ckpt = tf.train.Checkpoint(transformer = self.model, optimizer = optimizer)
        irr_ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(checkpoint_path, 'irr'), max_to_keep=1)
        if irr_ckpt_manager.latest_checkpoint:
            ckpt.restore(irr_ckpt_manager.latest_checkpoint).expect_partial()
            status = ckpt.restore(irr_ckpt_manager.latest_checkpoint)
            status.expect_partial()
            print('Last irr checkpoint restored')
        else:
            assert ValueError ('No checkpoint')
    
    
    def load_model(self):
        optimizer = tf.keras.optimizers.Adam(CustomSchedule(self.args.d_model, self.args.warmup), 
        beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        checkpoint_path = self.args.model_path
        ckpt = tf.train.Checkpoint(transformer = self.model, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = 1)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            status = ckpt.restore(ckpt_manager.latest_checkpoint)
            status.expect_partial()
            logging.info('I found the trained model!')
        else:
            assert ValueError ('No checkpoint')
    
    def forward (self, test_dataset):
        output_list = []
        try:
            for element in test_dataset.as_numpy_iterator():
                inp, tar = element['input_seq']
                output = tar[:, :1]
                bsz, seq_len = int(tf.shape(tar)[0]), int(tf.shape(tar)[1])
                output = self.generate(inp, [output], bsz, seq_len-1, tar)
                output_list.append(output)
        except:
            for (batch, (inp, tar)) in enumerate(test_dataset):
                output = tar[:, :1]
                bsz, seq_len = int(tf.shape(tar)[0]), int(tf.shape(tar)[1])
                output = self.generate(inp, [output], bsz, seq_len-1)
                output_list.append(output)
                
        if len(output_list) ==1:
            return output_list[0]
        else:
            return output_list
    
    def get_inp_tar(self, cand_indices, cand_beams, output_list):
        cand_indices = tf.expand_dims(cand_indices, axis =2)
        prev_output = tf.transpose(tf.convert_to_tensor(output_list),[1,0,2])
        prev_output = tf.gather(prev_output, cand_beams, batch_dims = 1)
        output = tf.transpose(tf.concat(axis = 2, values = [prev_output.numpy(), cand_indices.numpy()]), [1,0,2])
        return output
        
        
    def generate(self, inp, output_list, bsz: int, total_step:int, tar):
        beam_size = 5
        cand_scores = torch.zeros(int(bsz *beam_size), total_step)
        tar = tar
        for step in range(total_step):
            for output in output_list:
                predictions, _ = self.model([inp,output], training = False)
                try:
                    lprobs
                    lprobs = tf.concat(axis = 2, values = [lprobs, tf.nn.log_softmax(predictions, axis = 2)])
                except:
                    lprobs = tf.nn.log_softmax(predictions, axis = 2)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step, torch.from_numpy(lprobs.numpy()), cand_scores)
            if step<3:
                output_list = np.expand_dims(tar[:,:step+2], axis = 0)
            else:
                output_list = self.get_inp_tar(cand_indices, cand_beams, output_list)
        return output_list
        
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    args = config.get_args()
    args = config.process_args(args)
    
    ## random seeds
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    from importlib import reload
    
    reload(logging)
    
    log_file = os.path.join(args.model_path, 'log')
    handlers = [logging.FileHandler(log_file, mode = 'w+'), logging.StreamHandler()]
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M', level=logging.INFO, handlers=handlers)
    logging.info(args)
    
    logging.info(device_lib.list_local_devices())
    
    data, t = data_process.process_data(args)
    _, _, test_dataset, _ = data
    df_test = pd.read_csv(os.path.join(args.data_path_test),index_col = 0)

    ##settings
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    metrics = [accuracy]
    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(args.d_model, args.warmup), beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    
    
    add = args.label_spec
    
    ##create model
    vocab_num = len(t.word_counts) +1
            
    gen_model = Generate(args, t, 'mean')
    #dev_reg_acc, dev_irr_acc , _, _ = test_step(val_dataset, gen_model.model, t, add)
    #test_reg_acc, test_irr_acc, df_test_pred, test_entropy_list = test_step(test_dataset, gen_model.model,t, add)
    #df_nonce_pred, nonce_entropy_list = nonce_step(nonce_dataset, gen_model.model, t, add)
    
    #irr_gen_model = Generate(args, t, 'irr')
    #irr_dev_reg_acc, irr_dev_irr_acc , _, _ = test_step(val_dataset, irr_gen_model.model, t, add)
    #irr_test_reg_acc, irr_test_irr_acc, irr_df_test_pred, irr_test_entropy_list = test_step(test_dataset, irr_gen_model.model,t, add)
    #irr_df_nonce_pred, irr_nonce_entropy_list = nonce_step(nonce_dataset, irr_gen_model.model, t, add)
    
    #print('dev_reg_acc:', dev_reg_acc)
    #print('dev_irr_acc:', dev_irr_acc)
    #print('test_reg_acc:', test_reg_acc)
    #print('test_irr_acc:', test_irr_acc)
    
        
    #print('irr_dev_reg_acc:', irr_dev_reg_acc)
    #print('irr_dev_irr_acc:', irr_dev_irr_acc)
    #print('irr_test_reg_acc:', irr_test_reg_acc)
    #print('irr_test_irr_acc:', irr_test_irr_acc)
    
    #df_test_pred.to_csv(os.path.join(args.model_path, 'test_pred.csv'))
    #df_nonce_pred.to_csv(os.path.join(args.model_path, 'nonce_pred.csv'))
    #pd.DataFrame(test_entropy_list).to_csv(os.path.join(args.model_path, 'test_entropy.csv'))
    #pd.DataFrame(nonce_entropy_list).to_csv(os.path.join(args.model_path, 'nonce_entropy.csv'))

    #irr_df_test_pred.to_csv(os.path.join(args.model_path, 'irr/', 'test_pred.csv'))
    #irr_df_nonce_pred.to_csv(os.path.join(args.model_path, 'irr/', 'nonce_pred.csv'))
    #pd.DataFrame(irr_test_entropy_list).to_csv(os.path.join(args.model_path, 'irr/', 'test_entropy.csv'))
    #pd.DataFrame(irr_nonce_entropy_list).to_csv(os.path.join(args.model_path, 'irr/', 'nonce_entropy.csv'))

    test_pred_list = gen_model.forward(test_dataset)
    #nonce_pred_list = gen_model.forward(nonce_dataset)
    
    #irr_test_pred_list = irr_gen_model.forward(test_dataset)
    #irr_nonce_pred_list = irr_gen_model.forward(nonce_dataset)
    
    
    
    def clean(x):
        try:
            end = x.index('end')
        except:
            end = len(x)
        return ",".join(x[:end])
    
    df_top = pd.DataFrame(test_pred_list.numpy().tolist()).transpose()
    df_top = df_top.rename(columns = {0:'inf_pred'})
    #df_top_nonce = pd.DataFrame(nonce_pred_list.numpy().tolist()).transpose()
    
    #irr_df_top = pd.DataFrame(irr_test_pred_list.numpy().tolist()).transpose()
    #irr_df_top_nonce = pd.DataFrame(irr_nonce_pred_list.numpy().tolist()).transpose()
    df_top['inf_pred'] = df_top['inf_pred'].apply(lambda row: t.sequences_to_texts([row])[0].split(' ')[1:])
    df_top['inf_pred'] = df_top['inf_pred'].apply(lambda x: clean(x))
    if add == 'both':
        df_top['pred'] = df_top['inf_pred'].apply(lambda x: "".join(x.split(',')[2:])) 
    else:
        df_top['pred'] = df_top['inf_pred'].apply(lambda x: "".join(x.split(',')[1:]))
    df_top['target'] = df_test['ipa_word'].reset_index(drop=True)
    df_top['reg'] = df_test['reg'].reset_index(drop=True)
    df_top['cor1'] = np.where((df_top['pred'] == df_top['target']), 1, 0)
    
    test_reg_acc_top1 = df_top[df_top['reg']=='Reg']['cor1'].sum()/60
    test_irr_acc_top1 = df_top[df_top['reg']=='Irreg']['cor1'].sum()/20 
    
    
    print('test_reg_acc', test_reg_acc_top1)
    print('test_irr_acc', test_irr_acc_top1)
    
    df_acc = pd.DataFrame([test_reg_acc_top1, test_irr_acc_top1]).T
    df_acc.columns = ['test_reg_acc', 'test_irr_acc']
    
    df_top.to_csv(os.path.join(args.model_path, 'inf_test_pred.csv'))
    df_acc.to_csv(os.path.join(args.model_path, 'inf_test_acc.csv'))
