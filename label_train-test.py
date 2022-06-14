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

from model_no_embedding import *

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

def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp], 
                                 training = True)
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

def char_cor(row):
    verb = row['ipa_word']
    pred = row['pred']
    len = min(row['pred_len'], row['tar_len'])
    kk = 0
    for i in range(len):
        if verb[i] == pred[i]:
            kk+=1
    return kk

def correct(row):
    if row['ipa_word']==row['pred']:
        k = 1
    else:
        k = 0
    return k

def dev_step(val_dataset, t):
    start, end = t.word_index['start'], t.word_index['end']
    val_cor_all = []
    for element in val_dataset.as_numpy_iterator():
        inp, tar = element['input_seq']
        label = element['label']
        output = tar[:,:1]
        index_tar = np.where(tar==2)[0]
        bsz, seq_len = float(tf.shape(tar)[0]), float(tf.shape(tar)[1])
        for i in tf.range(seq_len -1):
            pred, _ = transformer([inp, output], training = False)
            predictions = pred[:, -1:, :]
            predicted_id = tf.argmax(predictions,axis = -1)
            output = tf.concat(values = [output, predicted_id], axis = 1)
        val_cor = []
        for line_tar, line_output, lab in zip(tar.tolist(), output.numpy().tolist(), label.tolist()):
            index_tar = line_tar.index(end)
            try:
                index_output = line_output.index(end)
            except:
                index_output = index_tar
            output_a = "".join(t.sequences_to_texts([line_output[1:index_output]])[0].split(' '))
            tar_a = "".join(t.sequences_to_texts([line_tar[1:index_tar]])[0].split(' '))
            val_cor.append([output_a, tar_a, lab])
        val_cor_all.append(val_cor)
    val_cor_all = list(itertools.chain(*val_cor_all))
    print('All val len:', len(val_cor_all))
    df_val_cor = pd.DataFrame(val_cor_all, columns = ['pred', 'ipa_word','reg'])
    df_val_cor['cor'] = df_val_cor.apply(lambda row: correct(row), axis=1)
    df_val_cor['tar_len'] = [len(i) for i in df_val_cor['ipa_word']]
    df_val_cor['pred_len'] = [len(i) for i in df_val_cor['pred']]
    df_val_cor['char_cor'] = df_val_cor.apply(lambda row:char_cor(row), axis =1)
    Total_cor = df_val_cor['cor'].sum()/len(df_val_cor)
    Reg = df_val_cor[df_val_cor['reg']==b'Reg']
    Irreg = df_val_cor[df_val_cor['reg']==b'Irreg']
    Reg_cor = Reg['cor'].sum()/len(Reg)
    Irreg_cor = Irreg['cor'].sum()/len(Irreg)
    Reg_char_cor = Reg['char_cor'].sum()/Reg['tar_len'].sum()
    Irreg_char_cor = Irreg['char_cor'].sum()/Irreg['tar_len'].sum()
    return Total_cor, Reg_cor, Irreg_cor,Reg_char_cor, Irreg_char_cor
    
def test_step(test_dataset, model, t):
    start, end = t.word_index['start'], t.word_index['end']
    test_cor_all = []
    for element in test_dataset.as_numpy_iterator():
        inp, tar = element['input_seq']
        label = element['label']
        output_list = []
        output = tar[:,:1]
        bsz, seq_len = float(tf.shape(tar)[0]), float(tf.shape(tar)[1])
        entropy_list = []
        for i in tf.range(seq_len -1):
            pred, _ = model([inp, output], training = False)
            predictions = pred[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis = -1)
            lprobs = tf.nn.log_softmax(predictions, axis = 2)
            entropy_list.append(entropy(np.exp(lprobs.numpy()), axis = 2).ravel())
            output = tf.concat(values = [output, predicted_id], axis= 1)
        output_list.append(output)
        test_cor = []
        for line_tar, line_output, lab in zip(tar.tolist(), output_list[0].numpy().tolist(), label.tolist()):
            index_tar = line_tar.index(end)
            try:
                index_output = line_output.index(end)
            except:
                index_output = index_tar
            output_all = ",".join(t.sequences_to_texts([line_output])[0].split(' '))
            output_a = "".join(t.sequences_to_texts([line_output[1:index_output]])[0].split(' '))
            tar_a = "".join(t.sequences_to_texts([line_tar[1:index_tar]])[0].split(' '))
            test_cor.append([output_all, output_a, tar_a, lab])
        test_cor_all.append(test_cor)
    test_cor_all = list(itertools.chain(*test_cor_all))
    df_test_cor = pd.DataFrame(test_cor_all, columns = ['all', 'pred','ipa_word', 'reg'])
    df_test_cor['cor'] = df_test_cor.apply(lambda row: correct(row), axis = 1)
    Reg = df_test_cor[df_test_cor['reg'] == b'Reg']
    Irreg = df_test_cor[df_test_cor['reg'] == b'Irreg']
    Reg_cor= Reg['cor'].sum()/len(Reg)
    Irreg_cor = Irreg['cor'].sum()/len(Irreg)
    return Reg_cor, Irreg_cor, df_test_cor, entropy_list

def nonce_step(nonce_dataset, model, t):
    start, end = t.word_index['start'], t.word_index['end']
    for (batch, (inp,tar)) in enumerate(nonce_dataset):
        output_list = []
        output = tar[:,:1]
        entropy_list = []
        bsz, seq_len = float(tf.shape(tar)[0]), float(tf.shape(tar)[1])
        for i in tf.range(seq_len -1):
            pred, _ = model([inp, output], training = False)
            predictions = pred[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis = -1)
            lprobs = tf.nn.log_softmax(predictions, axis = 2)
            entropy_list.append(entropy(np.exp(lprobs.numpy()), axis = 2).ravel())
            output = tf.concat(values = [output, predicted_id], axis= 1)
        output_list.append(output)
        nonce_cor = []
        for line_tar, line_output in zip(tar.numpy().tolist(), output_list[0].numpy().tolist()):
            index_tar = line_tar.index(end)
            try:
                index_output = line_output.index(end)
            except:
                index_output = index_tar
            output_all = ",".join(t.sequences_to_texts([line_output])[0].split(' '))
            output_a = "".join(t.sequences_to_texts([line_output[1:index_output]])[0].split(' '))
            tar_a = "".join(t.sequences_to_texts([line_tar[1:index_tar]])[0].split(' '))
            nonce_cor.append([output_all, output_a, tar_a])
        df_nonce_pred = pd.DataFrame(nonce_cor, columns = ['all', 'pred','ipa_word'])
    return df_nonce_pred, entropy_list
    
class BeamSearch(nn.Module):
    def __init__(self, args, pad_id, vocab_size):
        super().__init__()
        self.vk = args.vk
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
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.pad_id = 0
        self.vocab_size = vocab_num
        self.model = Transformer(
            num_layers = args.nlayers,
            d_model = args.d_model,
            num_heads = args.num_heads,
            dff = args.dff,
            input_vocab_size = vocab_num,
            target_vocab_size = vocab_num,
            rate = args.dropout)
        self.load_model()
        self.search = BeamSearch(args, self.pad_id, self.vocab_size)
        
    def load_model(self):
        optimizer = tf.keras.optimizers.Adam(CustomSchedule(self.args.d_model), 
        beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        checkpoint_path = self.args.model_path
        ckpt = tf.train.Checkpoint(transformer = self.model, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = 1)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            status = ckpt.restore(ckpt_manager.latest_checkpoint)
            status.expect_partial()
            print('Last checkpoint restored')
        else:
            assert ValueError ('No checkpoint')
    def forward (self, test_dataset):
        output_list = []
        try:
            for element in test_dataset.as_numpy_iterator():
                inp, tar = element['input_seq']
                output = tar[:, :1]
                bsz, seq_len = int(tf.shape(tar)[0]), int(tf.shape(tar)[1])
                output = self.generate(inp, [output], bsz, seq_len-1)
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
        
        
    def generate(self, inp, output_list, bsz: int, total_step:int):
        beam_size = 5
        cand_scores = torch.zeros(int(bsz *beam_size), total_step)
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
            output_list = self.get_inp_tar(cand_indices, cand_beams, output_list)
        return output_list
        
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
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
    
    dataset,t = data_process.process_data(args)
    train_dataset, val_dataset, test_dataset, nonce_dataset = dataset
    num_batches, val_batches = len(train_dataset), len(val_dataset)
    
    ##settings
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    metrics = [accuracy]
    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(args.d_model), beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    
    EPOCHS = args.EPOCHS
    
    ##create model
    vocab_num = len(t.word_counts) +1
    transformer = Transformer(
        num_layers = args.nlayers,
        d_model = args.d_model,
        num_heads = args.num_heads,
        dff = args.dff,
        input_vocab_size = vocab_num,
        target_vocab_size = vocab_num,
        rate = args.dropout)
    checkpoint_path = args.model_path
    ckpt = tf.train.Checkpoint(transformer = transformer, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
    irr_ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(checkpoint_path, 'irr'), max_to_keep=1)
    
    if ckpt_manager.latest_checkpoint:
        logging.info('this is already trained')
        status = ckpt.restore(ckpt_manager.latest_checkpoint)
        status.expect_partial()
        epoch, best_epoch, best_irreg_epoch = 0., 0., 1.
    else:
        best_model, best_dev_acc, best_epoch = None, 0.0, 0.0
        best_irr_model, best_irr_dev_acc, best_irreg_epoch = None, 0.0, 0.0
        for epoch in range(EPOCHS):
            start = time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()
            for element in train_dataset.as_numpy_iterator():
                inp, tar = element['input_seq']
                train_step(inp, tar)
            logging.info(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            total_acc, dev_reg, dev_irreg, dev_reg_char, dev_irreg_char = dev_step(val_dataset, t)
            logging.info(f'Epoch {epoch + 1} Dev Acc {total_acc:.4f} Dev Reg {dev_reg:.4f} Dev Irreg {dev_irreg:.4f} Dev Reg Char {dev_reg_char:.4f} Dev Irreg Char {dev_irreg_char:.4f}')
            print(f'Epoch {epoch + 1} Dev Acc {total_acc:.4f} Dev Reg {dev_reg:.4f} Dev Irreg {dev_irreg:.4f} Dev Reg Char {dev_reg_char:.4f} Dev Irreg Char {dev_irreg_char:.4f}')
            dev_acc = total_acc
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_epoch = epoch + 1
                ckpt_save_path = ckpt_manager.save()
                logging.info(f'Saving best model for epoch {epoch + 1} at {ckpt_save_path}')
            if dev_irreg > best_irr_dev_acc:
                best_irr_dev_acc = dev_irreg
                best_irr_epoch = epoch +1
                ckpt_save_path = irr_ckpt_manager.save()
                logging.info(f'Saving best irreg model for epoch {epoch + 1} at {ckpt_save_path}')
            logging.info(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
            
    gen_model = Generate(args, t)
    dev_reg_acc, dev_irr_acc , df_dev_pred, _ = test_step(val_dataset, gen_model.model, t)
    test_reg_acc, test_irr_acc, df_test_pred, test_entorpy_list = test_step(test_dataset, gen_model.model,t)
    df_nonce_pred, nonce_entropy_list = nonce_step(nonce_dataset, gen_model.model, t)
    
    print('dev_reg_acc:', dev_reg_acc)
    print('dev_irr_acc:', dev_reg_acc)
    print('test_reg_acc:', test_reg_acc)
    print('test_irr_acc:', test_irr_acc)

    test_pred_list = gen_model.forward(test_dataset)
    nonce_pred_list = gen_model.forward(nonce_dataset)
    
    df_top = pd.DataFrame(test_pred_list.numpy().tolist()).transpose()
    df_top_nonce = pd.DataFrame(nonce_pred_list.numpy().tolist()).transpose()
    for key in df_top.columns:
        df_top[key] = df_top[key].apply(lambda row: t.sequences_to_texts([row])[0].split(' ')[1:])
    df_top['root'] = df_test_pred['ipa_word'].reset_index(drop=True)
    df_top['original_pred'] = df_test_pred['pred'].reset_index(drop=True)
    
    for key in df_top_nonce.columns:
        df_top_nonce[key] = df_top_nonce[key].apply(lambda row: t.sequences_to_texts([row])[0].split(' ')[1:])
        
    df_acc = pd.DataFrame([dev_reg_acc, dev_irr_acc, test_reg_acc, test_irr_acc]).T
    df_acc.columns = ['dev_reg_acc', 'dev_irr_acc', 'test_reg_acc', 'test_irr_acc']

    df_top.to_csv(os.path.join(args.model_path, 'test_top_k.csv'))
    df_top_nonce.to_csv(os.path.join(args.model_path, 'nonce_top_k.csv'))
    df_acc.to_csv(os.path.join(args.model_path, 'dev_test_acc.csv'))
    df_dev_pred.to_csv(os.path.join(args.model_path, 'dev_pred.csv'))
    df_test_pred.to_csv(os.path.join(args.model_path, 'test_pred.csv'))
    df_nonce_pred.to_csv(os.path.join(args.model_path, 'nonce_pred.csv'))

