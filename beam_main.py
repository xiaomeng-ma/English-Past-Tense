from tensorflow.python.client import device_lib
import os
import sys
import time
import logging
import data_process
import config
import pandas as pd
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
    #if np.isnan(loss.numpy()) == False:
      #loss_list.append(loss.numpy())
    #else:
      #nan_string.append([inp, tar])
    #if np.isnan(loss.numpy())==True:
      #print("TAR_INP:", tar_inp)
      #print("TAR_Real:", tar_real)
      #print(predictions)
      
  gradients = tape.gradient(loss, transformer.trainable_variables) 
  #gradients = [(tf.clip_by_value(grad,-1,1)) for grad in gradients]
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))
  #print(accuracy_function(tar_real, predictions))

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

def dev_step(val_dataset):
  k = []
  start, end = 1, 2
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
      index_tar = line_tar.index(2)
      try:
        index_output = line_output.index(2)
      except:
        index_output = index_tar
      output_a = "".join(t.sequences_to_texts([line_output[1:index_output]])[0].split(' '))
      tar_a = "".join(t.sequences_to_texts([line_tar[1:index_tar]])[0].split(' '))
      val_cor.append([output_a, tar_a, lab])
    df_val_cor = pd.DataFrame(val_cor, columns = ['pred', 'ipa_word', 'reg'])
    df_val_cor['cor'] = df_val_cor.apply(lambda row: correct(row), axis=1)
    df_val_cor['tar_len'] = [len(i) for i in df_val_cor['ipa_word']]
    df_val_cor['pred_len'] = [len(i) for i in df_val_cor['pred']]
    df_val_cor['char_cor'] = df_val_cor.apply(lambda row:char_cor(row), axis =1)
    Reg = df_val_cor[df_val_cor['reg']==b'Reg']
    Irreg = df_val_cor[df_val_cor['reg']==b'Irreg']
    Reg_cor = Reg['cor'].sum()/len(Reg)
    Irreg_cor = Irreg['cor'].sum()/len(Irreg)
    Reg_char_cor = Reg['char_cor'].sum()/Reg['tar_len'].sum()
    Irreg_char_cor = Irreg['char_cor'].sum()/Irreg['tar_len'].sum()
    k.append(df_val_cor)
  return Reg_cor, Irreg_cor,Reg_char_cor, Irreg_char_cor, k

if __name__ == "__main__":
  

  
