import os
import tensorflow as tf
import pandas as pd
import numpy as np
import collections
import config
import argparse
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer



print('Process Data')
def to_char(row,col):
  return ",".join(row[col])
  
def proc_data(args, df):
    df.replace({'vowel change + t': 'vct', 'vowel change': 'vc', 'vowel change + d': 'vcd', 'level': 'lvl', 'ruckumlaut': 'ruc', 'weak':'wk', 'other':'or'}, inplace = True)
    if args.label_spec == 'reg':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df['reg'] + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ','  + 'END'
    elif args.label_spec == 'vc':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df['verb class'] + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + 'END'
    elif args.label_spec == 'both':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' +  df['reg'] + ',' + df['verb class'] + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1)  + ','+ 'END'
    elif args.label_spec == 'no':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + 'END'
    else:
        raise ValueError ('Give me label_spec')
            
    return df
def proc_data_nonce(args, df):
    if args.label_spec == 'reg':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + 'reg' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) +  ',' + 'END'
    elif args.label_spec == 'vc':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + '-d' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) +  ',' + 'END'
    elif args.label_spec == 'both':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + 'reg' + ',' + '-d' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + 'END'
    elif args.label_spec == 'no':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + 'END'
    else:
        raise ValueError ('Give me label_spec')
            
    return df

def partition (list_in, seed, n):
    random.Random(seed).shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def tokenize(args):
    df_train = pd.read_csv(os.path.join(args.data_path_train), index_col = 0)
    df_test = pd.read_csv(os.path.join(args.data_path_test),index_col = 0)
    df_nonce = pd.read_csv(os.path.join(args.data_path_nonce), index_col = 0)
    df_train = proc_data(args, df_train)
    df_test = proc_data(args, df_test)
    df_nonce = proc_data_nonce(args, df_nonce)
    df_train['label'] = df_train['reg']
    df_test['label'] = df_test['reg']
    t = Tokenizer(split = ',', filters = '!')
    t.fit_on_texts(list(df_train['verb'].values) + list(df_test['verb'].values))
    total_words = len(t.word_counts) + 1
    MAX_LEN = max(len(i.split(',')) for i in df_train['verb'].to_list() + df_test['verb'].to_list())
    return df_train, df_test, df_nonce, t, MAX_LEN

    
def process_data(args, df_train, df_test, df_nonce, t, MAX_LEN):
    BUFFER_SIZE = 60
    BATCH_SIZE = args.BATCH_SIZE
    seed = args.seed
    EPOCHS = args.EPOCHS
    if args.ratio == 'token_parent':
        ratio = 0.377
    elif args.ratio == 'token_reg':
        ratio = 2.195
    else:
        ratio = 1
    
    root_seq_test = t.texts_to_sequences(df_test['root'].values)
    verb_seq_test = t.texts_to_sequences(df_test['verb'].values)
    root_seq_nonce = t.texts_to_sequences(df_nonce['root'].values)
    verb_seq_nonce = t.texts_to_sequences(df_nonce['verb'].values)
    root_pad_test = pad_sequences(root_seq_test, maxlen=MAX_LEN, padding = 'post')
    verb_pad_test = pad_sequences(verb_seq_test, maxlen=MAX_LEN, padding = 'post')
    root_pad_nonce = pad_sequences(root_seq_nonce, maxlen = MAX_LEN, padding = 'post')
    verb_pad_nonce = pad_sequences(verb_seq_nonce, maxlen = MAX_LEN, padding = 'post')
    test_examples = tf.data.Dataset.from_tensor_slices(
        {"input_seq":(tf.cast(np.array(root_pad_test), tf.int64),
        tf.cast(np.array(verb_pad_test), tf.int64)),"label":df_test['label'].values})
    nonce_examples = tf.data.Dataset.from_tensor_slices((tf.cast(np.array(root_pad_nonce), tf.int64), 
                                                       tf.cast(np.array(verb_pad_nonce), tf.int64)))
                                                    
    test_dataset = test_examples.padded_batch(len(test_examples))
    nonce_dataset = nonce_examples.padded_batch(len(nonce_examples))
    
    df_train_reg = df_train[df_train['reg']=='Reg']
    df_train_irreg = df_train[df_train['reg']=='Irreg']
    train_irreg, val_irreg = train_test_split(
        df_train_irreg, test_size = 0.2,
        stratify=df_train_irreg[['verb class']], random_state=seed)
    train_reg, val_reg = train_test_split(
        df_train_reg, test_size = 0.01, 
        stratify = df_train_reg[['verb class']], random_state = seed)
    
    length = len(train_irreg)
    train_list = []
    for epoch in range(EPOCHS):
        train_reg_part = train_reg.sample(round(length*ratio), random_state = seed+epoch, replace = True)
        train_list.append(pd.concat([train_irreg, train_reg_part]))

    epoch_train_dataset = []
    for train_data in train_list:
        train_data = train_data.sample(frac = 1, random_state = seed)
        root_seq_train = t.texts_to_sequences(train_data['root'].values)
        verb_seq_train = t.texts_to_sequences(train_data['verb'].values)
        root_pad_train = pad_sequences(root_seq_train, maxlen = MAX_LEN, padding = 'post')
        verb_pad_train = pad_sequences(verb_seq_train, maxlen = MAX_LEN, padding = 'post')
        train_examples = tf.data.Dataset.from_tensor_slices(
            {"input_seq":(tf.cast(np.asarray(root_pad_train), tf.int64), 
            tf.cast(np.asarray(verb_pad_train), tf.int64)),
            "label": train_data['reg'].values})
        train_dataset = train_examples.cache()
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(BATCH_SIZE)
        epoch_train_dataset.append(train_dataset)    
        
        
    val_data = pd.concat([val_reg, val_irreg])
    root_seq_val = t.texts_to_sequences(val_data['root'].values)
    verb_seq_val = t.texts_to_sequences(val_data['verb'].values)
    root_pad_val = pad_sequences(root_seq_val, maxlen=20, padding = 'post')
    verb_pad_val = pad_sequences(verb_seq_val, maxlen=20, padding = 'post')
    val_examples = tf.data.Dataset.from_tensor_slices(
        {"input_seq":(tf.cast(np.array(root_pad_val), tf.int64), 
                    tf.cast(np.array(verb_pad_val), tf.int64)), 
         "label": val_data['reg'].values})
         
    val_dataset = val_examples.padded_batch(len(val_examples))     
    
    return epoch_train_dataset, val_dataset, test_dataset, nonce_dataset
    

    