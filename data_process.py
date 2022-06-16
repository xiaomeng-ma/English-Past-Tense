import os
import tensorflow as tf
import pandas as pd
import numpy as np
import collections
import config
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer



print('Process Data')
def to_char(row,col):
  return ",".join(row[col])
  
def proc_data(args, df):
    if args.label_spec == 'reg':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + df['reg'] + ',' + 'END'
    elif args.label_spec == 'vc':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + df['verb class'] + ',' + 'END'
    elif args.label_spec == 'both':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + df['reg'] + ',' + df['verb class'] + ',' + 'END'
    elif args.label_spec == 'no':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + 'END'
    else:
        raise ValueError ('Give me label_spec')
            
    return df
def proc_data_nonce(args, df):
    if args.label_spec == 'reg':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + 'reg' + ',' + 'END'
    elif args.label_spec == 'vc':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + '-d' + ',' + 'END'
    elif args.label_spec == 'both':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + 'reg' + ',' + '-d' + ',' + 'END'
    elif args.label_spec == 'no':
        df['root'] = 'START' + ',' + df.apply(lambda row:to_char(row, 'ipa_root'), axis = 1) + ',' + 'END'
        df['verb'] = 'START' + ',' + df.apply(lambda row: to_char(row, 'ipa_word'), axis = 1) + ',' + 'END'
    else:
        raise ValueError ('Give me label_spec')
            
    return df
def process_data(args):
    df_train = pd.read_csv(os.path.join(args.data_path_train), index_col = 0)
    df_test = pd.read_csv(os.path.join(args.data_path_test),index_col = 0)
    df_nonce = pd.read_csv(os.path.join(args.data_path_nonce), index_col = 0)
    df_train = proc_data(args, df_train)
    df_test = proc_data(args, df_test)
    df_nonce = proc_data_nonce(args, df_nonce)
    df_train['label'] = df_train['reg']
    df_test['label'] = df_test['reg']
    root_train, root_val, label_train, label_val, verb_train, verb_val = train_test_split(
      df_train['root'].values, df_train['label'].values, df_train['verb'].values, 
      test_size = 0.1, stratify=df_train[['label']], random_state=args.seed)
    t = Tokenizer(split = ',', filters = '!')
    t.fit_on_texts(verb_train)
    total_words = len(t.word_counts) + 1
    root_seq_train = t.texts_to_sequences(root_train)
    root_seq_val = t.texts_to_sequences(root_val)
    verb_seq_train = t.texts_to_sequences(verb_train)
    verb_seq_val = t.texts_to_sequences(verb_val)
    verb_seq_test = t.texts_to_sequences(df_test['verb'].values)
    root_seq_test = t.texts_to_sequences(df_test['root'].values)
    root_seq_nonce = t.texts_to_sequences(df_nonce['root'].values)
    verb_seq_nonce = t.texts_to_sequences(df_nonce['verb'].values)
    MAX_LEN = max(len(i) for i in root_seq_train + root_seq_val + verb_seq_train + verb_seq_val + verb_seq_test)
    root_pad_train = pad_sequences(root_seq_train, maxlen=MAX_LEN, padding = 'post')
    root_pad_val = pad_sequences(root_seq_val, maxlen=MAX_LEN, padding = 'post')
    verb_pad_train = pad_sequences(verb_seq_train, maxlen=MAX_LEN, padding = 'post')
    verb_pad_val = pad_sequences(verb_seq_val, maxlen=MAX_LEN, padding = 'post')
    root_pad_test = pad_sequences(root_seq_test, maxlen=MAX_LEN, padding = 'post')
    verb_pad_test = pad_sequences(verb_seq_test, maxlen=MAX_LEN, padding = 'post')
    root_pad_nonce = pad_sequences(root_seq_nonce, maxlen = MAX_LEN, padding = 'post')
    verb_pad_nonce = pad_sequences(verb_seq_nonce, maxlen = MAX_LEN, padding = 'post')
 
    train_examples = tf.data.Dataset.from_tensor_slices({"input_seq":(
                                                         tf.cast(np.asarray(root_pad_train), tf.int64), 
                                                         tf.cast(np.asarray(verb_pad_train), tf.int64)
                                                         ), 
                                                     "label": label_train})
    val_examples = tf.data.Dataset.from_tensor_slices({"input_seq":(tf.cast(np.array(root_pad_val), tf.int64), 
                                                       tf.cast(np.array(verb_pad_val), tf.int64)), 
                                                   "label": label_val})
    test_examples = tf.data.Dataset.from_tensor_slices({"input_seq":(tf.cast(np.array(root_pad_test), tf.int64), 
                                                       tf.cast(np.array(verb_pad_test), tf.int64)),
                                                    "label":df_test['label'].values})
    nonce_examples = tf.data.Dataset.from_tensor_slices((tf.cast(np.array(root_pad_nonce), tf.int64), 
                                                       tf.cast(np.array(verb_pad_nonce), tf.int64)))
                                                    
    BUFFER_SIZE = 60
    BATCH_SIZE = args.BATCH_SIZE
    train_dataset = train_examples.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(BATCH_SIZE)
    val_dataset = val_examples.padded_batch(BATCH_SIZE)
    test_dataset = test_examples.padded_batch(len(test_examples))
    nonce_dataset = nonce_examples.padded_batch(len(nonce_examples))
    return (train_dataset, val_dataset, test_dataset, nonce_dataset), t
    
