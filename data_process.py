import os
import tensorflow as tf
import pandas as pd
import numpy as np
import collections
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

def proc_data(args, df):
  if args.add_label:
    if args.label_spec == 'reg':
      df['root'] = '<' + df['ipa_root'] + '>' + df['reg'] + '$'
      df['verb'] = '<' + df['ipa_word'] + '>' + df['reg'] + '$'
      df['label'] = df['reg']
    elif args.label_spec == 'vc':
      df['root'] = '<' + df['ipa_root'] + '>' + df['verb class'] + '$'
      df['verb'] = '<' + df['ipa_word'] + '>' + df['verb class'] + '$'
      df['label'] = df['reg']
    elif args.label_spec == 'both':
      df['root'] = '<' + df['ipa_root'] + '>' + df['reg'] + df['verb class'] + '$'
      df['verb'] = '<' + df['ipa_word'] + '>' + df['reg'] + df['verb class'] + '$'
      df['label'] = df['reg']
    else:
      raise ValueError('Label Select Error')
  else:
    df['root'] = '<' + df['ipa_root'] + '>' 
    df['verb'] = '<' + df['ipa_word'] + '>' 
    df['label'] = df['reg']
  return df

def process_data(args):
  df_train = pd.read_csv(os.path.join(args.data_path_train), index_col = 0)
  df_test = pd.read_csv(os.path.join(args.data_path_test),index_col = 0)
  #df_nonce = pd.read_csv(os.path.join(args.data_path_nonce), index_col = 0)
  df_train = proc_df(args, df_train)
  df_test = proc_df(args, df_test)
  root_train, root_val, label_train, label_val, verb_train, verb_val = train_test_split(
      dftrain['ipa_root'].values, dftrain['reg'].values, dftrain['ipa_verb'].values, test_size = 0.1, stratify=dftrain[['reg']], random_state=args.seed)
  t = Tokenizer(char_level=True)
  t.fit_on_texts(root_train + verb_train)
  total_words = len(t.word_counts)
  t.num_words = total_words +2
  root_seq_train = t.texts_to_sequences(root_train)
  root_seq_val = t.texts_to_sequences(root_val)
  verb_seq_train = t.texts_to_sequences(verb_train)
  verb_seq_val = t.texts_to_sequences(verb_val)
  verb_seq_test = t.texts_to_sequences(dftest['ipa_verb'].values)
  root_seq_test = t.texts_to_sequences(dftest['ipa_root'].values)
  MAX_LEN = max(len(i) for i in root_seq_train + root_seq_val + verb_seq_train+verb_seq_val+verb_seq_test)
  root_pad_train = pad_sequences(root_seq_train, maxlen=MAX_LEN, padding = 'post')
  root_pad_val = pad_sequences(root_seq_val, maxlen=MAX_LEN, padding = 'post')
  verb_pad_train = pad_sequences(verb_seq_train, maxlen=MAX_LEN, padding = 'post')
  verb_pad_val = pad_sequences(verb_seq_val, maxlen=MAX_LEN, padding = 'post')
  root_pad_test = pad_sequences(root_seq_test, maxlen=MAX_LEN, padding = 'post')
  verb_pad_test = pad_sequences(verb_seq_test, maxlen=MAX_LEN, padding = 'post')
  train_examples = tf.data.Dataset.from_tensor_slices({"input_seq":(tf.cast(np.asarray(root_pad_train), tf.int64), 
                                                         tf.cast(np.asarray(verb_pad_train), tf.int64)), 
                                                     "label": label_train})
  val_examples = tf.data.Dataset.from_tensor_slices({"input_seq":(tf.cast(np.array(root_pad_val), tf.int64), 
                                                       tf.cast(np.array(verb_pad_val), tf.int64)), 
                                                   "label": label_val})
  test_examples = tf.data.Dataset.from_tensor_slices({"input_seq":(tf.cast(np.array(root_pad_test), tf.int64), 
                                                       tf.cast(np.array(verb_pad_test), tf.int64)),
                                                    "label":dftest['reg'].values})
  BUFFER_SIZE = 60
  BATCH_SIZE = 32
  train_dataset = train_examples.cache()
  train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  train_dataset = train_dataset.prefetch(BATCH_SIZE)
  val_dataset = val_examples.padded_batch(len(val_dataset))
  test_dataset = test_examples.padded_batch(len(dftest))

  print(dftrain.head())
  print(next(iter(train_dataset)))
  print(next(iter(val_dataset)))
  print(next(iter(test_dataset)))
  return (train_dataset, val_dataset, test_dataset), t
