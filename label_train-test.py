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

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp],
                                     training=True)
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


def num2char(tar, end, output, add):
    val_cor = []
    for line_tar, line_output in zip(tar.tolist(), output.numpy().tolist()):
        index_tar = line_tar.index(end)
        try:
            index_output = line_output.index(end)
        except:
            index_output = index_tar
        output_all = ",".join(t.sequences_to_texts([line_output[1:index_output]])[0].split(' '))
        if add == 'both':
            tar_a = "".join(t.sequences_to_texts([line_tar[3:index_tar]])[0].split(' '))
            out_a = "".join(t.sequences_to_texts([line_output[3:index_output]])[0].split(' '))
        elif add == 'no':
            tar_a = "".join(t.sequences_to_texts([line_tar[1:index_tar]])[0].split(' '))
            out_a = "".join(t.sequences_to_texts([line_output[1:index_output]])[0].split(' '))
        else:
            tar_a = "".join(t.sequences_to_texts([line_tar[2:index_tar]])[0].split(' '))
            out_a = "".join(t.sequences_to_texts([line_output[2:index_output]])[0].split(' '))
        val_cor.append([output_all, out_a, tar_a])
    return val_cor


def dev_step(val_dataset, t, add):
    start, end = t.word_index['start'], t.word_index['end']
    val_cor_all = []
    add = add
    for element in val_dataset.as_numpy_iterator():
        inp, tar = element['input_seq']
        label = element['label']
        output = tar[:, :1]
        index_tar = np.where(tar == 2)[0]
        bsz, seq_len = float(tf.shape(tar)[0]), float(tf.shape(tar)[1])
        for i in tf.range(seq_len - 1):
            pred, _ = transformer([inp, output], training=False)
            predictions = pred[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)
            output = tf.concat(values=[output, predicted_id], axis=1)
        val_cor = num2char(tar, end, output, add)
        for i in range(len(val_cor)):
            val_cor[i].append(label.tolist()[i])
        val_cor_all.append(val_cor)
    val_cor_all = list(itertools.chain(*val_cor_all))
    print('All val len:', len(val_cor_all))
    df_val_cor = pd.DataFrame(val_cor_all, columns=['pred_all', 'pred', 'tar', 'reg'])
    df_val_cor['cor'] = np.where((df_val_cor['pred'] == df_val_cor['tar']), 1, 0)
    Total_cor = df_val_cor['cor'].sum() / len(df_val_cor)
    Reg = df_val_cor[df_val_cor['reg'] == b'Reg']
    Irreg = df_val_cor[df_val_cor['reg'] == b'Irreg']
    Reg_cor = Reg['cor'].sum() / len(Reg)
    Irreg_cor = Irreg['cor'].sum() / len(Irreg)
    return Total_cor, Reg_cor, Irreg_cor, df_val_cor


def test_step(test_dataset, model, t, add):
    k = []
    start, end = 1, 2
    add = add
    for element in test_dataset.as_numpy_iterator():
        inp, tar = element['input_seq']
        label = element['label']
        output = tar[:, :1]
        index_tar = np.where(tar == 2)[0]
        entropy_list = []
        bsz, seq_len = float(tf.shape(tar)[0]), float(tf.shape(tar)[1])
        for i in tf.range(seq_len - 1):
            pred, _ = transformer([inp, output], training=False)
            predictions = pred[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)
            lprobs = tf.nn.log_softmax(predictions, axis=2)
            entropy_list.append(entropy(np.exp(lprobs.numpy()), axis=2).ravel())
            output = tf.concat(values=[output, predicted_id], axis=1)
        test_cor = num2char(tar, end, output, add)
        for i in range(len(test_cor)):
            test_cor[i].append(label.tolist()[i])
        df_test_cor = pd.DataFrame(test_cor, columns=['pred_all', 'pred', 'tar', 'reg'])
        df_test_cor['cor'] = np.where((df_test_cor['pred'] == df_test_cor['tar']), 1, 0)
        k.append(df_test_cor)
    df_test_all = pd.concat(k)
    Reg = df_test_all[df_test_all['reg'] == b'Reg']
    Irreg = df_test_all[df_test_all['reg'] == b'Irreg']
    Reg_cor = Reg['cor'].sum() / len(Reg)
    Irreg_cor = Irreg['cor'].sum() / len(Irreg)
    return Reg_cor, Irreg_cor, df_test_all, entropy_list


def nonce_step(nonce_dataset, model, t, add):
    start, end = t.word_index['start'], t.word_index['end']
    add = add
    for (batch, (inp, tar)) in enumerate(nonce_dataset):
        output = tar[:, :1]
        entropy_list = []
        bsz, seq_len = float(tf.shape(tar)[0]), float(tf.shape(tar)[1])
        for i in tf.range(seq_len - 1):
            pred, _ = transformer([inp, output], training=False)
            predictions = pred[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)
            lprobs = tf.nn.log_softmax(predictions, axis=2)
            entropy_list.append(entropy(np.exp(lprobs.numpy()), axis=2).ravel())
            output = tf.concat(values=[output, predicted_id], axis=1)
        nonce_cor = num2char(tar.numpy(), end, output, add)
    df_nonce_cor = pd.DataFrame(nonce_cor, columns=['pred_all', 'pred', 'tar'])
    return df_nonce_cor, entropy_list


class BeamSearch(nn.Module):
    def __init__(self, args, pad_id, vocab_size):
        super().__init__()
        self.vk = args.vk
        self.pad_id = pad_id
        self.vocab_size = vocab_size

    def step(self, step, lprobs, scores):
        bsz, beam_size, _ = lprobs.size()
        lprobs[:, :, self.pad_id] = -math.inf
        if step == 0:
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            assert scores is not None
            lprobs = lprobs[:, -1, :]
            scores = scores.repeat_interleave(self.vocab_size, dim=1)
            lprobs = lprobs + scores
        scores_buf, indices_buf = torch.topk(lprobs.view(bsz, -1), k=self.vk)
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
            num_layers=args.nlayers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            dff=args.dff,
            input_vocab_size=vocab_num,
            target_vocab_size=vocab_num,
            rate=args.dropout)
        if self.metric == 'irr':
            self.load_model_irr()
        else:
            self.load_model()
        self.search = BeamSearch(args, self.pad_id, self.vocab_size)

    def load_model_irr(self):
        optimizer = tf.keras.optimizers.Adam(CustomSchedule(self.args.d_model, self.args.warmup),
                                             beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        checkpoint_path = self.args.model_path
        ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=optimizer)
        irr_ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(checkpoint_path, 'irr'), max_to_keep=1)
        if irr_ckpt_manager.latest_checkpoint:
            ckpt.restore(irr_ckpt_manager.latest_checkpoint).expect_partial()
            status = ckpt.restore(irr_ckpt_manager.latest_checkpoint)
            status.expect_partial()
            print('Last irr checkpoint restored')
        else:
            assert ValueError('No checkpoint')

    def load_model(self):
        optimizer = tf.keras.optimizers.Adam(CustomSchedule(self.args.d_model, self.args.warmup),
                                             beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        checkpoint_path = self.args.model_path
        ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            status = ckpt.restore(ckpt_manager.latest_checkpoint)
            status.expect_partial()
            print('Last checkpoint restored')
        else:
            assert ValueError('No checkpoint')

    def forward(self, test_dataset):
        output_list = []
        try:
            for element in test_dataset.as_numpy_iterator():
                inp, tar = element['input_seq']
                output = tar[:, :1]
                bsz, seq_len = int(tf.shape(tar)[0]), int(tf.shape(tar)[1])
                output = self.generate(inp, [output], bsz, seq_len - 1)
                output_list.append(output)
        except:
            for (batch, (inp, tar)) in enumerate(test_dataset):
                output = tar[:, :1]
                bsz, seq_len = int(tf.shape(tar)[0]), int(tf.shape(tar)[1])
                output = self.generate(inp, [output], bsz, seq_len - 1)
                output_list.append(output)

        if len(output_list) == 1:
            return output_list[0]
        else:
            return output_list

    def get_inp_tar(self, cand_indices, cand_beams, output_list):
        cand_indices = tf.expand_dims(cand_indices, axis=2)
        prev_output = tf.transpose(tf.convert_to_tensor(output_list), [1, 0, 2])
        prev_output = tf.gather(prev_output, cand_beams, batch_dims=1)
        output = tf.transpose(tf.concat(axis=2, values=[prev_output.numpy(), cand_indices.numpy()]), [1, 0, 2])
        return output

    def generate(self, inp, output_list, bsz: int, total_step: int):
        beam_size = 5
        cand_scores = torch.zeros(int(bsz * beam_size), total_step)
        for step in range(total_step):
            for output in output_list:
                predictions, _ = self.model([inp, output], training=False)
                try:
                    lprobs
                    lprobs = tf.concat(axis=2, values=[lprobs, tf.nn.log_softmax(predictions, axis=2)])
                except:
                    lprobs = tf.nn.log_softmax(predictions, axis=2)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step, torch.from_numpy(lprobs.numpy()), cand_scores)
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
    handlers = [logging.FileHandler(log_file, mode='w+'), logging.StreamHandler()]
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M', level=logging.INFO, handlers=handlers)
    logging.info(args)

    logging.info(device_lib.list_local_devices())

    data, t = data_process.process_data(args)
    train_dataset, val_dataset, test_dataset, nonce_dataset = data
    num_batches, val_batches = len(train_dataset), len(val_dataset)

    ##settings
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    metrics = [accuracy]
    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(args.d_model, args.warmup), beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    EPOCHS = args.EPOCHS
    add = args.label_spec

    ##create model
    vocab_num = len(t.word_counts) + 1
    transformer = Transformer(
        num_layers=args.nlayers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        input_vocab_size=vocab_num,
        target_vocab_size=vocab_num,
        rate=args.dropout)
    checkpoint_path = args.model_path
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
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

            total_acc, dev_reg, dev_irreg, _ = dev_step(val_dataset, t, add)

            logging.info(f'Epoch {epoch + 1} Dev Acc {total_acc:.4f} Dev Reg {dev_reg:.4f} Dev Irreg {dev_irreg:.4f}')
            dev_acc = (dev_reg + dev_irreg) / 2
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_epoch = epoch + 1
                ckpt_save_path = ckpt_manager.save()
                logging.info(f'Saving best model for epoch {epoch + 1} at {ckpt_save_path}')
            if dev_irreg >= best_irr_dev_acc:
                best_irr_dev_acc = dev_irreg
                best_irr_epoch = epoch + 1
                ckpt_save_path = irr_ckpt_manager.save()
                logging.info(f'Saving best irreg model for epoch {epoch + 1} at {ckpt_save_path}')
            logging.info(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    gen_model = Generate(args, t, 'mean')
    dev_reg_acc, dev_irr_acc, _, _ = test_step(val_dataset, gen_model.model, t, add)
    test_reg_acc, test_irr_acc, df_test_pred, test_entropy_list = test_step(test_dataset, gen_model.model, t, add)
    df_nonce_pred, nonce_entropy_list = nonce_step(nonce_dataset, gen_model.model, t, add)

    irr_gen_model = Generate(args, t, 'irr')
    irr_dev_reg_acc, irr_dev_irr_acc, _, _ = test_step(val_dataset, irr_gen_model.model, t, add)
    irr_test_reg_acc, irr_test_irr_acc, irr_df_test_pred, irr_test_entropy_list = test_step(test_dataset,
                                                                                            irr_gen_model.model, t, add)
    irr_df_nonce_pred, irr_nonce_entropy_list = nonce_step(nonce_dataset, irr_gen_model.model, t, add)

    print('dev_reg_acc:', dev_reg_acc)
    print('dev_irr_acc:', dev_irr_acc)
    print('test_reg_acc:', test_reg_acc)
    print('test_irr_acc:', test_irr_acc)

    print('irr_dev_reg_acc:', irr_dev_reg_acc)
    print('irr_dev_irr_acc:', irr_dev_irr_acc)
    print('irr_test_reg_acc:', irr_test_reg_acc)
    print('irr_test_irr_acc:', irr_test_irr_acc)

    df_test_pred.to_csv(os.path.join(args.model_path, 'test_pred.csv'))
    df_nonce_pred.to_csv(os.path.join(args.model_path, 'nonce_pred.csv'))
    pd.DataFrame(test_entropy_list).to_csv(os.path.join(args.model_path, 'test_entropy.csv'))
    pd.DataFrame(nonce_entropy_list).to_csv(os.path.join(args.model_path, 'nonce_entropy.csv'))

    irr_df_test_pred.to_csv(os.path.join(args.model_path, 'irr/', 'test_pred.csv'))
    irr_df_nonce_pred.to_csv(os.path.join(args.model_path, 'irr/', 'nonce_pred.csv'))
    pd.DataFrame(irr_test_entropy_list).to_csv(os.path.join(args.model_path, 'irr/', 'test_entropy.csv'))
    pd.DataFrame(irr_nonce_entropy_list).to_csv(os.path.join(args.model_path, 'irr/', 'nonce_entropy.csv'))

    test_pred_list = gen_model.forward(test_dataset)
    nonce_pred_list = gen_model.forward(nonce_dataset)

    irr_test_pred_list = irr_gen_model.forward(test_dataset)
    irr_nonce_pred_list = irr_gen_model.forward(nonce_dataset)


    def clean_topk(x, label):
        try:
            end = x.index('end')
        except:
            end = len(x)
        if label == 'no':
            return "".join(x[:end])
        elif label == 'both':
            return "".join(x[2:end])
        else:
            return "".join(x[1:end])


    df_top = pd.DataFrame(test_pred_list.numpy().tolist()).transpose()
    df_top_nonce = pd.DataFrame(nonce_pred_list.numpy().tolist()).transpose()

    irr_df_top = pd.DataFrame(irr_test_pred_list.numpy().tolist()).transpose()
    irr_df_top_nonce = pd.DataFrame(irr_nonce_pred_list.numpy().tolist()).transpose()

    for key in df_top.columns:
        df_top[key] = df_top[key].apply(lambda row: t.sequences_to_texts([row])[0].split(' ')[1:])
    for i in df_top.columns:
        df_top[i] = df_top[i].apply(lambda x: clean_topk(x, add))
    df_top['target'] = df_test_pred['tar'].reset_index(drop=True)
    df_top['original_pred'] = df_test_pred['pred'].reset_index(drop=True)
    df_top['reg'] = df_test_pred['reg'].reset_index(drop=True)

    for key in irr_df_top.columns:
        irr_df_top[key] = irr_df_top[key].apply(lambda row: t.sequences_to_texts([row])[0].split(' ')[1:])
    for i in irr_df_top.columns:
        irr_df_top[i] = irr_df_top[i].apply(lambda x: clean_topk(x, add))

    irr_df_top['target'] = irr_df_test_pred['tar'].reset_index(drop=True)
    irr_df_top['original_pred'] = irr_df_test_pred['pred'].reset_index(drop=True)
    irr_df_top['reg'] = irr_df_test_pred['reg'].reset_index(drop=True)


    def topk_correct(row):
        cor = row['target']
        l = [row[col] for col in range(0, 5)]
        if cor in l:
            return 1
        else:
            return 0


    df_top['cor'] = df_top.apply(lambda row: topk_correct(row), axis=1)
    df_top['cor1'] = np.where((df_top[0] == df_top['target']), 1, 0)

    irr_df_top['cor'] = irr_df_top.apply(lambda row: topk_correct(row), axis=1)
    irr_df_top['cor1'] = np.where((irr_df_top[0] == irr_df_top['target']), 1, 0)

    test_reg_acc_topk = df_top[df_top['reg'] == b'Reg']['cor'].sum() / 60
    test_irr_acc_topk = df_top[df_top['reg'] == b'Irreg']['cor'].sum() / 20
    test_reg_acc_top1 = df_top[df_top['reg'] == b'Reg']['cor1'].sum() / 60
    test_irr_acc_top1 = df_top[df_top['reg'] == b'Irreg']['cor1'].sum() / 20

    irr_test_reg_acc_topk = irr_df_top[irr_df_top['reg'] == b'Reg']['cor'].sum() / 60
    irr_test_irr_acc_topk = irr_df_top[irr_df_top['reg'] == b'Irreg']['cor'].sum() / 20
    irr_test_reg_acc_top1 = irr_df_top[irr_df_top['reg'] == b'Reg']['cor1'].sum() / 60
    irr_test_irr_acc_top1 = irr_df_top[irr_df_top['reg'] == b'Irreg']['cor1'].sum() / 20

    print('test_reg_acc_topk', test_reg_acc_topk)
    print('test_irr_acc_tokp', test_irr_acc_topk)

    print('irr_test_reg_acc_topk', irr_test_reg_acc_topk)
    print('irr_test_irr_acc_tokp', irr_test_irr_acc_topk)

    for key in df_top_nonce.columns:
        df_top_nonce[key] = df_top_nonce[key].apply(lambda row: t.sequences_to_texts([row])[0].split(' ')[1:])
    for i in df_top_nonce.columns:
        df_top_nonce[i] = df_top_nonce[i].apply(lambda x: clean_topk(x, add))

    for key in irr_df_top_nonce.columns:
        irr_df_top_nonce[key] = irr_df_top_nonce[key].apply(lambda row: t.sequences_to_texts([row])[0].split(' ')[1:])
    for i in irr_df_top_nonce.columns:
        irr_df_top_nonce[i] = irr_df_top_nonce[i].apply(lambda x: clean_topk(x, add))

    df_acc = pd.DataFrame(
        [dev_reg_acc, dev_irr_acc, test_reg_acc, test_irr_acc, test_reg_acc_top1, test_irr_acc_top1, test_reg_acc_topk,
         test_irr_acc_topk]).T
    df_acc.columns = ['dev_reg_acc', 'dev_irr_acc', 'test_reg_acc', 'test_irr_acc', 'test_reg_acc_top1',
                      'test_reg_irr_top1', 'test_reg_acc_topk', 'test_irr_acc_topk']

    irr_df_acc = pd.DataFrame(
        [irr_dev_reg_acc, irr_dev_irr_acc, irr_test_reg_acc, irr_test_irr_acc, irr_test_reg_acc_top1,
         irr_test_irr_acc_top1, irr_test_reg_acc_topk, irr_test_irr_acc_topk]).T
    irr_df_acc.columns = ['irr_dev_reg_acc', 'irr_dev_irr_acc', 'irr_test_reg_acc', 'irr_test_irr_acc',
                          'irr_test_reg_acc_top1', 'irr_test_reg_irr_top1', 'irr_test_reg_acc_topk',
                          'irr_test_irr_acc_topk']

    df_top.to_csv(os.path.join(args.model_path, 'test_top_k.csv'))
    df_top_nonce.to_csv(os.path.join(args.model_path, 'nonce_top_k.csv'))
    df_acc.to_csv(os.path.join(args.model_path, 'dev_test_acc.csv'))
    # df_dev_pred.to_csv(os.path.join(args.model_path, 'dev_pred.csv'))

    irr_df_top.to_csv(os.path.join(args.model_path, 'irr/', 'test_top_k.csv'))
    irr_df_top_nonce.to_csv(os.path.join(args.model_path, 'irr/', 'nonce_top_k.csv'))
    irr_df_acc.to_csv(os.path.join(args.model_path, 'irr/', 'dev_test_acc.csv'))
