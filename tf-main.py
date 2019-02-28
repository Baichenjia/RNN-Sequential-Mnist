# -*- coding: utf-8 -*- 

import tensorflow as tf 
import numpy as np
import os
import tensorflow.contrib.eager as tfe 
from tensorflow.keras.datasets import mnist

tfe.enable_eager_execution()

# 导入数据 shape分别为  (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# 将第二个维度作为序列，第三个维度作为输入。相当于每个时间步输入图像的一行
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = (train_data / 255.).astype(np.float32)
test_data = (test_data / 255.).astype(np.float32)
train_labels, test_labels = train_labels.astype(np.int), test_labels.astype(np.int)

class RNNModel(tf.keras.Model):
    def __init__(self, cell_size=50, dense_size=100, num_classes=10):
        super(RNNModel, self).__init__()
        self.cell_size = cell_size
        self.dense_size = dense_size
        self.num_classes = num_classes

        # dense
        self.dense_layer = tf.keras.layers.Dense(dense_size, activation=tf.nn.relu)
        self.pred_layer = tf.keras.layers.Dense(num_classes, activation=None)
        
        # lstm
        self.rnn_cell = tf.nn.rnn_cell.LSTMCell(cell_size)


    def predict(self, X, seq_length, is_training):
        """
        """
        num_samples = tf.shape(X)[0]   # X.shape=(128, 28, 28)
        X = tf.unstack(X, axis=1)      # 将第二个维度作为序列，第三个维度作为输入。相当于每个时间步输入图像的一行

        # LSTM: 此处仅需要保留最后一个单元的输出用于预测，只在最后一个时间步计算损失
        state = self.rnn_cell.zero_state(num_samples, dtype=tf.float32)
        for input_step in X:
            output, state = self.rnn_cell(input_step, state)
        dropped_outputs = tf.layers.dropout(output, rate=0.3, training=is_training)

        # Predict
        dense = self.dense_layer(dropped_outputs)
        dense_drop = tf.layers.dropout(dense, rate=0.3, training=is_training)
        logits = self.pred_layer(dense_drop)
        return logits


    def loss_fn(self, X, y, seq_length, is_training):
        """"""
        preds = self.predict(X, seq_length, is_training)   # (batch,10)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=preds)
        return loss


    def grads_fn(self, X, y, seq_length, is_training):
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(X, y, seq_length, is_training)
        return loss, tape.gradient(loss, self.variables)


    def acc_fn(self, X, y, seq_length, is_training):
        preds = self.predict(X, seq_length, is_training).numpy()
        acc = np.sum(np.argmax(preds, axis=1) == y.numpy(), dtype=np.float32) / X.numpy().shape[0]
        return acc 


def train(model, dataset, test_data, test_labels, 
          checkpoint, checkpoint_prefix, optimizer, epoches=10):
    test_data, test_labels = tf.convert_to_tensor(test_data), tf.convert_to_tensor(test_labels)
    # train
    for epoch in range(epoches):
        losses = []
        for (batch, (inp, targ)) in enumerate(dataset):
            loss, gradients = model.grads_fn(inp, targ, seq_length=28, is_training=True)
            grad, _ = tf.clip_by_global_norm(gradients, 1.)      # clip梯度
            optimizer.apply_gradients(zip(grad, model.variables))
            losses.append(loss)
        print("Epoch :", epoch, ", train loss :", np.mean(losses))

        if epoch % 2 == 0:
            acc = model.acc_fn(test_data, test_labels, seq_length=28, is_training=False)
            print("**Epoch :", epoch, ", valid acc:", acc*100, "%")
            checkpoint.save(file_prefix=checkpoint_prefix)

# model 
learning_rate = tf.Variable(1e-4, name="learning_rate")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
model = RNNModel()

# dataset
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(60000)
dataset = dataset.batch(128, drop_remainder=True)

# checkpoint
checkpoint_dir = 'checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, learning_rate=learning_rate, model=model)

# train
train(model, dataset, test_data, test_labels, checkpoint, checkpoint_prefix, optimizer, epoches=30)

