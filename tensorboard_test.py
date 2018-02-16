from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Mnistに使うデータセットをインポートする
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data',
                                  one_hot=True)

# セッションの作成と初期化
sess = tf.InteractiveSession()

# test_layerスコープ
with tf.name_scope('test_layer'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')  # 入力するPlaceholder
    W = tf.Variable(tf.zeros([784, 10]), name='W')  # 重み
    b = tf.Variable(tf.zeros([10]), name='b')  # バイアス
    y = tf.matmul(x, W) + b  # 内積計算とバイアスの加算
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_') # 正解

# optimizerスコープ
with tf.name_scope('optimizer'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# evaluatorスコープ
with tf.name_scope('evaluator'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Tensorboardのサマリスコープ
with tf.name_scope('summary'):
    writer = tf.summary.FileWriter("./tensorboard_log", sess.graph) # ログを残すフォルダの指定とセッショングラフを可視化
    tf.summary.scalar('cross_entropy',  cross_entropy) # cross_entropyを可視化
    tf.summary.scalar('accuracy',  accuracy) # accuracyを可視化
    merged = tf.summary.merge_all()

# 学習部(1000回学習)
tf.global_variables_initializer().run() # 重みの初期化

for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs,y_: batch_ys})
    summary_str, _ = sess.run([merged, train_step], feed_dict={x: batch_xs,y_: batch_ys})
    if step % 100 == 0:
        writer.add_summary(summary_str, step)

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))