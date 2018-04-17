from process import DataPreprocessor
from mlbootstrap import Bootstrap
import tensorflow as tf

bootstrap = Bootstrap(
    'config.yaml',
    preprocessor=DataPreprocessor())
bootstrap.preprocess(force=False)

# data = bootstrap.dataset()
# max_doc_length = data['max_doc_length']
# embedding_size = 300

# with tf.name_scope('input'):
#     title = tf.placeholder(tf.int32, [None, max_doc_length], name='title')
#     significant = tf.placeholder(tf.float32, [None], name='significant')
#     target = tf.placeholder(tf.float32, [None], name='target')
#
#
# def weighted_variable(shape, name=None):
#     initializer = tf.random_uniform_initializer(-0.1, 0.1)
#     return tf.Variable(initializer(shape), name=name)
#
#
# def bias_variable(shape):
#     return tf.Variable(tf.constant(.0, shape=shape))
#
#
# with tf.name_scope('title'):
#     W = weighted_variable([data['vocab'], embedding_size])
#     title_emb = tf.reduce_sum(tf.nn.embedding_lookup(W, title), axis=1)
#
# with tf.name_scope('attention'):
#     A = weighted_variable([1, embedding_size])
#     attention = tf.reduce_sum(title_emb * A, axis=1)
#
# with tf.name_scope('score'):
#     W_fc = weighted_variable([2, 1])
#     B_fc = bias_variable([1])
#     score = tf.squeeze(tf.transpose(tf.stack([attention, significant])) @ W_fc + B_fc)
#
# with tf.name_scope('loss'):
#     l2_loss = tf.nn.l2_loss(score - target)
#     loss_op = l2_loss
#
# with tf.name_scope('train'):
#     optimizer = tf.train.AdamOptimizer()
#     train_op = optimizer.minimize(loss_op)
#
# with tf.name_scope('predict'):
#     _, indices = tf.nn.top_k(score, 3)
#     best_possible = tf.reduce_sum(tf.nn.top_k(target, 3)[0])
#     predict_op = tf.reduce_sum(tf.gather(target, indices)) / best_possible
#     _, indices = tf.nn.top_k(significant, 3)
#     baseline_op = tf.reduce_sum(tf.gather(target, indices)) / best_possible

# session = tf.Session()
# for step in range(1000):
#     session.run(tf.global_variables_initializer())
#     for batch in data['train']:
#         session.run([train_op], feed_dict={
#             title: batch['title'], significant: batch['significant'], target: batch['target']
#         })
#     if step % 50 == 0:
#         print('---------')
#         print('Epoch {}'.format(step))
#
#
#         def __print_validation(mode):
#             predict = 0
#             baseline = 0
#             for __batch in data[mode]:
#                 pred, ba = session.run([predict_op, baseline_op], feed_dict={
#                     title: __batch['title'], significant: __batch['significant'],
#                     target: __batch['target']
#                 })
#                 predict += pred
#                 baseline += ba
#             predict /= len(data['train'])
#             baseline /= len(data['train'])
#             print('{} predict: {}'.format(mode, predict))
#             print('{} baseline: {}'.format(mode, baseline))
#
#
#         __print_validation('train')
#         __print_validation('val')
#         __print_validation('test')
