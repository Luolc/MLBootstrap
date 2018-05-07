from process import DataPreprocessor
from mlbootstrap import Bootstrap
import tensorflow as tf

bootstrap = Bootstrap(
    'config.yaml',
    preprocessor=DataPreprocessor())
bootstrap.preprocess(force=False)

data = bootstrap.dataset()
max_doc_length = data['max_doc_length']
embedding_size = 64
max_idx = 100

with tf.name_scope('input'):
    title = tf.placeholder(tf.int32, [None, max_doc_length], name='title')
    significant = tf.placeholder(tf.float32, [None], name='significant')
    insight_type = tf.placeholder(tf.int32, [None], name='insight_type')
    cells = tf.placeholder(tf.float32, [None, 3], name='cells')
    index = tf.placeholder(tf.int32, [None], name='index')
    target = tf.placeholder(tf.float32, [None], name='target')


def weighted_variable(shape, name=None):
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    return tf.Variable(initializer(shape), name=name)


def bias_variable(shape):
    return tf.Variable(tf.constant(.0, shape=shape))


with tf.name_scope('title'):
    W_w = weighted_variable([data['vocab_size'], embedding_size])
    title_emb = tf.reduce_sum(tf.nn.embedding_lookup(W_w, title), axis=1)

with tf.name_scope('type'):
    W_type = weighted_variable([2, embedding_size])
    type_emb = tf.nn.embedding_lookup(W_type, insight_type)

with tf.name_scope('index'):
    W_idx = weighted_variable([max_idx, embedding_size])
    idx_emb = tf.nn.embedding_lookup(W_idx, index)

with tf.name_scope('conv'):
    kernel = weighted_variable([3, 16])
    W_c = weighted_variable([16, embedding_size])
    conv1 = tf.nn.leaky_relu(cells @ kernel) @ W_c

with tf.name_scope('significant'):
    W_s = weighted_variable([1, embedding_size])
    b_s = bias_variable([embedding_size])
    sig_den = tf.reshape(significant, [-1, 1]) @ W_s + b_s

with tf.name_scope('mem'):
    feature_den = title_emb + type_emb + idx_emb + conv1 + sig_den
    attn = title_emb @ tf.transpose(title_emb)
    attn_ex = tf.expand_dims(attn, axis=1)
    feature_ex = tf.expand_dims(feature_den, axis=-1)
    mem_output = tf.transpose(tf.reduce_sum(attn_ex * feature_ex, axis=0))

with tf.name_scope('score'):
    W_fc = weighted_variable([embedding_size])
    b_fc = bias_variable([1])
    score = tf.reduce_sum(mem_output * W_fc, axis=1) + b_fc

with tf.name_scope('loss'):
    l2_loss = tf.nn.l2_loss(score - target)
    loss_op = l2_loss

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss_op)

with tf.name_scope('predict'):
    _, indices = tf.nn.top_k(score, 3)
    best_possible = tf.reduce_sum(tf.nn.top_k(target, 3)[0])
    predict_op = tf.reduce_sum(tf.gather(target, indices)) / best_possible
    _, indices = tf.nn.top_k(significant, 3)
    baseline_op = tf.reduce_sum(tf.gather(target, indices)) / best_possible

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())

for step in range(10000):
    batch = data['train'][step % len(data['train'])]
    session.run([train_op], feed_dict={
        title: batch['title'],
        significant: batch['significant'],
        insight_type: batch['type'],
        cells: batch['cells'],
        index: batch['index'],
        target: batch['target']
    })
    if step % 100 == 0:
        print('---------')
        print('Epoch {}'.format(step))


        def __print_validation(mode):
            predict = 0
            baseline = 0
            for __batch in data[mode]:
                pred, ba = session.run([predict_op, baseline_op], feed_dict={
                    title: __batch['title'],
                    significant: __batch['significant'],
                    insight_type: __batch['type'],
                    cells: __batch['cells'],
                    index: __batch['index'],
                    target: __batch['target']
                })
                predict += pred
                baseline += ba
            predict /= len(data['train'])
            baseline /= len(data['train'])
            print('{} predict: {}'.format(mode, predict))
            print('{} baseline: {}'.format(mode, baseline))


        __print_validation('train')
        __print_validation('val')
        __print_validation('test')
