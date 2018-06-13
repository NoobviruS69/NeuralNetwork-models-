import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
numofclasses = 10

def weights(shape):
    weights_distribution = tf.truncated_normal(shape , stddev=0.1)
    return tf.Variable(weights_distribution)

def biases(shape):
    biases_init = tf.constant(0.1 , shape=shape)
    return tf.Variable(biases_init)

def conv2d(x , w):
    return tf.nn.conv2d(x , w , strides=[1,1,1,1] , padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x , ksize=[1,2,2,1], strides= [1,2,2,1], padding='SAME') #ksize = pool window where[batchsize , height , width , channels]

def conv_layer(inputx , shape):
    w = weights(shape)
    b = biases([shape[3]])
    return tf.nn.relu(conv2d(inputx , w) + b)

def Dense_connected(layer , size):
    input_size = int(layer.get_shape()[1])
    w = weights([input_size , size])
    b = biases([size])
    return tf.nn.relu(tf.matmul(layer , w) + b)

x = tf.placeholder(tf.float32 ,shape=[None , 784])
y = tf.placeholder(tf.float32 , shape=[None , 10])

ximage = tf.reshape(x, [-1, 28, 28, 1])

conv1 = conv_layer(ximage, shape=[5 , 5 ,1 ,32])
conv1_pool = maxPool(conv1)

conv2 = conv_layer(conv1_pool, shape=[5 , 5 ,32 ,64])
conv2_pool = maxPool(conv2)

conv_flat = tf.reshape(conv2_pool , [-1 , 7*7*64])
full_layer1 = tf.nn.relu(Dense_connected(conv_flat , 1024))

dropout_probability = tf.placeholder(tf.float32)
dropout1 = tf.nn.dropout(full_layer1 , keep_prob= dropout_probability)

ypred = Dense_connected(dropout1 ,10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y , logits=ypred))
optamizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optamizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    steps = 1000
    sess.run(init)

    for i in range(steps):
        batx , baty = mnist.train.next_batch(50)
        sess.run(train , feed_dict={x: batx , y:baty , dropout_probability: 0.5})

        if i%100 == 0:
            print('on step {}'.format(i))
            does_match = tf.equal(tf.argmax(ypred , 1) , tf.argmax(y , 1))
            acc = tf.reduce_mean(tf.cast(does_match , tf.float32))
            print(sess.run(acc ,feed_dict={x:mnist.test.images , y: mnist.test.labels , dropout_probability:1}))


