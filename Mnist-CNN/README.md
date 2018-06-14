# MNIST-Convolutional-NN
**************

This is a CNN model working on the MNIST digit classifiction dataset.But the model is not just confined to this dataset but in order to use your own images, the MNIST helper fuctions have to be self created.the fact that i am using a basic dataset like MNIST is purely due to ease of showing the concept of CNN through code.

## About MNIST:
* there are 55,000 traing images , 10,000 test images and 5000 validation images
* The images are 28x28 pixels with values between 0-1 as pixel densities.

## Library Imports:
```
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
```
## MNIST:
```
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
```

## Weights and Biases initialization:
The weight function returns variables in the form of a truncated_normal array of weights with a shape and  deviation of 0.1.
similarly the biases return a array of constant array with a shape.
```
def weights(shape):
    weights_distribution = tf.truncated_normal(shape , stddev=0.1)
    return tf.Variable(weights_distribution)

def biases(shape):
    biases_init = tf.constant(0.1 , shape=shape)
    return tf.Variable(biases_init)
```
## Convolution and pooling functions:
```
def conv2d(x , w):
    return tf.nn.conv2d(x , w , strides=[1,1,1,1] , padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x , ksize=[1,2,2,1], strides= [1,2,2,1], padding='SAME')
```
## Convolutional layer function:
```
def conv_layer(inputx , shape):
    w = weights(shape)
    b = biases([shape[3]])
    return tf.nn.relu(conv2d(inputx , w) + b)
```
Here we pass in the shape value to the weights to create them. the shape value will contain the [convh , convw , channels , no. of filters/bias shape].and then we return the rectified linear output of the conv2d of all the values.hence making code like this
```
conv1 = conv2d(x, weights['wc1'], biases['wc1'])
conv1 = maxpool(conv1)
conv1 = tf.nn.relu(conv1)

into 

conv1 = conv_layer(ximage, shape=[5 , 5 ,1 ,32])
conv1_pool = maxPool(conv1)
```
## CONV and Fully connected Model:
```
ximage = tf.reshape(x, [-1, 28, 28, 1])

conv1 = conv_layer(ximage, shape=[5 , 5 ,1 ,32])
conv1_pool = maxPool(conv1)

conv2 = conv_layer(conv1_pool, shape=[5 , 5 ,32 ,64])
conv2_pool = maxPool(conv2)

conv_flat = tf.reshape(conv2_pool , [-1 , 7*7*64])
full_layer1 = tf.nn.relu(Dense_connected(conv_flat , 1024))
```
## PlaceHolders and dropout:
```
x = tf.placeholder(tf.float32 ,shape=[None , 784])
y = tf.placeholder(tf.float32 , shape=[None , 10])

dropout_probability = tf.placeholder(tf.float32)
dropout1 = tf.nn.dropout(full_layer1 , keep_prob= dropout_probability)

ypred = Dense_connected(dropout1 ,10)
```
## cost , optamize and train:
```
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y , logits=ypred))
optamizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optamizer.minimize(cross_entropy)
```
## create and run session:
```
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
```

## License
copyright © N00bVrus69








