from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import cPickle
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import time

class AbstractLayer(object):
  def init_biases(self, name, n):
    values = np.zeros((n,), dtype = theano.config.floatX)
    return theano.shared(value = values, name = name)

  def init_weights(self, name, n_in, n_out, shape):
    # Y. Bengio, X. Glorot, Understanding the difficulty of
    # training deep feedforward neural networks, AISTATS 2010
    low = -np.sqrt(6. / (n_in + n_out))
    high = np.sqrt(6. / (n_in + n_out))
    values = np.asarray(np.random.uniform(low = low, high = high,
                                          size = shape),
                        dtype = theano.config.floatX)
    return theano.shared(value = values, name = name)

class ConvolutionPoolLayer(AbstractLayer):
  def __init__(self, input, f_shape, i_shape, p_shape = (2, 2)):
    n_in = np.prod(f_shape[1:])
    n_out = f_shape[0] * np.prod(f_shape[2:]) / np.prod(p_shape)
    W = self.init_weights('W', n_in, n_out, f_shape)
    b = self.init_biases('b', f_shape[0])
    f_map = conv.conv2d(input = input, filters = W,
                        filter_shape = f_shape,
                        image_shape = i_shape)
    f_map = downsample.max_pool_2d(input = f_map, ds = p_shape,
                                   ignore_border = True)
    self.params = [W, b]
    self.output = T.tanh(f_map + b.dimshuffle('x', 0, 'x', 'x'))

class FullyConnectedLayer(AbstractLayer):
  def __init__(self, input, n_in, n_out):
    W = self.init_weights('W', n_in, n_out, (n_in, n_out))
    b = self.init_biases('b', n_out)
    self.params = [W, b]
    self.output = T.tanh(theano.dot(input, W) + b)

class SoftmaxLayer(AbstractLayer):
  def __init__(self, X, n_in, n_out):
    W = self.init_weights('W', n_in, n_out)
    b = self.init_biases('b', n_out)
    self.params = [W, b]
    self.output = T.nnet.softmax(T.dot(X, W) + b)

  def init_weights(self, name, n_in, n_out):
    values = np.zeros((n_in, n_out), dtype = theano.config.floatX)
    return theano.shared(value = values, name = name)

class Lenet:
  def __init__(self, batch_sz, lr = .02):
    lr = theano.shared(np.cast[theano.config.floatX](lr))
    X = T.matrix('X')
    X_p = X.reshape((batch_sz, 3, 32, 32))
    Y = T.ivector('Y')
    # Create layers.
    f_shape, i_shape = (20, 3, 5, 5), (batch_sz, 3, 32, 32)
    layer_1 = ConvolutionPoolLayer(X_p, f_shape, i_shape)
    f_shape, i_shape = (50, 20, 5, 5), (batch_sz, 20, 14, 14)
    layer_2 = ConvolutionPoolLayer(layer_1.output, f_shape, i_shape)
    n_in, n_out =  (50 * 5 * 5), 512
    layer_3 = FullyConnectedLayer(layer_2.output.flatten(2), n_in, n_out)
    layer_4 = SoftmaxLayer(layer_3.output, n_out, 10)
    # Negative log likelihood cost function.
    cost = -T.sum(T.log(layer_4.output)[T.arange(Y.shape[0]), Y])
    # Computes the network parameter updates.
    updates = []
    for layer in [layer_1, layer_2, layer_3, layer_4]:
      for param in layer.params:
        gradient = T.grad(cost, param)
        g1_tm1 = theano.shared(np.zeros_like(param.get_value()))
        g2_tm1 = theano.shared(np.zeros_like(param.get_value()))
        g1_t = 0.95 * g1_tm1 + (1. - 0.95) * gradient
        g2_t = 0.95 * g2_tm1 + (1. - 0.95) * gradient * gradient
        rms = T.sqrt(g2_t - g1_t * g1_t + 1e-4)
        updates.append((param, param - lr * gradient / rms))
    # Backprop the error through the network.
    self.train = theano.function(inputs = [X, Y],
                                 outputs = cost,
                                 updates = updates,
                                 allow_input_downcast = True)
    # Make a prediction.
    self.predict = theano.function(inputs = [X], outputs = layer_4.output)

# Create a convolutional neural network for training.
batch_sz = 100
n_epochs = 1000
nn = Lenet(batch_sz, lr = .0001)
# Train the network on the cifar dataset.
train_errors = np.ndarray(n_epochs)
train_start = time.time()
for epoch in xrange(n_epochs):
  error = 0.
  files = ['data_batch_1', 'data_batch_2', 'data_batch_3',
           'data_batch_4', 'data_batch_5']
  for file in files:
    path = 'cifar-10-batches-py/%s' % file
    train_x, train_y = None, None
    with open(path, 'rb') as input:
      temp = cPickle.load(input)
      train_x = temp['data'].astype(theano.config.floatX) / 256
      train_y = temp['labels']
    n_batches = len(train_x) / batch_sz
    for batch in xrange(n_batches):
      start = batch * batch_sz
      end = start + batch_sz
      error += nn.train(train_x[start:end], train_y[start:end])
  train_errors[epoch] = error
  print 'Epoch [%i] Cost: %f' % (epoch, error)
train_end = time.time()
print
print 'Training Time: %g seconds' % (train_end - train_start)
print
# Plot the learning curve.
plt.plot(np.arange(n_epochs), train_errors, 'b-')
plt.xlabel('epochs')
plt.ylabel('error')
plt.show()