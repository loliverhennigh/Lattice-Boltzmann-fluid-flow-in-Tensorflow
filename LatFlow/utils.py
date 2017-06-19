
import tensorflow as tf
import numpy as np

def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=tf.float32)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  print(x.get_shape())
  print(k.get_shape())
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='VALID')
  return y

def pad_mobius(f):
  f_mobius = tf.concat(axis=1, values=[f[:,-1:], f, f[:,0:1]]) 
  f_mobius = tf.concat(axis=2, values=[f_mobius[:,:,-1:], f_mobius, f_mobius[:,:,0:1]])
  return f_mobius
 
