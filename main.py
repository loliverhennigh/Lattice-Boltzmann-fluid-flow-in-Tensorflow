
import time

import numpy as np
import tensorflow as tf
import math 
import cv2

import matplotlib.pyplot as plt 

# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

success = video.open('video.mov', fourcc, 60, (1024, 256), True)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('test', True,
                           """will save the state of the system for testing purposes""")

def make_car_boundary(car_name="cars/car_001.png", shape=[256,1024], car_shape=(256,128)):
  img = cv2.imread(car_name, 0)
  img = cv2.flip(img, 1)
  resized_img = cv2.resize(img, car_shape)
  resized_img = -(resized_img/255.0).astype(int).astype(np.float32) + 1.0
  resized_img = resized_img.reshape([1, car_shape[1], car_shape[0], 1])
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:, 64:64+car_shape[1], 64:64+car_shape[0], :] = resized_img
  return boundary

def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=tf.float32)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  #if k.get_shape()[0] == 3:
  #  x = tf.pad(
  #      x, [[0, 0], [1, 1], [1, 1],
  #      [0, 0]], "REFLECT")
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='VALID')
  return y

def transfer():
  transfer_kernel = np.zeros((3,3,9,9))

  # how to transfer states  
  transfer_kernel[1,0,0,0] = 1.0
  transfer_kernel[2,0,1,1] = 1.0
  transfer_kernel[2,1,2,2] = 1.0
  transfer_kernel[2,2,3,3] = 1.0
  transfer_kernel[1,2,4,4] = 1.0
  transfer_kernel[0,2,5,5] = 1.0
  transfer_kernel[0,1,6,6] = 1.0
  transfer_kernel[0,0,7,7] = 1.0
  # center
  transfer_kernel[1,1,8,8] = 1.0
  return transfer_kernel

def run():
  # 2D Lattice Boltzmann (BGK) model of a fluid.
  #  c4  c3   c2  D2Q9 model. At each timestep, particle densities propagate
  #    \  |  /    outwards in the directions indicated in the figure. An
  #  c5 -c9 - c1  equivalent 'equilibrium' density is found, and the densities
  #    /  |  \    relax towards that state, in a proportion governed by omega.
  #  c6  c7   c8      Iain Haslam, March 2006.
  # fig taken from http://exolete.com/lbm/

  # constants
  omega = 1.85
  density = 0.08
  t1 = 4.0/9 
  t2 = 1.0/9 
  t3 = 1.0/36
  c_squ = 1.0/3
  # size of grid
  nx = 256.0
  ny = 1024.0
  msize=nx*ny
  # make grid (F_i) 
  #F_i = np.zeros((1, int(nx), int(ny), 9), dtype=np.float32)
  #F_i = F_i + density/9
  #F_i[0, 20, 10, :] = 10.0
  #F_i = tf.Variable(F_i)
  #FEQ = tf.placeholder(tf.float32, shape=(1, int(nx), int(ny),9))
  # generated bounds 
  #bound = (np.random.rand(1, int(nx), int(ny), 1) > 1.0).astype(int).astype(np.float32)
  #bound[0,50:80,50:70,0] = 1.0
  bound = make_car_boundary()
  bound_display = bound[0,:,:,0]
  # remove bound on incoming fluid
  bound[0,:,0,0] = 0.0 
  # make tensors to kill bounds
  bound_inv = -(bound-1.0) 
  bound_9 = np.concatenate([bound, bound, bound, bound, bound, bound, bound, bound, np.zeros((1, int(nx), int(ny), 1))], axis=3)
  bound_9_inv = np.concatenate([bound_inv, bound_inv, bound_inv, bound_inv, bound_inv, bound_inv, bound_inv, bound_inv, np.zeros((1, int(nx), int(ny), 1)) + 1.0], axis=3)
  bound_9 = tf.constant(bound_9, dtype=tf.float32)
  bound_9_inv = tf.constant(bound_9_inv, dtype=tf.float32)
  bound = tf.constant(bound, dtype=tf.float32)

  # make grid (F_i) 
  F_i = np.zeros((1, int(nx), int(ny), 9), dtype=np.float32)
  F_i = F_i + density/9
  #F_i = F_i * bound_inv
  #F_i[0, 20, 10, :] = 10.0
  F_i = tf.Variable(F_i)
  FEQ = tf.placeholder(tf.float32, shape=(1, int(nx), int(ny),9))

  # bound kernel (this will flip the state of bound states ie. c4 goes to c8)
  bound_kernel = np.array([[ 0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0],
                           [ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                           [ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                           [ 0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0]])
  bound_kernel = np.expand_dims(bound_kernel, axis=0)
  bound_kernel = np.expand_dims(bound_kernel, axis=0)
  bound_kernel = tf.constant(bound_kernel, dtype=tf.float32)

  # propagate_kernel (3 by 3 conv filter that acts to propagate fluid)
  propagate_kernel = transfer()
  propagate_kernel = tf.constant(propagate_kernel, dtype=tf.float32)

  # x vel kernel (1 by 1 conv filter that calcs the x velocity)
  ux_kernel = np.array([ 1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0,  1.0,  0.0])
  ux_kernel = np.expand_dims(ux_kernel, axis=1)
  ux_kernel = np.expand_dims(ux_kernel, axis=0)
  ux_kernel = np.expand_dims(ux_kernel, axis=0)
  ux_kernel = tf.constant(ux_kernel, dtype=tf.float32)

  # y vel kernel (1 by 1 conv filter that calcs the y velocity)
  uy_kernel = np.array([ 0.0,  1.0,  1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0])
  uy_kernel = np.expand_dims(uy_kernel, axis=1)
  uy_kernel = np.expand_dims(uy_kernel, axis=0)
  uy_kernel = np.expand_dims(uy_kernel, axis=0)
  uy_kernel = tf.constant(uy_kernel, dtype=tf.float32)

  # make delta ux (add this tensor to add to the y velocity in the middle region)
  delta = 3e-3
  #delta = 0.0
  delta_ux = np.zeros((1, int(nx), int(ny), 1))
  delta_ux[0,:,0,0] = delta 
  #delta_ux_display = (delta_ux * (1/delta))[0,:,:,0]
  delta_ux_display = delta_ux[0,:,:,0]
  delta_ux = tf.constant(delta_ux, dtype=tf.float32)

  # now define the compution
  # make F_i mobius
  F_i_mobius = F_i
  F_i_mobius = tf.concat(axis=1, values=[F_i_mobius, F_i_mobius[:,0:1,:,:]]) 
  F_i_mobius = tf.concat(axis=1, values=[F_i_mobius[:,int(nx-1):int(nx),:,:], F_i_mobius]) 
  F_i_mobius = tf.concat(axis=2, values=[F_i_mobius, F_i_mobius[:,:,0:1,:]]) 
  F_i_mobius = tf.concat(axis=2, values=[F_i_mobius[:,:,int(ny-1):int(ny),:], F_i_mobius]) 
  #propagate
  F = simple_conv(F_i_mobius, propagate_kernel)
  # single out bounce back values
  bounce_back = tf.multiply(F, bound_9)
  F_test_1 = bounce_back 
  bounce_back = simple_conv(bounce_back, bound_kernel)
  F_test_2 = bounce_back 
  # calc density
  density = tf.expand_dims(tf.reduce_sum(F, 3), 3)
  # calc x vel
  ux = tf.div(simple_conv(F, ux_kernel), density)
  # calc y vel
  uy = tf.div(simple_conv(F, uy_kernel), density)
  # add delta
  ux = ux + delta_ux
  #uy = uy + delta_ux
  # kill bounded velocitys and density
  ux = tf.multiply(ux, bound_inv)
  uy = tf.multiply(uy, bound_inv)
  density = tf.multiply(density, bound_inv)
  # Calc more things
  u_squ = tf.square(ux) + tf.square(uy)
  u_c2 = ux + uy
  u_c4 = -ux + uy
  u_c6 = -u_c2
  u_c8 = -u_c4
  # now FEQ 
  # this could probably be heavelily optimized with conv layers but for now I will go with this
  FEQ_0 = t2*density*(1.0 + tf.div(ux, c_squ) + tf.multiply(tf.square(tf.div(ux, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_2 = t2*density*(1.0 + tf.div(uy, c_squ) + tf.multiply(tf.square(tf.div(uy, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_4 = t2*density*(1.0 - tf.div(ux, c_squ) + tf.multiply(tf.square(tf.div(ux, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_6 = t2*density*(1.0 - tf.div(uy, c_squ) + tf.multiply(tf.square(tf.div(uy, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  # next neighbour ones
  FEQ_1 = t3*density*(1.0 + tf.div(u_c2, c_squ) + tf.multiply(tf.square(tf.div(u_c2, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_3 = t3*density*(1.0 + tf.div(u_c4, c_squ) + tf.multiply(tf.square(tf.div(u_c4, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_5 = t3*density*(1.0 + tf.div(u_c6, c_squ) + tf.multiply(tf.square(tf.div(u_c6, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_7 = t3*density*(1.0 + tf.div(u_c8, c_squ) + tf.multiply(tf.square(tf.div(u_c8, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  # final one
  FEQ_8 = t1*density*(1.0 - tf.div(u_squ, 2.0*c_squ))
  # put them all together
  FEQ = tf.concat(axis=3, values=[FEQ_0, FEQ_1, FEQ_2, FEQ_3, FEQ_4, FEQ_5, FEQ_6, FEQ_7, FEQ_8])

  # compute f
  F = omega*FEQ+(1.0-omega)*F
  F_i_plus = tf.multiply(F, bound_9_inv) + bounce_back
  step = tf.group(
    F_i.assign(F_i_plus))

  # init things
  init = tf.global_variables_initializer()
  # start sess
  sess = tf.Session()
  # init variables
  sess.run(init)

  # run steps
  for i in range(100000):
    t = time.time()
    step.run(session=sess)
    elapsed = time.time() - t
    if i % 1000 == 0:
      print("step " + str(i))
      print("time per step " + str(elapsed))
    #print(F_test_1.eval(session=sess)[0,4,4,:])
    #print("next")
    #print(F_test_2.eval(session=sess)[0,4,4,:])
    if i % 50 == 0:
      ux_r = ux.eval(session=sess)
      uy_r = uy.eval(session=sess)
      ux_r = ux_r[0,:,:,:]
      uy_r = uy_r[0,:,:,:]
      frame = np.square(ux_r) + np.square(uy_r)
      frame = np.uint8(255 * frame/np.max(frame))
      frame = cv2.applyColorMap(frame[:,:,0], 2)
      video.write(frame)
    """
    if i % 100 == 0:
      ux_r = ux.eval(session=sess)
      uy_r = uy.eval(session=sess)
      ux_r = ux_r[0,:,:,0]
      uy_r = uy_r[0,:,:,0]
      #plt.imshow(np.square(ux_r) + np.square(uy_r) - delta_ux_display)
      plt.imshow(np.square(ux_r) + np.square(uy_r))
      plt.show()
    """
 
  if FLAGS.test:
    print("saving last F to compare with matlab version")
    state = F_test_1.eval(session=sess)
    np.save("test_tensorflow", state)
  

def main(argv=None):  # pylint: disable=unused-argument
  run()

if __name__ == '__main__':
  tf.app.run()




