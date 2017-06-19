
import time

import numpy as np
import tensorflow as tf
import math 
import cv2

from lb_solver import *

import matplotlib.pyplot as plt 

# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

shape = [128, 512]
success = video.open('video.mov', fourcc, 4, (shape[1], shape[0]), True)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('test', True,
                           """will save the state of the system for testing purposes""")

def make_car_boundary(car_name="cars/car_001.png", shape=[256,1024], car_shape=(256,128)):
  img = cv2.imread(car_name, 0)
  img = cv2.flip(img, 1)
  resized_img = cv2.resize(img, car_shape)
  resized_img = -np.rint(resized_img/255.0).astype(int).astype(np.float32) + 1.0
  resized_img = resized_img.reshape([1, car_shape[1], car_shape[0], 1])
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:, shape[0]-car_shape[1]:, 64:64+car_shape[0], :] = resized_img
  boundary[:,0,:,:] = 1.0
  boundary[:,shape[0]-1,:,:] = 1.0
  return boundary

def make_u_input(shape, value=0.001):
  u = np.zeros((1,shape[0],1,1))
  l = shape[0] - 2
  for i in xrange(shape[0]):
    yp = i - 1.5
    vx = value*4.0/(l*l)*(l*yp - yp*yp)
    u[0,i,0,0] = vx
  u = u.astype(np.float32)
  return u

def run():
  # constants
  density = 1.0
  input_vel = 0.1
  tau = 0.5 + 3.0*.02

  # make lattice state, boundary and input velocity
  f = zeros_f(shape, density=density)
  boundary = make_car_boundary(shape=shape, car_shape=(int(shape[1]/1.3), int(shape[0]/1.3)))
  boundary = tf.constant(boundary)
  u_in = make_u_input(shape, value=input_vel)
  u_in = tf.constant(u_in)
 
  # construc solver 
  step, u, f = lbm_step(f, boundary, u_in, density=density, tau=tau)

  # init things
  init = tf.global_variables_initializer()
  # start sess
  sess = tf.Session()
  # init variables
  sess.run(init)

  # run steps
  for i in range(1000):
    if i % 10 == 0:
      f_r = f.eval(session=sess)
      u_r = u.eval(session=sess)
      ux_r = u_r[0,:,:,0:1]
      #uy_r = u_r[0,:,:,1:2]
      uy_r = u_r[0,:,:,1:2]
      frame = np.square(ux_r) + np.square(uy_r)
      frame = np.uint8(255 * frame/np.max(frame))
      frame = cv2.applyColorMap(frame[:,:,0], 2)
      video.write(frame)
    t = time.time()
    step.run(session=sess)
    elapsed = time.time() - t
    if i % 100 == 0:
      print("step " + str(i))
      print("time per step " + str(elapsed))

def main(argv=None):  # pylint: disable=unused-argument
  run()

if __name__ == '__main__':
  tf.app.run()




