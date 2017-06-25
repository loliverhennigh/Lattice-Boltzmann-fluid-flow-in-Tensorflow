
import time

import numpy as np
import tensorflow as tf
import math 
import cv2

import LatFlow.Domain as dom
from   LatFlow.utils  import *

# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

shape = [128, 128, 512]

success = video.open('cube_3d_flow.mov', fourcc, 30, (shape[2]*3, shape[0]*3), True)

FLAGS = tf.app.flags.FLAGS

def make_cube_boundary(shape):
  boundary = np.zeros([1] + shape + [1], dtype=np.float32)
  boundary[:, shape[0]/2-20:shape[0]/2+20,  shape[1]/2-20:shape[1]/2+20, shape[0]/2-20:shape[0]/2+20] = 1.0
  return boundary

def cube_init_step(domain, value=0.04):
  vel_dir = tf.zeros_like(domain.Vel[0][:,:,:,:,0:1])
  vel = tf.concat([vel_dir+value, vel_dir, vel_dir], axis=4)
  vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=4), axis=4)
  vel_dot_c = tf.reduce_sum(tf.expand_dims(vel, axis=4) * domain.C, axis=5)
  feq = domain.W * (1.0 + (3.0/domain.Cs)*vel_dot_c + (4.5/(domain.Cs*domain.Cs))*(vel_dot_c*vel_dot_c) - (1.5/(domain.Cs*domain.Cs))*vel_dot_vel)

  vel = vel * (1.0 - domain.boundary)
  rho = (1.0 - domain.boundary)

  f_step = domain.F[0].assign(feq)
  rho_step = domain.Rho[0].assign(rho)
  vel_step = domain.Vel[0].assign(vel)
  initialize_step = tf.group(*[f_step, rho_step, vel_step])
  return initialize_step

def cube_setup_step(domain, value=0.004):
  u = np.zeros((1,shape[0],shape[1],1,1))
  for i in xrange(shape[0]):
    for j in xrange(shape[1]):
      u[0,i,j,0] = value
  u = u.astype(np.float32)
  u = tf.constant(u)

  # new vel
  vel = domain.Vel[0]
  vel_out = vel[:,:,:,1:]
  vel_edge = vel[:,:,:,:1]
  vel_edge = tf.split(vel_edge, 3, axis=4)
  vel_edge[0] = vel_edge[0]+value
  vel_edge = tf.concat(vel_edge, axis=4)
  vel = tf.concat([vel_edge,vel_out],axis=3)

  # make steps
  vel_step = domain.Vel[0].assign(vel)
  return vel_step

def cube_save(domain, sess):
  frame = sess.run(domain.Vel[0])
  frame = np.sqrt(np.square(frame[0,:,shape[1]/2,:,0]) + np.square(frame[0,:,shape[1]/2,:,1]) + np.square(frame[0,:,shape[1]/2,:,2]))
  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, 2)
  frame = cv2.resize(frame, (shape[2]*3, shape[0]*3))
  video.write(frame)

def run():
  # constants
  input_vel = 0.01
  nu = input_vel*(0.0015)
  Ndim = shape
  boundary = make_cube_boundary(shape=Ndim)

  # domain
  domain = dom.Domain("D3Q15", nu, Ndim, boundary)

  # make lattice state, boundary and input velocity
  initialize_step = cube_init_step(domain, value=input_vel)
  setup_step = cube_setup_step(domain, value=input_vel)

  # init things
  init = tf.global_variables_initializer()

  # start sess
  sess = tf.Session()

  # init variables
  sess.run(init)

  # run steps
  domain.Solve(sess, 2000, initialize_step, setup_step, cube_save, 1)

def main(argv=None):  # pylint: disable=unused-argument
  run()

if __name__ == '__main__':
  tf.app.run()




