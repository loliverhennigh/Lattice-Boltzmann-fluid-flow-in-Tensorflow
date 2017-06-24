
import time

import numpy as np
import tensorflow as tf
import math 
import cv2
from utils import *

import Domain as dom

import matplotlib.pyplot as plt 

# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

shape = [32, 32, 128]

success = video.open('video.mov', fourcc, 60, (4*128, 128), True)

FLAGS = tf.app.flags.FLAGS

def make_ball_boundary(shape):
  boundary = np.zeros([1] + shape + [1], dtype=np.float32)
  boundary[:, shape[0]/2-5:shape[0]/2+5,  shape[1]/2-5:shape[1]/2+5, shape[0]/2-5:shape[0]/2+5] = 1.0
  return boundary

def ball_init_step(domain, value=0.04):
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

def ball_setup_step(domain, value=0.004):
  u = np.zeros((1,shape[0],shape[1],1,1))
  for i in xrange(shape[0]):
    for j in xrange(shape[1]):
      u[0,i,j,0] = value
  u = u.astype(np.float32)
  u = tf.constant(u)

  # input vel on left side
  f_out = domain.F[0][:,:,:,1:]
  f_edge = tf.split(domain.F[0][:,:,:,0:1], 15, axis=4)

  # new in distrobution
  f_edge[1] = 1.0/3.0*(-2*f_edge[0]-4*f_edge[10]-4*f_edge[12]-4*f_edge[14]-f_edge[2]-2*f_edge[3]-2*f_edge[4]-2*f_edge[5]-2*f_edge[6]-4*f_edge[8]+2*(value+1.0))
  f_edge[7] = 1.0/24.0*(-2*f_edge[0]-4*f_edge[10]-4*f_edge[12]-4*f_edge[14]-4*f_edge[2] +f_edge[3]-5*f_edge[4]  +f_edge[5]-5*f_edge[6]+20*f_edge[8]+2*(value+1.0))
  f_edge[9] = 1.0/24.0*(-2*f_edge[0]+20*f_edge[10]-4*f_edge[12]-4*f_edge[14]-4*f_edge[2]+f_edge[3]-5*f_edge[4]-5*f_edge[5]+f_edge[6]-4*f_edge[8]+2*(value+1.0))
  f_edge[11]= 1.0/24.0*(-2*f_edge[0]-4*f_edge[10]+20*f_edge[12]-4*f_edge[14]-4*f_edge[2]-5*f_edge[3]+f_edge[4]  +f_edge[5]-5*f_edge[6]-4*f_edge[8]+2*(value+1.0))
  f_edge[13]= 1.0/24.0*(-2*f_edge[0]-4*f_edge[10]-4 *f_edge[12]+20*f_edge[14]-4*f_edge[2]-5*f_edge[3]+  f_edge[4]-5*f_edge[5]+f_edge[6]-4*f_edge[8]+2*(value+1.0))
  f_edge = tf.stack(f_edge, axis=4)[:,:,:,:,:,0]
  f = tf.concat([f_edge,f_out],axis=3)
 
  # new Rho
  rho = domain.Rho[0]
  rho_out = rho[:,:,:,1:]
  rho_edge = tf.expand_dims(tf.reduce_sum(f_edge, axis=4), axis=4)
  rho = tf.concat([rho_edge,rho_out],axis=3)

  # new vel
  vel = domain.Vel[0]
  vel_out = vel[:,:,:,1:]
  vel_edge = simple_conv(f_edge, domain.C)
  vel_edge = vel_edge/rho_edge
  vel = tf.concat([vel_edge,vel_out],axis=3)

  # remove vel on right side
  f_out = f[:,:,:,:-1]
  f_edge = tf.split(f[:,:,:,-1:], 15, axis=4)

  # new out distrobution
  f_edge[2] = 1/3.0* (-2*f_edge[0]-f_edge[1]-2*(2*f_edge[11]+2*f_edge[13]+f_edge[3]+f_edge[4]+f_edge[5]+f_edge[6]+2*f_edge[7]+2*f_edge[9]-1.0))
  f_edge[8] = 1/24.0*(-2*f_edge[0] - 4*f_edge[1] - 4*f_edge[11] - 4*f_edge[13] - 5*f_edge[3] + f_edge[4] - 5*f_edge[5] + f_edge[6] +20*f_edge[7] - 4*f_edge[9] + 2.0)
  f_edge[10]= 1/24.0*(-2*f_edge[0] - 4*f_edge[1] - 4*f_edge[11] - 4*f_edge[13] - 5*f_edge[3] + f_edge[4] + f_edge[5] - 5*f_edge[6] - 4*f_edge[7] + 20*f_edge[9] + 2.0)
  f_edge[12]= 1/24.0*(-2*f_edge[0] - 4*f_edge[1] + 20*f_edge[11] - 4*f_edge[13] + f_edge[3] - 5*f_edge[4] - 5*f_edge[5] + f_edge[6] -  4*f_edge[7] - 4*f_edge[9] + 2.0)
  f_edge[14]= 1/24.0*(-2*f_edge[0] - 4*f_edge[1] - 4*f_edge[11] + 20*f_edge[13] + f_edge[3] - 5*f_edge[4] + f_edge[5] - 5*f_edge[6] -  4*f_edge[7] - 4*f_edge[9] + 2.0)
  f_edge = tf.stack(f_edge, axis=4)[:,:,:,:,:,0]
  f = tf.concat([f_out,f_edge],axis=3)
 
  # new Rho
  rho_out = rho[:,:,:,:-1]
  rho_edge = tf.expand_dims(tf.reduce_sum(f_edge, axis=4), axis=4)
  rho = tf.concat([rho_out,rho_edge],axis=3)

  # new vel
  vel_out = vel[:,:,:,:-1]
  vel_edge = simple_conv(f_edge, domain.C)
  vel_edge = vel_edge/rho_edge
  vel = tf.concat([vel_out,vel_edge],axis=3)

  # make steps
  f_step =   domain.F[0].assign(f)
  rho_step = domain.Rho[0].assign(rho)
  vel_step = domain.Vel[0].assign(vel)
  setup_step = tf.group(*[f_step, rho_step, vel_step])
  return setup_step

def run():
  # constants
  input_vel = 0.03
  Re = 10000.0
  nu = input_vel*(2.0*30.)/Re
  Ndim = shape
  boundary = make_ball_boundary(shape=Ndim)

  # domain
  domain = dom.Domain("D3Q15", nu, Ndim, boundary)

  # make lattice state, boundary and input velocity
  initialize_step = ball_init_step(domain, value=input_vel)
  setup_step = ball_setup_step(domain, value=input_vel)

  # init things
  init = tf.global_variables_initializer()

  # start sess
  sess = tf.Session()

  # init variables
  sess.run(init)

  # run steps
  domain.Solve(sess, 500, initialize_step, setup_step, video)

def main(argv=None):  # pylint: disable=unused-argument
  run()

if __name__ == '__main__':
  tf.app.run()




