
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

# make video
shape = [256, 512]
cshape = [256, 512]
success = video.open('les.mov', fourcc, 30, (shape[1], shape[0]), True)

FLAGS = tf.app.flags.FLAGS

def make_flow_boundary(shape):
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:, shape[0]/2-4:shape[0]/2+4, shape[0]/2-4:shape[0]/2+4, :] = 1.0
  return boundary

def flow_init_step(domain, value=0.1):
  vel_dir = tf.zeros_like(domain.Vel[0][:,:,:,0:1])
  vel = tf.concat([vel_dir+value, vel_dir, vel_dir], axis=3)
  vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=3), axis=3)
  vel_dot_c = tf.reduce_sum(tf.expand_dims(vel, axis=3) * tf.reshape(domain.C, [1,1,1,domain.Nneigh,3]), axis=4)
  feq = tf.reshape(domain.W, [1,1,1,domain.Nneigh]) * (1.0 + 3.0*vel_dot_c/domain.Cs + 4.5*vel_dot_c*vel_dot_c/(domain.Cs*domain.Cs) - 1.5*vel_dot_vel/(domain.Cs*domain.Cs))

  vel = vel * (1.0 - domain.boundary)
  rho = (1.0 - domain.boundary)

  f_step = domain.F[0].assign(feq)
  rho_step = domain.Rho[0].assign(rho)
  vel_step = domain.Vel[0].assign(vel)
  initialize_step = tf.group(*[f_step, rho_step, vel_step])
  return initialize_step

def flow_setup_step(domain, value=0.1):
  u = np.zeros((1,shape[0],1,1))
  l = shape[0] - 2
  for i in xrange(shape[0]):
    yp = i - 1.5
    vx = value*4.0/(l*l)*(l*yp - yp*yp)
    #u[0,i,0,0] = vx
    u[0,i,0,0] = value
  u = u.astype(np.float32)
  u = tf.constant(u)

  # input vel on left side
  f_out = domain.F[0][:,:,1:]
  f_edge = tf.split(domain.F[0][:,:,0:1], 9, axis=3)

  # new in distrobution
  rho = (f_edge[0] + f_edge[2] + f_edge[4] + 2.0*(f_edge[3] + f_edge[6] + f_edge[7]))/(1.0 - u)
  f_edge[1] = f_edge[3] + (2.0/3.0)*rho*u
  f_edge[5] = f_edge[7] + (1.0/6.0)*rho*u - 0.5*(f_edge[2]-f_edge[4])
  f_edge[8] = f_edge[6] + (1.0/6.0)*rho*u + 0.5*(f_edge[2]-f_edge[4])
  f_edge = tf.stack(f_edge, axis=3)[:,:,:,:,0]
  f = tf.concat([f_edge,f_out],axis=2)
 
  # new Rho
  rho = domain.Rho[0]
  rho_out = rho[:,:,1:]
  rho_edge = tf.expand_dims(tf.reduce_sum(f_edge, axis=3), axis=3)
  rho = tf.concat([rho_edge,rho_out],axis=2)

  # new vel
  vel = domain.Vel[0]
  vel_out = vel[:,:,1:]
  vel_edge = simple_conv(f_edge, tf.reshape(domain.C, [1,1,domain.Nneigh, 3]))
  vel_edge = vel_edge/rho_edge
  vel = tf.concat([vel_edge,vel_out],axis=2)

  # remove vel on right side
  f_out = f[:,:,:-1]
  f_edge = tf.split(f[:,:,-1:], 9, axis=3)

  # new out distrobution
  vx = -1.0 + (f_edge[0] + f_edge[2] + f_edge[4] + 2.0*(f_edge[1] + f_edge[5] + f_edge[8]))
  f_edge[3] = f_edge[1] - (2.0/3.0)*vx
  f_edge[7] = f_edge[5] - (1.0/6.0)*vx + 0.5*(f_edge[2]-f_edge[4])
  f_edge[6] = f_edge[8] - (1.0/6.0)*vx - 0.5*(f_edge[2]-f_edge[4])
  f_edge = tf.stack(f_edge, axis=3)[:,:,:,:,0]
  f = tf.concat([f_out,f_edge],axis=2)
 
  # new Rho
  rho_out = rho[:,:,:-1]
  rho_edge = tf.expand_dims(tf.reduce_sum(f_edge, axis=3), axis=3)
  rho = tf.concat([rho_out,rho_edge],axis=2)

  # new vel
  vel_out = vel[:,:,:-1]
  vel_edge = simple_conv(f_edge, tf.reshape(domain.C, [1,1,domain.Nneigh, 3]))
  vel_edge = vel_edge/rho_edge
  vel = tf.concat([vel_out,vel_edge],axis=2)

  # make steps
  f_step =   domain.F[0].assign(f)
  rho_step = domain.Rho[0].assign(rho)
  vel_step = domain.Vel[0].assign(vel)
  setup_step = tf.group(*[f_step, rho_step, vel_step])
  return setup_step

def flow_save(domain, sess):
  frame = sess.run(domain.Vel[0])
  frame = np.sqrt(np.square(frame[0,:,:,0]) + np.square(frame[0,:,:,1]) + np.square(frame[0,:,:,2]))
  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, 2)
  video.write(frame)

def run():
  # simulation constants
  input_vel = 0.1
  nu = 0.025
  Ndim = shape
  boundary = make_flow_boundary(shape=Ndim)

  # les train details
  batch_size = 4
  les_ratio = 2

  # placeholders
  flow_in = tf.placeholder(tf.float32, [batch_size] + shape + [9], name="flow_in")

  # domains
  domain     = dom.Domain("D2Q9", nu, Ndim, boundary, les=False)
  domain_les = dom.Domain("D2Q9", nu, Ndim, boundary, les=True, train_les=True)

  # unroll solvers
  domain

  # make lattice state, boundary and input velocity
  initialize_step = flow_init_step(domain, value=input_vel)
  setup_step = flow_setup_step(domain, value=input_vel)

  # train op
  train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)

  # init things
  init = tf.global_variables_initializer()

  # start sess
  sess = tf.Session()

  # init variables
  sess.run(init)

  # run steps
  domain.Solve(sess, 4000, initialize_step, setup_step, flow_save, 60)

def main(argv=None):  # pylint: disable=unused-argument
  run()

if __name__ == '__main__':
  tf.app.run()




