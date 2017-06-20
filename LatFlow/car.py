
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

shape = [128, 512]
success = video.open('video.mov', fourcc, 4, (shape[1], shape[0]), True)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('test', True,
                           """will save the state of the system for testing purposes""")

def make_car_boundary(car_name="../cars/car_001.png", shape=[256,1024], car_shape=(256,128)):
  img = cv2.imread(car_name, 0)
  img = cv2.flip(img, 1)
  resized_img = cv2.resize(img, car_shape)
  resized_img = -np.rint(resized_img/255.0).astype(int).astype(np.float32) + 1.0
  resized_img = resized_img.reshape([1, car_shape[1], car_shape[0], 1])
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  #boundary[:, shape[0]-car_shape[1]:, 64:64+car_shape[0], :] = resized_img
  boundary[:,0,:,:] = 1.0
  boundary[:,shape[0]-1,:,:] = 1.0
  return boundary

def car_init_step(domain, value=0.08):
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

def car_setup_step(domain, value=0.001):
  u = np.zeros((1,shape[0],1,1))
  l = shape[0] - 2
  for i in xrange(shape[0]):
    yp = i - 1.5
    vx = value*4.0/(l*l)*(l*yp - yp*yp)
    u[0,i,0,0] = vx
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
  rho_out = domain.Rho[0][:,:,1:]
  rho_edge = tf.expand_dims(tf.reduce_sum(f_edge, axis=3), axis=3)
  rho = tf.concat([rho_edge,rho_out],axis=2)

  # new vel
  vel_out = domain.Vel[0][:,:,1:]
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
  rho_out = domain.Rho[0][:,:,:-1]
  rho_edge = tf.expand_dims(tf.reduce_sum(f_edge, axis=3), axis=3)
  rho = tf.concat([rho_out,rho_edge],axis=2)

  # new vel
  vel_out = domain.Vel[0][:,:,:-1]
  vel_edge = simple_conv(f_edge, tf.reshape(domain.C, [1,1,domain.Nneigh, 3]))
  vel_edge = vel_edge/rho_edge
  vel = tf.concat([vel_out,vel_edge],axis=2)

  # make steps
  f_step =   domain.F[0].assign(f)
  rho_step = domain.Rho[0].assign(rho)
  vel_step = domain.Vel[0].assign(vel)
  setup_step = tf.group(*[f_step, rho_step, vel_step])
  return setup_step

def run():
  # constants
  input_vel = 0.01
  Re = 40000.0
  nu = input_vel * (2*6*(1024./10))/Re
  Ndim = shape
  boundary = make_car_boundary(shape=Ndim, car_shape=(int(Ndim[1]/1.3), int(Ndim[0]/1.3)))

  # domain
  domain = dom.Domain("D2Q9", nu, Ndim, boundary)

  # make lattice state, boundary and input velocity
  initialize_step = car_init_step(domain, value=0.008)
  setup_step = car_setup_step(domain, value=input_vel)

  # init things
  init = tf.global_variables_initializer()

  # start sess
  sess = tf.Session()

  # init variables
  sess.run(init)

  # run steps
  domain.Solve(sess, 100, initialize_step, setup_step)

  for i in range(1000):
    if i % 10 == 0:
      f_r = f.eval(session=sess)
      u_r = u.eval(session=sess)
      ux_r = u_r[0,:,:,0:1]
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




