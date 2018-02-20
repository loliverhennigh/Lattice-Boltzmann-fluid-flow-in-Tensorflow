
import time

import numpy as np
import tensorflow as tf
import math 
import cv2
import os

import LatFlow.Domain as dom
from   LatFlow.utils  import *
import matplotlib.pyplot as plt

from tqdm import *

# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video_a = cv2.VideoWriter()
video_b = cv2.VideoWriter()
video_c = cv2.VideoWriter()

# make video
shape = [64 + 32, 256 + 128]
success = video_a.open('les_a.mov', fourcc, 30, (shape[1], shape[0]*2), True)
success = video_b.open('les_b.mov', fourcc, 30, (shape[1]/4, shape[0]*2/4), True)
success = video_c.open('les_c.mov', fourcc, 30, ((shape[1]/4)/2 - 10, (shape[0]/4)), True)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('run_mode', 'eval',
                          """ run mode """)
tf.app.flags.DEFINE_float('lr', 0.001,
                          """ learning rate """)
TRAIN_DIR = './network'

def make_flow_boundary(shape, compression_factor):
  square_size = shape[0]/16
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  for i in xrange(1):
    pos_x = np.random.randint(6*square_size/compression_factor, (shape[0] - 8*square_size)/compression_factor) * compression_factor
    pos_y = np.random.randint(12*square_size/compression_factor, (shape[0] - 2*square_size)/compression_factor) * compression_factor
    boundary[:, pos_x-square_size:pos_x+square_size, 
                pos_y-square_size:pos_y+square_size, :] = 1.0
  boundary[:,0:compression_factor,:,:] = 1.0
  boundary[:,-compression_factor:,:,:] = 1.0
  return boundary

def filter_function(lattice, compression_factor, filter_type="ave_pool"):
  # just ave pool for now
  if filter_type == "ave_pool":
    lattice = tf.nn.avg_pool(lattice,
                                [1, compression_factor, compression_factor, 1], 
                                [1, compression_factor, compression_factor, 1],
                                 padding='SAME')
  return lattice

def flow_init_step(domain, value=0.12, graph_unroll=False):
  shape = domain.F[0].get_shape()[0:3]
  shape = list(map(int, shape))
  u = np.zeros((shape[0],shape[1],shape[2],3))
  l = shape[1] - 2
  for i in xrange(shape[1]):
    yp = i - 0.5
    vx = value*4.0/(l*l)*(l*yp - yp*yp)
    u[:,i,:,0] = max(vx, 0.0)
    u[:,i,:,1] = 0.0
    u[:,i,:,2] = 0.0
  u = u.astype(np.float32)
  vel = tf.constant(u)

  vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=3), axis=3)
  vel_dot_c = tf.reduce_sum(tf.expand_dims(vel, axis=3) * tf.reshape(domain.C, [1,1,1,domain.Nneigh,3]), axis=4)
  feq = tf.reshape(domain.W, [1,1,1,domain.Nneigh]) * (1.0 + 3.0*vel_dot_c/domain.Cs + 4.5*vel_dot_c*vel_dot_c/(domain.Cs*domain.Cs) - 1.5*vel_dot_vel/(domain.Cs*domain.Cs))

  vel = vel * (1.0 - domain.boundary)
  rho = (1.0 - domain.boundary)

  if graph_unroll:
    return feq
  else:
    f_step = domain.F[0].assign(feq)
    rho_step = domain.Rho[0].assign(rho)
    vel_step = domain.Vel[0].assign(vel)
    initialize_step = tf.group(*[f_step, rho_step, vel_step])
    return initialize_step

def flow_setup_step(domain, value=0.12, graph_unroll=False):
  x_len = int(domain.F[0].get_shape()[1])
  u = np.zeros((1,x_len,1,1))
  l = x_len - 2
  for i in xrange(x_len):
    yp = i - 0.5
    vx = value*4.0/(l*l)*(l*yp - yp*yp)
    u[0,i,0,0] = max(vx, 0.0)
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
  if graph_unroll:
    domain.F[0] = f
    domain.Rho[0] = rho
    domain.Vel[0] = vel
  else:
    f_step = domain.F[0].assign(f)
    rho_step = domain.Rho[0].assign(rho)
    vel_step = domain.Vel[0].assign(vel)
    setup_step = tf.group(*[f_step, rho_step, vel_step])
    return setup_step

def flow_save(domain, sess):
  vel_frame = sess.run(domain.Vel[0])
  rho_frame = sess.run(domain.Rho[0])
  vel_frame = np.sqrt(np.square(vel_frame[0,:,:,0]) + np.square(vel_frame[0,:,:,1]) + np.square(vel_frame[0,:,:,2]))
  rho_frame = rho_frame[0,:,:,0] - 1.0
  frame = np.concatenate([vel_frame, rho_frame], axis=0)
  frame = frame - np.min(frame)
  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, 2)
  video_a.write(frame)

def train_save(sess, domain, domain_les, lattice_in, boundary_in, train_lattices, train_boundaries, loss_lattice):
  feed_dict={boundary_in: train_boundaries,
             lattice_in: train_lattices}
  vel_frame = sess.run(domain.Vel[0], feed_dict=feed_dict)
  rho_frame = sess.run(domain.Rho[0], feed_dict=feed_dict)
  vel_frame = np.sqrt(np.square(vel_frame[0,:,:,0]) + np.square(vel_frame[0,:,:,1]) + np.square(vel_frame[0,:,:,2]))
  rho_frame = rho_frame[0,:,:,0] - 1.0
  frame = np.concatenate([vel_frame, rho_frame], axis=0)
  frame = frame - np.min(frame)
  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, 2)
  video_a.write(frame)

  vel_frame = sess.run(domain_les.Vel[0], feed_dict=feed_dict)
  rho_frame = sess.run(domain_les.Rho[0], feed_dict=feed_dict)
  vel_frame = np.sqrt(np.square(vel_frame[0,:,:,0]) + np.square(vel_frame[0,:,:,1]) + np.square(vel_frame[0,:,:,2]))
  rho_frame = rho_frame[0,:,:,0] - 1.0
  frame = np.concatenate([vel_frame, rho_frame], axis=0)
  frame = frame - np.min(frame)
  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, 2)
  video_b.write(frame)

  loss_frame = sess.run(loss_lattice, feed_dict=feed_dict)
  loss_frame = np.sum(loss_frame[0,:,:,], axis=-1)
  frame = loss_frame
  frame = frame - np.min(frame)
  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, 2)
  video_c.write(frame)

def run():
  # simulation constants
  input_vel = 0.12
  nu = 0.02
  Ndim = shape

  # les train details
  batch_size = 8
  les_ratio = 2
  compression_factor = pow(2, les_ratio)
  nu_les = nu/compression_factor
  Ndim_les = [x / compression_factor for x in shape]
 
  if FLAGS.run_mode == "les_train":
    with tf.Session() as sess:
      # placeholders
      lattice_in = tf.placeholder(tf.float32, [batch_size] + shape + [9], name="lattice_in")
      boundary_in = tf.placeholder(tf.float32, [batch_size] + shape + [1], name="boundary_in")
      lattice_les_in = filter_function(lattice_in, compression_factor)
      boundary_les_in = tf.nn.max_pool(boundary_in,
                                      [1, compression_factor, compression_factor, 1], 
                                      [1, compression_factor, compression_factor, 1],
                                       padding='SAME')
    
      # domains
      domain     = dom.Domain("D2Q9", nu,     Ndim,     boundary_in,     les=False)
      domain_les = dom.Domain("D2Q9", nu_les, Ndim_les, boundary_les_in, les=True, train_les=True)

      # get inital lattice state
      init_lattice_in = flow_init_step(domain, value=input_vel, graph_unroll=True)
    
      # unroll solvers
      steps = 5
      lattice_out     = domain.Unroll(    lattice_in,     steps*compression_factor, flow_setup_step)
      lattice_les_out = domain_les.Unroll(lattice_les_in, steps,                  flow_setup_step)
   
      # loss
      lattice_true_out = filter_function(lattice_out, compression_factor)
      lattice_true_out = tf.stop_gradient(lattice_true_out[:,:,Ndim_les[1]/2:-10])
      lattice_les_out = lattice_les_out[:,:,Ndim_les[1]/2:-10]
      #lattice_true_out = tf.stop_gradient(lattice_true_out)
      #lattice_les_out = lattice_les_out
      loss_lattice = tf.abs((lattice_les_out - lattice_true_out) * (1.0 - domain_les.boundary[:,:,Ndim_les[1]/2:-10]))
      loss = tf.nn.l2_loss((lattice_les_out - lattice_true_out) * (1.0 - domain_les.boundary[:,:,Ndim_les[1]/2:-10]))
      #loss_lattice = tf.abs((lattice_les_out - lattice_true_out) * (1.0 - domain_les.boundary))
      #loss = tf.nn.l2_loss((lattice_les_out - lattice_true_out) * (1.0 - domain_les.boundary))
      tf.summary.scalar('loss', loss)
   
      # train op
      train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
  
      # Build a saver
      saver = tf.train.Saver(tf.global_variables())   
  
      # Summary op
      summary_op = tf.summary.merge_all()
   
      # Build an initialization operation to run below.
      init = tf.global_variables_initializer()
  
      # init if this is the very time training
      sess.run(init)
  
      # init from checkpoint
      variables_to_restore = tf.all_variables()
      saver_restore = tf.train.Saver(variables_to_restore)
      ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
      if ckpt is not None:
        pass
        #saver_restore.restore(sess, ckpt.model_checkpoint_path)
   
      # Summary op
      graph_def = sess.graph.as_graph_def(add_shapes=True)
      summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph_def=graph_def)
       
      # make train inital set
      train_boundaries = []
      for i in xrange(batch_size):
        train_boundaries.append(make_flow_boundary(shape, compression_factor))
      train_boundaries = np.concatenate(train_boundaries, axis=0)
      train_lattices = sess.run(init_lattice_in, feed_dict={boundary_in: train_boundaries})
      print("making dataset")
      for i in tqdm(xrange(2000/steps)): # run simulation a few steps to get rid of pressure waves at begining
        train_lattices = sess.run(lattice_out, feed_dict={boundary_in: train_boundaries,
                                                          lattice_in: train_lattices})

      les_error = []
      sm_constant = []
      for step in tqdm(xrange(1000)):
        _ , loss_value, Sc = sess.run([train_op, loss, domain_les.Sc],feed_dict={boundary_in: train_boundaries,
                                                              lattice_in: train_lattices})
        les_error.append(loss_value)
        sm_constant.append(Sc)
        train_lattices = sess.run(lattice_out, feed_dict={boundary_in: train_boundaries,
                                                          lattice_in: train_lattices})

        if step%(50/steps) == 0:
          train_save(sess, domain, domain_les, lattice_in, boundary_in, train_lattices, train_boundaries, loss_lattice)
  
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
  
        if step%(50/steps) == 0:
          summary_str = sess.run(summary_op, feed_dict={boundary_in: train_boundaries,
                                                        lattice_in: train_lattices})
          summary_writer.add_summary(summary_str, step) 
          print("loss value at " + str(loss_value))
          print("Sc constant at " + str(Sc))
  
        if step%500 == 0:
          checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)  
          print("saved to " + TRAIN_DIR)

      les_error = np.array(les_error)
      sm_constant = np.array(sm_constant)
      plt.plot(les_error)
      plt.title('Loss while training')
      plt.ylabel('Mean Squared Error Loss')
      plt.xlabel('Train Step')
      plt.savefig('loss.pdf')
      plt.show()
      plt.plot(sm_constant)
      plt.title('viscosity constant while training')
      plt.ylabel('Sc')
      plt.xlabel('Train Step')
      plt.savefig('sc.pdf')
      plt.show()

  elif FLAGS.run_mode == "eval":
  
    boundary = make_flow_boundary(shape=Ndim, compression_factor=compression_factor)

    # make lattice state, boundary and input velocity
    domain     = dom.Domain("D2Q9", nu, Ndim, boundary, les=False)
    initialize_step = flow_init_step(domain, value=input_vel)
    setup_step = flow_setup_step(domain, value=input_vel)

    # init things
    init = tf.global_variables_initializer()

    # start sess
    sess = tf.Session()

    # init variables
    sess.run(init)

    # run steps
    domain.Solve(sess, 20000, initialize_step, setup_step, flow_save, 60)

def main(argv=None):  # pylint: disable=unused-argument
  run()

if __name__ == '__main__':
  tf.app.run()




