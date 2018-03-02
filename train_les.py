
import time
import glob
import os

import numpy as np
import tensorflow as tf
import math 
import cv2

import LatFlow.Domain as dom
from   LatFlow.utils  import *
from tqdm import *

# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 

# define flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('run_mode', 'train',
                          """ run mode """)
tf.app.flags.DEFINE_string('data_dir', './data',
                          """ data dir """)
tf.app.flags.DEFINE_integer('num_train_examples', 1500,
                          """ num_train examples to save """)
tf.app.flags.DEFINE_string('train_dir', './network',
                          """ network dir """)
tf.app.flags.DEFINE_float('lr', 0.001,
                          """ learning rate """)
tf.app.flags.DEFINE_float('train_iters', 10000,
                          """ num_train_steps """)
tf.app.flags.DEFINE_integer('nr_downsamples', 2,
                          """ downsamples """)
tf.app.flags.DEFINE_float('visc', 0.02,
                          """ visc on DNS """)
tf.app.flags.DEFINE_float('input_vel', 0.0025,
                          """ input velocity on DNS """)
tf.app.flags.DEFINE_integer('nx', 256,
                          """ lattice size nx """)
tf.app.flags.DEFINE_integer('ny', 256,
                          """ lattice size ny """)
tf.app.flags.DEFINE_string('filter_type', "ave_pool",
                          """ what filter function to use """)
tf.app.flags.DEFINE_integer('batch_size', 32,
                          """ batch size to use """)
TRAIN_DIR = './network'

def filter_function(lattice):
  # just ave pool for now
  if FLAGS.filter_type == "ave_pool":
    compression_factor = pow(2, FLAGS.nr_downsamples)
    lattice = tf.nn.avg_pool(lattice,
                                [1, compression_factor, compression_factor, 1], 
                                [1, compression_factor, compression_factor, 1],
                                 padding='SAME')
  return lattice

def make_lid_boundary(nx, ny, simulation="DNS"):
  ratio = pow(2,FLAGS.nr_downsamples)
  if simulation == 'LES':
    boundary = np.zeros((1, nx/ratio, ny/ratio, 1), dtype=np.float32)
    width = 1
  else:
    boundary = np.zeros((1, nx, ny, 1), dtype=np.float32)
    width=ratio
    ratio = 1
  boundary[:,:,0:width,:] = 1.0
  boundary[:,  nx/ratio-(width+1):nx/ratio-1,:,:] = 1.0
  boundary[:,:,ny/ratio-(width+1):ny/ratio-1,:]   = 1.0
  return boundary

def lid_init_step(domain, graph_unroll=False):
  vel = tf.zeros_like(domain.Vel[0])
  vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=3), axis=3)
  vel_dot_c = tf.reduce_sum(tf.expand_dims(vel, axis=3) * tf.reshape(domain.C, [1,1,1,domain.Nneigh,3]), axis=4)
  feq = tf.reshape(domain.W, [1,1,1,domain.Nneigh]) * (1.0 + vel_dot_vel) 

  vel = vel * (1.0 - domain.boundary)
  rho = (1.0 - domain.boundary)

  f_step = domain.F[0].assign(feq)
  rho_step = domain.Rho[0].assign(rho)
  vel_step = domain.Vel[0].assign(vel)
  if graph_unroll:
    return feq
  else:
    initialize_step = tf.group(*[f_step, rho_step, vel_step])
    return initialize_step

def lid_setup_step(domain, simulation="DNS", graph_unroll=False):
  # inputing top velocity 
  if simulation == 'DNS':
    #width = pow(2,FLAGS.nr_downsamples)
    width = 1
  else:
    width = 1
  vel = domain.Vel[0]
  vel_out  = vel[:,1:]
  vel_edge = vel[:,:1]
  vel_edge = tf.split(vel_edge, 3, axis=3)
  vel_edge[0] = vel_edge[0]+FLAGS.input_vel
  vel_edge = tf.concat(vel_edge, axis=3)
  vel = tf.concat([vel_edge,vel_out],axis=1)

  # make steps
  if graph_unroll:
    domain.Vel[0] = vel
  else:
    vel_step = domain.Vel[0].assign(vel)
    return vel_step

def lid_save_data(domain, sess):
  state = sess.run(domain.F[0])
  np.save(FLAGS.data_dir + '/state_' + str(domain.time).zfill(7), state)

def lid_save_video(domain, sess, video):
  frame_vel = sess.run(domain.Vel[0])
  frame_rho = sess.run(domain.Rho[0])[0,:,:,0]
  frame_vel = np.sqrt(np.square(frame_vel[0,:,:,0]) + np.square(frame_vel[0,:,:,1]) + np.square(frame_vel[0,:,:,2]))
  frame_rho = frame_rho - np.min(frame_rho)
  frame = np.concatenate([frame_vel, frame_rho], axis=1)
  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, 2)
  video.write(frame)

class DataSet():
  def __init__(self):
    data_files = glob.glob(FLAGS.data_dir + "/*")
    data_files.sort()
    self.data = []
    print("reading in data")
    for i in tqdm(xrange(len(data_files))):
      self.data.append(np.load(data_files[i]))

  def batch(self, batch_size):
    batch_in  = []
    batch_out = []
    for i in xrange(batch_size): 
      index = np.random.randint(0, len(self.data)-1)
      batch_in.append(self.data[index])
      batch_out.append(self.data[index+1])
    batch_in  = np.concatenate(batch_in,  axis=0)
    batch_out = np.concatenate(batch_out, axis=0)
    return batch_in, batch_out

def make_data():
  # constants
  ratio = pow(2, FLAGS.nr_downsamples)
  Ndim = [FLAGS.nx, FLAGS.ny]
  boundary = make_lid_boundary(FLAGS.nx, FLAGS.ny, simulation='DNS')

  # domain
  domain = dom.Domain("D2Q9", FLAGS.visc, Ndim, boundary, les=False)

  # make lattice state, boundary and input velocity
  initialize_step = lid_init_step(domain)
  setup_step = lid_setup_step(domain, simulation="DNS")

  # init things
  init = tf.global_variables_initializer()

  # start sess
  sess = tf.Session()

  # init variables
  sess.run(init)

  # run steps
  domain.Solve(sess, FLAGS.num_train_examples * ratio, initialize_step, setup_step, lid_save_data, ratio)

def test_dns():

  # make video
  video = cv2.VideoWriter()
  success = video.open('test_dns_lid_video.mov', fourcc, 30, (FLAGS.ny*2, FLAGS.nx), True)

  # constants
  ratio = pow(2, FLAGS.nr_downsamples)
  Ndim = [FLAGS.nx, FLAGS.ny]
  boundary = make_lid_boundary(FLAGS.nx, FLAGS.ny, simulation='DNS')

  # domain
  domain = dom.Domain("D2Q9", FLAGS.visc, Ndim, boundary, les=False)

  # make lattice state, boundary and input velocity
  initialize_step = lid_init_step(domain)
  setup_step = lid_setup_step(domain, simulation="DNS")

  # init things
  init = tf.global_variables_initializer()

  # start sess
  sess = tf.Session()

  # init variables
  sess.run(init)

  # run steps
  lid_save = lambda x, y: lid_save_video(x, y, video=video)
  domain.Solve(sess, FLAGS.num_train_examples * ratio, initialize_step, setup_step, lid_save, ratio)

def test_les():

  # make video
  ratio = pow(2, FLAGS.nr_downsamples)
  video = cv2.VideoWriter()
  success = video.open('test_les_lid_video.mov', fourcc, 30, (FLAGS.ny*2/ratio, FLAGS.nx/ratio), True)

  # constants
  Ndim = [FLAGS.nx/ratio, FLAGS.ny/ratio]
  boundary = make_lid_boundary(FLAGS.nx, FLAGS.ny, simulation='LES')

  # domain
  domain = dom.Domain("D2Q9", FLAGS.visc/ratio, Ndim, boundary, les=True)

  # make lattice state, boundary and input velocity
  initialize_step = lid_init_step(domain)
  setup_step = lid_setup_step(domain, simulation="LES")

  # init things
  init = tf.global_variables_initializer()

  # start sess
  sess = tf.Session()

  # init variables
  sess.run(init)

  # run steps
  lid_save = lambda x, y: lid_save_video(x, y, video=video)
  domain.Solve(sess, FLAGS.num_train_examples, initialize_step, setup_step, lid_save, 1)

def train():
  # constants
  ratio = pow(2, FLAGS.nr_downsamples)
  Ndim_LES = [FLAGS.nx/ratio, FLAGS.ny/ratio]
  Ndim_DNS = [FLAGS.nx, FLAGS.ny]
  boundary = make_lid_boundary(FLAGS.nx, FLAGS.ny, simulation="LES") # TODO this should probably come from the dataset

  # make dataset
  dataset = DataSet()

  # start tf sesstion
  with tf.Session() as sess:
    # placeholder inputs
    lattice_in =  tf.placeholder(tf.float32, [FLAGS.batch_size] + Ndim_DNS + [9], name="lattice_in")
    lattice_out = tf.placeholder(tf.float32, [FLAGS.batch_size] + Ndim_DNS + [9], name="lattice_out")
    lattice_les_in  = filter_function(lattice_in)
    lattice_les_out = filter_function(lattice_out)
    
    # make trainable domain
    domain = dom.Domain("D2Q9", FLAGS.visc/ratio, Ndim_LES, boundary, les=True, train_les=True)

    # unroll solver
    lattice_les_out_g = domain.Unroll(lattice_les_in, 1, lid_setup_step)

    # loss
    buffer_edges = 4
    lattice_les_out = lattice_les_out[:,buffer_edges:-buffer_edges,
                                        buffer_edges:-buffer_edges]
    lattice_les_out_g = lattice_les_out_g[:,buffer_edges:-buffer_edges,
                                            buffer_edges:-buffer_edges]
    loss = tf.nn.l2_loss(lattice_les_out
                       - lattice_les_out_g)
    tf.summary.scalar('loss', loss) 
    tf.summary.scalar('Sc', domain.Sc[0]) 

    # image summary
    tf.summary.image('loss_image', tf.reduce_sum(tf.abs(lattice_les_out - lattice_les_out_g), axis=-1, keep_dims=True))
    
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
      #saver_restore.restore(sess, ckpt.model_checkpoint_path)
      pass
  
    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph_def=graph_def)
           
    for step in tqdm(xrange(FLAGS.train_iters)):
      np_lattice_in, np_lattice_out = dataset.batch(FLAGS.batch_size)
      _ , loss_value, Sc = sess.run([train_op, loss, domain.Sc],
                                     feed_dict={lattice_in:  np_lattice_in,
                                                lattice_out: np_lattice_out})

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
  
      if step%10 == 0:
        summary_str = sess.run(summary_op, feed_dict={lattice_in:  np_lattice_in,
                                                      lattice_out: np_lattice_out})
        summary_writer.add_summary(summary_str, step) 
        #print("loss value at " + str(loss_value))
        #print("Sc constant at " + str(Sc))
  
      if step%500 == 0:
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + TRAIN_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.run_mode == "train":
    train()
  elif FLAGS.run_mode == "make_data":
    make_data()
  elif FLAGS.run_mode == "test_DNS":
    test_dns()
  elif FLAGS.run_mode == "test_LES":
    test_les()
  

if __name__ == '__main__':
  tf.app.run()




