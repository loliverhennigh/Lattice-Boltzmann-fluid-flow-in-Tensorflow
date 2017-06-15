
import numpy as np
import tensorflow as tf

import D2Q9

class Domain():
  def __init__(method,
               nu, 
               Ndim,
               dx=1.0,
               dt=1.0)
    if method == "D2Q9":
      self.Nneigh = 9
      self.Dim    = 2
      self.W      = D2Q9.WEIGHTS
      self.C      = D2Q9.LVELOC
      self.Op     = D2Q9.OPPOSITE
      self.St     = D2Q9.STREAM
   
    self.time   = 0.0
    self.dt     = dt
    self.dx     = dx
    self.Cs     = dx/dt
    self.Step   = 1
    self.Sc     = 0.17
    self.Ndim   = Ndim
    self.Ncells = np.prod(np.array(Ndim))

    self.Nl     = len(nu)
    self.tau    = []
    self.G      = []
    self.Gx     = []
    self.Rhoref = []
    self.Psi    = []
    self.Gmix   = 0.0
      
    self.F       = []
    self.Ftemp   = []
    self.Vel     = []
    self.BForce  = []
    self.Rho     = []
    self.IsSolid = []
  
    for i in xrange(len(nu)):
      self.tau.append(     3.0*nu[i]*self.dt/(self.dx*self.dx)+0.5
      self.G.append(       0.0)
      self.Gs.append(      0.0)
      self.Rhoref.append(  200.0)
      self.Psi.append(     4.0)
      self.F.append(       tf.Variable([1] + Ndim + [self.Nneigh]))
      self.Ftemp.append(   tf.Variable([1] + Ndim + [self.Nneigh]))
      self.Vel.append(     tf.Variable([1] + Ndim + [3]))
      self.BForce.append(  tf.Variable([1] + Ndim + [3]))
      self.Rho.append(     tf.Variable([1] + Ndim + [1]))
      self.IsSolid.append( tf.Variable([1] + Ndim + [1]))

    self.EEk = tf.zeros((self.Nneigh))
    for n in xrange(3):
      for m in xrange(3):
        self.EEk = self.EEk + tf.abs(self.C[:,n] * self.C[:,m])

    def CollideSC():
      Vmix = self.Rho[0]

    def StreamSC():
      # stream f
      f_pad = pad_mobius(self.F[0])
      f_pad = simple_conv(f_pad, self.St)
      # calc new velocity and density
      Rho = tf.expand_dims(tf.reduce_sum(f_pad, 3), 3)
      Rho = tf.multiply(Rho, (1.0 - self.boundary))
      Vel = simple_conv(f_pad, self.C)
      Vel = tf.multiply(Vel, (1.0 - self.boundary))
      Vel = tf.multiply(Vel, Rho)
      # create steps
      stream_step = self.F[0].assign(f_pad)
      Rho_step = self.Rho[0].assign(Rho)
      Vel_step = self.Vel[0].assign(Vel)
      step = tf.group([stream_step, Rho_step, Vel_step])
      return step
  
      
      
       





      
  
