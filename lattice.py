
import tensorflow as tf
import numpy as np

from D2Q9 import *
from utils import *


class lattice(object):
    def __init__(self, lattice_type="D2Q9", nu=1.0, dims=[32,32], dx=1.0, dt=1.0):

       self.dims = dims
       self.ncells = np.product(np.array(dims))

       # set lattice type
       if (lattice_type == "D2Q9"):
           self.nneigh = 9
           self.w = WEIGHTSD2Q9
           self.c = LVELOCD2Q9
           self.op = OPPOSITED2Q9
           self.stream_kernel = STREAMD2Q9
           self.bounce_kernel = BOUNCED2Q9
           self.d2d = True

       # make boundary and f vars
       self.f = tf.Variable(np.zeros(dims + [self.nneigh]))
       self.f_temp = self.f
       self.boundary = tf.Variable(np.zeros(dims)) # boundary (1 if boundary 0 if not)
       self.gamma = tf.Variable(np.zeros(dims)) # ratio of fluid to solid

       # constants
       self.dx = dx
       self.dy = dx # need different collide kernel to make differnt
       self.Tau = 3.0*nu*dt/(dx*dx) + 0.5
       self.rhoref = 200.0;
       self.psiref = 4.0;
       self.g = 0.0;
       self.gs = 0.0;


    def _density(self):
        self.density = tf.reduce_sum(f_temp, axis=3)

    def _velocity(self):
        flat_f = self.f_flat()
        velocity = flat_f * self.c
        velocity = tf.reshape(velocity, self.dims + [3])
        self._density()
        self.velocity = velocity*(self.cs/self.density)

    def _zero_gamma(self):
        # not implemented 
        exit()

    def _stream(self):
        self.f_temp = simple_conv(self.f_temp, self.steam_kernel)

    def _psi(self):
        self.psi = self.psiref*tf.exp(-self.rhoref / self.rho)

    def _f_flat(self, f):
        return tf.reshape(f, (self.ncells, self.nneigh))

    def _f_no_boundary(self, f):
        f_flat = self._f_flat( 
        return flat_f * (-self.boundary + 1.0)

    def _f_boundary(self):
        f_flat = self._f_flat 
        return flat_f * self.boundary

    def _bounce_back(self):
        f_no_boundary = self._f_no_boundary()
        f_boundary = self._f_no_boundary()
        f_boundary = f_boundary * self.bounce_kernel
        f_flat = f_boundary + self.f_no_boundary
        self.f_temp = tf.reshape(f_flat, self.dims + [self.nneigh])

    def _bn(self):
        return (self.gamma*(self.tau-0.5))/((1-self.gamma)+(self.tau-0.5))

    def _feq(self):
        self._velocity()
        v_dot_c = self.v * tf.transpose(self.c, [1,0])
        v_dot_v = tf.reduce_sum(self.v * self.v, axis=3)
        feq = self.w * self.density * (1.0 + 3.0 * v_dot_c / self.cs + 4.5 * v_dot_c * v_dot_c /(self.cs*self.cs) - 1.5 * v_dot_v/(self.cs * self.cs))
        return feq

    def _collide(self):
        ome = 1.0/self.tau
        f_no_boundary = self._f_no_boundary()
        bn = self._bn() 
         
        self._velocity()

        


