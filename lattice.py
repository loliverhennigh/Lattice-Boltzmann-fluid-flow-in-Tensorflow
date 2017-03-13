
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
        self.density = tf.reduce_sum(f, axis=3)

    def _velocity(self):
        flat_f = tf.reshape(self.f, (self.ncells, self.nneigh))
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

    def _boundary_kill(self):
        flat_f = tf.reshape(self.f_temp, (self.ncells, self.nneigh))
        self.f_boundary_kill = flat_f * (-self.boundary + 1.0)

    def _boundary_not_kill(self):
        flat_f = tf.reshape(self.f_temp, (self.ncells, self.nneigh))
        self.f_boundary_not_kill = flat_f * self.boundary

    def _bounce_back(self):
        self._boundary_kill()
        self._boundary_kill()
        flat_f = tf.reshape(self.f_temp, (self.ncells, self.nneigh))
        flat_boundary = tf.reshape(self.boundary, (self.ncells, self.nneigh))
        self.flat_f_boundary_not_kill = self.flat_f_boundary_not_kill * self.bounce_kernel

    def _bn(self):
        self.bn = (self.gamma*(self.tau-0.5))/((1-self.gamma)+(self.tau-0.5))

    def _feq(self):
        flat_velocity = self.

    def _collide(self):
        ome = 1.0/self.tau
        self._boundary_kill
        self._velocity()

        


