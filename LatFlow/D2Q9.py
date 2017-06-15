
import tensorflow as tf
import numpy as np

# Lattice weights
WEIGHTSD2Q9 = tf.constant([4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.], dtype=1)

# Lattice lveloc
LVELOCD2Q9 = tf.constant( [ [0,0,0], [1,0,0], [0,1,0], [-1,0,0], [0,-1,0], [1,1,0], [-1,1,0], [-1,-1,0], [1,-1,0]     ], dtype=1)

# Lattice Opposite
OPPOSITED2Q9 = tf.constant([ 0, 3, 4, 1, 2, 7, 8, 5, 6 ], dtype=1)

# Lattice bounce back kernel
BOUNCED2Q9 = np.array([[   1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0],
                         [ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0],
                         [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0],
                         [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0]])
BOUNCED2Q9 = tf.constant(BOUNCED2Q9, dtype=1)

# how to transfer states  
STREAMD2Q9 = np.zeros((3,3,9,9))
STREAMD2Q9[0,1,0,0] = 1.0
STREAMD2Q9[0,0,1,1] = 1.0
STREAMD2Q9[1,0,2,2] = 1.0
STREAMD2Q9[2,0,3,3] = 1.0
STREAMD2Q9[2,1,4,4] = 1.0
STREAMD2Q9[2,2,5,5] = 1.0
STREAMD2Q9[1,2,6,6] = 1.0
STREAMD2Q9[0,2,7,7] = 1.0
STREAMD2Q9[1,1,8,8] = 1.0
STREAMD2Q9 = tf.constant(STREAMD2Q9, dtype=1)





