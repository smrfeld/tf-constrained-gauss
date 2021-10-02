from ..net_common import unit_mat_sym
from ..helpers import check_non_zero_idx_pairs

import tensorflow as tf

import numpy as np

from typing import List, Tuple

@tf.keras.utils.register_keras_serializable(package="tfConstrainedGauss")
class LayerMultPrecCov(tf.keras.layers.Layer):
    """Layer that multiplies precision by covarinace matrix
    """

    @classmethod
    def constructDiag(cls,
        n: int,
        non_zero_idx_pairs: List[Tuple[int,int]],
        init_diag_val: float = 1.0,
        **kwargs
        ):
        """Constructor for a diagonal precision matrix with elements specified

        Args:
            n (int): Size of matrix
            non_zero_idx_pairs (List[Tuple[int,int]]): List of non-zero index pairs in the matrix
            init_diag_val (float, optional): Value of the diagonal part of precision matrix. Defaults to 1.0.

        Returns:
            LayerMultPrecCov: layer
        """
        check_non_zero_idx_pairs(n, non_zero_idx_pairs)

        # Set all diagonal elements to one, rest zero (like a identity matrix)
        non_zero_vals = np.zeros(len(non_zero_idx_pairs))
        for i,pair in enumerate(non_zero_idx_pairs):
            if pair[0] == pair[1]:
                non_zero_vals[i] = init_diag_val

        return cls(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            non_zero_vals=non_zero_vals,
            **kwargs
            )

    def __init__(self,
        n: int,
        non_zero_idx_pairs: List[Tuple[int,int]],
        non_zero_vals: np.array,
        **kwargs
        ):
        """Constructor

        Args:
            n (int): Size of matrix
            non_zero_idx_pairs (List[Tuple[int,int]]): List of non-zero index pairs in the matrix
            non_zero_vals (np.array): Non-zero values in the precision matrix corresponding to non_zero_idx_pairs
        """
        super(LayerMultPrecCov, self).__init__(**kwargs)

        check_non_zero_idx_pairs(n, non_zero_idx_pairs)

        self.n = n
        self.non_zero_idx_pairs = non_zero_idx_pairs

        self.non_zero_vals = self.add_weight(
            name="non_zero_vals",
            shape=len(non_zero_vals),
            trainable=True,
            initializer=tf.constant_initializer(non_zero_vals),
            dtype='float32'
            )

    @property
    def n_non_zero(self) -> int:
        """Number of non-zero elements, not counting twice for symmetry!

        Returns:
            int: number of non-zero elements, not counting twice for symmetry!
        """
        return len(self.non_zero_idx_pairs)

    def get_config(self):
        """Get config for writing
        """
        config = super(LayerMultPrecCov, self).get_config()
        config.update({
            "n": self.n,
            "non_zero_idx_pairs": self.non_zero_idx_pairs,
            "non_zero_vals": self.non_zero_vals.numpy()
            })
        return config

    @classmethod
    def from_config(cls, config):
        """Construct from config
        """
        return cls(**config)

    def call(self, inputs):
        """Call the layer

        Args:
            inputs ([type]): Covariance matrix: batch_size x n x n

        Returns:
            [type]: Product of precision and covariance matrix: vector of length batch_size
        """
        # Inputs = cov mat

        prec_mat = tf.zeros_like(inputs)
        for i,pair in enumerate(self.non_zero_idx_pairs):
            prec_mat_0 = self.non_zero_vals[i] * unit_mat_sym(self.n,pair[0],pair[1])
            prec_mat += tf.map_fn(lambda val: prec_mat_0, prec_mat)

        return tf.matmul(prec_mat,inputs)

@tf.keras.utils.register_keras_serializable(package="tfConstrainedGauss")
class ModelID(tf.keras.Model):
    """Model for the Identidy method
    """

    def __init__(self, 
        mult_lyr: LayerMultPrecCov,
        **kwargs
        ):
        super(ModelID, self).__init__(**kwargs)

        self.mult_lyr = mult_lyr

    def get_config(self):
        # Do not call super.get_config !!!
        config = {
            "mult_lyr": self.mult_lyr
        }
        return config

    # from_config doesn't get called anyways?
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input_tensor, training=False):
        """Wrapper to call the mulitplication layer of precision and covariance matrices
        
        Args:
            input_tensor ([type]): Covariance matrix: batch_size x n x n

        Returns:
            [type]: Product of precision and covariance matrix: vector of length batch_size
        """
        return self.mult_lyr(input_tensor)