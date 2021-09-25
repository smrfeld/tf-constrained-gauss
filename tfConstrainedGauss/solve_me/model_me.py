from ..net_common import unit_mat_sym
from ..helpers import check_non_zero_idx_pairs

import tensorflow as tf
import numpy as np

from typing import List, Tuple

@tf.keras.utils.register_keras_serializable(package="tfConstrainedGauss")
class LayerPrecToCovMat(tf.keras.layers.Layer):

    @classmethod
    def constructDiag(cls,
        n: int,
        non_zero_idx_pairs: List[Tuple[int,int]],
        init_diag_val: float = 1.0,
        **kwargs
        ):
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
        super(LayerPrecToCovMat, self).__init__(**kwargs)

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
    def n_non_zero(self):
        return len(self.non_zero_idx_pairs)

    def get_config(self):
        config = super(LayerPrecToCovMat, self).get_config()
        config.update({
            "n": self.n,
            "non_zero_idx_pairs": self.non_zero_idx_pairs,
            "non_zero_vals": self.non_zero_vals.numpy()
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        
        # Form precision matrix
        batch_size = tf.shape(inputs)[0]
        prec_mat = tf.zeros((batch_size,self.n,self.n), dtype='float32')

        for i,pair in enumerate(self.non_zero_idx_pairs):
            prec_mat += tf.map_fn(lambda val: self.non_zero_vals[i] * unit_mat_sym(self.n,pair[0],pair[1]), prec_mat)
        
        # Invert to get Covariance matrix
        cov_mat = tf.linalg.inv(prec_mat)

        return cov_mat

@tf.keras.utils.register_keras_serializable(package="tfConstrainedGauss")
class ModelME(tf.keras.Model):

    def __init__(self, 
        inv_lyr: LayerPrecToCovMat,
        **kwargs
        ):
        super(ModelME, self).__init__(**kwargs)

        self.inv_lyr = inv_lyr

    def get_config(self):
        # Do not call super.get_config !!!
        config = {
            "inv_lyr": self.inv_lyr
        }
        return config

    # from_config doesn't get called anyways?
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input_tensor, training=False):
        cov_mat = self.inv_lyr(input_tensor)

        batch_size = tf.shape(cov_mat)[0]
        cov_mat_non_zero = tf.zeros((batch_size,self.inv_lyr.n_non_zero), dtype='float32')

        for i,pair in enumerate(self.inv_lyr.non_zero_idx_pairs):
            unit_vec = tf.one_hot(
                indices=i,
                depth=self.inv_lyr.n_non_zero,
                dtype='float32'
                )
            cov_mat_non_zero += tf.map_fn(lambda cov_mat_batch: cov_mat_batch[pair[0],pair[1]] * unit_vec, cov_mat)

        return cov_mat_non_zero