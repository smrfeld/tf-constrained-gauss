from ..helpers import convert_mat_non_zero_to_mat, check_symmetric, convert_mat_to_mat_non_zero
from .model_id import ModelID, LayerMultPrecCov

import tensorflow as tf

import numpy as np

from typing import List, Tuple

def invert_ggm(
    n: int,
    non_zero_idx_pairs: List[Tuple[int,int]],
    cov_mat: np.array,
    epochs: int = 1000,
    learning_rate: float = 0.1,
    batch_size : int = 2
    ) -> Tuple[np.array,float]:

    if batch_size == 1:
        raise ValueError("Batch size = 1 leads to peculiar problems; try anything higher, e.g. 2")
    
    assert(batch_size > 0)

    assert(cov_mat.shape == (n,n))
    assert(check_symmetric(cov_mat))
    
    # Invert 
    prec_mat_init = np.linalg.inv(cov_mat)
    prec_mat_non_zero_init = convert_mat_to_mat_non_zero(n,non_zero_idx_pairs,prec_mat_init)

    lyr = LayerMultPrecCov(
        n=n,
        non_zero_idx_pairs=non_zero_idx_pairs,
        non_zero_vals=prec_mat_non_zero_init
        )

    model = ModelID(mult_lyr=lyr)

    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                loss=loss_fn,
                run_eagerly=False)

    # Covariance matrix input
    inputs = {"cov_mat": [cov_mat]}

    # Output = identity
    outputs = [np.eye(n)]

    # Train!
    # Do NOT pass batch_size = 1 -> peculiar problems
    tf.get_logger().setLevel('ERROR')
    model.fit(
        inputs, 
        outputs, 
        epochs=epochs, 
        batch_size=20
        )

    # Return
    final_loss = loss_fn(outputs, model(inputs)).numpy()
    return (model.mult_lyr.non_zero_vals.numpy(),final_loss)

def invert_ggm_chol(
    n: int,
    non_zero_idx_pairs: List[Tuple[int,int]],
    cov_mat: np.array,
    epochs: int = 1000,
    learning_rate: float = 0.1,
    batch_size : int = 2
    ) -> Tuple[np.array,float]:

    prec_mat_non_zero, final_loss = invert_ggm(
        n=n,
        non_zero_idx_pairs=non_zero_idx_pairs,
        cov_mat=cov_mat,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
        )

    prec_mat = convert_mat_non_zero_to_mat(n,non_zero_idx_pairs,prec_mat_non_zero)

    chol = np.linalg.cholesky(prec_mat)
    return (chol, final_loss)