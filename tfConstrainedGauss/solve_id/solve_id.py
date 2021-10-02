from ..helpers import convert_mat_non_zero_to_inv_mat, check_symmetric, \
    convert_mat_non_zero_to_inv_mat_non_zero, convert_mat_non_zero_to_mat, \
        convert_mat_to_mat_non_zero
from .model_id import ModelID, LayerMultPrecCov

import tensorflow as tf

import numpy as np

from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class InputsID:
    n : int
    non_zero_idx_pairs: List[Tuple[int,int]]
    target_cov_mat : np.array
    epochs: int = 100
    learning_rate: float = 0.01

    def convert_mat_non_zero_to_inv_mat_non_zero(self, mat_non_zero: np.array) -> np.array:
        return convert_mat_non_zero_to_inv_mat_non_zero(
            n=self.n,
            non_zero_idx_pairs=self.non_zero_idx_pairs,
            mat_non_zero=mat_non_zero
        )
    
    def convert_mat_non_zero_to_inv_mat(self, mat_non_zero: np.array) -> np.array:
        return convert_mat_non_zero_to_inv_mat(
            n=self.n,
            non_zero_idx_pairs=self.non_zero_idx_pairs,
            mat_non_zero=mat_non_zero
        )

    def convert_mat_to_mat_non_zero(self, mat: np.array) -> np.array:
        return convert_mat_to_mat_non_zero(
            n=self.n,
            non_zero_idx_pairs=self.non_zero_idx_pairs,
            mat=mat
        )
    
    @property
    def n_non_zero(self):
        return len(self.non_zero_idx_pairs)

    def report(self):
        print("----- Inputs -----")

        print("Constraints on the prec mat: non-zero elements are:")
        print(self.non_zero_idx_pairs)

        print("Target cov mat:")
        print(self.target_cov_mat)

        print("Solver options:")
        print("epochs:", self.epochs)
        print("learning_rate:", self.learning_rate)

        print("----- End inputs -----")

@dataclass
class ResultsID:
    inputs: InputsID
    trained_model : ModelID

    init_prec_mat_non_zero : np.array
    init_cov_mat_reconstructed : np.array
    learned_prec_mat_non_zero : np.array
    learned_cov_mat : np.array

    @property
    def learned_prec_mat(self):
        return convert_mat_non_zero_to_mat(
            n=self.inputs.n,
            non_zero_idx_pairs=self.inputs.non_zero_idx_pairs,
            mat_non_zero=self.learned_prec_mat_non_zero
            )

    def report(self):
        print("----- Results -----")

        print("Prec mat initial guess for non-zero elements:")
        print(self.init_prec_mat_non_zero)

        print("-> Learned prec mat non-zero elements:")
        print(self.learned_prec_mat_non_zero)

        print("Initial cov mat corresponding to initial prec mat guess:")
        print(self.init_cov_mat_reconstructed)

        print("-> Learned cov mat:")
        print(self.learned_cov_mat)

        print("--> Target cov mat:")
        print(self.inputs.target_cov_mat)

        print("Matrix product (prec.cov) learned:")
        print(np.dot(self.learned_prec_mat,self.learned_cov_mat))

        print("----- End results -----")

def solve_id(
    inputs: InputsID
    ) -> ResultsID:
    """Solve the identity proble

    Args:
        inputs (InputsID): Inputs

    Returns:
        ResultsID: Results
    """

    assert(inputs.target_cov_mat.shape == (inputs.n,inputs.n))
    assert(check_symmetric(inputs.target_cov_mat))
    
    # Invert
    init_prec_mat = np.linalg.inv(inputs.target_cov_mat)
    init_prec_mat_non_zero = inputs.convert_mat_to_mat_non_zero(init_prec_mat)
    init_cov_mat_reconstructed = inputs.convert_mat_non_zero_to_inv_mat(init_prec_mat_non_zero)

    lyr = LayerMultPrecCov(
        n=inputs.n,
        non_zero_idx_pairs=inputs.non_zero_idx_pairs,
        non_zero_vals=init_prec_mat_non_zero
        )

    model = ModelID(mult_lyr=lyr)

    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=inputs.learning_rate)
    model.compile(optimizer=opt,
                loss=loss_fn,
                run_eagerly=False)

    # Covariance matrix input
    # NOTE: do not use batch size = 1, anything higher eg 2 is OK - 1 is peculiar
    batch_size = 2
    train_inputs = np.full(shape=(batch_size,inputs.n,inputs.n),fill_value=inputs.target_cov_mat)

    # Output = identity
    train_outputs = np.full(shape=(batch_size,inputs.n,inputs.n),fill_value=np.eye(inputs.n))

    # Train!
    # Do NOT pass batch_size = 1 -> peculiar problems
    tf.get_logger().setLevel('ERROR')
    model.fit(
        train_inputs, 
        train_outputs, 
        epochs=inputs.epochs, 
        batch_size=batch_size
        )

    # Return
    learned_prec_mat_non_zero = model.mult_lyr.non_zero_vals.numpy()
    learned_cov_mat = inputs.convert_mat_non_zero_to_inv_mat(learned_prec_mat_non_zero)

    return ResultsID(
        inputs=inputs,
        trained_model=model,
        init_prec_mat_non_zero=init_prec_mat_non_zero,
        init_cov_mat_reconstructed=init_cov_mat_reconstructed,
        learned_prec_mat_non_zero=learned_prec_mat_non_zero,
        learned_cov_mat=learned_cov_mat
        )