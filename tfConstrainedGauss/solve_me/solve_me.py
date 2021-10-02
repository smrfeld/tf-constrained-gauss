from ..helpers import convert_mat_to_mat_non_zero, convert_mat_non_zero_to_mat, \
    convert_mat_non_zero_to_inv_mat_non_zero, convert_mat_non_zero_to_inv_mat
from .model_me import ModelME, LayerPrecToCovMat

import tensorflow as tf
import numpy as np
from dataclasses import dataclass

from typing import List, Tuple

@dataclass
class InputsME:
    """Inputs to the MaxEnt problem
    """

    n : int
    non_zero_idx_pairs: List[Tuple[int,int]]
    target_cov_mat_non_zero : np.array
    epochs: int = 100
    learning_rate: float = 0.01
    use_weighted_loss : bool = False

    verbose : bool = False

    def convert_mat_non_zero_to_inv_mat_non_zero(self, mat_non_zero: np.array):
        """Wrapper around convert_mat_non_zero_to_inv_mat_non_zero
        """
        return convert_mat_non_zero_to_inv_mat_non_zero(
            n=self.n,
            non_zero_idx_pairs=self.non_zero_idx_pairs,
            mat_non_zero=mat_non_zero
        )
    
    @property
    def n_non_zero(self):
        """The number of unique non-zero elements in the precision matrix (excluding symmetry)

        Returns:
            int: Count
        """
        return len(self.non_zero_idx_pairs)

    def report(self):
        """Print some information
        """
        print("----- Inputs -----")

        print("Constraints on the prec mat: non-zero elements are:")
        print(self.non_zero_idx_pairs)

        print("Constraints on the cov mat: specified values corresponding to above idxs:")
        print(self.target_cov_mat_non_zero)

        print("Solver options:")
        print("epochs:", self.epochs)
        print("learning_rate:", self.learning_rate)
        print("use_weighted_loss:", self.use_weighted_loss)

        print("----- End inputs -----")

@dataclass
class ResultsME:
    """Results of the MaxEnt problem
    """
    inputs: InputsME
    trained_model : ModelME

    init_prec_mat_non_zero : np.array
    init_cov_mat_reconstructed_non_zero : np.array
    learned_prec_mat_non_zero : np.array
    learned_cov_mat : np.array

    @property
    def learned_prec_mat(self) -> np.array:
        return convert_mat_non_zero_to_mat(
            n=self.inputs.n,
            non_zero_idx_pairs=self.inputs.non_zero_idx_pairs,
            mat_non_zero=self.learned_prec_mat_non_zero
            )

    @property
    def learned_cov_mat_non_zero(self) -> np.array:
        return convert_mat_to_mat_non_zero(
            n=self.inputs.n,
            non_zero_idx_pairs=self.inputs.non_zero_idx_pairs,
            mat=self.learned_cov_mat
            )

    def report(self):
        """Print some information
        """
        print("----- Results -----")

        print("Prec mat initial guess for non-zero elements:")
        print(self.init_prec_mat_non_zero)

        print("-> Learned prec mat non-zero elements:")
        print(self.learned_prec_mat_non_zero)

        print("Initial cov mat non-zero elements corresponding to initial prec mat guess:")
        print(self.init_cov_mat_reconstructed_non_zero)

        print("-> Learned cov mat non-zero elements:")
        print(self.learned_cov_mat_non_zero)

        print("--> Target cov mat non-zero elements:")
        print(self.inputs.target_cov_mat_non_zero)

        print("----- End results -----")

def custom_weighted_mse(class_weights):
    def weighted_mse(gt, pred):
        # Formula: 
        # w_1*(y_1-y'_1)^2 + ... + w_100*(y_100-y'_100)^2 / sum(weights)
        return tf.keras.backend.sum(class_weights * tf.keras.backend.square(gt - pred)) / tf.keras.backend.sum(class_weights)
    return weighted_mse

def solve_me(inputs: InputsME) -> ResultsME:
    """Solve the MaxEnt problem

    Args:
        inputs (InputsME): Inputs

    Returns:
        ResultsME: Results
    """

    assert(inputs.target_cov_mat_non_zero.shape == (len(inputs.non_zero_idx_pairs),))

    # Invert cov mat to get initial guess for precision matrix
    init_prec_mat_non_zero = inputs.convert_mat_non_zero_to_inv_mat_non_zero(inputs.target_cov_mat_non_zero)
    if inputs.verbose:
        print("Prec mat initial guess for non-zero elements", init_prec_mat_non_zero)

    init_cov_mat_reconstructed_non_zero = inputs.convert_mat_non_zero_to_inv_mat_non_zero(init_prec_mat_non_zero)
    if inputs.verbose:
        print("Initial cov mat corresponding non-zero elements", init_cov_mat_reconstructed_non_zero)

    # Make layer and model
    lyr = LayerPrecToCovMat(
        n=inputs.n,
        non_zero_idx_pairs=inputs.non_zero_idx_pairs,
        non_zero_vals=init_prec_mat_non_zero
        )
    model = ModelME(inv_lyr=lyr)

    # Compile
    opt = tf.keras.optimizers.Adam(learning_rate=inputs.learning_rate)
    if inputs.use_weighted_loss:
        weights_loss = 1.0 / pow(inputs.target_cov_mat_non_zero,2)
        weights_loss = weights_loss.astype('float32')
        if inputs.verbose:
            print("Using weighted loss functing with weights:", weights_loss)
        loss_fn = custom_weighted_mse(weights_loss)
    else:
        loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt,
                loss=loss_fn,
                run_eagerly=False)
    
    # Inputs outputs
    # NOTE: Do NOT use batch_size=1, use anything greater like batch_size=2
    # Make just enough for the batch size to cover
    batch_size = 2
    train_inputs = np.full(
        shape=(batch_size,1),
        fill_value=np.array([2])
        )

    train_outputs = np.full(
        shape=(batch_size,inputs.n_non_zero),
        fill_value=inputs.target_cov_mat_non_zero
        )

    # Train!
    # NOTE: Do NOT use batch_size=1, use anything greater like batch_size=2
    model.fit(
        train_inputs, 
        train_outputs, 
        epochs=inputs.epochs, 
        batch_size=batch_size
        )

    # Return solution & model
    learned_prec_mat_non_zero = model.inv_lyr.non_zero_vals.numpy()
    learned_cov_mat = convert_mat_non_zero_to_inv_mat(
        n=inputs.n,
        non_zero_idx_pairs=inputs.non_zero_idx_pairs,
        mat_non_zero=learned_prec_mat_non_zero
        )

    return ResultsME(
        inputs=inputs,
        trained_model=model,
        init_prec_mat_non_zero=init_prec_mat_non_zero,
        init_cov_mat_reconstructed_non_zero=init_cov_mat_reconstructed_non_zero,
        learned_prec_mat_non_zero=learned_prec_mat_non_zero,
        learned_cov_mat=learned_cov_mat
        )
