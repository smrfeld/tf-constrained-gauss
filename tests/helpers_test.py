import tensorflow as tf
import numpy as np

from typing import List, Tuple

def random_non_zero_idx_pairs(n: int) -> List[Tuple[int,int]]:
    # Non zero elements
    non_zero_idx_pairs = []
    # All diagonal (required)
    for i in range(0,n):
        non_zero_idx_pairs.append((i,i))
    # Some off diagonal < n choose 2
    max_no_off_diagonal = int((n-1)*n/2)
    no_off_diagonal = np.random.randint(low=0,high=max_no_off_diagonal)
    print("No non-zero off-diagonal elements:",no_off_diagonal,"max possible:",max_no_off_diagonal)
    idx = 0
    while idx < no_off_diagonal:
        i = np.random.randint(low=1,high=n)
        j = np.random.randint(low=0,high=i)
        if not (i,j) in non_zero_idx_pairs:
            non_zero_idx_pairs.append((i,j))
            idx += 1

    return non_zero_idx_pairs

# Random cov mat using chol decomposition
# Diagonal = positive => unique
def random_cov_mat(n: int) -> np.array:
    chol = np.tril(np.random.rand(n,n))
    return np.dot(chol,np.transpose(chol))

@tf.keras.utils.register_keras_serializable(package="physDBD")
class SingleLayerModel(tf.keras.Model):

    def __init__(self, lyr, **kwargs):
        super(SingleLayerModel, self).__init__(name='')
        self.lyr = lyr

    def get_config(self):
        return {
            "lyr": self.lyr
            }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input_tensor, training=False):
        return self.lyr(input_tensor)

def assert_equal_dicts(x_out, x_out_true):
    for key, val_true in x_out_true.items():
        val = x_out[key]

        assert_equal_arrs(val,val_true)

def assert_equal_arrs(x_out, x_out_true):
    tol = 5.e-4
    assert np.max(abs(x_out-x_out_true)) < tol

def save_load_model(lyr, x_in):

    # Test save; call the model once to build it first
    model = SingleLayerModel(lyr)
    x_out = model(x_in)

    print(model)
    model.save("saved_models/model", save_traces=False)

    # Test load
    model_rel = tf.keras.models.load_model("saved_models/model")
    print(model_rel)

    # Check types match!
    # Otherwise we may have: tensorflow.python.keras.saving.saved_model.load.XYZ instead of XYZ
    assert type(model_rel) is type(model)
