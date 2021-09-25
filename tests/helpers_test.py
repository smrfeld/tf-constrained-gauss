import tensorflow as tf
import numpy as np

from typing import List, Tuple

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

def save_load_layer(lyr, x_in):
    model = SingleLayerModel(lyr)
    save_load_model(model, x_in)

def save_load_model(model, x_in):

    # Test save; call the model once to build it first
    x_out = model(x_in)

    print(model)
    model.save("saved_models/model", save_traces=False)

    # Test load
    model_rel = tf.keras.models.load_model("saved_models/model")
    print(model_rel)

    # Check types match!
    # Otherwise we may have: tensorflow.python.keras.saving.saved_model.load.XYZ instead of XYZ
    assert type(model_rel) is type(model)

