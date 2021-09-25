from helpers_test import assert_equal_arrs, save_load_layer, \
    random_non_zero_idx_pairs, random_cov_mat, save_load_model
from tfConstrainedGauss import InputsID, solve_id, LayerMultPrecCov, ModelID

import numpy as np
import tensorflow as tf

class TestID:

    def test_layer_mult(self):
        
        n=2
        lyr = LayerMultPrecCov.constructDiag(
            n=n,
            non_zero_idx_pairs=[(0,0),(1,0),(1,1)],
            init_diag_val=10.0
            )

        batch_size = 3
        x_in = np.full(
            shape=(batch_size,n,n),
            fill_value=np.array([[0.1,0.0],[0.0,0.1]])
        )

        x_out = lyr(x_in)
        assert_equal_arrs(x_out[0], np.eye(n))
    
    def test_save(self):

        n=2
        lyr = LayerMultPrecCov.constructDiag(
            n=n,
            non_zero_idx_pairs=[(0,0),(1,0),(1,1)],
            init_diag_val=10.0
            )

        batch_size = 3
        x_in = np.full(
            shape=(batch_size,n,n),
            fill_value=np.array([[0.1,0.0],[0.0,0.1]])
        )

        save_load_layer(lyr,x_in)

        model = ModelID(mult_lyr=lyr)

        save_load_model(model,x_in)

    def test_id_n2(self):

        inputs = InputsID(
            n=2,
            non_zero_idx_pairs=[(0,0),(1,0),(1,1)],
            target_cov_mat=np.array([[0.1,0.0],[0.0,0.1]])
            )
        inputs.report()

        results = solve_id(inputs)
        results.report()

        identity_check = np.dot(results.learned_prec_mat,results.learned_cov_mat)
        assert_equal_arrs(identity_check, np.eye(inputs.n))

    def test_id_n3(self):

        inputs = InputsID(
            n=3,
            non_zero_idx_pairs=[(0,0),(1,0),(1,1),(2,1),(2,2)],
            target_cov_mat=np.array([
                [10.0,5.0,2.0],
                [5.0,20.0,4.0],
                [2.0,4.0,30.0]
                ]),
            epochs=100,
            learning_rate=0.001
            )
        inputs.report()

        results = solve_id(inputs)
        results.report()

        identity_check = np.dot(results.learned_prec_mat,results.learned_cov_mat)
        assert_equal_arrs(identity_check, np.eye(inputs.n))

    def test_id_n6(self):

        n = 6
        non_zero_idx_pairs = random_non_zero_idx_pairs(n)
        cov_mat = random_cov_mat(n)

        inputs = InputsID(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            target_cov_mat=cov_mat,
            epochs=100,
            learning_rate=0.001
            )
        inputs.report()

        results = solve_id(inputs)
        results.report()

        identity_check = np.dot(results.learned_prec_mat,results.learned_cov_mat)
        assert_equal_arrs(identity_check, np.eye(inputs.n))