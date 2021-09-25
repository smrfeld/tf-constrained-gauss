from helpers_test import assert_equal_arrs, save_load_model, random_non_zero_idx_pairs, random_cov_mat
from tfConstrainedGauss import InputsID, solve_id

import numpy as np
import tensorflow as tf

class TestID:

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