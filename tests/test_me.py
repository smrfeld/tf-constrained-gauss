from helpers_test import assert_equal_arrs, save_load_model, random_non_zero_idx_pairs, random_cov_mat
from tfConstrainedGauss import LayerPrecToCovMat, solve_me, InputsME, convert_mat_to_mat_non_zero

import numpy as np
import tensorflow as tf

class TestME:

    def test_inv_prec_to_cov_mat(self):

        n = 2
        non_zero_idx_pairs = [(0,0),(1,0),(1,1)]

        # Create layer
        lyr = LayerPrecToCovMat.constructDiag(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            init_diag_val=10.0
            )

        # Input/Output
        # NOTE: do not use batch size =1, anything else OK like 2, 1 is peculiar results
        batch_size = 2
        x_in = np.random.rand(batch_size,1)
        x_out = lyr(x_in)

        print("Outputs:", x_out)

        x_out_true = np.full(
            shape=(batch_size,2,2),
            fill_value=np.array([[0.1,0.0],[0.0,0.1]])
            )

        assert_equal_arrs(x_out, x_out_true)

        save_load_model(lyr, x_in)

    def test_me_n3(self):

        inputs = InputsME(
            n=3,
            non_zero_idx_pairs=[(0,0),(1,0),(1,1),(2,1),(2,2)],
            target_cov_mat_non_zero=np.array([10.0,5.0,20.0,4.0,30.0]),
            epochs=100,
            learning_rate=0.001
            )
        
        results = solve_me(inputs)

        results.report()

    def test_me_n6(self):

        n = 6
        
        # Non zero elements
        non_zero_idx_pairs = random_non_zero_idx_pairs(n)

        # Cov mat
        cov_mat = random_cov_mat(n)
        cov_mat_non_zero = convert_mat_to_mat_non_zero(n,non_zero_idx_pairs,cov_mat)

        inputs = InputsME(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            target_cov_mat_non_zero=cov_mat_non_zero,
            epochs=100,
            learning_rate=0.01,
            use_weighted_loss=False
            )
        inputs.report()
        
        results = solve_me(inputs)
        results.report()

    def test_me_weighted_n6(self):

        n = 6

        # Non zero elements
        non_zero_idx_pairs = random_non_zero_idx_pairs(n)

        # Cov mat
        cov_mat = random_cov_mat(n)
        cov_mat_non_zero = convert_mat_to_mat_non_zero(n,non_zero_idx_pairs,cov_mat)

        inputs = InputsME(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            target_cov_mat_non_zero=cov_mat_non_zero,
            epochs=100,
            learning_rate=0.01,
            use_weighted_loss=True
            )
        inputs.report()

        results = solve_me(inputs)
        results.report()