from helpers_test import assert_equal_arrs, save_load_layer, save_load_model
from tfConstrainedGauss import LayerPrecToCovMat, solve_me, InputsME, ModelME, random_cov_mat

import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

class TestME:

    def test_random_cov_mat(self):
        cov_mat = random_cov_mat(n=5, unit_diag=True)
        print(cov_mat)

        assert(is_pos_def(cov_mat))
        assert(check_symmetric(cov_mat))

    def test_inv_prec_to_cov_mat(self):

        n = 2
        lyr = LayerPrecToCovMat.constructDiag(
            n=n,
            non_zero_idx_pairs=[(0,0),(1,0),(1,1)],
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

    def test_save(self):
        
        n = 2
        lyr = LayerPrecToCovMat.constructDiag(
            n=n,
            non_zero_idx_pairs=[(0,0),(1,0),(1,1)],
            init_diag_val=10.0
            )

        # Input/Output
        # NOTE: do not use batch size =1, anything else OK like 2, 1 is peculiar results
        batch_size = 2
        x_in = np.random.rand(batch_size,1)
        
        save_load_layer(lyr, x_in)

        model = ModelME(inv_lyr=lyr)

        save_load_model(model, x_in)

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