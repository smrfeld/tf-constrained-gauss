import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="physDBD")
def unit_mat_sym(n: int, i: int, j: int):
    """Construct the symmetric unit matrix of size nxn
       1 at (i,j) AND (j,i)
       0 elsewhere

    Args:
        n (int): Size of square matrix
        i (int): First idx
        j (int): Second idx

    Returns:
        tf.Constant: Matrix that is 1 at (i,j) AND (j,i) and 0 everywhere else
    """
    idx = i * n + j
    one_hot = tf.one_hot(indices=idx,depth=n*n, dtype='float32')
    
    if i != j:
        idx = j * n + i
        one_hot += tf.one_hot(indices=idx,depth=n*n, dtype='float32')

    return tf.reshape(one_hot,shape=(n,n))

@tf.keras.utils.register_keras_serializable(package="physDBD")
def unit_mat(n: int, i: int, j: int):
    """Construct the non-symmetric unit matrix of size nxn
       1 at (i,j) ONLY
       0 elsewhere

    Args:
        n (int): Size of square matrix
        i (int): First idx
        j (int): Second idx

    Returns:
        tf.Constant: Matrix that is 1 at (i,j) ONLY and 0 everywhere else
    """
    idx = i * n + j
    one_hot = tf.one_hot(indices=idx,depth=n*n, dtype='float32')
    return tf.reshape(one_hot,shape=(n,n))