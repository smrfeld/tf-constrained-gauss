import numpy as np
import pandas as pd
from dataclasses import astuple
from typing import List, Tuple

def random_non_zero_idx_pairs(n: int) -> List[Tuple[int,int]]:
    """Generate a random list of non-zero index pairs. The number of index pairs is less than n choose 2 (max).
    The indices are lower triangular.

    Args:
        n (int): Size of the matrix

    Returns:
        List[Tuple[int,int]]: List of index pairs in the lower triangular matrix.
    """
    
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

# Random realistic cov mat following Madar 2015 
# Stat Probab Lett. 2015 August 1; 103: 142–147. doi:10.1016/j.spl.2015.03.014.
def random_cov_mat(n: int, unit_diag: bool, diag_scale_std_dev: float = 0.1) -> np.array:
    """Generate a random realistic covariance matrix following
    Madar 2015
    Stat Probab Lett. 2015 August 1; 103: 142–147. doi:10.1016/j.spl.2015.03.014.

    Args:
        n (int): Size of matrix
        unit_diag (bool): Whether the diagonal should be all 1 (True), otherwise the diagonal is scaled as 
            described by diag_scale_std_dev
        diag_scale_std_dev (float): If unit_diag is False, a scaling matrix is constructed as 
            A = diag(randomNormal(mean=1, std_dev=diag_scale_std_dev)). The covariance matrix is transformed as A.cov.Transpose[A]
            corresponding to scaling the data matrix Z by A.Z. Default = 0.1.

    Returns:
        np.array: Covariance matrix
    """
    
    # Step 1
    u = np.random.rand(n)
    
    # Sort descending
    u = -np.sort(-u)

    l = np.zeros(shape=(n,n))
    l[0,0] = 1.0
    l[1,1] = np.sqrt(u[1])
    for j in range(2,n):
        l[j,j] = np.sqrt(u[j]/u[j-1])

    # Step 2
    um = np.zeros(shape=(n,n))
    # Original idx: 2 to n-1 inclusive = 2 to n
    # Zero idx: 1 to n
    for j in range(1,n):
        um[0,j] = 1.0
        um[j,j] = pow(l[j,j],2)

        # No additional random variables = j-2 (original)
        # So j-1 in zero indexed version
        if j-1 > 0:
            # Random decreasing
            um_extra = np.random.uniform(low=l[j,j],high=1.0,size=(j-1))
            um_extra = - np.sort(-um_extra)

            # Assign
            um[1:j,j] = um_extra

        # Original idx: 1 to j-1 inclusive = 1 to j
        # Zero idx: 0 to j
        for i in range(0,j):
            l[j,i] = np.sqrt(um[i,j] - um[i+1,j])

    # Step 3
    for i in range(0,n):
        for j in range(i+1,n):
            bij = np.random.binomial(n=n,p=0.5)
            l[j,i] *= pow(-1,bij)
    
    # Step 4
    cov_mat = np.dot(l, np.transpose(l))

    # Scaling (optional)
    if not unit_diag:

        # Scaling matrix
        a_diag = np.random.normal(
            loc=1.0, 
            scale=diag_scale_std_dev, 
            size=n
            )
        a = np.diag(a_diag)

        # Scale data matrix z linearly by a.z
        # Covariance scales by cov(a.z) = a.cov(z).transpose(a)
        cov_mat = np.dot(a, np.dot(cov_mat, np.transpose(a)))

    return cov_mat

def check_symmetric(a, rtol=1e-05, atol=1e-08) -> bool:
    """Check a matrix is symmetric

    Args:
        a ([type]): Matrix
        rtol ([type], optional): Tol. Defaults to 1e-05.
        atol ([type], optional): Tol. Defaults to 1e-08.

    Returns:
        bool: True if symmetric
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def convert_mat_non_zero_to_mat(n: int, non_zero_idx_pairs: List[Tuple[int,int]], mat_non_zero: np.array) -> np.array:
    """Convert list of matrice's non-zero elements into the matrix

    Args:
        n (int): Size of matrix
        non_zero_idx_pairs (List[Tuple[int,int]]): List of non-zero index pairs in the matrix
        mat_non_zero (np.array): Corresponding non-zero matrix elements

    Returns:
        np.array: nxn matrix with specified non-zero elements
    """
    mat = np.zeros((n,n))
    for i,pair in enumerate(non_zero_idx_pairs):
        mat[pair[0],pair[1]] = mat_non_zero[i]
        mat[pair[1],pair[0]] = mat_non_zero[i]
    return mat

def convert_mat_non_zero_to_inv_mat(n: int, non_zero_idx_pairs: List[Tuple[int,int]], mat_non_zero: np.array) -> np.array:
    """Convert list of matrice's non-zero elements into the matrice's inverse 

    Args:
        n (int): Size of matrix
        non_zero_idx_pairs (List[Tuple[int,int]]): List of non-zero index pairs in the matrix
        mat_non_zero (np.array): Corresponding non-zero matrix elements

    Returns:
        np.array: Inverse of the nxn matrix
    """
    mat = convert_mat_non_zero_to_mat(
        n=n,
        non_zero_idx_pairs=non_zero_idx_pairs,
        mat_non_zero=mat_non_zero
        )
    inv_mat = np.linalg.inv(mat)
    return inv_mat

def convert_mat_non_zero_to_inv_mat_non_zero(n: int, non_zero_idx_pairs: List[Tuple[int,int]], mat_non_zero: np.array) -> np.array:
    """Convert list of matrice's non-zero elements into the matrice's inverse's list of elements

    Args:
        n (int): Size of matrix
        non_zero_idx_pairs (List[Tuple[int,int]]): List of non-zero index pairs in the matrix
        mat_non_zero (np.array): Corresponding non-zero matrix elements

    Returns:
        np.array: List of elements corresponding to indices non_zero_idx_pairs in the inverse of the matrix which has non-zero elements given by mat_non_zero
    """
    inv_mat = convert_mat_non_zero_to_inv_mat(
        n=n,
        non_zero_idx_pairs=non_zero_idx_pairs,
        mat_non_zero=mat_non_zero
        )
    inv_mat_non_zero = convert_mat_to_mat_non_zero(
        n=n,
        non_zero_idx_pairs=non_zero_idx_pairs,
        mat=inv_mat
        )
    return inv_mat_non_zero

def convert_mat_to_mat_non_zero(n: int, non_zero_idx_pairs: List[Tuple[int,int]], mat: np.array) -> np.array:
    """Convert a matrix to it's list of non-zero elements for specified non-zero indices

    Args:
        n (int): Size of matrix
        non_zero_idx_pairs (List[Tuple[int,int]]): List of non-zero index pairs in the matrix
        mat_non_zero (np.array): nxn matrix
    
    Returns:
        np.array: List of elements specified by non_zero_idx_pairs
    """
    assert(check_symmetric(mat))

    mat_non_zero = np.zeros(len(non_zero_idx_pairs))
    for i,pair in enumerate(non_zero_idx_pairs):
        mat_non_zero[i] = mat[pair[0],pair[1]]
    return mat_non_zero

def check_non_zero_idx_pairs(n: int, non_zero_idx_pairs: List[Tuple[int,int]]):
    """Check list of non-zero index pairs is valid, i.e. lower triangular and that all diagonal elements are given

    Args:
        n (int): Size of matrix
        non_zero_idx_pairs (List[Tuple[int,int]]): List of non-zero index pairs in the matrix

    Raises:
        ValueError: If not all diagonal elements are specified
        ValueError: If some indices are not lower-triangular
    """
    for i in range(0,n):
        if not (i,i) in non_zero_idx_pairs:
            raise ValueError("All diagonal elements must be specified as non-zero.")

    for pair in non_zero_idx_pairs:
        if pair[0] < pair[1]:
            raise ValueError("Only provide lower triangular indexes.")

def convert_np_to_pd(arr_with_times: np.array, nv: int, nh: int) -> pd.DataFrame:
    """Convert a numpy array of wt, b, sig2, muh, varh_diag to a pandas dataframe with named columns

    Args:
        arr_with_times (np.array): Array of size TxN where T is the number of timepoints and 
            N is the size of (wt,b,sig2,muh,varh_diag) = (nv*nh, nv, 1, nh, nh) = nv*nh + nv + 1 + 2*nh
        nv (int): No. visible species
        nh (int): No. hidden species

    Returns:
        pd.DataFrame: Pandas data frame
    """

    # Convert to pandas
    columns = ["t"]
    for ih in range(0,nh):
        for iv in range(0,nv):
            columns += ["wt%d%d" % (ih,iv)]
    for iv in range(0,nv):
        columns += ["b%d" % iv]
    columns += ["sig2"]
    for ih in range(0,nh):
        columns += ["muh%d" % ih]
    for ih in range(0,nh):
        columns += ["varh_diag%d" % ih]

    df = pd.DataFrame(arr_with_times, columns=columns)
    return df

def normalize(vec: np.array) -> np.array:
    """Normalize 1D np arr

    Args:
        vec (np.array): 1D vec to normalize

    Returns:
        np.array: normalized
    """
    return vec / np.sqrt(np.sum(vec**2))

# Equality of np arrays in data class
# https://stackoverflow.com/a/51743960/1427316
def array_safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays. 
        Needed for checking equality with @dataclass(eq=False) decorator"""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and np.max(abs(a - b)) < 1e-8
    if isinstance(a,float) and isinstance(b,float):
        return np.max(abs(a-b)) < 1e-8
    try:
        return a == b
    except TypeError:
        return NotImplemented

def dc_eq(dc1, dc2) -> bool:
   """checks if two dataclasses which hold numpy arrays are equal. 
    Needed for checking equality with @dataclass(eq=False) decorator"""
   if dc1 is dc2:
        return True
   if dc1.__class__ is not dc2.__class__:
       return NotImplemented  # better than False
   t1 = astuple(dc1)
   t2 = astuple(dc2)
   return all(array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))