from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Union

def setup_dataset(toy_ds: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Reads and preprocesses dataset.
    
    Args:
        toy_ds: `bool` indicating whether to load a toy dataset with a 4x3 Matrix instead of laoding the Netflix dataset.
        
    Returns:
        Two numpy ndarrays of shape (m,n). The first array corresponds to the rating matrix R and contains user-item ratings
          in the range [1, 5]. Zero entries in this matrix indicate that the user has not rated the movie. The second matrix, I,
          indicates whether a user has rated a movie. This means, r_{ij} = 1 iff user i has rated movie j.    
    """
    if toy_ds:
        R = [[5,4,0],
             [5,5,1],
             [1,0,5],
             [1,1,4]]
        R = np.array(R,dtype=np.float64)
    else:
        num_users = 943
        num_movies = 404
        R = np.load(Path(__file__).parent.joinpath('netflix-reduced.npy'))

    I = (R > 0).astype('int')
    return R, I


def visualize_sparsity_pattern(matrix: np.array) -> plt.figure:
    """This function returns a visualization of the sparsity pattern of the provided `matrix`.
    
    Args:
        matrix: `np.ndarray` of shape (m, n)
        
    Returns:
        A `plt.figure` visualizing the sparsity pattern. A black pixel means that the corresponding 
          entry in the matrix is non-zero.
    """
    print('A black pixel means that the corresponding entry in the matrix is non-zero')
    return plt.spy(matrix)
