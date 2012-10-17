"""
Data from Hogg et al 2010; useful for testing robust regression methods
"""
import numpy as np


def fetch_hogg2010test(structured=False):
    """Fetch the Hogg et al 2010 test data
    """
    data = np.array([[1, 201, 592, 61, 9, -0.84],
                     [2, 244, 401, 25, 4, 0.31],
                     [3, 47, 583, 38, 11, 0.64],
                     [4, 287, 402, 15, 7, -0.27],
                     [5, 203, 495, 21, 5, -0.33],
                     [6, 58, 173, 15, 9, 0.67],
                     [7, 210, 479, 27, 4, -0.02],
                     [8, 202, 504, 14, 4, -0.05],
                     [9, 198, 510, 30, 11, -0.84],
                     [10, 158, 416, 16, 7, -0.69],
                     [11, 165, 393, 14, 5, 0.30],
                     [12, 201, 442, 25, 5, -0.46],
                     [13, 157, 317, 52, 5, -0.03],
                     [14, 131, 311, 16, 6, 0.50],
                     [15, 166, 400, 34, 6, 0.73],
                     [16, 160, 337, 31, 5, -0.52],
                     [17, 186, 423, 42, 9, 0.90],
                     [18, 125, 334, 26, 8, 0.40],
                     [19, 218, 533, 16, 6, -0.78],
                     [20, 146, 344, 22, 5, -0.56]])
    dtype = [("ID", np.int32),
             ("x", np.float64),
             ("y", np.float64),
             ("sigma_x", np.float64),
             ("sigma_y", np.float64),
             ("rho_xy", np.float64)]

    recarray = np.empty(data.shape[0], dtype=dtype)
    recarray['ID'] = data[:, 0]
    recarray['x'] = data[:, 1]
    recarray['y'] = data[:, 2]
    recarray['sigma_x'] = data[:, 4]
    recarray['sigma_y'] = data[:, 3]
    recarray['rho_xy'] = data[:, 5]

    return recarray
