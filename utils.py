import mesh as m
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def gaussData(order, elementDim):
    if order == 1:
        if elementDim == 2: # triangles
            gfs = np.array([1/2])
            if m.mesh['problemDimension'] == 2:
                gps = np.array([[1/3, 1/3]])
            elif m.mesh['problemDimension'] == 3:
                gps = np.array([[1/3, 1/3, 0]])
        elif elementDim == 3: # tetraeder
            gfs = np.array([1/6])
            gps = np.array([[1/4, 1/4, 1/4]])
    elif order == 2:
        if elementDim == 2: # triangles
            gfs = np.array([1/6, 1/6, 1/6])
            if m.mesh['problemDimension'] == 2:
                gps = np.array([[1/6, 1/6],
                                [2/3, 1/6],
                                [1/6, 2/3]])
            elif m.mesh['problemDimension'] == 3:
                gps = np.array([[1/6, 1/6, 0],
                                [2/3, 1/6, 0],
                                [1/6, 2/3, 0]])
        elif elementDim == 3: # tetrahedron
            gfs = np.array([0.041666666666666699, 0.041666666666666699, 0.041666666666666699, 0.041666666666666699])
            if m.mesh['problemDimension'] == 2:
                assert True # this should never happen
            elif m.mesh['problemDimension'] == 3:
                gps = np.array([[0.13819660112500001, 0.13819660112500001, 0.13819660112500001],
                                [0.58541019662500005, 0.13819660112500001, 0.13819660112500001],
                                [0.13819660112500001, 0.58541019662500005, 0.13819660112500001],
                                [0.58541019662500005, 0.13819660112500001, 0.13819660112500001]])
    return gfs,gps
