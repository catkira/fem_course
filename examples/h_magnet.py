import numpy as np
import time
import os
import sys
import inspect
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from formulation import *

import contextlib
import ctypes
from ctypes.util import find_library

# Prioritize hand-compiled OpenBLAS library over version in /usr/lib/
# from Ubuntu repos
try_paths = ['/opt/OpenBLAS/lib/libopenblas.so',
             '/lib/libopenblas.so',
             '/usr/lib/libopenblas.so.0',
             '/root/.venv/python3.10/lib/python3.10/site-packages/petsc/lib/libopenblas.so',
             find_library('openblas')]
openblas_lib = None
for libpath in try_paths:
    try:
        openblas_lib = ctypes.cdll.LoadLibrary(libpath)
        break
    except OSError:
        continue
if openblas_lib is None:
    raise EnvironmentError('Could not locate an OpenBLAS shared library', 2)


def set_num_threads(n):
    """Set the current number of threads used by the OpenBLAS server."""
    openblas_lib.openblas_set_num_threads(int(n))


# At the time of writing these symbols were very new:
# https://github.com/xianyi/OpenBLAS/commit/65a847c
try:
    openblas_lib.openblas_get_num_threads()
    def get_num_threads():
        """Get the current number of threads used by the OpenBLAS server."""
        return openblas_lib.openblas_get_num_threads()
except AttributeError:
    def get_num_threads():
        """Dummy function (symbol not present in %s), returns -1."""
        return -1
    pass

try:
    openblas_lib.openblas_get_num_procs()
    def get_num_procs():
        """Get the total number of physical processors"""
        return openblas_lib.openblas_get_num_procs()
except AttributeError:
    def get_num_procs():
        """Dummy function (symbol not present), returns -1."""
        return -1
    pass


@contextlib.contextmanager
def num_threads(n):
    """Temporarily changes the number of OpenBLAS threads.

    Example usage:

        print("Before: {}".format(get_num_threads()))
        with num_threads(n):
            print("In thread context: {}".format(get_num_threads()))
        print("After: {}".format(get_num_threads()))
    """
    old_n = get_num_threads()
    set_num_threads(n)
    try:
        yield
    finally:
        set_num_threads(old_n)

def run_h_magnet(verify=False, dirichlet='soft', gauge=True, legacy=False):
    print(f"number of threads {get_num_threads()}")
    #set_num_threads(8)
    loadMesh("examples/h_magnet.msh")
    mu0 = 4*np.pi*1e-7
    mur_frame = 1000
    b_r_magnet = 1.5    
    # regions
    magnet = 0
    frame = 1
    air = 2
    inf = 3

    start = time.time()
    nu = Parameter()
    nu.set(frame, 1/(mu0*mur_frame))
    nu.set([magnet, air], 1/mu0)
    #storeInVTK(mu, "mu.vtk")
    
    br = Parameter(3)
    br.set(magnet, [0, 0, b_r_magnet])
    br.set([frame, air], [0, 0, 0])
    hr = Parameter(3)
    hr.set(magnet, [0, 0, b_r_magnet/mu0])
    hr.set([frame, air], [0, 0, 0])
    #storeInVTK(br, "br.vtk")

    volumeRegion = Region([magnet, frame, air])
    boundaryRegion = Region([inf])

    field = FieldHCurl([magnet, frame, air])
    if gauge:
        if True:
            spanningtree = st.spanningtree(excludedRegions=[inf]) # regions with Dirichlet BCs need to be excluded
            spanningtree.write("h_magnet_spanntree.pos")
        else:
            spanningtree = st.spanningtree(create=False)
            spanningtree.load("h_magnet_spanntree_SL.pos")
            spanningtree.write("h_magnet_spanntree_imported.pos")
        field.setGauge(spanningtree)
    if dirichlet == 'soft':
        alpha = Parameter()
        alpha.set(inf, 1e9) # Dirichlet BC
        K = stiffnessMatrixCurl(field, nu, volumeRegion, legacy=legacy)
        B = massMatrixCurl(field, alpha, boundaryRegion, verify=verify)
        rhs = fluxRhsCurl(field, hr, volumeRegion)
        A = K+B
    else:
        field.setDirichlet([inf])  # this has to come before any assembly!
        K = stiffnessMatrixCurl(field, nu, volumeRegion, legacy=legacy)
        #is_symmetric(K)
        rhs = fluxRhsCurl(field, hr, volumeRegion)
        A = K
    stop = time.time()
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")
    print(f'max(rhs) = {max(rhs)}')

    if False:
        fig, axs = plt.subplots(2)
        fig.suptitle('python')
        indexptr = A.indptr
        nnzs = np.empty(len(indexptr)-1)
        for i in range(len(indexptr)-1):
            nnzs[i] = indexptr[i+1] - indexptr[i]
        axs[0].hist(nnzs, bins=range(30))

        axs[1].hist(A.data, bins=(np.arange(31)-15)/15*np.max(A.data))
        plt.show()

    # A.nnz = 1324368 but it should be 1299676 !!!
    solve(A, rhs, 'petsc')
    u = field.solution
    print(f'max(u) = {max(u)}')
    #storeInVTK(u, "h_magnetCurl_u.vtk", writePointData=True)
    b = field.curl(u, dim=3)
    storeInVTK(b, "h_magnetCurl_b.vtk")
    print(f'b_max = {max(np.linalg.norm(b,axis=1)):.4f}')
    if field.isGauged():
        assert(abs(max(np.linalg.norm(b,axis=1)) - 3.1892) < 2e-3)
    else:
        assert(abs(max(np.linalg.norm(b,axis=1)) - 2.9294) < 2e-3)

if __name__ == "__main__":
    run_h_magnet(dirichlet='hard', gauge=True, legacy=False)
