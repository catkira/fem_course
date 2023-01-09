import numpy as np
import time
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from formulation import *
from mesh import *
from ioHelper import storeInVTK
from field import FieldHCurl
import spanningtree as st

def run_magmesh(verify=False, dirichlet='soft', coarse=True, gauge=True):
    if coarse:
        loadMesh("examples/magmesh_coarse.msh")
    else:
        loadMesh("examples/magmesh.msh")
    mu0 = 4*np.pi*1e-7
    mur_frame = 1000
    # regions
    conductor = 1
    shield = 2
    air = 3
    inf = 4

    start = time.time()
    nu = Parameter()
    nu.set(shield, 1/(mu0*mur_frame))
    nu.set([conductor, air], 1/mu0)
    #storeInVTK(nu,"nu.vtk")

    currentDensity = Parameter(3)
    currentDensity.set(conductor, [0, 0, 1])
    currentDensity.set([shield, air], [0, 0, 0])

    volumeRegion = Region([conductor, shield, air])
    boundaryRegion = Region([inf])

    field = FieldHCurl([conductor, shield, air])
    if gauge:
        spanningtree = st.spanningtree([inf])
        spanningtree.write("magmesh_spanntree.pos")
        field.setGauge(spanningtree)
    if dirichlet == 'soft':
        alpha = Parameter()
        alpha.set(inf, 1e9) # Dirichlet BC
        B = massMatrixCurl(field, field, alpha, boundaryRegion, verify=verify)
        K = stiffnessMatrixCurl(field, nu, volumeRegion, legacy = False)
        rhs = loadRhs(field, currentDensity, volumeRegion)    
        A = K+B    
    else:
        field.setDirichlet([inf])
        K = stiffnessMatrixCurl(field, nu, volumeRegion)
        rhs = loadRhs(field, currentDensity, volumeRegion)    
        A = K
    stop = time.time()
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")       
    print(f'max(rhs) = {max(rhs)}')
    solve(A, rhs, 'mumps')
    u = field.solution
    print(f'max(u) = {max(u)}')
    #storeInVTK(u, "magmesh_u.vtk", writePointData=True)
    b = field.curl(u, dim=3)
    storeInVTK(b, "magmesh_b.vtk")
    print(f'b_max = {max(np.linalg.norm(b.elementValues.values, axis=1)):.8f}')
    assert abs(max(np.linalg.norm(b.elementValues.values ,axis=1)) - 2.8919e-8) < 2e-3

if __name__ == "__main__":
    run_magmesh(dirichlet='soft', gauge=True, coarse=False)
