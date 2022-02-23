import numpy as np
import time
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from formulation import *


def run_h_magnet_octant(vectorized=True, legacy=False, dirichlet='soft'):
    loadMesh("examples/h_magnet_octant.msh")
    mu0 = 4*np.pi*1e-7
    mur_frame = 1000
    b_r_magnet = 1.5    
    # regions
    magnet = 1
    frame = 2
    air = 3
    inf = 4
    innerXZBoundary = 5
    innerYZBoundary = 6
    innerXYBoundary = 7
    magnetXYBoundary = 8

    start = time.time()
    mu = Parameter()
    mu.set(frame, mu0*mur_frame)
    mu.set([magnet, air], mu0)
    #storeInVTK(mu, "mu.vtk")
    
    br = Parameter(3)
    br.set(magnet, [0, 0, b_r_magnet])
    br.set([frame, air], [0, 0, 0])
    #storeInVTK(br, "br.vtk")    

    volumeRegion = Region()
    volumeRegion.append([magnet, frame, air])

    boundaryRegion = Region()
    boundaryRegion.append([inf, innerXYBoundary, magnetXYBoundary])

    field = FieldH1([magnet, frame, air, inf])
    if dirichlet == 'soft':
        K = stiffnessMatrix(field, mu, volumeRegion, vectorized=vectorized, legacy=legacy)
        alpha = Parameter()
        alpha.set([inf, innerXYBoundary, magnetXYBoundary], 1e9) # Dirichlet BC
        B = massMatrix(field, alpha, boundaryRegion, vectorized=vectorized)
        rhs = fluxRhs(field, br, volumeRegion, vectorized=vectorized)    
        A = K+B    
    elif dirichlet == 'hard':
        field.setDirichlet([inf, innerXYBoundary, magnetXYBoundary])        
        K = stiffnessMatrix(field, mu, volumeRegion, vectorized=vectorized, legacy=legacy)
        rhs = fluxRhs(field, br, volumeRegion, vectorized=vectorized)    
        A = K
    stop = time.time()
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")       
    solve(A, rhs, 'petsc')    
    u = field.solution
    storeInVTK(u, "h_magnet_octant_u.vtk", writePointData=True)    
    h = -field.grad(u, dim=3)
    storeInVTK(h, "h_magnet_octant_h.vtk")
    mus = mu.getValues()  
    brs = br.getValues()
    b = np.column_stack([mus,mus,mus])*h + brs  # this is a bit ugly
    storeInVTK(b, "h_magnet_octant_b.vtk")        
    print(f'b_max = {max(np.linalg.norm(b, axis=1)):.4f}')    
    assert(abs(max(np.linalg.norm(b, axis=1)) - 3.3684) < 1e-3)

if __name__ == "__main__":
    run_h_magnet_octant(dirichlet='hard', vectorized=True, legacy=False)
