import numpy as np
import time
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from formulation import *

def run_magnet_in_room():
    loadMesh("examples/magnet_in_room.msh")
    mu0 = 4*np.pi*1e-7
    mur_wall = 1000
    b_r_magnet = 1.5    
    # regions
    magnet = 1
    insideAir = 2
    wall = 3
    outsideAir = 4
    inf = 5

    start = time.time()
    mu = Parameter()
    mu.set(wall, mu0*mur_wall)
    mu.set([magnet, insideAir, outsideAir], mu0)
    #storeInVTK(mu, "mu.vtk")
    
    br = Parameter(2)
    br.set(magnet, [b_r_magnet, 0])
    br.set([wall, insideAir, outsideAir], [0, 0])
    #storeInVTK(br, "br.vtk")

    alpha = Parameter()
    alpha.set(inf, 1e9) # Dirichlet BC

    surfaceRegion = Region()
    surfaceRegion.append([wall, magnet, insideAir, outsideAir])

    boundaryRegion = Region()
    boundaryRegion.append(inf)

    field = FieldH1()
    K = stiffnessMatrix(field, mu, surfaceRegion)
    B = massMatrix(field, alpha, boundaryRegion)
    rhs = fluxRhs(field, br, surfaceRegion)
    b = rhs
    A = K+B
    stop = time.time()    
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")        
    solve(A, b, 'petsc')
    u = field.solution
    storeInVTK(u,"magnet_in_room_phi.vtk", writePointData=True)
    m = numberOfTriangles()   
    h = -field.grad(u)
    storeInVTK(h,"magnet_in_room_h.vtk")
    mus = mu.getValues()  
    brs = np.column_stack([br.getValues(), np.zeros(m)])
    b = np.column_stack([mus,mus,mus])*h + brs  # this is a bit ugly
    storeInVTK(b,"magnet_in_room_b.vtk")
    print(f'b_max = {max(np.linalg.norm(b,axis=1)):.4f}')    
    assert(abs(max(np.linalg.norm(b,axis=1)) - 1.6330) < 1e-3)

if __name__ == "__main__":
    run_magnet_in_room()
