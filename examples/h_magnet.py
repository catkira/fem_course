import numpy as np
import time
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from formulation import *


def run_h_magnet(verify=False, dirichlet='soft', gauge=True, legacy=False):
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

    volumeRegion = Region()
    volumeRegion.append([magnet, frame, air])

    boundaryRegion = Region()
    boundaryRegion.append(inf)

    field = FieldHCurl()
    if gauge:
        spanningtree = st.spanningtree(excludedRegions=[inf]) # regions with Dirichlet BCs need to be excluded!
        spanningtree.write("h_magnet_spanntree.pos")
        setGauge(spanningtree)
    if dirichlet == 'soft':
        alpha = Parameter()
        alpha.set(inf, 1e9) # Dirichlet BC
        K = stiffnessMatrixCurl(field, nu, volumeRegion, legacy=legacy)
        B = massMatrixCurl(field, alpha, boundaryRegion, verify=verify)
        rhs = fluxRhsCurl(field, hr, volumeRegion)    
        A = K+B    
    else:
        setDirichlet([inf])  # this has to come before any assembly!!!
        K = stiffnessMatrixCurl(field, nu, volumeRegion, legacy=legacy)
        rhs = fluxRhsCurl(field, hr, volumeRegion)    
        A = K
    stop = time.time()
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")       
    print(f'max(rhs) = {max(rhs)}')
    u = solve(A, rhs, 'petsc')    
    print(f'max(u) = {max(u)}')
    storeInVTK(u, "h_magnetCurl_u.vtk", writePointData=True)    
    b = field.curl(u, dim=3)
    storeInVTK(b, "h_magnetCurl_b.vtk")
    print(f'b_max = {max(np.linalg.norm(b,axis=1)):.4f}')    
    if isGauged():
        assert(abs(max(np.linalg.norm(b,axis=1)) - 3.1931) < 2e-3)
    else:
        assert(abs(max(np.linalg.norm(b,axis=1)) - 2.9374) < 2e-3)


if __name__ == "__main__":
    run_h_magnet(dirichlet='hard', gauge=True)
