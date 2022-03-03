import numpy as np
import time
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from formulation import *

def run_inductionheating(verify=False, dirichlet='soft', gauge=True):
    loadMesh("examples/inductionheating.msh")
    mu0 = 4*np.pi*1e-7
    mur_frame = 1000
    # regions
    coil = 1
    tube = 2
    air = 3
    coilskin = 4
    tubeskin = 5
    vin = 6
    vout = 7
    domainboundary = 8

    #conductor = selectunion([coil,tube])
    #wholedomain = selectunion([coil, tube, air, coilskin, tubeskin, vin, vout, domainboundary])
    #domainboundary = selectunion([domainboundary, vin, vout])

    start = time.time()
    nu = Parameter()
    nu.set([coil, tube, air], 1/mu0)
    #storeInVTK(nu,"nu.vtk")

    sigma = Parameter()
    sigma.set(coil, 6e7)
    sigma.set(tube, 3e7)
    sigma.set(air, 0)  # this is only needed for storeInVTK
    #storeInVTK(sigma, "inductionheating_sigma.vtk")

    volumeRegion = Region([coil, tube, air])
    conductorRegion = Region([coil, tube])

    fieldA = FieldHCurl([coil, tube, air, coilskin, tubeskin, vin, vout, domainboundary])
    fieldV = FieldH1([coil, tube])
    if gauge:
        spanningtree = st.spanningtree([domainboundary, vin, vout])
        spanningtree.write("inductionheating_spanntree.pos")
        fieldA.setGauge(spanningtree)

    fieldA.setDirichlet([domainboundary])
    fieldV.setDirichlet([vout])
    
    if True:
        alpha = Parameter()
        alpha.set([vin], 1e9)
        VinRegion = Region([vin])
        B = massMatrix(fieldV, alpha, VinRegion)
        vinElements = fieldV.getElements(region=VinRegion)
        pd = np.zeros(countAllFreeDofs())
        pd[np.unique(vinElements.ravel())] = 1
        rhs = B @ pd
    else:
        # inhomogeneous Dirichlet BCs are not yet implemented!
        fieldV.setDirichlet([vin], 1) # WIP
    
    K_A = stiffnessMatrixCurl(fieldA, nu, volumeRegion)
    K_V = stiffnessMatrix(fieldV, sigma, conductorRegion)
    
    # TODO: coupling terms
    # setFundamentalFrequency(50)
    # magdyn += integral(conductor, sigma*dt(dof(a))*tf(a))
    # magdyn += integral(conductor, sigma*dt(dof(a))*grad(tf(v)))

    A = K_V + K_A
    stop = time.time()
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")       
    print(f'max(rhs) = {max(rhs)}')
    solve(A, rhs, 'petsc')    
    u = fieldV.solution
    E = -fieldV.grad(u, dim=3)
    #
    # TODO: implement storeInVTK on regions
    #
    storeInVTK(E, "inductionheating_E.vtk", field=fieldV)    
    u = fieldA.solution
    B = fieldA.curl(u, dim=3)
    storeInVTK(B, "inductionheating_B.vtk", field=fieldA)    
    #print(f'max(u) = {max(u)}')
    #storeInVTK(u, "magmesh_u.vtk", writePointData=True)    
    #b = field.curl(u, dim=3)
    #storeInVTK(b, "magmesh_b.vtk")
    #print(f'b_max = {max(np.linalg.norm(b,axis=1)):.8f}')    
    #assert(abs(max(np.linalg.norm(b,axis=1)) - 2.8919e-8) < 2e-3)


if __name__ == "__main__":
    run_inductionheating(gauge=True)
