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

    # regions
    coil = 1
    tube = 2
    air = 3
    coilskin = 4
    tubeskin = 5
    vin = 6
    vout = 7
    domainboundary = 8

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

    fieldA1 = FieldHCurl([coil, tube, air])  # 1st harmonic inphase component
    fieldA2 = FieldHCurl([coil, tube, air])  # 1st harmonic quadrature component
    fieldV1 = FieldH1([coil, tube])
    fieldV2 = FieldH1([coil, tube])
    if gauge:
        spanningtree = st.spanningtree([domainboundary, vin, vout])
        spanningtree.write("inductionheating_spanntree.pos")
        fieldA1.setGauge(spanningtree)
        fieldA2.setGauge(spanningtree)

    fieldA1.setDirichlet([domainboundary])
    fieldA2.setDirichlet([domainboundary])
    fieldV1.setDirichlet([vout])
    fieldV2.setDirichlet([vout])
    
    if True:
        alpha = Parameter()
        alpha.set([vin], 1e9)
        VinRegion = Region([vin])
        B = massMatrix(fieldV1, alpha, VinRegion)
        vinElements = fieldV1.getElements(region=VinRegion)
        pd = np.zeros(countAllFreeDofs())
        pd[np.unique(vinElements.ravel())] = 1
        rhs = B @ pd
    else:
        # inhomogeneous Dirichlet BCs are not yet implemented!
        fieldV.setDirichlet([vin], 1) # WIP
    
    K_A1 = stiffnessMatrixCurl(fieldA1, nu, volumeRegion)
    K_A2 = stiffnessMatrixCurl(fieldA2, nu, volumeRegion)
    # magdyn += integral(conductor, sigma*dt(dof(a))*tf(a))
    K_A1A2 = matrix_dtDofA_tfA(fieldA1, fieldA2, sigma, conductorRegion, 50)  # untested
    K_V1 = stiffnessMatrix(fieldV1, sigma, conductorRegion)
    K_V2 = stiffnessMatrix(fieldV2, sigma, conductorRegion)
    
    # TODO: coupling terms
    # magdyn += integral(conductor, sigma*grad(dof(v))*tf(a))
    K_V_A = matrix_gradDofV_tfA(fieldV1, fieldA1, sigma, conductorRegion)  # WIP
    # magdyn += integral(conductor, sigma*dt(dof(a))*grad(tf(v)))

    A = K_V1 + K_V2 + K_A1 + K_A2
    A += K_V_A
    # A += K_A1A2
    stop = time.time()
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")       
    print(f'max(rhs) = {max(rhs)}')
    solve(A, rhs, 'petsc')    

    # TODO: check fieldA2.dt()
    E1 = -fieldV1.grad(fieldV1.solution, dim=3)
    E1 -= fieldA2.dt(fieldA2.solution, dim=3, frequency=50)[0:len(E1),:] # HACK
    E2 = -fieldV2.grad(fieldV2.solution, dim=3)
    E2 -= fieldA1.dt(fieldA1.solution, dim=3, frequency=50)[0:len(E2),:] # HACK
    #
    # TODO: implement storeInVTK on regions
    #
    storeInVTK(E1, "inductionheating_E1.vtk", field=fieldV1)    
    storeInVTK(E2, "inductionheating_E2.vtk", field=fieldV2)    
    B1 = fieldA1.curl(fieldA1.solution, dim=3)
    B2 = fieldA2.curl(fieldA2.solution, dim=3)
    storeInVTK(B1, "inductionheating_B1.vtk", field=fieldA1)    
    storeInVTK(B2, "inductionheating_B2.vtk", field=fieldA2)
    #print(f'max(u) = {max(u)}')
    #storeInVTK(u, "magmesh_u.vtk", writePointData=True)    
    #b = field.curl(u, dim=3)
    #storeInVTK(b, "magmesh_b.vtk")
    #print(f'b_max = {max(np.linalg.norm(b,axis=1)):.8f}')    
    #assert(abs(max(np.linalg.norm(b,axis=1)) - 2.8919e-8) < 2e-3)


if __name__ == "__main__":
    run_inductionheating(gauge=True)
