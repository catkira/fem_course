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
from field import FieldH1


def run_bookExample2(scalarSigma, anisotropicInclusion=False, method="petsc", mesh="msh"):
    if mesh == "msh":
        loadMesh("examples/example2.msh")
        computeEdges2d()
        computeBoundary()
    else:
        rectangularCriss(50, 50)
        computeEdges2d()
        computeBoundary()
        getMesh()["xp"][:, 0] = getMesh()["xp"][:, 0] * 5
        getMesh()["xp"][:, 1] = getMesh()["xp"][:, 1] * 4
    # example from book page 34
    n = numberOfVertices()
    m = numberOfTriangles()
    r = numberOfBoundaryEdges()
    start = time.time()
    if scalarSigma:
        sigmas = np.zeros(m)
        sigmaTensor1 = 1e-3
        sigmaTensor2 = 1
    else:
        sigmas = np.zeros((m, 2, 2))
        sigmaTensor2 = np.eye(2)
        if anisotropicInclusion:
            sigmaTensor1 = np.array([[1.0001, 0.9999], [0.9999, 1.0001]])
        else:
            sigmaTensor1 = 1e-3 * np.eye(2)
    for t in range(m):
        cog = np.sum(getMesh()["xp"][getMesh()["pt"][t, :], :], 0) / 3
        if 1 < cog[0] and cog[0] < 2 and 1 < cog[1] and cog[1] < 2:
            sigmas[t] = sigmaTensor1
        elif 3 < cog[0] and cog[0] < 4 and 2 < cog[1] and cog[1] < 3:
            sigmas[t] = sigmaTensor1
        else:
            sigmas[t] = sigmaTensor2
    alphas = np.zeros(r)
    for e in range(r):
        cog = np.sum(getMesh()["xp"][getMesh()["pe"][getMesh()["eb"][e], :], :], 0) / 2
        if abs(cog[0] - 5) < 1e-6:
            alphas[e] = 1e9  # Dirichlet BC
        elif abs(cog[0]) < 1e-6:
            alphas[e] = 1e-9  # Neumann BC
        else:
            alphas[e] = 0  # natural Neumann BC
    pd = np.zeros(n)
    for i in range(n):
        x = getMesh()["xp"][i, :]
        if abs(x[0] - 5) < 1e-6:
            pd[i] = 4 - x[1]  # Dirichlet BC
        elif abs(x[0] < 1e-6):
            pd[i] = -1e9  # Neumann BC
    stop = time.time()
    print(f"parameters prepared in {stop - start:.2f} s")
    start = time.time()
    field = FieldH1()
    K = stiffnessMatrix(field, sigmas)
    B = boundaryMassMatrix(field, alphas)
    b = B @ pd
    A = K + B
    stop = time.time()
    print(f"assembled in {stop - start:.2f} s")
    solve(A, b, method)
    u = field.solution
    print(f"u_max = {max(u):.4f}")
    assert abs(max(u) - 4) < 1e-3
    if anisotropicInclusion:
        # storeFluxInVTK(u,sigmas,"example2_anisotropicInclusions.vtk")
        pass
    else:
        if scalarSigma:
            storeInVTK(u, "example2_scalar_isotropicInclusions.vtk", writePointData=True)
        else:
            storeInVTK(u, "example2_tensor_isotropicInclusions.vtk", writePointData=True)


def run_bookExample2Parameter(scalarSigma, anisotropicInclusion=False, method="mumps"):
    loadMesh("examples/example2.msh")
    # example from book page 34
    n = numberOfVertices()
    # regions
    incl1 = 0
    incl2 = 1
    air = 2
    infBottom = 3
    infLeft = 4
    infRight = 5
    infTop = 6

    start = time.time()
    sigma = Parameter()
    sigma.set([incl1, incl2], 1e-3)
    sigma.set(air, 1)

    alpha = Parameter()
    alpha.set([infBottom, infTop], 0)
    alpha.set(infLeft, 1e-9)  # Neumann BC
    alpha.set(infRight, 1e9)  # Dirichlet BC

    pd = np.zeros(n)
    for i in range(n):
        x = getMesh()["xp"][i, :]
        if abs(x[0] - 5) < 1e-6:
            pd[i] = 4 - x[1]  # Dirichlet BC
        elif abs(x[0] < 1e-6):
            pd[i] = -1e9  # Neumann BC
    stop = time.time()
    print(f"parameters prepared in {stop - start:.2f} s")
    start = time.time()

    surfaceRegion = Region()
    surfaceRegion.append([incl1, incl2, air])

    boundaryRegion = Region()
    boundaryRegion.append([infBottom, infTop, infLeft, infRight])

    field = FieldH1()
    K = stiffnessMatrix(field, sigma, surfaceRegion)
    B = massMatrix(field, alpha, boundaryRegion)
    b = B @ pd
    A = K + B
    stop = time.time()
    print(f"assembled in {stop - start:.2f} s")
    solve(A, b, method)
    u = field.solution
    if anisotropicInclusion:
        # storeFluxInVTK(u,sigma.triangleValues,"example2_anisotropicInclusions_p.vtk")
        pass
    else:
        if scalarSigma:
            storeInVTK(u, "example2_scalar_isotropicInclusions_p.vtk", writePointData=True)
        else:
            storeInVTK(u, "example2_tensor_isotropicInclusions_p.vtk", writePointData=True)
    print(f"u_max = {max(u):.4f}")
    assert abs(max(u) - 4) < 1e-3


if __name__ == "__main__":
    run_bookExample2Parameter(scalarSigma=True)
    run_bookExample2(False, anisotropicInclusion=True, method='mumps')
    run_bookExample2(False, anisotropicInclusion=True, method='mumps', mesh='criss')
