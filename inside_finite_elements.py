import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyvista as pv
import vtk
import time
import scipy as sp
import meshio
from scipy.sparse import *

from mesh import *

# unofficial python 3.10 pip wheels for vtk 
# pip install https://github.com/pyvista/pyvista-wheels/raw/main/vtk-9.1.0.dev0-cp310-cp310-win_amd64.whl
# pip install https://github.com/pyvista/pyvista-wheels/raw/main/vtk-9.1.0.dev0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

def globalCoordinate(G, t, xi):
    B, x1 = transformationJacobian(G, t)
    return x1 + B @ xi    

def localCoordinate(G, t, x):
    B, x1 = transformationJacobian(G, t)
    xi,_,_,_ = np.linalg.lstsq(B, x-x1, rcond=None)
    return xi

def shapeFunctionGradients():
    return np.array([[-1, -1],
        [1, 0],
        [0, 1]])

def shapeFunctionValues(xi):
    return [1, 0, 0] + shapeFunctionGradients() @ xi

def plotShapeFunctions():
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    grid = np.meshgrid(x,y)
    coords = np.array([grid[0].flatten(), grid[1].flatten()]).T
    coordsMask = [coords[i][0] + coords[i][1] <= 1 for i in range(0,coords.shape[0])]
    triangleCoords = coords[coordsMask]
    val = np.zeros([triangleCoords.shape[0],3])
    for i in range (triangleCoords.shape[0]):
        val[i] = shapeFunctionValues([triangleCoords[i][0], triangleCoords[i][1]])

    if False:
        fig = plt.figure(figsize =(14, 9))
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.plot_trisurf(triangleCoords[:,0],triangleCoords[:,1],val[:,0])
        ax.azim = -90
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.plot_trisurf(triangleCoords[:,0],triangleCoords[:,1],val[:,1])
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.plot_trisurf(triangleCoords[:,0],triangleCoords[:,1],val[:,2])
        plt.show()
    else:
        # from plotly.subplots import make_subplots
        # fig = make_subplots(rows=1, cols=1)
        fig = go.Figure()

        data = np.ones(triangleCoords.shape[0]) - triangleCoords[:,0] - triangleCoords[:,1]
        fig.add_trace(go.Mesh3d(x=triangleCoords[:,0], y=triangleCoords[:,1], z=data, color='green',opacity=0.90))
        fig.add_trace(go.Mesh3d(x=triangleCoords[:,0], y=triangleCoords[:,1], z=val[:,1], color='blue',opacity=0.90))
        fig.add_trace(go.Mesh3d(x=triangleCoords[:,0], y=triangleCoords[:,1], z=val[:,2], color='red',opacity=0.90))
        #fig.add_trace(go.Mesh3d(x=triangleCoords[:,0], y=triangleCoords[:,1], z=np.zeros(triangleCoords.shape[0]), color='gray'))
        fig.update_layout(
            scene = dict(
                xaxis = dict(nticks=4, range=[0,1]),
                            yaxis = dict(nticks=4, range=[0,1]),
                            zaxis = dict(nticks=4, range=[0,1]),),
            width=1000,
            height=1000,
            margin=dict(r=10, l=10, b=10, t=10))        
        fig.show()

def stiffnessMatrix(sigmas):
    Grads = shapeFunctionGradients()
    # area of ref triangle is 0.5, integrands are constant within integral
    B_11 = 0.5 * Grads @ np.array([[1,0],[0,0]]) @ Grads.T 
    B_12 = 0.5 * Grads @ np.array([[0,1],[0,0]]) @ Grads.T
    B_21 = 0.5 * Grads @ np.array([[0,0],[1,0]]) @ Grads.T
    B_22 = 0.5 * Grads @ np.array([[0,0],[0,1]]) @ Grads.T

    n = numberOfVertices()
    K = np.zeros([n,n])
    #K = csr_Tensor((n,n))
    P_T = np.zeros([n,3])
    for triangleIndex, triangle in enumerate(mesh()['pt']):
        jac,_ = transformationJacobian(triangleIndex)
        detJac = np.abs(np.linalg.det(jac))
        if len(sigmas.shape) == 1:
            gamma1 = sigmas[triangleIndex]*1/detJac*np.dot(jac[:,1],jac[:,1])
            gamma2 = -sigmas[triangleIndex]*1/detJac*np.dot(jac[:,0],jac[:,1])
            gamma3 = -sigmas[triangleIndex]*1/detJac*np.dot(jac[:,1],jac[:,0])
            gamma4 = sigmas[triangleIndex]*1/detJac*np.dot(jac[:,0],jac[:,0])
        else:
            invJac = np.linalg.inv(jac)
            sigma_dash = invJac @ sigmas[triangleIndex] @ invJac.T * detJac
            gamma1 = sigma_dash[0,0] 
            gamma2 = sigma_dash[1,0]
            gamma3 = sigma_dash[0,1]
            gamma4 = sigma_dash[1,1]
        K_T = gamma1*B_11 + gamma2*B_12 + gamma3*B_21 + gamma4*B_22
        K[np.ix_(triangle[:],triangle[:])] = K[np.ix_(triangle[:],triangle[:])] + K_T
    return K


def massMatrix(rhos):
    Grads = shapeFunctionGradients()
    Mm = 1/24 * np.array([[2,1,1],
                        [1,2,1],
                        [1,1,2]])
    n = numberOfVertices()
    M = np.zeros([n,n])
    for triangleIndex, triangle in enumerate(mesh()['pt']):
        detJac = np.abs(np.linalg.det(transformationJacobian(triangleIndex)[0]))
        M_T = rhos[triangleIndex]*detJac*Mm
        M[np.ix_(triangle[:],triangle[:])] = M[np.ix_(triangle[:],triangle[:])] + M_T
    return M

def boundaryMassMatrix(alphas):
    Grads = shapeFunctionGradients()
    Bb = 1/6 * np.array([[2,1],
                        [1,2]])
    r = numberOfBoundaryEdges()
    n = numberOfVertices()
    B = np.zeros([n,n]) 
    for edgeCount, edgeIndex in enumerate(mesh()['eb']):
        ps = mesh()['pe'][edgeIndex]
        detJac = np.abs(np.linalg.norm(mesh()['xp'][ps[0]] - mesh()['xp'][ps[1]]))
        B_T = alphas[edgeCount]*detJac*Bb
        B[np.ix_(ps[:],ps[:])] = B[np.ix_(ps[:],ps[:])] + B_T
    return B

def storePotentialInVTK(u,filename):
    n = numberOfVertices()    
    m = numberOfTriangles()    
    points = np.hstack([mesh()['xp'], np.zeros((n,1))]) # add z coordinate
    cells = (np.hstack([(3*np.ones((m,1))), mesh()['pt']])).ravel().astype(np.int64)
    celltypes = np.empty(m, np.uint8)
    celltypes[:] = vtk.VTK_TRIANGLE    
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid.point_data["u"] = u
    grid.save(filename) 

def storeFluxInVTK(u,sigmas,filename):
    n = numberOfVertices()    
    m = numberOfTriangles()    
    points = np.hstack([mesh()['xp'], np.zeros((n,1))]) # add z coordinate
    cells = (np.hstack([(3*np.ones((m,1))), mesh()['pt']])).ravel().astype(np.int64)
    celltypes = np.empty(m, np.uint8)
    celltypes[:] = vtk.VTK_TRIANGLE    
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid.point_data["u"] = u
    grid = grid.compute_derivative(scalars='u', gradient='velocity')
    v = grid.get_array('velocity')    
    # Problem: compute_derivative returns gradients for every point, 
    # but we need gradients for every triangle, because sigmas are indexed by triangles
    flux = np.zeros((len(v),3))
    for i in range(len(v)):
        # rough fix: find a triangle thas includes point i
        # this is super slow though
        for triangleIndex, triangle in enumerate(mesh()['pt']):
            if triangle[0] == i or triangle[1] == i or triangle[2] == i:
                sigmaIndex = triangleIndex
                break
        flux[i][0:2] = sigmas[sigmaIndex] @ v[i][0:2]
    grid.point_data["flux"] = flux
    grid.save(filename) 

def solve(A, b, method='np'):
    start = time.time()
    if method == 'sparse':
        from scipy.sparse.linalg import inv    
        A = csc_matrix(A)
        u = inv(A) @ b
    elif method == 'petsc':
        from petsc4py import PETSc
        A = PETSc.Mat().create()  
        n = numberOfVertices()              
        A.setSizes([n**3, n**3])
        A.setType('python')

        # TODO complete code
        ksp = PETSc.KSP().create()
        pc = ksp.getPC()
        ksp.setType('cg')
        pc.setType('none')        
    elif method == 'np':
        u = np.linalg.inv(A) @ b
    else:
        print("unknown method")
        abort()
    stop = time.time()
    print(f'solved in {stop - start:.2f} s')    
    return u

def bookExample1():
    # example from book page 33
    n = numberOfVertices()
    m = numberOfTriangles()
    r = numberOfBoundaryEdges()
    sigmas = np.ones(m)
    rhos = np.ones(m)
    alphas = 1e9*np.ones(r)  # dirichlet BC
    f = np.ones(n)

    K = stiffnessMatrix(sigmas)
    M = massMatrix(rhos)
    B = boundaryMassMatrix(alphas)
    b = M @ f
    A = K+B
    u = solve(A,b)
    print(f'u_max = {max(u):.4f}')
    assert(abs(max(u) - 0.0732) < 1e-3)

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.plot_trisurf(G['xp'][:,0], G['xp'][:,1], u)
    #plt.show()

    storePotentialInVTK(u,"example1.vtk")

def bookExample2(scalarSigma, anisotropicInclusion=False):
    # example from book page 34
    n = numberOfVertices()
    m = numberOfTriangles()
    r = numberOfBoundaryEdges()
    if scalarSigma:
        sigmas = np.zeros(m)
        sigmaTensor1 = 1e-3
        sigmaTensor2 = 1
    else:
        sigmas = np.zeros((m,2,2))
        sigmaTensor2 = np.eye(2)        
        if anisotropicInclusion:
            sigmaTensor1 = np.array([[1.0001, 0.9999],
                                    [0.9999, 1.0001]])
        else:
            sigmaTensor1 = 1e-3*np.eye(2)
    for t in range(m):
        cog = np.sum(mesh()['xp'][mesh()['pt'][t,:],:],0)/3
        if 1<cog[0] and cog[0]<2 and 1<cog[1] and cog[1]<2:
            sigmas[t] = sigmaTensor1
        elif 3<cog[0] and cog[0]<4 and 2<cog[1] and cog[1]<3:
            sigmas[t] = sigmaTensor1
        else:
            sigmas[t] = sigmaTensor2
    alphas = np.zeros(r) 
    for e in range(r):
        cog = np.sum(mesh()['xp'][mesh()['pe'][mesh()['eb'][e],:],:],0)/2
        if abs(cog[0]-5) < 1e-6: 
            alphas[e] = 1e9 # Dirichlet BC
        elif abs(cog[0]) < 1e-6:
            alphas[e] = 1e-9 # Neumann BC
        else:
            alphas[e] = 0 # natural Neumann BC
    pd = np.zeros(n)
    for i in range(n):
        x = mesh()['xp'][i,:]
        if (abs(x[0]-5) < 1e-6):
            pd[i] = 4-x[1] # Dirichlet BC
        elif abs(x[0] < 1e-6):
            pd[i] = -1e9 # Neumann BC

    start = time.time()
    K = stiffnessMatrix(sigmas)
    B = boundaryMassMatrix(alphas)
    b = B @ pd
    A = K+B
    stop = time.time()    
    print(f'assembled in {stop - start:.2f} s')        
    u = solve(A,b)
    print(f'u_max = {max(u):.4f}')    
    assert(abs(max(u) - 4) < 1e-3)
    if anisotropicInclusion:
        storeFluxInVTK(u,sigmas,"example2_anisotropicInclusions.vtk")
    else:
        if scalarSigma:
            storePotentialInVTK(u,"example2_scalar_isotropicInclusions.vtk")
        else:
            storePotentialInVTK(u,"example2_tensor_isotropicInclusions.vtk")
  

def main():
    if False:
        G = {}
        # store point coordinates 'xp'
        G['xp'] = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]])
        # store the points which make up an element in 'pt'
        G['pt'] = np.array([
            [0, 1, 2],
            [0, 2, 3]])        
        print("number of Vertices(G) = " + str(numberOfVertices(G)))
        print("number of Triangles(G) = " + str(numberOfTriangles(G)))
        xi = np.array([0, 1])
        p = globalCoordinate(G, 0, xi)
        print(f'point ({xi[0]:d}, {xi[1]:d}) of ref triangle transformed to global triangle = ({p[0]:d}, {p[1]:d})')
        xi = localCoordinate(G, 0, p)
        print(f'point ({p[0]:d}, {p[1]:d}) of global triangle transformed to ref triangle = ({xi[0]:f}, {xi[1]:f})')
    
    #plotShapeFunctions()
    loadMesh("air_box_2d.msh")
    # G = rectangularCriss(50,50)
    # printEdgesofTriangle(G,1)
    # plotMesh(G)

    bookExample1()
    # scale mesh to size [0,5] x [0,4]
    mesh()['xp'][:,0] = mesh()['xp'][:,0]*5
    mesh()['xp'][:,1] = mesh()['xp'][:,1]*4

    loadMesh("example2.msh")

    bookExample2(True)
    bookExample2(False)
    bookExample2(False, True)

if __name__ == "__main__":
    main()