import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyvista as pv
import vtk
import time
import sys
import pkg_resources
from scipy.sparse import *
from parameter import parameter
from region import region

if 'petsc4py' in pkg_resources.working_set.by_key:
    hasPetsc = True
    import petsc4py
    petsc4py.init(sys.argv)        
    from petsc4py import PETSc
else:
    print("Warning: no petsc4py found, solving will be very slow!")

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

# integral grad(u) * sigma * grad(tf(u)) 
def stiffnessMatrix(sigmas, region=[]):
    Grads = shapeFunctionGradients()
    # area of ref triangle is 0.5, integrands are constant within integral
    B_11 = 0.5 * Grads @ np.array([[1,0],[0,0]]) @ Grads.T 
    B_12 = 0.5 * Grads @ np.array([[0,1],[0,0]]) @ Grads.T
    B_21 = 0.5 * Grads @ np.array([[0,0],[1,0]]) @ Grads.T
    B_22 = 0.5 * Grads @ np.array([[0,0],[0,1]]) @ Grads.T

    n = numberOfVertices()
    m = numberOfTriangles()
    #K_Ts = np.zeros([m,3,3])
    rows = np.zeros(m*9)
    cols = np.zeros(m*9)
    data = np.zeros(m*9)
    if region == []:
        elements = mesh()['pt']
    else:
        elements = region.getElements()
        sigmas = sigmas.getValues(region)
    for triangleIndex, triangle in enumerate(elements):
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
        range = np.arange(start=triangleIndex*9, stop=triangleIndex*9+9)            
        rows[range] = np.tile(triangle[:],3).astype(np.int64)
        cols[range] = np.repeat(triangle[:],3).astype(np.int64)
        data[range] = (gamma1*B_11 + gamma2*B_12 + gamma3*B_21 + gamma4*B_22).ravel()
        #K_Ts[triangleIndex] = gamma1*B_11 + gamma2*B_12 + gamma3*B_21 + gamma4*B_22
        #K[np.ix_(triangle[:],triangle[:])] = K[np.ix_(triangle[:],triangle[:])] + K_T        
    K = csr_matrix((data, (rows, cols)), shape=[n,n]) 
    return K

# integral br * grad(tf(u))
def fluxRhs(br, region=[]):
    Grads = shapeFunctionGradients()    
    if region == []:
        elements = mesh()['pt']
    else:
        elements = region.getElements()
        br = br.getValues(region)
    n = numberOfVertices()
    rhs = np.zeros(n)
    for triangleIndex, triangle in enumerate(elements):
        jac,_ = transformationJacobian(triangleIndex)
        invJac = np.linalg.inv(jac)
        detJac = np.abs(np.linalg.det(jac))        
        temp = 0.5 * invJac.T @ Grads.T * detJac
        rhs[triangle[0]] = rhs[triangle[0]] + np.dot(br[triangleIndex], temp.T[0])
        rhs[triangle[1]] = rhs[triangle[1]] + np.dot(br[triangleIndex], temp.T[1])
        rhs[triangle[2]] = rhs[triangle[2]] + np.dot(br[triangleIndex], temp.T[2])
    return rhs

# integral rho * u * tf(u)
def massMatrix(rhos, region=[]):
    Grads = shapeFunctionGradients()
    Mm = 1/24 * np.array([[2,1,1],
                        [1,2,1],
                        [1,1,2]])
    n = numberOfVertices()
    m = numberOfTriangles()
    rows = np.zeros(m*9)
    cols = np.zeros(m*9)
    data = np.zeros(m*9)    
    if region == []:
        elements = mesh()['pt']
    else:
        elements = region.getElements()
        rhos = rhos.getValues(region)        
    for triangleIndex, triangle in enumerate(elements):
        detJac = np.abs(np.linalg.det(transformationJacobian(triangleIndex)[0]))
        range = np.arange(start=triangleIndex*9, stop=triangleIndex*9+9)
        rows[range] = np.tile(triangle,3).astype(np.int64)
        cols[range] = np.repeat(triangle,3).astype(np.int64)
        data[range] = (rhos[triangleIndex]*detJac*Mm).ravel()
        #M_T = rhos[triangleIndex]*detJac*Mm
        #M[np.ix_(triangle[:],triangle[:])] = M[np.ix_(triangle[:],triangle[:])] + M_T
    M = csr_matrix((data, (rows, cols)), shape=[n,n]) 
    return M    

def boundaryMassMatrix(alphas, region=[]):
    Grads = shapeFunctionGradients()
    Bb = 1/6 * np.array([[2,1],
                        [1,2]])
    r = numberOfBoundaryEdges()
    n = numberOfVertices()
    #B_Ts = np.zeros([r,2,2])
    rows = np.zeros(r*4)
    cols = np.zeros(r*4)
    data = np.zeros(r*4)
    if region == []:
        elements = mesh()['pe'][mesh()['eb']]
    else:
        elements = region.getElements()
        alphas = alphas.getValues(region)
    for edgeCount, ps in enumerate(elements):
#    for edgeCount, edgeIndex in enumerate(mesh()['eb']):
#        ps = mesh()['pe'][edgeIndex]
        detJac = np.abs(np.linalg.norm(mesh()['xp'][ps[0]] - mesh()['xp'][ps[1]]))
        range = np.arange(start=edgeCount*4, stop=edgeCount*4+4)        
        rows[range] = np.tile(ps[:],2).astype(np.int64)
        cols[range] = np.repeat(ps[:],2).astype(np.int64)
        data[range] = (alphas[edgeCount]*detJac*Bb).ravel()
        #B_Ts[edgeCount] = alphas[edgeCount]*detJac*Bb
        #B[np.ix_(ps[:],ps[:])] = B[np.ix_(ps[:],ps[:])] + B_T
    B = csr_matrix((data, (rows, cols)), shape=[n,n]) 
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
    # rough fix: create average sigma for each point, some pointSigmas might be assigned multiple times
    pointSigmas = np.zeros((n,2,2))
    for i, triangle in enumerate(mesh()['pt']):
        pointSigmas[triangle[0]] = sigmas[i]
        pointSigmas[triangle[1]] = sigmas[i]
        pointSigmas[triangle[2]] = sigmas[i]
    flux = np.zeros((len(v),3))
    for i in range(len(v)):
        flux[i][0:2] = pointSigmas[i] @ v[i][0:2]
    grid.point_data["flux"] = flux
    grid.save(filename) 

def solve(A, b, method='np'):
    start = time.time()
    if method == 'sparse':
        from scipy.sparse.linalg import inv    
        A = csc_matrix(A)
        u = inv(A) @ b
    elif method == 'petsc':
        if not hasPetsc:
            print("petsc is not available on this system")
            sys.exit()            
        n = numberOfVertices()    
        csr_mat=csr_matrix(A)
        Ap = PETSc.Mat().createAIJ(size=(n, n),  csr=(csr_mat.indptr, csr_mat.indices, csr_mat.data))        
        Ap.setUp()
        Ap.assemblyBegin()
        Ap.assemblyEnd()
        bp = PETSc.Vec().createSeq(n) 
        bp.setValues(range(n),b)        
        up = PETSc.Vec().createSeq(n)        
        ksp = PETSc.KSP().create()
        ksp.setOperators(Ap)        
        ksp.setFromOptions()
        print(f'Solving with: {ksp.getType():s}')
        ksp.solve(bp, up)
        print(f'Converged in {ksp.getIterationNumber():d} iterations.')
        u = np.array(up)
    elif method == 'np':
        u = np.linalg.inv(A.toarray()) @ b
    else:
        print("unknown method")
        sys.exit()
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

def bookExample2(scalarSigma, anisotropicInclusion=False, method='petsc'):
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
    stop = time.time()    
    print(f'parameters prepared in {stop - start:.2f} s')        
    start = time.time()
    K = stiffnessMatrix(sigmas)
    B = boundaryMassMatrix(alphas)
    b = B @ pd
    A = K+B
    stop = time.time()    
    print(f'assembled in {stop - start:.2f} s')        
    u = solve(A, b, method)
    print(f'u_max = {max(u):.4f}')    
    assert(abs(max(u) - 4) < 1e-3)
    if anisotropicInclusion:
        storeFluxInVTK(u,sigmas,"example2_anisotropicInclusions.vtk")
    else:
        if scalarSigma:
            storePotentialInVTK(u,"example2_scalar_isotropicInclusions.vtk")
        else:
            storePotentialInVTK(u,"example2_tensor_isotropicInclusions.vtk")
  

def bookExample2Parameter(scalarSigma, anisotropicInclusion=False, method='petsc'):
    # example from book page 34
    n = numberOfVertices()
    m = numberOfTriangles()
    r = numberOfBoundaryEdges()
    # regions
    incl1 = 0
    incl2 = 1
    air = 2
    infBottom = 3
    infLeft = 4
    infRight = 5
    infTop = 6

    start = time.time()    
    sigma = parameter()
    sigma.set(incl1, 1e-3)
    sigma.set(incl2, 1e-3)
    sigma.set(air, 1)

    alpha = parameter()
    alpha.set(infBottom, 0)
    alpha.set(infLeft, 1e-9) # Neumann BC
    alpha.set(infRight, 1e9) # Dirichlet BC
    alpha.set(infTop, 0)

    pd = np.zeros(n)
    for i in range(n):
        x = mesh()['xp'][i,:]
        if (abs(x[0]-5) < 1e-6):
            pd[i] = 4-x[1] # Dirichlet BC
        elif abs(x[0] < 1e-6):
            pd[i] = -1e9 # Neumann BC
    stop = time.time()    
    print(f'parameters prepared in {stop - start:.2f} s')        
    start = time.time()

    surfaceRegion = region()
    surfaceRegion.append(incl1)
    surfaceRegion.append(incl2)
    surfaceRegion.append(air)
    surfaceRegion.calculateElements()

    boundaryRegion = region()
    boundaryRegion.append(infBottom)
    boundaryRegion.append(infTop)
    boundaryRegion.append(infLeft)
    boundaryRegion.append(infRight)
    boundaryRegion.calculateElements()

    K = stiffnessMatrix(sigma, surfaceRegion)    
    B = boundaryMassMatrix(alpha, boundaryRegion)
    b = B @ pd
    A = K+B
    stop = time.time()    
    print(f'assembled in {stop - start:.2f} s')        
    u = solve(A, b, method)
    print(f'u_max = {max(u):.4f}')    
    assert(abs(max(u) - 4) < 1e-3)
    if anisotropicInclusion:
        storeFluxInVTK(u,sigma.triangleValues,"example2_anisotropicInclusions_p.vtk")
    else:
        if scalarSigma:
            storePotentialInVTK(u,"example2_scalar_isotropicInclusions_p.vtk")
        else:
            storePotentialInVTK(u,"example2_tensor_isotropicInclusions_p.vtk")

def exampleMagnetInRoom():
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
    mu = parameter()
    mu.set(wall, mu0*mur_wall)
    mu.set(magnet, mu0)
    mu.set(insideAir, mu0)
    mu.set(outsideAir, mu0)
    mu.set(inf, mu0)

    br = parameter(2)
    br.set(magnet, [-b_r_magnet, 0])
    br.set(wall, [0, 0])
    br.set(insideAir, [0, 0])
    br.set(outsideAir, [0, 0])
    br.set(inf, [0, 0])

    alpha = parameter()
    alpha.set(inf, 1e9) # Dirichlet BC

    surfaceRegion = region()
    surfaceRegion.append(wall)
    surfaceRegion.append(magnet)
    surfaceRegion.append(insideAir)
    surfaceRegion.append(outsideAir)
    surfaceRegion.calculateElements()    

    boundaryRegion = region()
    boundaryRegion.append(inf)
    boundaryRegion.calculateElements()

    K = stiffnessMatrix(mu, surfaceRegion)
    B = boundaryMassMatrix(alpha, boundaryRegion)
    rhs = fluxRhs(br, surfaceRegion)
    stop = time.time()    
    b = rhs
    A = K+B
    print(f'assembled in {stop - start:.2f} s')        
    u = solve(A, b, 'petsc')
    print(f'u_max = {max(u):.4f}')
    storePotentialInVTK(u,"magnet_in_room.vtk")                      

def exampleHMagnet():
    loadMesh("examples/h_magnet.msh")

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
    loadMesh("examples/air_box_2d.msh")
    # rectangularCriss(50,50)
    # printEdgesofTriangle(G,1)
    # plotMesh(G)

    bookExample1()
    
    # scale mesh to size [0,5] x [0,4]
    #rectangularCriss(50,50)
    #mesh()['xp'][:,0] = mesh()['xp'][:,0]*5
    #mesh()['xp'][:,1] = mesh()['xp'][:,1]*4

    loadMesh("examples/example2.msh")
    bookExample2Parameter(True, anisotropicInclusion=False, method='petsc')
    #bookExample2(False, 'petsc')
    #bookExample2(False, True, 'petsc')
    # exampleHMagnet()
    exampleMagnetInRoom()
    print('finished')

if __name__ == "__main__":
    main()