import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyvista as pv
import vtk
import time
import scipy as sp
import meshio
from scipy.sparse import *

# unofficial python 3.10 pip wheels for vtk 
# pip install https://github.com/pyvista/pyvista-wheels/raw/main/vtk-9.1.0.dev0-cp310-cp310-win_amd64.whl
# pip install https://github.com/pyvista/pyvista-wheels/raw/main/vtk-9.1.0.dev0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

def numberOfVertices(G):
    return G['xp'].shape[0]

def numberOfTriangles(G):
    return G['pt'].shape[0]

# triangleIndices start with 1, because 0 is used for 'no triangle'
# G['pe'] translates from points to edges, every row contains two points which form an edge
# G['te'] translates from edges to triangles, every row contains the triangles to which the edge belongs
def computeEdges(G):
    # first triangle is stored in triu, second triangle in tril
    E = lil_matrix((len(G['xp']),len(G['xp']))) 
    for triangleIndex, triangle in enumerate(G['pt'], start=1):
        for numEdge in range(3):
            lowIndex = triangle[numEdge]
            numEdge = numEdge + 1 if numEdge < 2 else 0
            highIndex = triangle[numEdge]
            if lowIndex > highIndex:
                lowIndex, highIndex = highIndex, lowIndex
            # indices in E are shifted up by 1 because 0 means 'no triangle' (for border edges)
            if E[lowIndex, highIndex] == 0:
                E[lowIndex, highIndex] = triangleIndex
            else:
                E[highIndex, lowIndex] = triangleIndex
    [p1, p2, t1] = find(triu(E))
    G['pe'] = np.vstack([p1, p2]).T
    numEdges = G['pe'].shape[0]
    G['te'] = np.zeros([numEdges,2], dtype=np.int64)
    G['te'][:,0] = t1
    for edgeIndex in range(numEdges):
        ps = G['pe'][edgeIndex]
        G['te'][edgeIndex,1] = E[ps[1],ps[0]]
    return G

def computeBoundary(G):
    G['eb'] = []
    for edgeIndex, edge in enumerate(G['te']):
        if edge[1] == 0:
            G['eb'].append(edgeIndex)
    G['eb'] = np.array(G['eb'])
    return G

def numberOfEdges(G):
    return G['pe'].shape[0]

def numberOfBoundaryEdges(G):
    return G['eb'].shape[0]

def transformationJacobian(G, t):
    ps = G['pt'][t,:]
    x1 = G['xp'][ps[0],:]
    B = np.array([G['xp'][ps[1],:]-x1, G['xp'][ps[2],:]-x1]).T
    return (B, x1)

# t is id of triangle
# xi is point in reference triangle
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

def plotMesh(G):
    plt.scatter(G['xp'][:,0],G['xp'][:,1])
    for edge in G['pt']:
        plt.plot([G['xp'][edge[0],0], G['xp'][edge[1],0], G['xp'][edge[2],0], G['xp'][edge[0],0]],[G['xp'][edge[0],1], G['xp'][edge[1],1], G['xp'][edge[2],1], G['xp'][edge[0],1]])
    x = []
    y = []
    for boundaryEdge in G['eb']:
        [p0, p1] = G['pe'][boundaryEdge]
        x = [G['xp'][p0][0], G['xp'][p1][0]]
        y = [G['xp'][p0][1], G['xp'][p1][1]]
        plt.plot(x,y,'k-')
    plt.show()

def rectangularCriss(w, h):
    start = time.time()
    G = dict()
    G['xp'] = np.zeros([w*h,2])
    G['pt'] = []
    for y in range (0,h):
        for x in range (0,w):
            G['xp'][x+w*y] = [x,y]
            if x < (w-1) and y < (h-1):
                G['pt'].append([x+w*y, (x+1)+w*y, x+w*(y+1)])
                G['pt'].append([(x+1)+w*y, (x+1)+w*(y+1), x+w*(y+1)])
    G['pt'] = np.array(G['pt'])
    G['xp'] = G['xp']*1/np.max([G['xp'][:,0], G['xp'][:,1]]) # scale max dimension of grid to 1
    G = computeEdges(G)
    G = computeBoundary(G)
    stop = time.time()
    print(f'loaded mesh in {stop - start:.2f} s')    
    print(f'mesh contains {numberOfTriangles(G):d} triangles')
    print(f'mesh contains {numberOfVertices(G):d} vertices')
    print(f'mesh contains {numberOfEdges(G):d} edges')
    print(f'mesh contains {numberOfBoundaryEdges(G):d} boundaryEdges')    
    return G

def printEdgesofTriangle(G, triangleIndex):
    G['pt'][triangleIndex]
    for edgeIndex in range(numberOfEdges(G)):
        if G['te'][edgeIndex][0] == triangleIndex:
            print(f'edge {edgeIndex} belongs to triangle {triangleIndex}')
        if G['te'][edgeIndex][1] == triangleIndex:
            print(f'edge {edgeIndex} belongs to triangle {triangleIndex}')

def stiffnessMatrix(G, sigmas):
    Grads = shapeFunctionGradients()
    # area of ref triangle is 0.5, integrands are constant within integral
    B_11 = 0.5 * Grads @ np.array([[1,0],[0,0]]) @ Grads.T 
    B_12 = 0.5 * Grads @ np.array([[0,1],[0,0]]) @ Grads.T
    B_21 = 0.5 * Grads @ np.array([[0,0],[1,0]]) @ Grads.T
    B_22 = 0.5 * Grads @ np.array([[0,0],[0,1]]) @ Grads.T

    n = numberOfVertices(G)
    K = np.zeros([n,n])
    #K = csr_Tensor((n,n))
    P_T = np.zeros([n,3])
    for triangleIndex, triangle in enumerate(G['pt']):
        jac,_ = transformationJacobian(G,triangleIndex)
        detJac = np.linalg.det(jac)
        if len(sigmas.shape) == 1:
            gamma1 = sigmas[triangleIndex]*1/detJac*np.dot(jac[:,1],jac[:,1])
            gamma2 = -sigmas[triangleIndex]*1/detJac*np.dot(jac[:,0],jac[:,1])
            gamma3 = -sigmas[triangleIndex]*1/detJac*np.dot(jac[:,1],jac[:,0])
            gamma4 = sigmas[triangleIndex]*1/detJac*np.dot(jac[:,0],jac[:,0])
        else: # not yet working
            gamma1x = sigmas[triangleIndex][0,0]*1/detJac*np.dot(jac[:,1],jac[:,1])
            gamma2x = -sigmas[triangleIndex][0,0]*1/detJac*np.dot(jac[:,0],jac[:,1])
            gamma3x = -sigmas[triangleIndex][0,0]*1/detJac*np.dot(jac[:,1],jac[:,0])
            gamma4x = sigmas[triangleIndex][0,0]*1/detJac*np.dot(jac[:,0],jac[:,0])
            invJac = np.linalg.inv(jac)
            sigma_dash = invJac @ sigmas[triangleIndex] @ invJac.T
            gamma1 = sigma_dash[0,0] * detJac
            gamma2 = sigma_dash[1,0] * detJac
            gamma3 = sigma_dash[0,1] * detJac
            gamma4 = sigma_dash[1,1] * detJac
            if False == (np.round(gamma1,5) == np.round(gamma1x,5) 
                and np.round(gamma2,5) == np.round(gamma2x,5) 
                and np.round(gamma3,5) == np.round(gamma3x,5) 
                and np.round(gamma4,5) == np.round(gamma4x,5)):
                print("error")
        K_T = gamma1*B_11 + gamma2*B_12 + gamma3*B_21 + gamma4*B_22
        K[np.ix_(triangle[:],triangle[:])] = K[np.ix_(triangle[:],triangle[:])] + K_T
    return K


def massMatrix(G, rhos):
    Grads = shapeFunctionGradients()
    Mm = 1/24 * np.array([[2,1,1],
                        [1,2,1],
                        [1,1,2]])
    n = numberOfVertices(G)
    M = np.zeros([n,n])
    for triangleIndex, triangle in enumerate(G['pt']):
        detJac = np.linalg.det(transformationJacobian(G,triangleIndex)[0])
        M_T = rhos[triangleIndex]*detJac*Mm
        M[np.ix_(triangle[:],triangle[:])] = M[np.ix_(triangle[:],triangle[:])] + M_T
    return M

def boundaryMassMatrix(G, alphas):
    Grads = shapeFunctionGradients()
    Bb = 1/6 * np.array([[2,1],
                        [1,2]])
    r = numberOfBoundaryEdges(G)
    n = numberOfVertices(G)
    B = np.zeros([n,n]) 
    for edgeCount, edgeIndex in enumerate(G['eb']):
        ps = G['pe'][edgeIndex]
        detJac = np.linalg.norm(G['xp'][ps[0]] - G['xp'][ps[1]])
        B_T = alphas[edgeCount]*detJac*Bb
        B[np.ix_(ps[:],ps[:])] = B[np.ix_(ps[:],ps[:])] + B_T
    return B

def storePotentialInVTK(G,u,filename):
    r = numberOfBoundaryEdges(G)
    n = numberOfVertices(G)    
    m = numberOfTriangles(G)    
    points = np.hstack([G['xp'], np.zeros((n,1))]) # add z coordinate
    cells = (np.hstack([(3*np.ones((m,1))), G['pt']])).ravel().astype(np.int64)
    celltypes = np.empty(m, np.uint8)
    celltypes[:] = vtk.VTK_TRIANGLE    
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid.point_data["u"] = u
    grid.save(filename) 

def bookExample1(G):
    # example from book page 33
    n = numberOfVertices(G)
    m = numberOfTriangles(G)
    r = numberOfBoundaryEdges(G)
    sigmas = np.ones(m)
    rhos = np.ones(m)
    alphas = 1e9*np.ones(r)  # dirichlet BC
    f = np.ones(n)

    K = stiffnessMatrix(G,sigmas)
    M = massMatrix(G, rhos)
    B = boundaryMassMatrix(G, alphas)
    b = M @ f
    A = K+B
    u = np.linalg.inv(A) @ b
    print(f'u_max = {max(u):.4f}')
    assert(abs(max(u) - 0.0732) < 1e-3)

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.plot_trisurf(G['xp'][:,0], G['xp'][:,1], u)
    #plt.show()

    storePotentialInVTK(G,u,"example1.vtk")

def bookExample2(G, scalarSigma):
    # example from book page 34
    # scale mesh to size [0,5] x [0,4]
    G['xp'][:,0] = G['xp'][:,0]*5
    G['xp'][:,1] = G['xp'][:,1]*4
    n = numberOfVertices(G)
    m = numberOfTriangles(G)
    r = numberOfBoundaryEdges(G)
    if scalarSigma:
        sigmas = np.zeros(m)
        sigmaTensor = 1
    else:
        sigmas = np.zeros((m,2,2))
        sigmaTensor = np.eye(2)
    for t in range(m):
        cog = np.sum(G['xp'][G['pt'][t,:],:],0)/3
        if 1<cog[0] and cog[0]<2 and 1<cog[1] and cog[1]<2:
            sigmas[t] = 1e-3*sigmaTensor
        elif 3<cog[0] and cog[0]<4 and 2<cog[1] and cog[1]<3:
            sigmas[t] = 1e-3*sigmaTensor
        else:
            sigmas[t] = 1*sigmaTensor
    alphas = np.zeros(r) 
    for e in range(r):
        cog = np.sum(G['xp'][G['pe'][G['eb'][e],:],:],0)/2
        if abs(cog[0]-5) < 1e-6: 
            alphas[e] = 1e9 # Dirichlet BC
        elif abs(cog[0]) < 1e-6:
            alphas[e] = 1e-9 # Neumann BC
        else:
            alphas[e] = 0 # natural Neumann BC
    pd = np.zeros(n)
    for i in range(n):
        x = G['xp'][i,:]
        if (abs(x[0]-5) < 1e-6):
            pd[i] = 4-x[1] # Dirichlet BC
        elif abs(x[0] < 1e-6):
            pd[i] = -1e9 # Neumann BC

    start = time.time()
    print("starting assembly")
    K = stiffnessMatrix(G, sigmas)
    B = boundaryMassMatrix(G, alphas)
    b = B @ pd
    A = K+B
    stop = time.time()
    print(f'assembly done in {stop - start:.2f} s')
    start = time.time()
    if True:
        from scipy.sparse.linalg import inv    
        A = csc_matrix(A)
        u = inv(A) @ b
    else:
        u = np.linalg.inv(A) @ b
    stop = time.time()
    print(f'solved in {stop - start:.2f} s')
    print(f'u_max = {max(u):.4f}')    
    assert(abs(max(u) - 4) < 1e-3)
    storePotentialInVTK(G,u,"example2.vtk")

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
    G = rectangularCriss(20,20)

    #printEdgesofTriangle(G,1)
    #plotMesh(G)

    bookExample1(G)
    bookExample2(G, True)
    bookExample2(G, False)


if __name__ == "__main__":
    main()