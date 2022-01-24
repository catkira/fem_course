import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.sparse import *


def numberOfVertices(G):
    return G['xp'].shape[0]

def numberOfTriangles(G):
    return G['pt'].shape[0]

# triangleIndices start with 1, because 0 is used for 'no triangle'
# G['pe'] translates from points to edges, every row contains two points which form an edge
# G['te'] translates from edges to triangles, every row contains the triangles to which the edge belongs
def computeEdges(G):
    # first triangle is stored in triu, second triangle in tril
    E = csr_matrix((len(G['xp']),len(G['xp']))) 
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
    for triangleIndex, triangle in enumerate(G['pt']):
        jac,_ = transformationJacobian(G,triangleIndex)
        detJac = np.linalg.det(jac)
        gamma1 = sigmas[triangleIndex]*1/detJac*np.dot(jac[:,1],jac[:,1])
        gamma2 = sigmas[triangleIndex]*1/detJac*np.dot(jac[:,0],jac[:,1])
        gamma3 = sigmas[triangleIndex]*1/detJac*np.dot(jac[:,1],jac[:,0])
        gamma4 = sigmas[triangleIndex]*1/detJac*np.dot(jac[:,0],jac[:,0])
        K_T = gamma1*B_11 + gamma2*B_12 + gamma3*B_21 + gamma4*B_22
        P_T = np.zeros([n,3])
        P_T[triangle[0],0] = 1
        P_T[triangle[1],1] = 1
        P_T[triangle[2],2] = 1
        K = K + P_T @ K_T @ P_T.T # optimize later
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
        P_T = np.zeros([n,3])
        P_T[triangle[0],0] = 1
        P_T[triangle[1],1] = 1
        P_T[triangle[2],2] = 1
        M = M + P_T @ M_T @ P_T.T  # optimize later
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
        P_T = np.zeros([n,2])
        P_T[ps[0],0] = 1
        P_T[ps[1],1] = 1
        B = B + P_T @ B_T @ P_T.T  # optimize later
    return B

def main():
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
    G = computeEdges(G)
    G = computeBoundary(G)
    print(f'mesh contains {numberOfEdges(G):d} edges')
    print(f'mesh contains {numberOfBoundaryEdges(G):d} boundaryEdges')

    printEdgesofTriangle(G,1)
    #plotMesh(G)

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

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(G['xp'][:,0], G['xp'][:,1], u)
    plt.show()


if __name__ == "__main__":
    main()