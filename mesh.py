import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyvista as pv
import vtk
import time
import scipy as sp
import meshio
from scipy.sparse import *

mesh = dict()

def mesh():
    return mesh

def numberOfVertices():
    global mesh
    return mesh['xp'].shape[0]

def numberOfTriangles():
    global mesh
    return mesh['pt'].shape[0]

# triangleIndices start with 1, because 0 is used for 'no triangle'
# G['pe'] translates from points to edges, every row contains two points which form an edge
# G['te'] translates from edges to triangles, every row contains the triangles to which the edge belongs
def computeEdges():
    global mesh
    # first triangle is stored in triu, second triangle in tril
    E = lil_matrix((len(mesh['xp']),len(mesh['xp'])))
    for triangleIndex, triangle in enumerate(mesh['pt'], start=1):
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
    mesh['pe'] = np.vstack([p1, p2]).T
    numEdges = mesh['pe'].shape[0]
    mesh['te'] = np.zeros([numEdges,2], dtype=np.int64)
    mesh['te'][:,0] = t1
    for edgeIndex in range(numEdges):
        ps = mesh['pe'][edgeIndex]
        mesh['te'][edgeIndex,1] = E[ps[1],ps[0]]

def computeBoundary():
    global mesh
    mesh['eb'] = []
    for edgeIndex, edge in enumerate(mesh['te']):
        if edge[1] == 0:
            mesh['eb'].append(edgeIndex)
    mesh['eb'] = np.array(mesh['eb'])

def numberOfEdges():
    global mesh
    return mesh['pe'].shape[0]

def numberOfBoundaryEdges():
    global mesh
    return mesh['eb'].shape[0]

def transformationJacobian(t):
    global mesh
    ps = mesh['pt'][t,:]
    x1 = mesh['xp'][ps[0],:]
    B = np.array([mesh['xp'][ps[1],:]-x1, mesh['xp'][ps[2],:]-x1]).T
    return (B, x1)    

def plotMesh():
    global mesh
    G = mesh
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

def printEdgesofTriangle(G, triangleIndex):
    G['pt'][triangleIndex]
    for edgeIndex in range(numberOfEdges()):
        if G['te'][edgeIndex][0] == triangleIndex:
            print(f'edge {edgeIndex} belongs to triangle {triangleIndex}')
        if G['te'][edgeIndex][1] == triangleIndex:
            print(f'edge {edgeIndex} belongs to triangle {triangleIndex}')    

def rectangularCriss(w, h):
    global mesh
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
    mesh = G
    computeEdges()
    computeBoundary()
    stop = time.time()
    print(f'loaded mesh in {stop - start:.2f} s')    
    print(f'mesh contains {numberOfTriangles():d} triangles')
    print(f'mesh contains {numberOfVertices():d} vertices')
    print(f'mesh contains {numberOfEdges():d} edges')
    print(f'mesh contains {numberOfBoundaryEdges():d} boundaryEdges')    

def loadMesh(filename):
    global mesh
    start = time.time()
    meshioMesh = meshio.read(filename)
    G = dict()
    problemDimension = 2
    if problemDimension == 2:
        G['xp'] = meshioMesh.points[:,0:2] # tale only x,y coordinates
        G['pt'] = meshioMesh.cells_dict['triangle']
    mesh = G
    computeEdges()
    computeBoundary()
    stop = time.time()
    print(f'loaded mesh in {stop - start:.2f} s')    
    print(f'mesh contains {numberOfTriangles():d} triangles')
    print(f'mesh contains {numberOfVertices():d} vertices')
    print(f'mesh contains {numberOfEdges():d} edges')
    print(f'mesh contains {numberOfBoundaryEdges():d} boundaryEdges')          