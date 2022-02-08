import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyvista as pv
import vtk
import time
import scipy as sp
import meshio
import sys
from scipy.sparse import *
import region

mesh = dict()

def mesh():
    return mesh

def numberOfVertices():
    global mesh
    return mesh['xp'].shape[0]

def numberOfTriangles():
    global mesh
    return mesh['pt'].shape[0]

def numberOfTetraeders():
    global mesh
    return mesh['ptt'].shape[0]

def numberOfEdges():
    global mesh
    return mesh['ett'].shape[0]

# triangleIndices start with 1, because 0 is used for 'no triangle'
# G['pe'] translates from edges to points, every row contains two points which form an edge
# G['te'] translates from edges to triangles, every row contains the triangles to which the edge belongs
# this function is only used when the mesh is created with rectangularCriss
def computeEdges2d():
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

def computeSigns():
    global mesh
    if mesh['problemDimension'] == 2:
        tmp = mesh['pt'][:,[1,2,0]] - mesh['pt'][:,[2,0,1]]
        mesh['signs2d'] = np.multiply(tmp, 1/abs(tmp)).astype(np.int8)
    elif mesh['problemDimension'] == 3:
        tmp = mesh['ptt'][:,[0,0,0,1,2,3]] - mesh['ptt'][:,[1, 2, 3, 2, 3, 1]]
        mesh['signs3d'] = np.multiply(tmp, 1/abs(tmp)).astype(np.int8)

# computes edges for use with edge elements
def computeEdges3d():
    global mesh
    if mesh['pe'] != []:
        return   # edges are already computed
    start = time.time()
    # compute tetraeder-to-edges list
    if mesh['problemDimension'] == 3:
        if True:
            vertices3d = np.zeros((mesh['ptt'].shape[0]*6,2))
            vertices3d[0::6] = mesh['ptt'][:,[0,1]]            
            vertices3d[1::6] = mesh['ptt'][:,[0,2]]            
            vertices3d[2::6] = mesh['ptt'][:,[0,3]]            
            vertices3d[3::6] = mesh['ptt'][:,[1,2]]            
            vertices3d[4::6] = mesh['ptt'][:,[2,3]]            
            vertices3d[5::6] = mesh['ptt'][:,[3,1]]            
            vertices3d.sort(axis=1)
            _,J,I = np.unique(vertices3d, return_index=True, return_inverse=True, axis=0)
            mesh['ett'] = I.reshape(mesh['ptt'].shape[0],6)
            mesh['pe'] = vertices3d[J,:]
        if False:
            mesh['ett'] = np.zeros((numberOfTetraeders(), 6))
            mesh['pe'] = []
            E = dok_matrix((len(mesh['xp']),len(mesh['xp'])),dtype=np.int64)    # dok_matrix is faster here than lil_matrix
            edgePoints = np.array([[0,1],[1,2],[2,3],[3,0],[0,2],[1,3]])
            allEdges = mesh['ptt'][:,edgePoints]        
            allEdges.sort(axis=2)    # make sure lowest edge index is in front        
            for tetraederIndex, edges in enumerate(allEdges):
                for edgeIndex, edge in enumerate(edges):
                    storedIndex = E[edge[0],edge[1]]
                    if storedIndex == 0:
                        globalEdgeIndex = len(mesh['pe']) 
                        mesh['ett'][tetraederIndex][edgeIndex] = globalEdgeIndex
                        mesh['pe'].append(edge)
                        E[edge[0],edge[1]] = globalEdgeIndex + 1
                    else:
                        mesh['ett'][tetraederIndex][edgeIndex] = storedIndex - 1
     # compute triangle-to-edges list
        if True:
            vertices2d = np.zeros((mesh['pt'].shape[0]*6,2))
            vertices2d[0::6] = mesh['pt'][:,[1,2]]            
            vertices2d[1::6] = mesh['pt'][:,[2,0]]            
            vertices2d[2::6] = mesh['pt'][:,[0,1]]            
            vertices2d.sort(axis=1)
            verticesMixed = np.row_stack((vertices3d, vertices2d))
            _,J,I = np.unique(verticesMixed, return_index=True, return_inverse=True, axis=0)
            mesh['et'] = I[mesh['ett'].shape[0]*6:].reshape(mesh['pt'].shape[0],6)[:,0:3]
            mesh['pe'] = verticesMixed[J,:]  # there should not be much new edges from surface elements, but it can happen
        if False:      
            for triangleIndex, triangle in enumerate(mesh['pt']):
                mesh['et'] = np.zeros((numberOfTriangles(), 3))  
                for edgeIndex in range(3):
                    lowIndex = triangle[edgeIndex]
                    edgeIndex = edgeIndex + 1 if edgeIndex < 2 else 0
                    highIndex = triangle[edgeIndex]
                    if lowIndex > highIndex:
                        lowIndex, highIndex = highIndex, lowIndex
                    if E[lowIndex, highIndex] == 0:
                        print("Error: all edges should be contained in volume elements!")
                        sys.abort()
                    else:
                        mesh['et'][triangleIndex][edgeIndex] = E[highIndex, lowIndex] - 1
    stop = time.time()
    mesh['pe'] = np.array(mesh['pe'])
    numEdges = len(mesh['pe'])
    numTT = len(mesh['ptt'])
    numT = len(mesh['pt'])
    print(f'calculated {numEdges:d} edges, from {numTT:d} tetraeders and {numT:d} triangles in {stop - start:.2f} s')                       

# this function is only used when the mesh is created with rectangularCriss
def computeBoundary():
    global mesh
    if mesh['problemDimension'] == 2:
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

def dimensionOfRegion(id):
    global mesh
    if id in mesh['physical'][0]:
        return 1
    elif id in mesh['physical'][1]:
        return 2
    elif id in mesh['physical'][2]:
        return 3
    else:
        print(f'Error: region with id {id:d} not found!')
        sys.exit()

def transformationJacobians(reg = []):
    global mesh
    if reg == []:
        if mesh['problemDimension'] == 2:
            ps = mesh['pt']
        elif mesh['problemDimension'] == 3:
            ps = mesh['ptt']
    elif isinstance(reg, region.Region):
        ps = reg.getElements()
    else:
        ps = reg
    elementDim = ps.shape[1]-1
    if elementDim == 2:
        x1 = mesh['xp'][ps[:,0],:]
        if mesh['problemDimension'] == 2:
            B = np.array([mesh['xp'][ps[:,1],:]-x1, mesh['xp'][ps[:,2],:]-x1]).T
        elif mesh['problemDimension'] == 3:
            # extend jacobi matrix with a [0,0,1] vector, so that the determinant can be calculated
            B = np.array([mesh['xp'][ps[:,1],:]-x1, mesh['xp'][ps[:,2],:]-x1, np.tile([0,0,1],len(ps)).reshape((len(ps),3))]).T
    elif elementDim == 3:
        x1 = mesh['xp'][ps[:,0],:]        
        B = np.array([mesh['xp'][ps[:,1],:]-x1, mesh['xp'][ps[:,2],:]-x1, mesh['xp'][ps[:,3],:]-x1]).T
    B = B.swapaxes(0,1)
    return B

def transformationJacobian(t):
    global mesh
    dim = len(mesh['xp'][0])
    if dim == 2:
        ps = mesh['pt'][t,:]
        x1 = mesh['xp'][ps[0],:]
        B = np.array([mesh['xp'][ps[1],:]-x1, mesh['xp'][ps[2],:]-x1]).T
    elif dim == 3:
        ps = mesh['ptt'][t,:]
        x1 = mesh['xp'][ps[0],:]
        B = np.array([mesh['xp'][ps[1],:]-x1, mesh['xp'][ps[2],:]-x1, mesh['xp'][ps[3],:]-x1]).T
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

def printMeshInfo():
    global mesh
    if mesh['problemDimension'] == 2:
        print(f'mesh contains {numberOfTriangles():d} triangles, {numberOfVertices():d} vertices')
    elif mesh['problemDimension'] == 3:
        print(f'mesh contains {numberOfTetraeders():d} tetraeder, {numberOfTriangles():d} triangles, {numberOfVertices():d} vertices')

def regionDimension(id):
    global mesh
    if id in mesh['physical'][0]: # check lines
        return 1
    elif id in mesh['physical'][1]: # check triangles
        return 2
    elif id in mesh['physical'][2]: # check tetraeders
        return 3
    else:
        print(f'Error: Region with id {id:d} not found!')
        sys.exit()

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
    G['problemDimension'] = 2    
    mesh = G
    stop = time.time()
    print(f'loaded mesh in {stop - start:.2f} s')    
    printMeshInfo()

def loadMesh(filename):
    global mesh
    start = time.time()
    meshioMesh = meshio.read(filename)
    G = dict()
    if 'tetra' in meshioMesh.cells_dict:
        problemDimension = 3
    else:
        problemDimension = 2
    if problemDimension == 2 or problemDimension == 3:
        G['xp'] = meshioMesh.points[:,0:2] # take only x,y coordinates
        if 'line' in meshioMesh.cells_dict:
            G['pl'] = meshioMesh.cells_dict['line']
        G['pt'] = meshioMesh.cells_dict['triangle']
    if problemDimension == 3:
        G['xp'] = meshioMesh.points
        if 'line' in meshioMesh.cells_dict:
            G['pl'] = meshioMesh.cells_dict['line']
        if 'triangle' in meshioMesh.cells_dict:
           G['pt'] = meshioMesh.cells_dict['triangle']
        G['ptt'] = meshioMesh.cells_dict['tetra']
    #G['physical'] = meshioMesh.cell_data['gmsh:physical']
    G['physical'] = [np.empty(0), np.empty(0), np.empty(0)]
    if 'line' in meshioMesh.cell_data_dict['gmsh:physical']:
        G['physical'][0] = meshioMesh.cell_data_dict['gmsh:physical']['line']
    if 'triangle' in meshioMesh.cell_data_dict['gmsh:physical']:
        G['physical'][1] = meshioMesh.cell_data_dict['gmsh:physical']['triangle']
    if 'tetra' in meshioMesh.cell_data_dict['gmsh:physical']:
        G['physical'][2] = meshioMesh.cell_data_dict['gmsh:physical']['tetra']
    G['problemDimension'] = problemDimension
    mesh = G
    mesh['meshio'] = meshioMesh # will be removed later
    mesh['eb'] = []
    mesh['ett'] = []
    mesh['pe'] = []
    mesh['signs2d'] = []
    mesh['signs3d'] = []
    stop = time.time()
    print(f'loaded mesh in {stop - start:.2f} s')    
    printMeshInfo()
