import numpy as np
import mesh as m
import sys
import field

class DofManagerData:
    def __init__(self):
        self.freeNodesMask = np.repeat(True, m.numberOfVertices())
        self.freeEdgesMask = np.repeat(True, m.numberOfEdges())
        if m.getMesh()['problemDimension'] == 3:
            self.dirichletMask = np.repeat(False, m.numberOfTriangles()) # assume dirichlet BCs are always on problemDimension -1

dofManagerData = DofManagerData()

def countAllDofs():
    if field.elementType == 0:
        return m.numberOfVertices()
    elif field.elementType == 1:
        return m.numberOfEdges()

def countFreeDofs():
    if field.elementType == 0:
        return np.count_nonzero(dofManagerData.freeNodesMask)
    elif field.elementType == 1:
        return np.count_nonzero(dofManagerData.freeEdgesMask)

def freeDofMask():
    if field.elementType == 0:
        return dofManagerData.freeNodesMask
    elif field.elementType == 1:
        return dofManagerData.freeEdgesMask

def resetDofManager():
    global dofManagerData
    dofManagerData = DofManagerData()

def translateDofIndices(elements, dir='forward'):
    mask = freeDofMask()
    if dir == 'forward':
        idx = np.repeat(-1, countAllDofs()).astype(np.int)
        addr = 0
        for id in range(len(idx)):
            if mask[id]:
                idx[id] = addr
                addr += 1
        elements2 = np.take(idx, elements.ravel()).reshape(elements.shape)
    else:
        elements2 = np.zeros(countAllDofs())
        id = 0
        for id2 in range(countAllDofs()):
            if mask[id2]:
                elements2[id2] = elements[id]
                id += 1
            else:
                pass  # TODO implement inhomogeneous Dirichlet BCs
    return elements2

# TODO implement inhomogeneous Dirichlet BCs
def setDirichlet(regions, value = []):
    global dofManagerData
    meshDim = m.getMesh()['problemDimension']
    dim = meshDim - 2  # assume dirichlet BCs are always on problemDimension -1
    for region in regions:
        if region in m.getMesh()['physical'][dim]:
            dofManagerData.dirichletMask |= (m.getMesh()['physical'][dim] == region)

    elementType = field.elementType
    if meshDim == 3:
        if elementType == 0:
            dofManagerData.freeNodesMask[np.unique(m.getMesh()['pt'][dofManagerData.dirichletMask].ravel())] = False
        elif elementType == 1:
            dofManagerData.freeEdgesMask[np.unique(m.getMesh()['et'][dofManagerData.dirichletMask].ravel())] = False
    else:
        print("Error: not implemented!")
        sys.exit()
    

def setGauge(tree):
    print("Error: not implemented!")
    sys.exit()
