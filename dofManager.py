import numpy as np
import mesh as m
import sys
import field as fd
from scipy.sparse import *

class DofFieldData:
    def __init__(self, field):
        self.freeNodesMask = np.repeat(False, m.numberOfVertices())
        self.freeEdgesMask = np.repeat(False, m.numberOfEdges())
        self.edgeIds = []
        self.isGauged = False
        if m.getMesh()['problemDimension'] == 3:
            self.dirichletMask = np.repeat(False, m.numberOfTriangles()) # assume dirichlet BCs are always on problemDimension -1
        self.field = field
        self.startIndex = 0

    def freeDofMask(self):
        if self.field.isEdgeField():
            return self.freeEdgesMask        
        else:
            return self.freeNodesMask

    def countFreeDofs(self):
        if self.field.isEdgeField():
            return np.count_nonzero(self.freeEdgesMask)      
        else:
            return np.count_nonzero(self.freeNodesMask)

    def applyDirichletMask(self):
        meshDim = m.getMesh()['problemDimension']    
        if meshDim == 3:
            if self.field.isEdgeField():
                self.freeEdgesMask[np.unique(m.getMesh()['et'][self.dirichletMask].ravel())] = False
            else:
                self.freeNodesMask[np.unique(m.getMesh()['pt'][self.dirichletMask].ravel())] = False
        elif meshDim == 2:
            print("TODO: implemente applyDirichletMask() for 2d meshes!")

    def applyGaugeMask(self):
        self.freeEdgesMask[self.edgeIds] = False

class DofManagerData:
    def __init__(self):
        self.fields = np.empty(0, dtype=object)

    def updateStartIndices(self):
        startIndex = 0
        for field in self.fields:
            field.startIndex = startIndex
            startIndex += field.countFreeDofs()

dofManagerData = DofManagerData()

# every field registers itself here in its constructor
def registerField(field):
    dofManagerData.fields = np.append(dofManagerData.fields, DofFieldData(field))
    dofManagerData.updateStartIndices()
    return dofManagerData.fields.shape[0] - 1

# this needs to be called whenever the regions of a field change
# it initializes all free dofs that are inside the specified regions to True
# the free dofs will then be reduced by dirichlet constraints and gauge conditions
def updateFieldRegions(field):
    global dofManagerData
    for region in field.regions:
        for dim in range(m.getMesh()['problemDimension']):
            if np.any(m.getMesh()['physical'][dim]) != None and region in m.getMesh()['physical'][dim]:
                elements = m.getElements(field.isEdgeField(), dim)
                if elements is None:
                    continue
                if field.isEdgeField():
                    dofManagerData.fields[field.id].freeEdgesMask[elements[region == m.getMesh()['physical'][dim]]] = True
                else:
                    dofManagerData.fields[field.id].freeNodesMask[elements[region == m.getMesh()['physical'][dim]]] = True
    dofManagerData.fields[field.id].applyDirichletMask()
    dofManagerData.fields[field.id].applyGaugeMask()
    dofManagerData.updateStartIndices()

def createMatrix(rows, cols, data):
    # delete all rows and cols with index -1
    idx = np.append(np.where(rows == -1)[0], np.where(cols == -1)[0])
    #idx = np.where((rows == -1) | (cols == -1))
    data = np.delete(data, idx)
    rows = np.delete(rows, idx)
    cols = np.delete(cols, idx)
    numFreeDofs = countAllFreeDofs()
    K = csr_matrix((data, (rows, cols)), shape=[numFreeDofs, numFreeDofs])
    return K

def createVector(rows, data):
    # delete all rows and cols with index -1
    idx = np.where(rows == -1)[0]
    rows = np.delete(rows, idx)
    data = np.delete(data, idx)
    #
    numFreeDofs = countAllFreeDofs()
    rhs = csr_matrix((data, (rows, np.zeros(len(rows)))), shape=[numFreeDofs, 1]).toarray().ravel()
    return rhs    

def countAllDofs():
    dofs = 0
    for fieldData in dofManagerData.fields:
        if fieldData.field.elementType == 0:
            dofs += m.numberOfVertices()
        elif fieldData.field.elementType == 1:
            dofs += m.numberOfEdges()
    return dofs

def countAllFreeDofs():
    dofs = 0
    for fieldData in dofManagerData.fields:
        dofs += countFreeDofs(fieldData.field)
    return dofs

def countFreeDofs(field):
    return dofManagerData.fields[field.id].countFreeDofs()

def getStartIndex(field):
    return dofManagerData.fields[field.id].startIndex

def resetDofManager():
    global dofManagerData
    dofManagerData = DofManagerData()

def translateDofIndices(field, elements):
    mask = dofManagerData.fields[field.id].freeDofMask()
    idx = np.repeat(-1, len(mask)).astype(np.int64)
    addr = dofManagerData.fields[field.id].startIndex
    for id in range(len(idx)):
        if mask[id]:
            idx[id] = addr
            addr += 1
    elements2 = np.take(idx, elements.ravel()).reshape(elements.shape)
    return elements2

def putSolutionIntoFields(u):
    global dofManagerData
    for fieldData in dofManagerData.fields:
        mask = fieldData.freeDofMask()
        solution = np.zeros(len(mask))
        id = fieldData.startIndex
        for id2 in range(len(mask)):
            if mask[id2]:
                solution[id2] = u[id]
                id += 1
            else:
                pass  # TODO implement inhomogeneous Dirichlet BCs
        fieldData.field.solution = solution

# TODO implement inhomogeneous Dirichlet BCs
def setDirichlet(field, regions, value = []):
    global dofManagerData
    meshDim = m.getMesh()['problemDimension']
    dim = meshDim - 2  # assume dirichlet BCs are always on problemDimension -1
    for region in regions:
        if region in m.getMesh()['physical'][dim]:
            dofManagerData.fields[field.id].dirichletMask |= (m.getMesh()['physical'][dim] == region)
    dofManagerData.fields[field.id].applyDirichletMask()
    dofManagerData.updateStartIndices()        
    
def setGauge(field, tree):
    global dofManagerData
    dofManagerData.fields[field.id].edgeIds = tree.edgeIds
    dofManagerData.fields[field.id].isGauged = True
    dofManagerData.fields[field.id].applyGaugeMask()    
    dofManagerData.updateStartIndices()        

def isGauged(field):
    return dofManagerData.fields[field.id].isGauged