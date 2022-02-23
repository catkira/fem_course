import numpy as np
import mesh as m
import sys
import field as fd

class DofFieldData:
    def __init__(self, field):
        self.freeNodesMask = np.repeat(True, m.numberOfVertices())
        self.freeEdgesMask = np.repeat(True, m.numberOfEdges())
        self.isGauged = False
        if m.getMesh()['problemDimension'] == 3:
            self.dirichletMask = np.repeat(False, m.numberOfTriangles()) # assume dirichlet BCs are always on problemDimension -1
        self.field = field

    def freeDofMask(self):
        if self.field.elementType == 0:
            return self.freeNodesMask
        elif self.field.elementType == 1:
            return self.freeEdgesMask        

class DofManagerData:
    def __init__(self):
        self.fields = np.empty(0, dtype=object)
        fields = np.empty(0, object)

dofManagerData = DofManagerData()

# every field registers itself here in its constructor
def registerField(field):
    dofManagerData.fields = np.append(dofManagerData.fields, DofFieldData(field))
    return dofManagerData.fields.shape[0] - 1

def countAllDofs():
    dofs = 0
    for field in dofManagerData.fields:
        if field.field.elementType == 0:
            dofs += m.numberOfVertices()
        elif field.field.elementType == 1:
            dofs += m.numberOfEdges()
    return dofs

def countAllFreeDofs():
    dofs = 0
    for field in dofManagerData.fields:
        dofs += countFreeDofs(field.field)
    return dofs

def countFreeDofs(field):
    if dofManagerData.fields[field.id].field.elementType == 0:
        return np.count_nonzero(dofManagerData.fields[field.id].freeNodesMask)
    elif dofManagerData.fields[field.id].field.elementType == 1:
        return np.count_nonzero(dofManagerData.fields[field.id].freeEdgesMask)

def resetDofManager():
    global dofManagerData
    dofManagerData = DofManagerData()

def translateDofIndices(field, elements):
    mask = dofManagerData.fields[field.id].freeDofMask()
    idx = np.repeat(-1, countAllDofs()).astype(np.int)
    addr = 0
    for id in range(len(idx)):
        if mask[id]:
            idx[id] = addr
            addr += 1
    elements2 = np.take(idx, elements.ravel()).reshape(elements.shape)
    return elements2

def putSolutionIntoFields(u):
    global dofManagerData
    for field in dofManagerData.fields:
        mask = field.freeDofMask()
        solution = np.zeros(countAllDofs())
        id = 0
        for id2 in range(countAllDofs()):
            if mask[id2]:
                solution[id2] = u[id]
                id += 1
            else:
                pass  # TODO implement inhomogeneous Dirichlet BCs
        field.field.solution = solution

# TODO implement inhomogeneous Dirichlet BCs
def setDirichlet(field, regions, value = []):
    global dofManagerData
    meshDim = m.getMesh()['problemDimension']
    dim = meshDim - 2  # assume dirichlet BCs are always on problemDimension -1
    for region in regions:
        if region in m.getMesh()['physical'][dim]:
            dofManagerData.fields[field.id].dirichletMask |= (m.getMesh()['physical'][dim] == region)

    elementType = field.elementType
    if meshDim == 3:
        if elementType == 0:
            dofManagerData.fields[field.id].freeNodesMask[np.unique(m.getMesh()['pt'][dofManagerData.fields[field.id].dirichletMask].ravel())] = False
        elif elementType == 1:
            dofManagerData.fields[field.id].freeEdgesMask[np.unique(m.getMesh()['et'][dofManagerData.fields[field.id].dirichletMask].ravel())] = False
    else:
        print("Error: not implemented!")
        sys.exit()
    
def setGauge(field, tree):
    global dofManagerData
    dofManagerData.fields[field.id].freeEdgesMask[tree.branches] = False
    dofManagerData.fields[field.id].isGauged = True

def isGauged(field):
    return dofManagerData.fields[field.id].isGauged