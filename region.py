import numpy as np
import sys
import mesh as m
import field as fd

regionList = []

class Region:
    def __init__(self, ids = None):
        global regionList
        self.ids = None
        self.elements = []
        self.edgeElements = []
        self.regionDimension = 0
        if ids is not None:
            self.ids = list(ids)
        regionList.append(self)

    def append(self, ids):
        if isinstance(ids, (list, np.ndarray)):
            if self.ids is not None:
                for id in ids:
                    self.ids.append(id)
            else:
                self.ids = list(ids)
        elif isinstance(ids, int):
            if self.ids is None:
                self.ids = [ids]
            else:
                self.ids.append(ids)
        self.updateRegionDimension()

    # if elementType is not specified, take elementType of Field
    def getElements(self, nodesOnly=False, field=-1):
        if nodesOnly == True:
            edges = False
        else:
            if field == -1: # this is used if only one global field is defined
                edges = fd.isEdgeField()
            else:
                edges = field.isEdgeField()
        if edges:
            if self.edgeElements == []:
                self.calculateElements(edges=True)
            return self.edgeElements
        else:
            if self.elements == []:
                self.calculateElements()
            return self.elements

    def updateRegionDimension(self):
        for id in self.ids:
            for dim in np.arange(start=1,stop=4):
                if id in m.mesh['physical'][dim-1]:
                    if self.regionDimension != 0 and self.regionDimension != dim:
                        print("cannot mix dimensions in single region!")
                        sys.exit()
                    self.regionDimension = dim                 

    def calculateElements(self, edges=False):
        if edges:
            m.computeEdges3d()
        # always sort physicalIds so that numbering of region elements and parameters match
        self.ids.sort()
        if edges:
            self.edgeElements = m.getElementsInRegion(elementType=1, regions = self.ids)
        else:
            self.elements = m.getElementsInRegion(elementType=0, regions = self.ids)

        for id in self.ids:
            for dim in np.arange(start=1,stop=4):
                if id in m.mesh['physical'][dim-1]:
                    if self.regionDimension != 0 and self.regionDimension != dim:
                        print("cannot mix dimensions in single region!")
                        sys.exit()
                    self.regionDimension = dim
        #             matches = (m.mesh['physical'][dim-1] == id)
        #             if edges:
        #                 if self.edgeElements == []:
        #                     self.edgeElements = m.getElements(edges, dim-1)[matches]
        #                 else:
        #                     self.edgeElements = np.row_stack((self.edgeElements, m.getElements(edges, dim-1)[matches]))
        #             else:
        #                 if self.elements == []:
        #                     self.elements = m.getElements(edges, dim-1)[matches]
        #                 else:
        #                     self.elements = np.row_stack((self.elements, m.getElements(edges, dim-1)[matches]))
