import numpy as np
import sys
import mesh as m

regionList = []

class Region:
    def __init__(self):
        global regionList
        self.ids = []
        self.elements = []
        self.edgeElements = []
        self.regionDimension = 0
        regionList.append(self)

    def append(self, ids):
        if isinstance(ids, list):
            for id in ids:
                self.ids.append(id)
        else:
            self.ids.append(ids)

    def getElements(self, edges=False):
        if edges:
            if self.edgeElements == []:
                self.calculateElements(edges)
            return np.array(self.edgeElements)
        else:
            if self.elements == []:
                self.calculateElements(edges)
            return np.array(self.elements)

    def calculateElements(self, edges=False):
        if edges:
            m.computeEdges3d()
        # always sort physicalIds so that numbering of region elements and parameters match
        self.ids.sort()
        for id in self.ids:
            for dim in np.arange(start=1,stop=4):
                if id in m.mesh['physical'][dim-1]:
                    if self.regionDimension != 0 and self.regionDimension != dim:
                        print("cannot mix dimensions in single region!")
                        sys.exit()
                    self.regionDimension = dim
                    for i in range(len(m.mesh['physical'][dim-1])):
                        if m.mesh['physical'][dim-1][i] == id:
                            if dim == 1:
                                self.elements.append(m.mesh['pl'][i])
                            elif dim==2:
                                if edges:
                                    self.edgeElements.append(m.mesh['et'][i])
                                else:
                                    self.elements.append(m.mesh['pt'][i])
                            elif dim==3:
                                if edges:
                                    self.edgeElements.append(m.mesh['ett'][i])
                                else:
                                    self.elements.append(m.mesh['ptt'][i])
