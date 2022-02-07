import numpy as np
import sys
from mesh import *

regionList = []

class Region:
    def __init__(self):
        global regionList
        self.ids = []
        self.elements = []
        self.regionDimension = 0
        regionList.append(self)

    def append(self, ids):
        if isinstance(ids, list):
            for id in ids:
                self.ids.append(id)
        else:
            self.ids.append(ids)

    def getElements(self, edges=False):
        if self.elements == []:
            self.calculateElements(edges)
        return self.elements

    def calculateElements(self, edges=False):
        # always sort physicalIds so that numbering of region elements and parameters match
        self.ids.sort()
        for id in self.ids:
            for dim in np.arange(start=1,stop=4):
                if id in mesh()['physical'][dim-1]:
                    if self.regionDimension != 0 and self.regionDimension != dim:
                        print("cannot mix dimensions in single region!")
                        sys.exit()
                    self.regionDimension = dim
                    for i in range(len(mesh()['physical'][dim-1])):
                        if mesh()['physical'][dim-1][i] == id:
                            if dim == 1:
                                self.elements.append(mesh()['pl'][i])
                            elif dim==2:
                                self.elements.append(mesh()['pt'][i])
                            elif dim==3:
                                if edges:
                                    self.elements.append(mesh()['ett'][i])
                                else:
                                    self.elements.append(mesh()['ptt'][i])
