import numpy as np
import sys
from mesh import *

regionList = []

class region:
    def __init__(self):
        global regionList
        self.ids = []
        self.elements = []
        self.regionDimension = 0
        regionList.append(self)

    def append(self, id):
        self.ids.append(id)

    def getElements(self):
        if self.elements == []:
            self.calculateElements()
        return self.elements

    def calculateElements(self):
        # always sort physicalIds so that numbering of region elements and parameters match
        self.ids.sort()
        for id in self.ids:
            if id in mesh()['physical'][0]: # check lines
                if self.regionDimension != 0 and self.regionDimension != 2:
                    print("cannot mix dimensions in single region!")
                    sys.exit()
                self.regionDimension = 2
                for i in range(len(mesh()['physical'][0])):
                    if mesh()['physical'][0][i] == id:
                        self.elements.append(mesh()['pl'][i])
            if id in mesh()['physical'][1]: # check triangles
                if self.regionDimension != 0 and self.regionDimension != 3:
                    print("cannot mix dimensions in single region!")
                    sys.exit()
                self.regionDimension = 3
                for i in range(len(mesh()['physical'][1])):
                    if mesh()['physical'][1][i] == id:
                        self.elements.append(mesh()['pt'][i])
