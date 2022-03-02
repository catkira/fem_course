import numpy as np
import sys
from mesh import *
from field import *
from region import Region

parameterList = []

# TODO: make parameter able to handle regions of different dimensions

class Parameter:
    def __init__(self, rows=1):
        self.settings = []
        self.preparedValues = dict()
        self.parameterDimension = 0
        self.rows = rows
        parameterList.append(self)

    def checkDimensions(self):
        for setting in self.settings:
            if self.parameterDimension == 0:
                self.parameterDimension = regionDimension(setting[0])
            elif regionDimension(setting[0]) != self.parameterDimension:
                print(f'Error: cannot mix regions with different dimensions in a parameter!')
                sys.exit()

    def set(self, regions, value):
        if ((not isinstance(value, list) and self.rows != 1) or 
            (isinstance(value, list) and self.rows != len(value))):
            print(f'Error: values for this parameter need to have {self.rows:d} rows!')
            sys.exit()
        if not isinstance(regions, list):
            regions = [regions]
        for region in regions:
            if not [region, value] in self.settings:
                self.settings.append([region, value])
        self.settings.sort()
        self.checkDimensions()

    def prepareValues(self, ids):
        self.preparedValues[str(ids)] = np.empty((0,self.rows))    
        preparedValues2 = dict()    
        preparedValues2[str(ids)] = []        
        for setting in self.settings:
            if not setting[0] in ids:
                continue
            if self.rows == 1: # there is probably also a way to do it without this if-else
                self.preparedValues[str(ids)] = np.append(self.preparedValues[str(ids)], 
                    np.repeat(setting[1], np.count_nonzero(getMesh()['physical'][0] == setting[0])))
                self.preparedValues[str(ids)] = np.append(self.preparedValues[str(ids)], 
                    np.repeat(setting[1], np.count_nonzero(getMesh()['physical'][1] == setting[0])))
                self.preparedValues[str(ids)] = np.append(self.preparedValues[str(ids)], 
                    np.repeat(setting[1], np.count_nonzero(getMesh()['physical'][2] == setting[0])))
            else:
                self.preparedValues[str(ids)] = np.row_stack((self.preparedValues[str(ids)], 
                    np.tile(setting[1], (np.count_nonzero(getMesh()['physical'][0] == setting[0]),1))))
                self.preparedValues[str(ids)] = np.row_stack((self.preparedValues[str(ids)], 
                    np.tile(setting[1], (np.count_nonzero(getMesh()['physical'][1] == setting[0]),1))))
                self.preparedValues[str(ids)] = np.row_stack((self.preparedValues[str(ids)], 
                    np.tile(setting[1], (np.count_nonzero(getMesh()['physical'][2] == setting[0]),1))))

    def getValues(self, region=[]):
        ids = []
        for x in self.settings:
            ids.append(x[0])

        if region == []: # use all regions if no region is specified
            regionIds = ids
        else:
            # check if all ids exist in parameter
            for id in region.ids:
                if not id in ids:
                    print(f'Error: Region id {id:d} not present in parameter!')
                    sys.exit()
            regionIds = region.ids
            
        # use calculated values if possible
        if not str(regionIds) in self.preparedValues:
            self.prepareValues(regionIds)

        return self.preparedValues[str(regionIds)]
    
    # this function is quite inefficient, but its only for debug purpose anyway
    # Problem: vertexValues at the border of a region get overwritten by neighbouring region
    # but its only a problem for visualization, since this function is not used in calculation
    def getVertexValues(self):
        triangleValues = self.getValues()
        if len(triangleValues) != numberOfTriangles():
            print("getVertexValues() only works for parameters that are defined on all mesh regions!")
            sys.exit()
        if len(triangleValues.shape) > 1:
            vertexValues = np.zeros((numberOfVertices(), triangleValues.shape[1]))
        else:
            vertexValues = np.zeros((numberOfVertices()))
        for n in range(numberOfTriangles()):
            vertexValues[mesh()['pt'][n][0]] = triangleValues[n]
            vertexValues[mesh()['pt'][n][1]] = triangleValues[n]
            vertexValues[mesh()['pt'][n][2]] = triangleValues[n]
        return vertexValues     