import numpy as np
import sys
from mesh import *
import region

parameterList = []

class parameter:
    def __init__(self, rows=1):
        self.settings = []
        self.preparedValues = dict()
        self.lineValues = []
        self.triangleValues = []
        self.rows = rows
        parameterList.append(self)

    def set(self, region, value):
        if ((not isinstance(value, list) and self.rows != 1) or 
            (isinstance(value, list) and self.rows != len(value))):
            print(f'Error: values for this parameter need to have {self.rows:d} rows!')
            sys.exit()
        self.settings.append([region, value])

    def prepareValues(self, ids):
        self.lineValues = np.zeros(len(mesh()['physical'][0]), dtype = np.float128)
        self.triangleValues = np.zeros(len(mesh()['physical'][1]), dtype = np.float128)
        self.preparedValues[str(ids)] = []        
        for setting in self.settings:
            for i in range(len(mesh()['physical'][0])):
                if mesh()['physical'][0][i] == setting[0]:
                    self.lineValues[i] = setting[1]
                    self.preparedValues[str(ids)].append(setting[1])
            for i in range(len(mesh()['physical'][1])):
                if mesh()['physical'][1][i] == setting[0]:
                    self.triangleValues[i] = setting[1]
                    self.preparedValues[str(ids)].append(setting[1])

    def getValues(self, region):
        ids = []
        for x in self.settings:
            ids.append(x[0]) 

        # check if all ids exist in parameter
        for id in region.ids:
            if not id in ids:
                print(f'Error: Region id {id:d} not present in parameter!')
                sys.exit()
            
        # use calculated values if possible
        if not str(region.ids) in self.preparedValues:
            self.prepareValues(region.ids)

        return np.array(self.preparedValues[str(region.ids)])