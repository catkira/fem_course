import numpy as np
import sys
from mesh import *
from field import *
import region

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
        self.preparedValues[str(ids)] = []        
        for setting in self.settings:
            for i in range(len(mesh()['physical'][0])): # check lines
                if mesh()['physical'][0][i] == setting[0]:
                    self.preparedValues[str(ids)].append(setting[1])
            for i in range(len(mesh()['physical'][1])): # check triangles
                if mesh()['physical'][1][i] == setting[0]:
                    self.preparedValues[str(ids)].append(setting[1])
            for i in range(len(mesh()['physical'][2])): # check tetraeders
                if mesh()['physical'][2][i] == setting[0]:
                    self.preparedValues[str(ids)].append(setting[1])

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

        return np.array(self.preparedValues[str(regionIds)])
    
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

# calculates gradient for each element
def grad(u, dim=2):
    if dim == 2:
        m = numberOfTriangles()
        grads = np.zeros((m,3))
        sfGrads = shapeFunctionGradients()
        for elementIndex, element in enumerate(mesh()['pt']):    
            jac,_ = transformationJacobian(elementIndex)        
            invJac = np.linalg.inv(jac)
            grads[elementIndex] = np.append(invJac.T @ sfGrads.T @ u[element], 0)
    else:
        m = numberOfTetraeders()
        grads = np.zeros((m,3))
        sfGrads = shapeFunctionGradients()
        for elementIndex, element in enumerate(mesh()['ptt']):    
            jac,_ = transformationJacobian(elementIndex)        
            invJac = np.linalg.inv(jac)
            grads[elementIndex] = invJac.T @ sfGrads.T @ u[element]
    return grads
    # points = np.hstack([mesh()['xp'], np.zeros((n,1))]) # add z coordinate
    # cells = (np.hstack([(3*np.ones((m,1))), mesh()['pt']])).ravel().astype(np.int64)
    # celltypes = np.empty(m, np.uint8)
    # celltypes[:] = vtk.VTK_TRIANGLE    
    # grid = pv.UnstructuredGrid(cells, celltypes, points)
    # grid.point_data["u"] = u
    # grid = grid.compute_derivative(scalars='u', gradient='velocity')
    # return grid.get_array('velocity')       

# double values have around 16 decimals
def f2s(inputValue):
    return ('%.15f' % inputValue).rstrip('0').rstrip('.')

# u can be of type parameter or a list
def storeInVTK(u, filename, writePointData = False):
    start = time.time()    
    if isinstance(u, Parameter):
        if writePointData:
            u = u.getVertexValues()  # this function is problematic -> see definition
        else:
            u = u.getValues() 
    if mesh()['problemDimension'] == 2:
        m = numberOfTriangles()    
        elementContainer = mesh()['pt']
        ppe = 3 # points per element
        cellType = vtk.VTK_LAGRANGE_TRIANGLE
    else:
        m = numberOfTetraeders()    
        elementContainer = mesh()['ptt']
        ppe = 4 # points per element
        cellType = vtk.VTK_LAGRANGE_TETRAHEDRON
    scalarValue = (not isinstance(u[0], list)) and (not type(u[0]) is np.ndarray)
    with open(filename, 'w') as file:
        vtktxt = str()
        vtktxt += "# vtk DataFile Version 4.2\nu\nASCII\nDATASET UNSTRUCTURED_GRID\n\n"
        vtktxt += f'POINTS {m*ppe:d} double\n'
        for element in elementContainer:
            for point in element:
                coords = mesh()['xp'][point]
                if len(coords) == 2:
                    vtktxt += f2s(coords[0]) + " " + f2s(coords[1]) + " 0 "
                else:
                    vtktxt += f2s(coords[0]) + " " + f2s(coords[1]) + " " + f2s(coords[2]) + " "
            vtktxt += '\n'
        vtktxt += f'\nCELLS {m:d} {m*(ppe+1):d}\n'
        if ppe == 3:
            for elementIndex, element in enumerate(elementContainer):
                vtktxt += f'3 {elementIndex*ppe:d} {elementIndex*ppe+1:d} {elementIndex*ppe+2:d}\n'
        else:
            for elementIndex, element in enumerate(elementContainer):
                vtktxt += f'4 {elementIndex*ppe:d} {elementIndex*ppe+1:d} {elementIndex*ppe+2:d} {elementIndex*ppe+3:d}\n'
        vtktxt += f'\nCELL_TYPES {m:d}\n'
        for element in elementContainer:
            vtktxt += f"{cellType:d}\n"
        if writePointData:
            if scalarValue:
                vtktxt += f'\nPOINT_DATA {m*ppe:d}\nSCALARS u double\nLOOKUP_TABLE default\n'
            else:
                vtktxt += f'\nPOINT_DATA {m*ppe:d}\nVECTORS u double\n'
            for element in elementContainer:
                for point in element:
                    if scalarValue:
                        vtktxt += f2s(u[point]) + "\n"
                    else:
                        if mesh()['problemDimension'] == 2:
                            vtktxt += f2s(u[point][0]) + " " + f2s(u[point][1]) + " 0"
                        else:
                            vtktxt += f2s(u[point][0]) + " " + f2s(u[point][1]) + " " + f2s(u[point][2])
                        vtktxt += "\n"
        else:
            if scalarValue:
                vtktxt += f'\nCELL_DATA {m:d}\nSCALARS u double\nLOOKUP_TABLE default\n'
            else:
                vtktxt += f'\nCELL_DATA {m:d}\nVECTORS u double\n'
            for elementIndex, element in enumerate(elementContainer):
                if scalarValue:
                    vtktxt += f2s(u[elementIndex]) + "\n"
                else:
                    if mesh()['problemDimension'] == 2:
                        vtktxt += f2s(u[elementIndex][0]) + " " + f2s(u[elementIndex][1]) + " 0"
                    else:
                        vtktxt += f2s(u[elementIndex][0]) + " " + f2s(u[elementIndex][1]) + " " + f2s(u[elementIndex][2])
                    vtktxt += "\n"
        file.write(vtktxt)
    stop = time.time()                    
    print(f'written {filename:s} in {stop-start:.2f}s')
                
# using pyvista save writes vtk files in version 5, which generates wrong gradients
# when using the gradient filter in paraview
def storeInVTKpv(u, filename, writePointData = False):
    n = numberOfVertices()    
    m = numberOfTriangles()    
    points = np.hstack([mesh()['xp'], np.zeros((n,1))]) # add z coordinate
    cells = (np.hstack([(3*np.ones((m,1))), mesh()['pt']])).ravel().astype(np.int64)
    celltypes = np.empty(m, np.uint8)
    #celltypes[:] = vtk.VTK_TRIANGLE    
    celltypes[:] = vtk.VTK_LAGRANGE_TRIANGLE    
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    if not isinstance(u, parameter):
        grid.point_data["u"] = u
    elif isinstance(u, parameter):
        if writePointData:
            grid.point_data["u"] = u.getVertexValues()
        grid.cell_data["u"] = u.getValues()
    grid.save(filename, binary=False) 

# functions expects sigmas to be triangleData if its not a parameter object
# until now this function can only write point_data, because compute_derivative()
# returns point_data
def storeFluxInVTK(u, sigmas, filename):
    n = numberOfVertices()    
    m = numberOfTriangles()    
    v = grad(u)   
    if isinstance(sigmas, parameter):
        sigmaValues = sigmas.getValues()
    else:
        sigmaValues = sigmas
    if len(sigmaValues) != m:
        print("Error: parameter is not defined for all triangles!")
        exit()
    pointSigmas = np.zeros((n,2,2))
    for i, triangle in enumerate(mesh()['pt']):
        pointSigmas[triangle[0]] = sigmaValues[i]
        pointSigmas[triangle[1]] = sigmaValues[i]
        pointSigmas[triangle[2]] = sigmaValues[i]
    flux = np.zeros((len(v),3))
    for i in range(len(v)):
        flux[i][0:2] = pointSigmas[i] @ v[i][0:2]
    points = np.hstack([mesh()['xp'], np.zeros((n,1))]) # add z coordinate
    cells = (np.hstack([(3*np.ones((m,1))), mesh()['pt']])).ravel().astype(np.int64)
    celltypes = np.empty(m, np.uint8)
    celltypes[:] = vtk.VTK_TRIANGLE    
    grid = pv.UnstructuredGrid(cells, celltypes, points)        
    grid.point_data["u"] = u
    grid.point_data["flux"] = flux
    grid.save(filename)         