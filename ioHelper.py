import numpy as np
import time
import sys
import field as fd
from parameter import Parameter
from mesh import *
from region import Region

# double values have around 16 decimals
def f2s(inputValue):
    return ('%.15f' % inputValue).rstrip('0').rstrip('.')

# u can be of type parameter or a list
# if its a parameter, it has to be defined for all regions in its dimension
# parameter can only be plotted if its on the highest dimension of the mesh
def storeInVTK(u, filename, writePointData : np.bool8 = False, field = []):
    if isinstance(u, Parameter):
        if field == []:
            field = fd.globalField  # TODO: this only works if only one field is defined !
        if writePointData:
            u = u.getVertexValues()  # this function is problematic -> see definition
        else:
            u = u.getValues()
    elif isinstance(u, fd.Field):
        field = u
        u = u.elementValues.values
    else: # assume u is just an array with values
        if field == []:
            field = fd.globalField  # TODO: this only works if only one field is defined !
    storeInVTK2(u, filename, writePointData, field)

def storeInVTK2(u, filename, writePointData, field):
    start = time.time()    
    if getMesh()['problemDimension'] == 2:
        ppe = 3 # points per element
        #elementContainer = getMesh()['pt']
        elementContainer = field.getElements(dim=2, nodesOnly=True, translate=False)
        cellType = vtk.VTK_LAGRANGE_TRIANGLE
    else:
        #elementContainer = getMesh()['ptt']
        elementContainer = field.getElements(dim=3, nodesOnly=True, translate=False)
        ppe = 4 # points per element
        cellType = vtk.VTK_LAGRANGE_TETRAHEDRON
    m = len(elementContainer)

    if writePointData:
        if field.isEdgeField():
            if getMesh()['problemDimension'] == 3:
                ppe = 6
            # TODO: calculate barycenter of each edge
            # but its not so easy, because the barycenter point is not in mesh['xp']
            # so for now, just take one point of the edge
            elementContainer = getMesh()['ett'] 
            elementContainer = ((getMesh()['pe'][elementContainer.ravel()])[:,0]).reshape((len(elementContainer),ppe))
            assert len(u) >= np.max(elementContainer) + 1, "u has to be defined for all elements in the mesh"
        else:
            #assert len(u) == np.max(elementContainer) + 1, "u has to be defined for all elements in the mesh"
            # TODO: figure out while this assert is not working with inductionheating
            pass
    else:
        assert len(u) == m, "u has to be defined for all elements in the field"

    scalarValue = (not isinstance(u[0], list)) and (not type(u[0]) is np.ndarray)
    with open(filename, 'w') as file:
        vtktxt = str()
        vtktxt += "# vtk DataFile Version 4.2\nu\nASCII\nDATASET UNSTRUCTURED_GRID\n\n"
        vtktxt += f'POINTS {m*ppe:d} double\n'
        for element in elementContainer:
            for point in element:
                coords = getMesh()['xp'][point]
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
            # TODO implement write edge data
            if scalarValue:
                vtktxt += f'\nPOINT_DATA {m*ppe:d}\nSCALARS u double\nLOOKUP_TABLE default\n'
            else:
                vtktxt += f'\nPOINT_DATA {m*ppe:d}\nVECTORS u double\n'
            for element in elementContainer:
                for point in element:
                    if scalarValue:
                        vtktxt += f2s(u[point]) + "\n"
                    else:
                        if getMesh()['problemDimension'] == 2:
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
                    if getMesh()['problemDimension'] == 2:
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
    import pyvista as pv
    import vtk    
    n = numberOfVertices()    
    m = numberOfTriangles()    
    points = np.hstack([mesh()['xp'], np.zeros((n,1))]) # add z coordinate
    cells = (np.hstack([(3*np.ones((m,1))), mesh()['pt']])).ravel().astype(np.int64)
    celltypes = np.empty(m, np.uint8)
    #celltypes[:] = vtk.VTK_TRIANGLE    
    celltypes[:] = vtk.VTK_LAGRANGE_TRIANGLE    
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    if not isinstance(u, Parameter):
        grid.point_data["u"] = u
    elif isinstance(u, Parameter):
        if writePointData:
            grid.point_data["u"] = u.getVertexValues()
        grid.cell_data["u"] = u.getValues()
    grid.save(filename, binary=False) 

# functions expects sigmas to be triangleData if its not a parameter object
# until now this function can only write point_data, because compute_derivative()
# returns point_data
def storeFluxInVTK(field, u, sigmas, filename):
    n = numberOfVertices()    
    m = numberOfTriangles()    
    v = field.grad(u)   
    if isinstance(sigmas, Parameter):
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