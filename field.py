import numpy as np
import mesh as m
import dofManager as dm
import sys
import region as rg
import parameter
import copy

# this is only a hack to allow simpler function calls when only one field is defined
globalField = []

#
# if no regions are given, field is defined on all regions with the highest dimension of the current mesh
#
class Field:
    def __init__(self, regionIDs):
        global globalField
        globalField = self
        self.id = dm.registerField(self)
        self.solution = np.empty(0)
        if regionIDs == []:
            regionIDs = m.getAllRegions(dim=m.dimensionOfMesh())
        self.regions = regionIDs
        dm.updateFieldRegions(self)

    def __mul__(self, other):
        if isinstance(other, parameter.Parameter):
            pass
        else: # assume its just a number
            self.solution *= other

    def __neg__(self):
        newField = copy.deepcopy(self)
        newField.solution *= -1
        return newField

    # TODO: what should self.elementType be, if a H1 and Hcurl field are added ??
    def __add__(self, other):
        newField = copy.deepcopy(self)
        if self.regions != other.regions:
            newField.regions = np.intersect1d(self.regions, other.regions)
            newField.solution = np.empty((0,self.solution.shape[1]))
            selfRegionStartIdx = 0
            otherRegionStartIdx = 0
            for region in np.union1d(self.regions, other.regions):
                regionLen = self.getNumberOfElements(region)
                if region in newField.regions:
                    newField.solution = np.row_stack((newField.solution, self.solution[selfRegionStartIdx:selfRegionStartIdx+regionLen]
                        + other.solution[otherRegionStartIdx:otherRegionStartIdx+regionLen]))
                if region in other.regions:
                    otherRegionStartIdx += regionLen
                if region in self.regions:
                    selfRegionStartIdx += regionLen
        else:
            newField.solution = self.solution + other.solution
        return newField

    def __sub__(self, other):
        return self + (-other)

    def numBasisFunctions(self, elementDim):
        if elementDim == 2:
            return 3
        elif elementDim == 3:
            return 4                

    def setDirichlet(self, regions, value = []):
        dm.setDirichlet(self, regions, value)

    def isEdgeField(self):
        return self.elementType 

    def isGauged(self):
        return dm.isGauged(self)

    # Call to this function implicitely defines the regions where this field has dofs
    # TODO: consider whether this is good
    def getElements(self, dim : int = -1, region = -1, nodesOnly = False, translate = True):    
        if region != -1:
            if dim != -1:
                print("Error: cannot call with dim and region set at the same time!")
                sys.exit()
            #oldRegions = self.regions
            #self.regions = np.unique(np.append(self.regions, region.ids)).astype(np.int)
            #if np.any(oldRegions != self.regions) or len(oldRegions) == 0:
            #    dm.updateFieldRegions(self)
            elements =  region.getElements(field=self, nodesOnly = nodesOnly)
        else:
            assert dim != -1
            elements = self.getAllElements(dim, nodesOnly = nodesOnly, translate=False)
        if translate:
            return dm.translateDofIndices(self, elements)        
        else:
            return elements

    def getAllElements(self, dim, nodesOnly, translate=False):    
        if dim == 2:
            region = rg.Region(self.regions)   
            elements = region.getElements(field=self, nodesOnly=nodesOnly)             
        elif dim == 3:
            region = rg.Region(self.regions)   
            elements = region.getElements(field=self, nodesOnly=nodesOnly)             
        if translate:            
            return dm.translateDofIndices(self, elements)                
        else:
            return elements            

    def getNumberOfElements(self, region):
        if isinstance(region, rg.Region):
            elements = region.getElements(field=self)
        else:
            elements = rg.Region([region]).getElements(field=self)
        return len(elements)
    
    def addRegion(self, u, id):
        print("Warning: this works only if added region has higher id than regions already present")
        assert id > np.max(self.regions)
        # dm.dofManagerData.fields[self.id]
        num = np.count_nonzero(m.getMesh()['physical'][2] == id)
        u = np.row_stack((u, np.zeros((num, u.shape[1]))))
        return u
    
# HCurl only makes sense for mesh()['problemDimension'] = 3 !
class FieldHCurl(Field):
    def __init__(self, regionIDs=[]):
        self.elementType = 1
        super().__init__(regionIDs)

    def numBasisFunctions(self, elementDim):
        if elementDim == 2:
            return 3
        elif elementDim == 3:
            return 6

    def setGauge(self, tree):
        dm.setGauge(self, tree)       

    def shapeFunctionCurls(self, elementDim = 2):
        if elementDim == 2:
            return np.array([2,
                            2,
                            2])
        elif elementDim == 3:
            return np.array([[0, -2, 2],
                            [2, 0, -2],
                            [-2, 2, 0],
                            [0, 0, 2],
                            [2, 0, 0],
                            [0, 2, 0]], dtype=np.float64)

    def shapeFunctionValues(self, xi, elementDim = 3):
        # lambda[0] = 1 - xi[0] - xi[1] 
        # lambda[1] = xi[0]
        # lambda[2] = xi[1]
        # shapeFunction_e1,e2 = lambda[e1]*grad(lambda[e2]) - lambda[e2]*grad(lambda[e1])
        # edges in tetraeda are ordered like (1,2), (2,0), (0,1)       
        if elementDim == 2:
            if m.getMesh()['problemDimension'] == 2:
                return np.array([[-xi[1],    xi[0]],        # edge (1,2)
                                [-xi[1],     xi[0]],        # edge (2,0)
                                [1-xi[1],    xi[0]]])       # edge (0,1)
            if m.getMesh()['problemDimension'] == 3:
                return np.array([[-xi[1],    xi[0],     0],
                                [-xi[1],     xi[0]-1,   0],
                                [1-xi[1],    xi[0],     0]])
        # lambda[0] = 1 - xi[0] - xi[1] - xi[2]                                
        # lambda[1] = xi[0]
        # lambda[2] = xi[1]
        # lambda[3] = xi[2]
        # shapeFunction_e1,e2 = lambda[e1]*grad(lambda[e2]) - lambda[e2]*grad(lambda[e1])
        # edges in tetraeda are ordered like (0,1), (0,2), (0,3), (1,2), (2,3), (3,1)           
        elif elementDim == 3:
            return np.array([[1-xi[2]-xi[1], xi[0],          xi[0]],            # edge (0,1)
                            [xi[2],         1-xi[2]-xi[0],  xi[1]],             # edge (0,2)
                            [xi[2],         xi[2],          1-xi[1]-xi[0]],     # edge (0,3)
                            [-xi[1],        xi[0],          0],                 # edge (1,2)
                            [0,             -xi[2],         xi[1]],             # edge (2,3)
                            [xi[2],         0,              -xi[0]]],           # edge (3,1)
                            dtype=np.float64)
    
    def curl(self, u, dim=3):
        if dim == 2:
            numEdges = m.numberOfEdges()
            curls = np.zeros((numEdges,1))
            # TODO
        elif dim == 3:
            elements = m.getMesh()['ett']
            sfCurls = self.shapeFunctionCurls(dim)
            jacs = m.transformationJacobians([], dim)
            detJacs = np.linalg.det(jacs)
            signs = m.getMesh()['signs3d']
            curls = np.einsum('i,ijk,lk,il,il->ij', 1/detJacs, jacs, sfCurls, signs, u[elements])   
        newField = copy.deepcopy(self)
        newField.solution = curls
        return newField

    def dt(self, u, frequency, dim=3):
        if dim == 3:
            elements = m.getMesh()['ett']
            xiBarycenter = [1/3, 1/3, 1/3]
            values = self.shapeFunctionValues(xiBarycenter)
            jacs = m.transformationJacobians([], dim)
            invJacs = np.linalg.inv(jacs)
            signs = m.getMesh()['signs3d']
            dts = np.einsum('ikj,lk,il,il->ij', invJacs, values, signs, u[elements]) * 2*np.pi*frequency  # TODO: is this correct?   
        elif dim == 2:
            print("Error: not yet implemented!")
            sys.exit()
        newField = copy.deepcopy(self)
        newField.solution = dts
        return newField

class FieldH1(Field):    
    def __init__(self, regionIDs=[]):
        self.elementType = 0
        super().__init__(regionIDs)        

    def shapeFunctionGradients(self, elementDim = 2):
        if elementDim == 2:
            if m.getMesh()['problemDimension'] == 2:
                return np.array([[-1, -1],
                                [1, 0],
                                [0, 1]])
            elif m.getMesh()['problemDimension'] == 3:
                return np.array([[-1, -1, 0],
                                [1, 0, 0],
                                [0, 1, 0]])
        elif elementDim == 3:
            return np.array([[-1, -1, -1],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

    def shapeFunctionValues(self, xi, elementDim = 2):
        if m.getMesh()['problemDimension'] == 2:        
            return [1, 0, 0] + self.shapeFunctionGradients(elementDim) @ xi
        elif m.getMesh()['problemDimension'] == 3:        
            if elementDim == 2:
                return [1, 0, 0] + self.shapeFunctionGradients(elementDim) @ xi
            elif elementDim == 3:
                return [1, 0, 0, 0] + self.shapeFunctionGradients(elementDim) @ xi

    # calculates gradient for each element
    def grad(self, u, dim=2):
        if dim == 2:
            grads = np.zeros((m.numberOfTriangles(),3))
            sfGrads = self.shapeFunctionGradients(dim)
            for elementIndex, element in enumerate(m.getMesh()['pt']):    
                jac,_ = m.transformationJacobian(elementIndex)        
                invJac = np.linalg.inv(jac)
                grads[elementIndex] = np.append(invJac.T @ sfGrads.T @ u[element], 0)
        else:
            region = rg.Region(self.regions)
            elements = self.getElements(region=region, translate=False)
            grads = np.zeros((len(elements),3))
            sfGrads = self.shapeFunctionGradients(dim)
            for elementIndex, element in enumerate(elements):    
                jac,_ = m.transformationJacobian(elementIndex)        
                invJac = np.linalg.inv(jac)
                grads[elementIndex] = invJac.T @ sfGrads.T @ u[element]
        newField = copy.deepcopy(self)
        newField.solution = grads
        return newField
        # points = np.hstack([mesh()['xp'], np.zeros((n,1))]) # add z coordinate
        # cells = (np.hstack([(3*np.ones((m,1))), mesh()['pt']])).ravel().astype(np.int64)
        # celltypes = np.empty(m, np.uint8)
        # celltypes[:] = vtk.VTK_TRIANGLE    
        # grid = pv.UnstructuredGrid(cells, celltypes, points)
        # grid.point_data["u"] = u
        # grid = grid.compute_derivative(scalars='u', gradient='velocity')
        # return grid.get_array('velocity')             

    def plotShapeFunctions(self):
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go        
        x = np.linspace(0,1,100)
        y = np.linspace(0,1,100)
        grid = np.meshgrid(x,y)
        coords = np.array([grid[0].flatten(), grid[1].flatten()]).T
        coordsMask = [coords[i][0] + coords[i][1] <= 1 for i in range(0,coords.shape[0])]
        triangleCoords = coords[coordsMask]
        val = np.zeros([triangleCoords.shape[0],3])
        for i in range (triangleCoords.shape[0]):
            val[i] = self.shapeFunctionValues([triangleCoords[i][0], triangleCoords[i][1]])

        if False:
            fig = plt.figure(figsize =(14, 9))
            ax = fig.add_subplot(1, 3, 1, projection='3d')
            ax.plot_trisurf(triangleCoords[:,0],triangleCoords[:,1],val[:,0])
            ax.azim = -90
            ax = fig.add_subplot(1, 3, 2, projection='3d')
            ax.plot_trisurf(triangleCoords[:,0],triangleCoords[:,1],val[:,1])
            ax = fig.add_subplot(1, 3, 3, projection='3d')
            ax.plot_trisurf(triangleCoords[:,0],triangleCoords[:,1],val[:,2])
            plt.show()
        else:
            # from plotly.subplots import make_subplots
            # fig = make_subplots(rows=1, cols=1)
            fig = go.Figure()

            data = np.ones(triangleCoords.shape[0]) - triangleCoords[:,0] - triangleCoords[:,1]
            fig.add_trace(go.Mesh3d(x=triangleCoords[:,0], y=triangleCoords[:,1], z=data, color='green',opacity=0.90))
            fig.add_trace(go.Mesh3d(x=triangleCoords[:,0], y=triangleCoords[:,1], z=val[:,1], color='blue',opacity=0.90))
            fig.add_trace(go.Mesh3d(x=triangleCoords[:,0], y=triangleCoords[:,1], z=val[:,2], color='red',opacity=0.90))
            #fig.add_trace(go.Mesh3d(x=triangleCoords[:,0], y=triangleCoords[:,1], z=np.zeros(triangleCoords.shape[0]), color='gray'))
            fig.update_layout(
                scene = dict(
                    xaxis = dict(nticks=4, range=[0,1]),
                                yaxis = dict(nticks=4, range=[0,1]),
                                zaxis = dict(nticks=4, range=[0,1]),),
                width=1000,
                height=1000,
                margin=dict(r=10, l=10, b=10, t=10))        
            fig.show()    