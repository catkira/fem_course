import numpy as np
import matplotlib.pyplot as plt
from mesh import *

# HCurl only makes sense for mesh()['problemDimension'] = 3 !
class FieldHCurl:
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
                            [0, 2, 0]])

    def shapeFunctionValues(self, xi, elementDim = 3):
        # lambda[0] = 1 - xi[0] - xi[1] 
        # lambda[1] = xi[0]
        # lambda[2] = xi[1]
        # shapeFunction_e1,e2 = lambda[e1]*grad(lambda[e2]) - lambda[e2]*grad(lambda[e1])
        # edges in tetraeda are ordered like (1,2), (2,0), (0,1)       
        if elementDim == 2:
            if mesh()['problemDimension'] == 2:
                return np.array([[-xi[1],    xi[0]],        # edge (1,2)
                                [-xi[1],     xi[0]],        # edge (2,0)
                                [1-xi[1],    xi[0]]])       # edge (0,1)
            if mesh()['problemDimension'] == 3:
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
                            [xi[2],         0,              -xi[0]]])           # edge (3,1)

    def curl(self, u, dim=3):
        if dim == 2:
            m = numberOfEdges()
            curls = np.zeros((m,1))
            # TODO
        elif dim == 3:
            elements = mesh()['ett']
            curls = np.zeros((elements.shape[0],3))
            sfCurls = self.shapeFunctionCurls(dim)
            jacs = transformationJacobians([], dim)
            detJacs = np.linalg.det(jacs)
            signs = mesh()['signs3d']
            curls2 = np.einsum('i,ijk,lk,il,il->ij', 1/detJacs, jacs, sfCurls, signs, u[elements])   
        return curls2

class FieldH1:    
    def shapeFunctionGradients(self, elementDim = 2):
        if elementDim == 2:
            if mesh()['problemDimension'] == 2:
                return np.array([[-1, -1],
                                [1, 0],
                                [0, 1]])
            elif mesh()['problemDimension'] == 3:
                return np.array([[-1, -1, 0],
                                [1, 0, 0],
                                [0, 1, 0]])
        elif elementDim == 3:
            return np.array([[-1, -1, -1],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

    def shapeFunctionValues(self, xi, elementDim = 2):
        if mesh()['problemDimension'] == 2:        
            return [1, 0, 0] + self.shapeFunctionGradients(elementDim) @ xi
        elif mesh()['problemDimension'] == 3:        
            if elementDim == 2:
                return [1, 0, 0] + self.shapeFunctionGradients(elementDim) @ xi
            elif elementDim == 3:
                return [1, 0, 0, 0] + self.shapeFunctionGradients(elementDim) @ xi

    # calculates gradient for each element
    def grad(self, u, dim=2):
        if dim == 2:
            m = numberOfTriangles()
            grads = np.zeros((m,3))
            sfGrads = self.shapeFunctionGradients(dim)
            for elementIndex, element in enumerate(mesh()['pt']):    
                jac,_ = transformationJacobian(elementIndex)        
                invJac = np.linalg.inv(jac)
                grads[elementIndex] = np.append(invJac.T @ sfGrads.T @ u[element], 0)
        else:
            m = numberOfTetraeders()
            grads = np.zeros((m,3))
            sfGrads = self.shapeFunctionGradients(dim)
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

    def plotShapeFunctions(self):
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