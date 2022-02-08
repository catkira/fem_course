import numpy as np
import matplotlib.pyplot as plt
from mesh import *

class FieldHCurl:
    def shapeFunctionCurls(self):
        if mesh()['problemDimension'] == 2:
            return np.array([2,
                            2,
                            2])
        else:
            return np.array([[0, -2, 2],
                            [2, 0, -2],
                            [-2, 2, 0],
                            [0, 0, 2],
                            [2, 0, 0],
                            [0, 2, 0]])

    def shapeFunctionValues(self, xi):
        return np.array([[1-xi[2]-xi[1], xi[0],          xi[0]],
                        [xi[2],         1-xi[2]-xi[0],  xi[1]],
                        [xi[2],         xi[2],          1-xi[1]-xi[0]],
                        [-xi[1],        xi[0],          0],
                        [0,             -xi[2],         xi[1]],
                        [xi[2],         0,              -xi[0]]])

class FieldH1:    
    def shapeFunctionGradients(self):
        if mesh()['problemDimension'] == 2:
            return np.array([[-1, -1],
                            [1, 0],
                            [0, 1]])
        else:
            return np.array([[-1, -1, -1],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

    def shapeFunctionValues(self, xi):
        return [1, 0, 0] + self.shapeFunctionGradients() @ xi

    # calculates gradient for each element
    def grad(self, u, dim=2):
        if dim == 2:
            m = numberOfTriangles()
            grads = np.zeros((m,3))
            sfGrads = self.shapeFunctionGradients()
            for elementIndex, element in enumerate(mesh()['pt']):    
                jac,_ = transformationJacobian(elementIndex)        
                invJac = np.linalg.inv(jac)
                grads[elementIndex] = np.append(invJac.T @ sfGrads.T @ u[element], 0)
        else:
            m = numberOfTetraeders()
            grads = np.zeros((m,3))
            sfGrads = self.shapeFunctionGradients()
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