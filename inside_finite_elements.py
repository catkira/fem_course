import numpy as np
import matplotlib.pyplot as plt
import plotly

def numberOfVertices(G):
    return G['xp'].shape[0]

def numberOfTriangles(G):
    return G['pt'].shape[0]

def transformationJacobian(G, t):
    ps = G['pt'][t,:]
    x1 = G['xp'][ps[0],:]
    B = np.array([G['xp'][ps[1],:]-x1, G['xp'][ps[2],:]-x1]).T
    return (B, x1)

# t is id of triangle
# xi is point in reference triangle
def globalCoordinate(G, t, xi):
    B, x1 = transformationJacobian(G, t)
    return x1 + B @ xi    

def localCoordinate(G, t, x):
    B, x1 = transformationJacobian(G, t)
    xi,_,_,_ = np.linalg.lstsq(B, x-x1, rcond=None)
    return xi

def shapeFunctionGradients():
    return np.array([[-1, -1],
        [1, 0],
        [0, 1]])

def shapeFunctionValues(xi):
    return [1, 0, 0] + shapeFunctionGradients() @ xi

def main():
    G = {}
    # store point coordinates 'xp'
    G['xp'] = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]])
    # store the points which make up an element in 'pt'
    G['pt'] = np.array([
        [0, 1, 2],
        [0, 2, 3]])        
    print("number of Vertices(G) = " + str(numberOfVertices(G)))
    print("number of Triangles(G) = " + str(numberOfTriangles(G)))
    xi = np.array([0, 1])
    p = globalCoordinate(G, 0, xi)
    print(f'point ({xi[0]:d}, {xi[1]:d}) of ref triangle transformed to global triangle = ({p[0]:d}, {p[1]:d})')
    xi = localCoordinate(G, 0, p)
    print(f'point ({p[0]:d}, {p[1]:d}) of global triangle transformed to ref triangle = ({xi[0]:f}, {xi[1]:f})')
    
    fig = plt.figure(figsize =(14, 9))
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    grid = np.meshgrid(x,y)
    coords = np.array([grid[0].flatten(), grid[1].flatten()]).T
    coordsMask = [coords[i][0] <= coords[i][1] for i in range(0,coords.shape[0])]
    triangleCoords = coords[coordsMask]
    val = np.zeros([triangleCoords.shape[0],3])
    for i in range (triangleCoords.shape[0]):
        val[i] = shapeFunctionValues([triangleCoords[i][0], triangleCoords[i][1]])
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_trisurf(triangleCoords[:,0],triangleCoords[:,1],val[:,0])
    ax.azim = -90
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot_trisurf(triangleCoords[:,0],triangleCoords[:,1],val[:,1])
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot_trisurf(triangleCoords[:,0],triangleCoords[:,1],val[:,2])
    plt.show()

if __name__ == "__main__":
    main()