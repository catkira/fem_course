import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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

def plotShapeFunctions():
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    grid = np.meshgrid(x,y)
    coords = np.array([grid[0].flatten(), grid[1].flatten()]).T
    coordsMask = [coords[i][0] + coords[i][1] <= 1 for i in range(0,coords.shape[0])]
    triangleCoords = coords[coordsMask]
    val = np.zeros([triangleCoords.shape[0],3])
    for i in range (triangleCoords.shape[0]):
        val[i] = shapeFunctionValues([triangleCoords[i][0], triangleCoords[i][1]])

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

def rectangularCriss(w, h):
    G = dict()
    G['xp'] = np.zeros([w*h,2])
    G['pt'] = []
    for y in range (0,h):
        for x in range (0,w):
            G['xp'][x+w*y] = [x,y]
            if x < (w-1) and y < (h-1):
                G['pt'].append([x+w*y, (x+1)+w*y, x+w*(y+1)])
                G['pt'].append([(x+1)+w*y, (x+1)+w*(y+1), x+w*(y+1)])
    return G

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
    
    #plotShapeFunctions()
    G = rectangularCriss(10,5)
    plt.scatter(G['xp'][:,0],G['xp'][:,1])
    for edge in G['pt']:
        plt.plot([G['xp'][edge[0],0], G['xp'][edge[1],0], G['xp'][edge[2],0], G['xp'][edge[0],0]],[G['xp'][edge[0],1], G['xp'][edge[1],1], G['xp'][edge[2],1], G['xp'][edge[0],1]])
    plt.show()

if __name__ == "__main__":
    main()