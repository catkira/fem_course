import numpy as np
import time
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from formulation import *


def run_bookExample1(verify=False):
    # example from book page 33
    n = numberOfVertices()
    m = numberOfTriangles()
    r = numberOfBoundaryEdges()
    sigmas = np.ones(m)
    rhos = np.ones(m)
    alphas = 1e9*np.ones(r)  # dirichlet BC
    f = np.ones(n)

    spanningtree = st.spanningtree()
    spanningtree.write("example1_spanntree.pos")

    field = FieldH1()
    K = stiffnessMatrix(field, sigmas)
    M = massMatrix(field, rhos, region=getMesh()['pt'])
    #B = boundaryMassMatrix(alphas) # this function is only here to illustrate the most simple way to do it
    B = massMatrix(field, alphas, region=getMesh()['pe'][getMesh()['eb']], elementDim=1)
    b = M @ f
    A = K+B
    u = solve(A,b)
    print(f'u_max = {max(u):.4f}')
    assert(abs(max(u) - 0.0732) < 1e-3)

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.plot_trisurf(G['xp'][:,0], G['xp'][:,1], u)
    #plt.show()

    storeInVTK(u,"example1.vtk", writePointData=True)


if __name__ == "__main__":
    run_bookExample1()
