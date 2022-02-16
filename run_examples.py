from examples.h_magnet import run_h_magnet
from examples.bookExample1 import run_bookExample1
from examples.bookExample2 import *
from formulation import *

def main():
    if True:
        rectangularCriss(50,50)
        computeEdges2d()
        computeBoundary()    
        run_bookExample1()

        loadMesh("examples/air_box_2d.msh")
        computeEdges2d()
        computeBoundary()        
        run_bookExample1()

        loadMesh("examples/example2.msh")
        run_bookExample2Parameter(True, anisotropicInclusion=False, method='petsc')
        computeEdges2d()
        computeBoundary()       
        run_bookExample2(False, anisotropicInclusion=True, method='petsc')    

        rectangularCriss(50,50)
        computeEdges2d()
        computeBoundary()       
        mesh()['xp'][:,0] = mesh()['xp'][:,0]*5
        mesh()['xp'][:,1] = mesh()['xp'][:,1]*4    
        run_bookExample2(False, anisotropicInclusion=True, method='petsc')    

        run_h_magnet()
    print('finished')

if __name__ == "__main__":
    main()