from examples.h_magnet import run_h_magnet
from examples.bookExample1 import run_bookExample1
from examples.bookExample2 import *
from examples.h_magnet_octant import *
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

        run_bookExample2Parameter(True, anisotropicInclusion=False, method='petsc')
        run_bookExample2(False, anisotropicInclusion=True, method='petsc')     
        run_bookExample2(False, anisotropicInclusion=True, method='petsc', mesh='criss')    

        run_h_magnet_octant(dirichlet='hard')
        run_h_magnet_octant(vectorized=False)
        run_h_magnet_octant(vectorized=False, legacy=True)

        run_h_magnet(dirichlet='soft')
        run_h_magnet(dirichlet='hard')
    print('finished')

if __name__ == "__main__":
    main()