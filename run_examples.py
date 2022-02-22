from examples.h_magnet import run_h_magnet
from examples.bookExample1 import run_bookExample1
from examples.bookExample2 import *
from examples.h_magnet_octant import *
from examples.magnet_in_room import *
from examples.magmesh import *
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

        run_h_magnet_octant(dirichlet='soft', vectorized=True)
        run_h_magnet_octant(dirichlet='soft', vectorized=False)
        run_h_magnet_octant(dirichlet='hard', vectorized=True)
        run_h_magnet_octant(dirichlet='hard', vectorized=False, legacy=True)
        run_h_magnet_octant(dirichlet='hard', vectorized=False, legacy=False)        

        #run_h_magnet(dirichlet='soft', gauge=True)  # this gives wrong result
        run_h_magnet(dirichlet='soft', gauge=False, verify=True)
        run_h_magnet(dirichlet='hard', gauge=True, verify=True)
        run_h_magnet(dirichlet='hard', gauge=True, verify=True, legacy=True)

        run_magnet_in_room()

        run_magmesh(dirichlet='hard', gauge=True, coarse=False)
        run_magmesh(dirichlet='hard', gauge=False, coarse=False)
        run_magmesh(dirichlet='soft', gauge=False, coarse=False)
    print('finished')

if __name__ == "__main__":
    main()