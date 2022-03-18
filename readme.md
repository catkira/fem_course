# What is this?
This is a minimal FEM implementation entirely in Python. It is only an educational project with the goal to better understand FEM in the context of electromagnetic fields. Even though it is not particularly speed optimized, vectorized assembly with numpy and petsc4py for solving is used so that the speed is pretty fast (I have used up to 200k elements, but 1M elements should also work).
This is not intended to be used for real applications, there are plenty of open source FEM tools with much more features available - I think the best one is sparselizard.

Features implemented so far are:
- phi-formulation for magnetostatics
- A-formulation for magnetostatics
- 1st order node and edge elements
- homogeneous Dirichlet boundary conditions
- mesh import from GMSH including use of physical regions
- solution export as vtk-file for ParaView
- tree-cotree gauge
- eddy current example in A-phi-formulation
- harmonics

Planned features:
- higher order elements
- pos-file export for GMSH
- inhomogeneous Dirichlet boundary conditions

# Install on Ubuntu 18
    sudo apt update
    sudo apt install -y software-properties-common lsb-release
    sudo apt install -y gfortran gcc g++
    sudo apt install -y libblas-dev liblapack-dev # needed for pip install petsc
    sudo apt install -y libopenmpi-dev

    sudo apt install python3.10 python3.10-dev python3.10-venv
    mkdir ~/.venv
    cd ~/.venv
    python3.10 -m venv python3.10
    source ./python3.10/bin/activate
    https://github.com/pyvista/pyvista-wheels/raw/main/vtk-9.1.0.dev0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install scipy numpy matplotlib plotly vtk pyvista meshio
    pip install mpi4py
    export PETSC_CONFIGURE_OPTIONS="--with-openmp --with-mpi=0 --with-shared-libraries=1 --with-mumps-serial=1 --download-mumps --download-openblas --download-openblas-commit=origin/develop --download-metis --download-slepc --with-debugging=0 --with-scalar-type=real --with-x=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3"
    pip install  -v petsc
    pip install  -v petsc4py
    
# Examples
examples/h_magnet.py
![h_magnet](https://github.com/catkira/fem_course/blob/master/examples/h_magnet.png?raw=true)
examples/magnet_in_room.py
![magnet_in_room](https://github.com/catkira/fem_course/blob/master/examples/magnet_in_room.png?raw=true)
examples/magmesh.py
![magnet_in_room](https://github.com/catkira/fem_course/blob/master/examples/magmesh.png?raw=true)


# Recommended literature/papers
- Inside Finite Elements by Martin Weiser
- Finite-Elemente-Methode by Jörg Frochte
- Numerische Methoden in der Berechnung elektromagnetischer Felder by Arnulf Kost
- Die Finite-Elemente Methode für Anfänger by Goering, Roos, Lutz
- [Manges et al., A generalized tree-cotree gauge for magnetic field computation, 1995](https://ieeexplore.ieee.org/document/376275)
- [Anjam et al., Fast MATLAB assembly of FEM matrices in 2D and 3D: Edge elements, 2014](https://arxiv.org/abs/1409.4618)
- [Cendes, Vector finite elements for electromagnetic field computation, 1991](https://ieeexplore.ieee.org/document/104970)
- [Bossavit, A rationale for 'edge-elements' in 3-D fields computations, 1988](https://ieeexplore.ieee.org/document/43860)
