[![Python package](https://github.com/catkira/fem_course/actions/workflows/python-package.yml/badge.svg)](https://github.com/catkira/fem_course/actions/workflows/python-package.yml)
[![Pylint](https://catkira.github.io/fem_course/pylint.svg)](https://github.com/catkira/fem_course/actions/)

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
- tree gauge
- eddy current example in A-V/A formulation
- harmonics (partly implemented)

Possible features to be implemented (difficulty in in brackets):
- higher order elements for node and edge elements (++)
- pos-file export for GMSH (+)
- inhomogeneous Dirichlet boundary conditions (++)
- h-adaptation and .msh mesh export (++)
- symbolic expressions (+++)
- harmonics (++)
- output animated harmonic solution (+)
- time stepping (+++)

# Install on Ubuntu 22
    sudo apt update
    sudo apt install -y software-properties-common lsb-release
    sudo apt install -y gfortran gcc g++
    sudo apt install -y libblas-dev liblapack-dev # needed for pip install petsc
    sudo apt install -y libopenmpi-dev
    sudo apt install -y python3.10-venv
    
    mkdir ~/.venv
    cd ~/.venv
    python3 -m venv python_fem
    source ./python_fem/bin/activate
    pip install scipy numpy matplotlib plotly vtk pyvista meshio
    pip install mpi4py    
    export PETSC_CONFIGURE_OPTIONS="--with-openmp --with-mpi=0 --with-shared-libraries=1 --with-mumps-serial=1 --download-mumps --download-openblas --download-openblas-commit=origin/develop --download-metis --download-slepc --with-debugging=0 --with-scalar-type=real --with-x=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3"
    pip install -r requirements.txt
    
    Alternatively to install petsc manually:
    git clone -b release https://gitlab.com/petsc/petsc
    cd petsc
    ./configure --with-openmp --with-mpi=0 --with-shared-libraries=1 --with-mumps-serial=1 --download-mumps --download-openblas --download-openblas-commit=origin/develop --download-metis --download-slepc --with-debugging=0 --with-scalar-type=real --with-x=0 --with-petsc4py COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3
    sudo make -j$(nproc) install
    
# Examples
examples/h_magnet.py
![h_magnet](https://github.com/catkira/fem_course/blob/master/examples/h_magnet.png?raw=true)
examples/magnet_in_room.py
![magnet_in_room](https://github.com/catkira/fem_course/blob/master/examples/magnet_in_room.png?raw=true)
examples/magmesh.py
![magnet_in_room](https://github.com/catkira/fem_course/blob/master/examples/magmesh.png?raw=true)
examples/inductionheating.py
![inductionheating1](https://github.com/catkira/fem_course/blob/master/examples/inductionheating_current.png?raw=true)
![inductionheating2](https://github.com/catkira/fem_course/blob/master/examples/inductionheating_current_arrows.png?raw=true)
![inductionheating3](https://github.com/catkira/fem_course/blob/master/examples/inductionheating_cut.png?raw=true)


# Recommended literature/papers
- Inside Finite Elements by Martin Weiser
- Finite-Elemente-Methode by Jörg Frochte
- Numerische Methoden in der Berechnung elektromagnetischer Felder by Arnulf Kost
- Die Finite-Elemente Methode für Anfänger by Goering, Roos, Lutz
- [Manges et al., A generalized tree-cotree gauge for magnetic field computation, 1995](https://ieeexplore.ieee.org/document/376275)
- [Anjam et al., Fast MATLAB assembly of FEM matrices in 2D and 3D: Edge elements, 2014](https://arxiv.org/abs/1409.4618)
- [Cendes, Vector finite elements for electromagnetic field computation, 1991](https://ieeexplore.ieee.org/document/104970)
- [Bossavit, A rationale for 'edge-elements' in 3-D fields computations, 1988](https://ieeexplore.ieee.org/document/43860)
