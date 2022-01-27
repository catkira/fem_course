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
pip install  -v petsc
pip install  -v petsc4py