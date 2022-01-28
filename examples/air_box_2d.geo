SetFactory("OpenCASCADE");
  
DefineConstant[
  mm = 1.e-3,
  lc1 = {5*mm, Name "Parameters/3Mesh size on magnets [m]"}
  lc2 = {50*mm, Name "Parameters/4Mesh size at infinity [m]"}
  inf = {1000*mm, Name "Parameters/1Air box distance [m]"}
];

// change global Gmsh options
Mesh.Optimize = 1; // optimize quality of tetrahedra
Mesh.VolumeEdges = 0; // hide volume edges
Geometry.ExactExtrusion = 0; // to allow rotation of extruded shapes
Solver.AutoMesh = 2; // always remesh if necessary (don't reuse mesh on disk)
Geometry.OCCBooleanPreserveNumbering = 1;

// create air box 
air = news; Rectangle(air) = {0, 0, 0, inf, inf};
airBoundary() = Boundary{ Surface{air};};

//set mesh size
MeshSize{ PointsOf{ Surface{air}; } } = lc2;

Physical Volume(0) = {air};
Physical Surface(1) = {airBoundary[]};

