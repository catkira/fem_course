SetFactory("OpenCASCADE");
  
DefineConstant[
  mm = 1.e-3,
  lc1 = {20*mm, Name "Parameters/3Mesh size on inclusions [m]"}
  lc2 = {50*mm, Name "Parameters/4Mesh size at infinity [m]"}
  inf = {1000*mm, Name "Parameters/1Air box distance [m]"}
];

// change global Gmsh options
Mesh.Optimize = 1; // optimize quality of tetrahedra
Mesh.VolumeEdges = 0; // hide volume edges
Geometry.ExactExtrusion = 0; // to allow rotation of extruded shapes
Solver.AutoMesh = 2; // always remesh if necessary (don't reuse mesh on disk)
Geometry.OCCBooleanPreserveNumbering = 1;

//create inclusions
incl1 = news; Rectangle(incl1) = {1, 1, 0, 1, 1};
incl2 = news; Rectangle(incl2) = {3, 2, 0, 1, 1};

// create air box around inclusions
air = news; Rectangle(air) = {0, 0, 0, 5, 4};
MeshSize{ PointsOf{ Surface{air}; } } = lc2;
airBoundary() = {Boundary{ Surface{air()}; }};
air = BooleanDifference{Surface{air}; Delete;}{Surface{incl1}; Surface{incl2};};

//set mesh size
MeshSize{ PointsOf{ Surface{incl1, incl2}; } } = lc1;

Coherence;
Physical Surface(0) = {incl1};
Physical Surface(1) = {incl2};
Physical Surface(2) = {air};
Physical Line(3) = {airBoundary[0]};
Physical Line(4) = {airBoundary[1]};
Physical Line(5) = {airBoundary[2]};
Physical Line(6) = {airBoundary[3]};

