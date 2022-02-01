SetFactory("OpenCASCADE");
  
DefineConstant[
  mm = 1.e-3,
  cub = {10*mm, Name "Parameters/2Magnet bottom size [m]"}
  hite = {20*mm, Name "Parameters/2Magnet hieght [m]"}
  lc1 = {TotalMemory <= 2048 ? 5*mm : 2*mm, Name "Parameters/3Mesh size on magnets [m]"}
  lc2 = {TotalMemory <= 2048 ? 20*mm : 10*mm, Name "Parameters/4Mesh size at infinity [m]"}
  inf = {100*mm, Name "Parameters/1Air box distance [m]"}
];

// change global Gmsh options
Mesh.Optimize = 1; // optimize quality of tetrahedra
Mesh.VolumeEdges = 0; // hide volume edges
Geometry.ExactExtrusion = 0; // to allow rotation of extruded shapes
Solver.AutoMesh = 2; // always remesh if necessary (don't reuse mesh on disk)
Geometry.OCCBooleanPreserveNumbering = 1;

//create magnet
mag = newv; Box(mag) = {-cub, -cub, -hite, 2*cub, 2*cub, 2*hite};

//create frame
hite2 = hite + cub;
frameOutside = newv; Box(frameOutside) = {-4*cub, -cub, -hite2, 8*cub, 2*cub, 2*hite2};
frameInside = newv; Box(frameInside) = {-2*cub, -cub, -hite, 4*cub, 2*cub, 2*hite};
frame = BooleanDifference{ Volume{frameOutside}; Delete;}{ Volume{frameInside}; Delete;};

// create air box around magnets
air = newv; Box(air) = {-inf, -inf, -inf, 2*inf, 2*inf, 2*inf};
airBoundary() = {Boundary{ Volume{air()}; }};
air = BooleanDifference{Volume{air}; Delete;}{Volume{frame}; Volume{mag};};

// cut out octant
tempBox = newv; Box(tempBox) = {-2*inf, -2*inf, 0, 4*inf, 4*inf, -4*inf};
half = BooleanDifference{Volume{mag};Volume{frame};Volume{air}; Delete;}{Volume{tempBox}; Delete;};
tempBox = newv; Box(tempBox) = {0, -2*inf, -2*inf, -4*inf, 4*inf, 4*inf};
quadrant = BooleanDifference{Volume{mag};Volume{frame};Volume{air}; Delete;}{Volume{tempBox}; Delete;};
tempBox = newv; Box(tempBox) = {-inf, 0, -inf, 4*inf, -2*inf, 4*inf};
octant = BooleanDifference{Volume{mag};Volume{frame};Volume{air}; Delete;}{Volume{tempBox}; Delete;};

// this is a dirty hack
//airBoundary() = {Boundary{ Volume{air()}; }};
airBoundary() = {73, 72, 70};
innerXZBoundary() = {59,64, 71};
innerYZBoundary() = {62,69, 75};
innerXYBoundary() = {66, 74};

//set mesh size
MeshSize{ PointsOf{ Volume{frame, mag}; } } = lc1;

Coherence;
Physical Volume(1) = {mag};
Physical Volume(2) = {frame};
Physical Volume(3) = {air};
Physical Surface(4) = {airBoundary[]};
Physical Surface(5) = {innerXZBoundary[]};
Physical Surface(6) = {innerYZBoundary[]};
Physical Surface(7) = {innerXYBoundary[]};
Physical Surface(8) = {61};

