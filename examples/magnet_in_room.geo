SetFactory("OpenCASCADE");
//
airBoxSizeX = 5;
airBoxSizeY = 4;
airMeshSize = 0.05;
roomSizeX = 4;
roomSizeY = 3;
wallThickness = 0.05;
wallMeshSize = 0.01;
magnetSizeX = 1;
magnetSizeY = 0.4;
magnetMeshSize = 0.05;
//magnet
Point(1) = {-magnetSizeX/2, -magnetSizeY/2, 0, magnetMeshSize};
Point(2) = {-magnetSizeX/2, magnetSizeY/2, 0, magnetMeshSize};
Point(3) = {magnetSizeX/2, magnetSizeY/2, 0, magnetMeshSize};
Point(4) = {magnetSizeX/2, -magnetSizeY/2, 0, magnetMeshSize};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(4) = {1,2,3,4};
Plane Surface(1) = {4};
//wall
Point(5) = {-roomSizeX/2, -roomSizeY/2, 0, wallMeshSize};
Point(6) = {-roomSizeX/2, roomSizeY/2, 0, wallMeshSize};
Point(7) = {roomSizeX/2, roomSizeY/2, 0, wallMeshSize};
Point(8) = {roomSizeX/2, -roomSizeY/2, 0, wallMeshSize};
Line(6) = {5,6};
Line(7) = {6,7};
Line(8) = {7,8};
Line(9) = {8,5};
Line Loop(10) = {6,7,8,9}; // outside wall
Point(9) = {-roomSizeX/2+wallThickness, -roomSizeY/2+wallThickness, 0, wallMeshSize};
Point(10) = {-roomSizeX/2+wallThickness, roomSizeY/2-wallThickness, 0, wallMeshSize};
Point(11) = {roomSizeX/2-wallThickness, roomSizeY/2-wallThickness, 0, wallMeshSize};
Point(12) = {roomSizeX/2-wallThickness, -roomSizeY/2+wallThickness, 0, wallMeshSize};
Line(11) = {9,10};
Line(12) = {10,11};
Line(13) = {11,12};
Line(14) = {12,9};
Line Loop(15) = {11,12,13,14}; // inside wall
Plane Surface(2) = {15,4};  // inside air
Plane Surface(3) = {10,15};  // wall
//airbox
Point(13) = {-airBoxSizeX/2, -airBoxSizeY/2, 0, airMeshSize};
Point(14) = {-airBoxSizeX/2, airBoxSizeY/2, 0, airMeshSize};
Point(15) = {airBoxSizeX/2, airBoxSizeY/2, 0, airMeshSize};
Point(16) = {airBoxSizeX/2, -airBoxSizeY/2, 0, airMeshSize};
Line(16) = {13,14};
Line(17) = {14,15};
Line(18) = {15,16};
Line(19) = {16,13};
Line Loop(20) = {16,17,18,19};
Plane Surface(4) = {20,10};
//
Coherence;
Physical Surface(1) = {1}; // magnet 
Physical Surface(2) = {2}; // inside air
Physical Surface(3) = {3}; // wall
Physical Surface(4) = {4}; // outside air
Physical Line(5) = {16,17,18,19}; // inf
Mesh.MshFileVersion = 2.0;