// Wire position and radius:
r = 0.1;
xpos = -0.5;
// Shield position and radius:
xposshield = 0.7;
rshieldin = 0.35;
rshieldout = 0.5;
// Domain size around:
rout = 1.5;
// Height:
h = 0.3;

SetFactory("OpenCASCADE");
Disk(1) = {xpos, 0, 0, r, r};
Disk(2) = {xposshield, 0, 0, rshieldout, rshieldout};
Disk(3) = {xposshield, 0, 0, rshieldin, rshieldin};
Disk(4) = {0, 0, 0, rout, rout};

Characteristic Length {4, 2, 1} = 0.04;
Characteristic Length {3} = 0.04;

Extrude {0,0,h} { Surface{1};}
Extrude {0,0,h} { Surface{2};}
Extrude {0,0,h} { Surface{3};}
Extrude {0,0,h} { Surface{4};}

Coherence;

// Define the physical regions:
conductor = 1; shield = 2; air = 3; contour = 4;

Physical Volume(conductor) = {1};
Physical Volume(air) = {3,5};
Physical Volume(shield) = {4};
// conductor surface = {1,6}
Physical Surface(contour) = {16,15,14,12,13,10,8,3,13,14,1,6,3};
//Physical Surface(5) = {5,9,11};