# Camera Simulation and Calibration using DLT

This project involves simulating a camera in Python using homogeneous coordinates and performing camera calibration. Camera simulation is the process of modeling the behavior of a camera by defining its intrinsic and extrinsic parameters, which can be used to transform 3D world coordinates to 2D image coordinates. Camera calibration is the process of estimating the camera's intrinsic and extrinsic parameters from known 3D world points and their corresponding 2D image coordinates.<br><br>

## DLT

DLT stands for Direct Linear Transform, and it is a method for camera calibration and pose estimation in computer vision. The DLT algorithm estimates the parameters of a perspective camera model, which describes the mapping between 3D world points and their corresponding 2D image points in a camera. The camera model includes both intrinsic and extrinsic parameters, such as the focal length, principal point, camera position, and orientation.<br>

Here are the general steps involved in the DLT algorithm:

1. Capture images of a calibration pattern with known 3D coordinates, such as a checkerboard or a set of calibration points.
2. Extract the image coordinates of the calibration pattern using feature detection and matching techniques.
3. Generate the correspondence between the 3D world points and their 2D image coordinates.
4. Normalize the image coordinates and form the homogeneous system of linear equations.
5. Solve the linear system of equations using techniques such as singular value decomposition (SVD) or Gaussian elimination.
6. Recover the camera parameters from the solution of the linear system, including the intrinsic and extrinsic parameters.
7. Evaluate the accuracy of the camera calibration by computing the reprojection error, which measures the distance between the observed image points and the projected points from the estimated camera parameters.
8. If the reprojection error is high, refine the camera parameters using techniques such as bundle adjustment or nonlinear optimization.<br>

The DLT algorithm is widely used in computer vision and has many applications, such as 3D reconstruction, pose estimation, and augmented reality. However, it has limitations and assumptions, such as the need for known correspondences between 3D world points and their 2D image coordinates, and the assumption of idealized perspective projection.<br>

## Mathematics for Camera Simulation

Projection Matrix :  $P_{mat}= M_{int} * M_{ext}$

So we have : 

1. Intrinsic Paramters:

$M_{int}=
\left[\begin{matrix}
fk_x & S & C_x & 0\\
0 & fk_y & C_y & 0\\
0 & 0 & 1 & 0
\end{matrix}\right]$

2. Extrinsic Paramters:
$M_{ext}=
\left[\begin{matrix}
R & T\\
0 & 1
\end{matrix}\right]$


$ T=
\left[\begin{matrix}
x \\
y \\
z \\
1
\end{matrix}\right]$

We multiply these rotations to have the total rotation of the system: 

$R_t = R_z * R_y * R_z $

with $\beta$ is the angle of rotation around _'x'_, $\alpha$ around _'y'_ and $\theta$ around _'z'_ 

$R_{z}=
\left[\begin{matrix}
cos(\theta) & -sin(\theta) & 0\\
sin(\theta) & cos(\theta) & 0 \\
0 & 0 & 1 
\end{matrix}\right]$


$R_{y}=
\left[\begin{matrix}
cos(\alpha) & 0 & sin(\alpha)\\
 0 & 1 & 0\\
-sin(\alpha) & 0 & cos(\alpha) 
\end{matrix}\right]$


$R_{x}=
\left[\begin{matrix}
1 & 0 & 0\\
0 & cos(\beta) & -sin(\beta)\\
0 & sin(\beta) & cos(\beta) 
\end{matrix}\right]$

## Procedure

The steps involved in this project are as follows:

1. Generate random world points in 3D space.
2. Define the intrinsic and extrinsic parameters of the camera, including the focal length, principal point, and camera position and orientation.
3. Project the world points onto the camera coordinates using the camera matrix.
4. Project the camera coordinates onto the image plane using perspective projection.
5. Convert the image coordinates to pixel coordinates.
6. Use the world points and their corresponding pixel coordinates to estimate the camera calibration parameters, including the intrinsic matrix and distortion coefficients.<br><br>

This project is useful for understanding the fundamental principles of camera simulation and calibration, which are essential for many applications in computer vision and robotics. It also provides a practical implementation of these concepts using Python and homogeneous coordinates. The project report includes details on the theory, implementation, and results.
