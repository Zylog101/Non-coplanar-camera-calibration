# Non-coplanar-camera-calibration

Determined the extrinsic and intrinsic camera parameters from a given set of image and world points with the world points not lieing on a plane. Non-coplanar camera calibration is done under the assumption of noisy data. Outliers are eliminated by robust estimation through RANSAC algorithm. 

Using the computed parameter values new 3D point is projected on to image plane

Overview:
* Extract feature points by manually clicking on the image and providing the corresponding world points in a file (Detailed description present in the manual section)
* Create 3D-2D point correspondence out of the feature point extraction step
* Develop system of equation on which SVD is applied to obtain M (projection matrix)
* After obtaining the projection matrix solve for the camera parameters by computing systematically a series of equation
* Use RANSAC algorithm to extract the best camera parameters

![image](https://github.com/Zylog101/Non-coplanar-camera-calibration/blob/master/Image/CCalib.JPG)
