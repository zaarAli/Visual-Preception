# Corner Detection & Feature Tracking

Corner Detection & Feature Tracking are important concepts in the field of computer vision and image processing. The primary goal of corner detection is to identify the regions in an image that exhibit sharp changes in intensity, which are typically referred to as corners. These corners are then used as features for various computer vision applications, including object recognition, image registration, and tracking. Feature tracking, on the other hand, is the process of detecting and following these features over multiple frames in a sequence of images. In this exercise, we will be implementing our own Harris corner detector and using it to track corners over time with patch templates and SIFT descriptors on the popular KITTI Visual
Odometry dataset. This will give us hands-on experience in the practical implementation of corner detection and feature tracking techniques.

# Libraries Used
The Following libraries were used in this project:
- numpy
- scipy
- matplotlib
- OpenCV (cv2)
- os (for file handling)
- skimage

## Corner Detection using Harris Corner Detection

Corner detection is a fundamental operation in computer vision that involves identifying points in an image where the brightness or color changes rapidly in multiple directions. Corners are important features in images because they are distinctive and invariant to translation, rotation, and scale changes. They can be used for various applications such as image registration, object detection, tracking, and 3D reconstruction.<br>

Corner detection algorithms typically work by analyzing the gradient of the image intensity or color, which is a measure of the rate of change in brightness or color at each pixel. The most common corner detection algorithm is the Harris corner detector, which uses the second moment matrix of the image gradient to estimate the corner response function. The Harris corner detector computes the corner response function for each pixel in the image, and then applies a threshold to select the pixels with the highest response values as corners.<br>

Other corner detection algorithms include the Shi-Tomasi corner detector, which is a modification of the Harris corner detector that selects the best N corners based on their response values, and the FAST (Features from Accelerated Segment Test) corner detector, which uses a heuristic algorithm to detect corners based on local intensity comparisons.<br>

Corner detection is a challenging problem in computer vision due to the presence of noise, occlusions, and changes in lighting and viewpoint. Therefore, researchers are continuously developing new corner detection algorithms and improving existing ones to increase their robustness and accuracy.<br>

For this project, I used the Harris **Corner Detection algorithm.** The Harris Corner detector is a popular algorithm used for corner detection in computer vision. It was proposed by Chris Harris and Mike Stephens in 1988 and has since become one of the most widely used methods for corner detection due to its simplicity and effectiveness.<br>

The Harris corner detector works by analyzing the gradient of the image intensity or color, which is a measure of the rate of change in brightness or color at each pixel.

The steps involved in the Harris corner detection algorithm are:

1. Denoise the image.

2. Compute the gradient of the image using a filter such as the Sobel operator 

3. Compute the second moment matrix M for each pixel in the image, which is a measure of the local structure of the image. The second moment matrix is defined as $$M = [[A, B], [B, C]]$$
where: A, B, and C are the elements of the matrix computed from the gradient of the image.

4. Compute the corner response function R for each pixel using the formula:
$$ R = det(M) - k * trace(M)^2$$
where det(M) and trace(M) are the determinant and trace of the second moment matrix, respectively, and k is an empirically determined constant that controls the sensitivity of the detector.

5. Apply a threshold to the corner response function to select the pixels with the highest response values as corners.

6. Apply non-maximum suppression to eliminate spurious responses and refine the location of the corners.

7. Optionally, repeat the process with different scales and orientations to detect corners at different levels of detail and to account for rotations and scale changes in the image.

8. The Harris corner detector is robust to noise and changes in lighting and viewpoint, and it can detect corners with sub-pixel accuracy. However, it has limitations, such as the sensitivity to the choice of the threshold and the lack of rotation invariance. Therefore, researchers have proposed various extensions and modifications to the Harris corner detector to address these issues and improve its performance.

In detail procedure of Harris Corner Detection is af following:

### Gaussian Filter -- Image smoothing
Gaussian smoothing filter is applied to remove noise from an image because it has the ability to preserve edges while blurring out the noise present in the image. The Gaussian filter works by replacing each pixel value with the average of its neighboring pixels, weighted by the Gaussian function. This has the effect of smoothing out the noise and preserving the important details of the image, such as edges and contours. The size of the filter determines the extent of smoothing and the standard deviation of the Gaussian function determines the degree of smoothing. Gaussian smoothing is a simple and effective technique for removing noise from images, and it is widely used in various computer vision and image processing applications.

The Gaussian filter is defined as:

$$  
f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{\frac{-1}{2}\left(\frac{x-\mu}{\sigma}\right)^2} 
$$

We used a $3X3$ Gaussian kernal with a $\sigma$ of 0.1.

To apply this kernal to the image, we convolve it with the image and the result is the smoothed image.

The function is:
``` python
def gaussian_kernel(h, std):
    kernel = np.zeros((h, h), dtype=np.float32)
    center = h // 2
    variance = std**2
    coeff = 1. / (2 * np.pi * variance)
    sum = 0
    for x in range(h):
        for y in range(h):
            x_ = x - center
            y_ = y - center
            kernel[x, y] = coeff * np.exp(-(x_**2 + y_**2) / (2 * variance))
            sum += kernel[x, y]
    return kernel / sum

def remove_noise(image, kernel_size=3, sigma=0.1):
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(image, kernel, mode='reflect')
```

### Image Gradient -- Sobel Gradient

To find the gradient of the image, we convolved the smoothed image with the Sobel operators.


The Sobel operators are given as:


$$
G_x = 
    \left[\begin{matrix}
        -1 & 0 & +1\\
        -2 & 0 & +2\\
        -1 & 0 & +1
    \end{matrix}\right]
    G_y =
    \left[\begin{matrix}
        +1 & +2 & +1\\
        0 & 0 & 0\\
        -1 & -2 & -1
    \end{matrix}\right]
$$

The function is as following:
```python
def gradients(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Ix = convolve(image, sobel_x, mode='reflect')
    Iy = convolve(image, sobel_y, mode='reflect')
    return Ix, Iy

def gradient_images(image):
    # Calculate the gradient images
    gradient_x, gradient_y = gradients(image)

    # Compute the three images
    I_xx = gradient_x * gradient_x
    I_yy = gradient_y * gradient_y
    I_xy = gradient_x * gradient_y

    return I_xx, I_yy, I_xy 
```

### Harris Corner Detector

To implement the Harris Corner detector, we compute the elements of the matrix **M** by the gradients gained by the previous step in the *x, y and xy* direction convolved with the Gaussian kernal (5x5) with $\sigma = 1$. The elements of the matrix M are:

$$ det = I_{xx} * I_{yy} - I_{xy}^2 $$

$$ trace = I_{xx} + I_{yy} $$

The function is:

```py
def harris_response(I_xx, I_yy, I_xy, gaussian_w=gaussian_kernel(5, 1), k=0.06):

    I_xx = convolve(I_xx, gaussian_w, mode='reflect')
    I_yy = convolve(I_yy, gaussian_w, mode='reflect')
    I_xy = convolve(I_xy, gaussian_w, mode='reflect')


    # Compute the elements of the M matrix
    det = I_xx * I_yy - I_xy * I_xy
    trace = I_xx + I_yy
    R = det - k * trace **2

    # R = M / (I_xx + I_yy + k)

    # Return the image with Harris corner response R
    return R #np.clip(R, 0, 1)
```
### Non Max Supression -- NMS
Non-Max Suppression (NMS) is a technique used in computer vision and image processing to reduce the number of detected objects in a set of bounding boxes. The main idea behind NMS is to keep only the most confident detection while suppressing all other overlapping bounding boxes that have a lower confidence score.

The algorithm typically works by sorting the bounding boxes by their confidence scores, and then iterating over the boxes, retaining only the one with the highest score and suppressing all the other overlapping boxes that have an Intersection over Union (IoU) ratio higher than a pre-defined threshold.

Using the built in OpenCV function, we applied NMS and were able to obtain the coordinates of the keypoints of the corners detected in the previous step.

The function is:

```py
 def non_max_suppression(R):
    filtered_coords = feature.peak.peak_local_max(
        R,
        min_distance=7, 
        threshold_rel=0.01,
        exclude_border=True, 
        )
    return filtered_coords 
```
