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

## Corner Tracking with Patch Templates

Corner tracking with patch templates is a simple and straightforward computer vision technique for detecting and tracking corners in an image or video stream. It is based on the idea of using pixel intensity values of image patches around each keypoint to track its movement over time.

The process starts by detecting corners or keypoints in the first frame of the video using a corner detection algorithm. For each keypoint, a small patch or template is created by extracting the pixel intensity values around the keypoint. This patch is then used to search for the keypoint in subsequent frames by comparing the pixel intensity values of the patch in the current frame with the template. The location of the patch with the highest match is considered as the new location of the keypoint in the current frame.

This process is repeated for each keypoint in each subsequent frame, allowing the keypoints to be tracked over time. The corner tracking with patch templates approach is simple, efficient, and can be used in a variety of computer vision applications, such as object tracking and scene reconstruction. However, it can be sensitive to changes in illumination and can lose track of keypoints in the presence of occlusions or rotations.

### Patch Extraction

To extract patches and normailized patches, I wrote a function which takes in three arguments:

1. **image**: The input image from which patches are to be extracted.

2. **keypoints**: The keypoints or corners in the image for which patches are to be extracted.

3. **window\_size**: The size of the patches to be extracted

This function performs the following operations:

1. Pad the input image with replicas of the border pixels using the `cv2.copyMakeBorder` function from OpenCV library. This is done to ensure that the patches are extracted from the central portion of the keypoints, even if they are near the edges of the image.

2. Create two empty lists, *patches* and *normalized\_patches*, to store the extracted patches and normalized patches, respectively.

3. Loop over the *keypoints *in the keypoints argument and extract patches of size *window\_size* centered at each keypoint. The patch is extracted from the padded image, and the x and y coordinates of the keypoint are adjusted to account for the padding. The extracted patch is then appended to the *patches* list.

4. Normalize the intensity of each patch in the *patches* list by subtracting the mean intensity and dividing by the standard deviation. The mathematical equation is:

$$  \bar X = \frac{patch - \mu}{\sigma + \epsilon} $$

The normalized patch is then appended to the *normalized\_patches* list.

5. Finally, the function returns the list of normalized patches. The normalization step helps to account for variations in lighting and other environmental factors, which can affect the appearance of the patches and affect the performance of the tracking algorithm

The python Implementation is:

```py
 def extract_normalized_patches(image, keypoints, window_size=32):
    # Pad the image with replicas of the border pixels
    image = cv2.copyMakeBorder(image, window_size//2, window_size//2, window_size//2, window_size//2, cv2.BORDER_REPLICATE)
    patches = []
    normalized_patches = []
    for kp in keypoints:
        x, y = kp[1] + window_size//2, kp[0] + window_size//2
        patch = image[y-window_size//2 : y+window_size//2, x-window_size//2 : x+window_size//2]
        patches.append(patch)

    for patch in patches:
        # Normalize the intensity of the patch
        mean = np.mean(patch)
        std = np.std(patch)
        if std > 0:
            normalized_patch = (patch - mean) / (std + 1e-7)
            normalized_patches.append(normalized_patch)
    return normalized_patches
```

### Correspondence

To find the correspondence between two patches, I wrote a function which alculates the distance matrix between two sets of patches. The function takes in two arguments:

1. **Patches_1:** A list of patches or arrays representing the first set of patches.

2. **Patches_2:** A list of patches or arrays representing the second set of patches.

The function performs the following operations:

1. Initialize the number of patches in each set patches1 and patches2 as *n* and *m*, respectively.

2. Create a 2D numpy array *D* of shape $(n, m)$ to store the distance between each pair of patches.

3. Loop over the patches in the two sets and calculate the Euclidean distance between each pair of patches. The Euclidean distance is calculated as the sum of the squares of the difference between the intensity values of the two patches.

4. Before computing the Euclidean distance, I checks if the shapes of the two patches are equal. If they are not, it pads the smaller patch with zeros to match the shape of the larger patch.

5. The calculated distances are stored in the D array.

The function is:
```py
def distance_matrix(patches1, patches2):
    n = len(patches1)
    m = len(patches2)
    D = np.zeros((n, m))
    max_shape = max(patches1[i].flatten().shape[0] for i in range(n))
    for i in range(n):
        for j in range(m):
            patch1 = patches1[i].flatten()
            patch2 = patches2[j].flatten()
            patch1_shape = patch1.shape[0]
            patch2_shape = patch2.shape[0]
            if patch1_shape == patch2_shape:
                D[i, j] = np.sum((patch1 - patch2) ** 2)
            else:
                if patch1_shape < max_shape:
                    patch1 = np.pad(patch1, (0, max_shape - patch1_shape), 'constant')
                if patch2_shape < max_shape:
                    patch2 = np.pad(patch2, (0, max_shape - patch2_shape), 'constant')
                D[i, j] = np.sum((patch2 - patch1) ** 2)
    return D
```

###  Finding Matches

To find the correspondences between keypoints of two successive frames, I used the *argmin* function.

In the previous step, I had alot of ambiguous correspondences. To remove them, we used the **1NN/2NN** ratio test and the **cross-validation check**. For this, I created a function which takes the following inputs:

1. **Distance matrix D** between two sets of patches

2. **Ratio test threshold**

The function works as follows:

1. Initialize an empty list *matches*

2. Loop over the columns of the distance matrix *D*
    1. Sort the distances in the column and retrieve the indices of the two smallest distances

    2. Calculate the ratio between the two smallest distances and store the match only if the ratio is smaller than ratio\_test\_threshold.

3. Initialize an empty list **cross\_validated\_matches**

4. Loop over the matches found in the previous step. 

5. Check if the match is unique by comparing it with the other matches and only storing it if it's not a duplicate match.

6. Return the cross-validated matches.

The function implements a robust matching strategy based on the "ratio test". The idea is that if a feature in one set of patches has a much closer match with a patch in the other set than with any other patches in that set, then it's likely that the match is correct. The ratio test threshold is used to filter out the matches where the distance between the first and the second closest patch is too close to each other, indicating that the match is unreliable. The cross-validation step is used to further filter out the matches that are not unique.

## Corner Tracking with SIFT Features

SIFT (Scale-Invariant Feature Transform) is a widely used computer
vision algorithm for detecting and describing local features in images.
The SIFT features are invariant to changes in scale and orientation,
making them robust to changes in the scene or viewpoint.  
Corner tracking using SIFT features involves detecting corners in an
image using the SIFT algorithm, and then tracking these corners across
multiple frames in a video to understand the motion of the scene.  
Here is a high-level overview of the process:

1.  Detect keypoints and extract SIFT features: The first step is to
    detect keypoints in the image, which are corners or other
    distinctive regions in the image. SIFT features are then extracted
    for each of these keypoints.

2.  Match SIFT features between frames: The SIFT features are then
    matched between frames in the video. The matching process involves
    computing the Euclidean distance between the features in the current
    frame and the features in the previous frame.

3.  Track corners: The matches between the SIFT features are then used
    to track the corners across frames. The corners are tracked by
    associating the SIFT features in the current frame with the SIFT
    features in the previous frame based on their similarity.

4.  Compute motion: Finally, the motion of the scene is computed by
    analyzing the displacement of the corners between frames. This
    information can then be used for tasks such as object tracking,
    camera calibration, and structure from motion.

For the implimentation of Corner Tracking with SIFT Features, I used the
built in functions of OpenCV. The function is as following:

``` python
def extract_sift_descriptors(image, keypoints):
    # Convert the keypoints list to a list of cv.KeyPoint objects
    keypoints = [cv2.KeyPoint(float(pt[1]), float(pt[0]), size=32, angle=0) for pt in keypoints]
    
    # Create an instance of the SIFT feature extractor
    sift = cv2.SIFT_create()
    
    # Compute the keypoints and descriptors
    kp, desc = sift.compute(image, keypoints)
    
    return kp, desc
```

This function takes an image and a list of keypoints as input and
computes the SIFT (Scale-Invariant Feature Transform) descriptors for
the given keypoints.  
It starts by converting the keypoints list into a list of cv2.KeyPoint
objects. This is necessary for the SIFT feature extractor in OpenCV.  
Next, an instance of the SIFT feature extractor is created using
**cv2.SIFT_create()**. This creates an instance of the SIFT class in
OpenCV.  
Finally, the function compute is called on the SIFT instance to compute
the keypoints and the descriptors. The inputs to this function are the
image and the keypoints. The output is a tuple containing the computed
keypoints and descriptors. The keypoints and descriptors are returned by
the function.  
For finding the correspondence and the matches, we used the same
functions as defined in the previous part.  
Now our Corner Tracking with SIFT features is done. To compute the
distance between two patches was taking around 25 seconds. Due to this
reason, rather than making a compilation for all the 200 images, I only
applied this to the first 50 images in the data set. I hope you
understand professor.

## Part V - Corner Tracking with Prior Motion Fitting Model

RANSAC (Random Sample Consensus) is a widely used algorithm in computer
vision and image processing for estimating the parameters of a
mathematical model from a set of observed data. It is commonly used in
the context of solving the problem of fitting a model to noisy data,
where there may be outliers that don’t fit the model.  
In the context of Corner Tracking with Prior Motion Fitting Model,
RANSAC is used to estimate the parameters of a motion model from a set
of corresponding points between two images. The model could be a simple
translation, an affine transformation, or a more complex model such as a
homography.  
The RANSAC algorithm operates in two main steps:

1.  **Sampling**: Select a random set of correspondences, typically a
    small number, that are used to fit the model.

2.  **Model Validation**: For each point, use the fitted model to
    estimate its expected location in the second image. The point is
    considered an inlier if it is within a certain distance threshold
    from the expected location. The number of inliers is used to
    determine the quality of the model.

The process of sampling and model validation is repeated multiple times
with different random samples, and the model with the largest number of
inliers is considered the best fit.  
In this way, the RANSAC algorithm is able to robustly estimate the
motion model, even in the presence of outliers, and can be used to track
corners between two images.  
I was able to make the logic for this part but was not able to
succesfully complete it. This is what I have done so far:

``` python
def estimate_homography(previous_frame, current_frame, method=cv2.RANSAC):
    """
    Estimate the homography transformation between two frames.
    """
    previous_corners = non_max_suppression(harris_response
                        (*gradient_images(remove_noise
                        (rgb2gray(previous_frame)))))
    current_corners = non_max_suppression(harris_response
                     (*gradient_images(remove_noise
                    (rgb2gray(current_frame)))))
    # Find the corresponding points between the previous frame and the current frame
    previous_points = np.float32([previous_corners[i] for i in range
                     (len(previous_corners))]).reshape(-1,1,2)
    current_points = np.float32([current_corners[i] for i in range
                    (len(current_corners))]).reshape(-1,1,2)
    # Use the OpenCV function cv.findHomography to estimate the homography transformation
    M, mask = cv2.findHomography(previous_points, current_points, method, 5.0)
    # Return the mask that specifies which inliers were found
    return M, mask

def estimate_affine_transform(previous_frame, current_frame, method=cv2.RANSAC):
    """
    Estimate the affine transformation between two frames.
    """
    previous_corners = non_max_suppression(harris_response(
                    *gradient_images(remove_noise
                    (rgb2gray(previous_frame)))))
                    
    current_corners = non_max_suppression(harris_response
                    (*gradient_images(remove_noise
                    (rgb2gray(current_frame)))))
                    
    # Find the corresponding points between the previous frame and the current frame
    previous_points = np.float32([previous_corners[i] for i in range
                     (len(previous_corners))]).reshape(-1,1,2)
    current_points = np.float32([current_corners[i] for i in range
                     (len(current_corners))]).reshape(-1,1,2)
    # Use the OpenCV function cv.getAffineTransform to estimate the affine transformation
    M = cv2.getAffineTransform(previous_points, current_points)
    return M
```

# Results

## Harris Corner Detector

Here are a few results of the Harris Corner detector:

![000000](https://user-images.githubusercontent.com/86875043/219847308-a8d51c43-e5ed-4034-9fed-8c3b879db4af.png)

![000027](https://user-images.githubusercontent.com/86875043/219847310-85c80203-49ac-46d3-8d93-aa528bd779ab.png)

![000042](https://user-images.githubusercontent.com/86875043/219847313-a7b1d960-08e9-497f-a76e-bae1794ae3d3.png)

The rest of the results can be seen in the uploaded folder.

## Corner Tracking with Patch Templates based on pixel intensities

![result_0](https://user-images.githubusercontent.com/86875043/219847331-a7913bd9-d3eb-485e-9809-13e54e8d7c85.png)
![result_1](https://user-images.githubusercontent.com/86875043/219847333-05a26d99-0493-4819-80da-67ed0f5ea9ea.png)
![result_2](https://user-images.githubusercontent.com/86875043/219847335-348c2d5e-c31e-43ef-a372-81f319b12c6b.png)
![result_34](https://user-images.githubusercontent.com/86875043/219847340-ed8f72d6-36ee-49d5-9055-eb7f50d1f1cc.png)
![result_35](https://user-images.githubusercontent.com/86875043/219847347-edca0198-498e-4869-99b6-62c56f2643a8.png)
![result_36](https://user-images.githubusercontent.com/86875043/219847349-ce42a4bd-681e-442b-be78-ea67917a6598.png)



The rest of the result can be seen in the uploaded folder with the same name. Also, you can see the video that has been uplaoaded.

## Corner Tracking with SIFT Features
![result_2](https://user-images.githubusercontent.com/86875043/219847359-94e4d839-606a-4ecf-8658-d64fffdf6c5b.png)
![result_3](https://user-images.githubusercontent.com/86875043/219847360-d0bcf51c-2ca6-4ef1-ae17-48c9a3f172d3.png)
![result_4](https://user-images.githubusercontent.com/86875043/219847361-01ed5918-4563-4997-b826-466523ea692f.png)
![result_34](https://user-images.githubusercontent.com/86875043/219847365-4472622e-1724-4581-800b-c43219a1d51f.png)
![result_35](https://user-images.githubusercontent.com/86875043/219847369-9ecac5c7-e84e-4d93-8662-105cb095fcef.png)
![result_36](https://user-images.githubusercontent.com/86875043/219847373-c28cbf7d-63d8-4a83-b681-3268ca116319.png)


# Conclusion

In conclusion, we successfully implemented various techniques for corner
tracking including Harris corner detection, tracking with patch
templates using simple intensity values, and using SIFT features. We
improved the match finding by using the 1NN/2NN ratio test and
cross-validation check to eliminate ambiguous correspondences. The
distance matrix was also computed to measure the distance between
patches for match finding. The various techniques we employed
demonstrate the importance of considering multiple factors in corner
tracking and the effectiveness of using feature-based methods to improve
accuracy.

# Sources

[Computing Keypoints (Cyrill Stachniss)](https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-10-features-keypoints.pptx.pdf)




