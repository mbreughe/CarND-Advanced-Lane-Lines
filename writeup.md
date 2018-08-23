## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistort]: ./images/test2_undistort.jpg "Undistorted"
[color_thresh]: ./images/test2_Sobel.jpg "Color thresholded"
[perspective]: ./images/test2_transform.jpg "Perspective transform"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

TBD

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Below is an image before and after distortation is applied:
![undistort][undistort]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


I attempted various combinations of color and gradient thresholds. A combination of using the X-Sobel-gradient and color thresholds in HLS space was useful, however, in the end I relied on color thresholds only. In pipeline.py, lines 178 through 195 define the "sobel\_color" function. I set thresholds for both the Saturation, as well as the Hue (carefully selecting it, e.g., to avoid detecting too much green). On top of that, I used an extra RGB threshold to detect white lines.

An example can be seen below, before and after thresholding is applied:

![Thresholded image][color_thresh]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I opened a test image that contains straight lines in paint.net. I carefully identified points that, in a birds-eye-view should be rectangular, but are on a trapezoid in the image. I hard-coded the values of the image (source) as class values of my LaneDetector class on line 250. Further, I hard-coded the destination points of where I want the source points to show up after the transformation. These points can also be found in the table below.

Because the transformation is the same in all the images, we only need to compute the transformation matrices, M and Minv once. Therefore it is part of the get\_calibration\_data functio non lines 322 to 351 in pipeline.py, which gets called only once, from LaneDetector.\_\_init\_\_.

By using the cv2.warpPerspective we can see that the lane lines are indeed parallel. See also the example below.

![perspective transform][perspective]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
