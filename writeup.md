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
[color_thresh]: ./images/test2_sobel.jpg "Color thresholded"
[perspective]: ./images/test2_transform.jpg "Perspective transform"
[warp_detect]: ./images/test2_detect.jpg "Lane line detection"
[result]: ./images/test2_invplt.jpg "Result"
[video1]: ./result.mp4 "Video"

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

Below is an image before and after distortion-correction is applied:
![undistort][undistort]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


I attempted various combinations of color and gradient thresholds. A combination of using the X-Sobel-gradient and color thresholds in HLS space was useful, however, in the end I relied on color thresholds only. In pipeline.py, lines 178 through 195 define the "sobel\_color" function. I set thresholds for both the Saturation, as well as the Hue (carefully selecting it, e.g., to avoid detecting too much green). On top of that, I used an extra RGB threshold to detect white lines.

An example can be seen below, before and after thresholding is applied:

![Thresholded image][color_thresh]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I opened a test image that contains straight lines in paint.net. I carefully identified points that, in a birds-eye-view should be rectangular, but are on a trapezoid in the image. I hard-coded the values of the image (source) as class variables of my LaneDetector class on line 250. Further, I hard-coded the destination points of where I want the source points to show up after the transformation. These points can also be found in the table below.

Because the transformation is the same in all the images, we only need to compute the transformation matrices, M and Minv once. Therefore it is part of the get\_calibration\_data function on lines 322 to 351 in pipeline.py, which gets called only once, from LaneDetector.\_\_init\_\_.

By using the cv2.warpPerspective we can see that the lane lines are indeed parallel. See also the example below.

![perspective transform][perspective]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the "find\_window\_centroids" on line 18 through 96 in pipeline.py. This is a (convolution based) sliding window approach. We slice the image from bottom to top and identify the portion of the two lane lines in every slice. This is done by sliding two windows (one per lane line) from left to right whithin a given margin.

Besides tuning parameters, I made several changes to what the course taught us:
* I am not using the initial left and right centers. These are the average postion of the the lane lines of the bottom quarter of the image. I only use these to find around what location I should start looking. For bends with a very high curvature this tends to be helpful.
* I added a threshold for the minimum amount of pixels we need to detect in a lane line
* When no pixels are detected as being part of a lane line in a vertical slice, I don't add a center. 
* I keep track of strides: in case we didn't detect any pixels in a slice, we do want to make sure that in a next slice, we can shift far enough. (Remember that we can only slide our windows whithin a given margin).
* I also keep track of how many slices in a row did not find any lane lines, and stop detecting if we missed too many of them. Otherwise, we run into the issue that we might end up picking pixels from the other lane.

After detecting points that belong to lane lines, I fit a second order polynomial through them. This is done in the fitPolynomial function on lines 353 through 371.

In the image below, you can see the perspective transformed image on the left, and the detected lanes on the right, along whith activated pixels from color thresholding.

![Lane line detection][warp_detect]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Besides fitting a polynomial on pixel values, I also fitted a polynomial for each lane line in real space values. This allows me to calculate the curvature in real values, through the calcCurvature function on line 263 through 268. I calculate curvature for both left and right lane, and then take the average. Further, I keep track of a running average over 2 seconds of video. These last two steps are done in the updateFrames function on line 435 through 452.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The image below is the result of all the previous steps. It marks out the identified lane and lists curvature. Please ignore the graph on this image. This is only meaningful in the video where curvature is recorded over time.

![Result][result]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced several issues during the project of which I learned some useful lessons:
1. Visualize every step of the pipeline. This helps debugging. Initially, I didn't overlay my warped image with the polynomials that I found. I spent lots of time finetuning other parameters, only to find out that I was plotting polynomials from top to bottom of the image instead of bottom to top (resulting in very weird looking lanes).
2. Having a sort of sanity check in place is really helpful. It is tough to perfectly cover every single frame. If you can tell the program something about the quality of lane detection, detection in previous frames can at least be a backup.

Improvements:
* Searching around previously perfectly detected lane lines could improve the accuracy of my lane detection. Currently, I redo the searching for every frame.
* I've spent a lot of time tuning the various paramerers manually (sobel parameters, color thresholds, window sizes). A more structural approach would be helpful here. E.g., a GUI where you can turn knobs and see the impact on the detection and various intermediate steps.
