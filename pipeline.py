import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.image as mpimg

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)

    np.convolve(window,l_sum)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2

    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

def map_windows_to_centroids(window_centroids, warped_e):    
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped_e)
        r_points = np.zeros_like(warped_e)

        # Go through each level and draw the windows     
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped_e,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped_e,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((warped_e, warped_e, warped_e))*255 # making the original road pixels 3 color channels
        warpage = warpage.astype(np.uint8)
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped_e,warped_e,warped_e)),np.uint8)
    
    return output

def sobel(fname):
    image = mpimg.imread(fname) 
    # Choose a Sobel kernel size
    ksize = 31 # Choose a larger odd number to smooth gradient measurements
    x_thresh = (20, 100)
    y_thresh = (20, 100)
    mag_t = (30, 100)
    dir_thresh = (0.7, 1.3)

    # Apply each of the thresholding functions
    gradx = sobel_abs_axis(image, orient='x', sobel_kernel=ksize, thresh=x_thresh)
    grady = sobel_abs_axis(image, orient='y', sobel_kernel=ksize, thresh=y_thresh)
    mag_binary = sobel_magnitude(image, sobel_kernel=ksize, mag_thresh=mag_t)
    dir_binary = sobel_dir(image, sobel_kernel=ksize, thresh=dir_thresh)

    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined

def sobel_abs_axis(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary

def sobel_magnitude(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel_mag = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary    

def sobel_dir(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_x = np.absolute(sobelx)
    abs_y = np.absolute(sobely)
    
    arc_tan = np.arctan2(abs_y, abs_x)
    binary_output = np.zeros_like(arc_tan)
    binary_output[(arc_tan >= thresh[0]) & (arc_tan <= thresh[1])] = 1
    return binary_output

    return dir_binary

def undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
    
def getCalibrationPoints(fname):
    # Try various values of nx and ny
    for nx in [9, 8, 7]:
        for ny in [6, 5, 7]:
            objp = np.zeros((nx*ny,3), np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
            
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
            if ret:
                return (objp, corners)
    
    return (None, None)
    
def test_before_after(img_before, img_after, ofname):
    print(img_before.shape)
    print(img_after.shape)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(ofname)

if __name__ == "__main__":
    cal_dir = "camera_cal"
    cal_save = "cal_data.p"

    transform_src = np.array([[250, 680], [535, 490], [752, 490], [1058, 680]], np.float32)
    transform_dst = np.array([[250, 680], [250, 490], [1058, 490], [1058, 680]], np.float32)
    
    transform_src = np.array([[250, 680], [593, 450], [687, 450], [1058, 680]], np.float32)
    transform_dst = np.array([[250, 680], [250, 450], [1058, 450], [1058, 680]], np.float32)
    
    #transform_src = np.array([[250, 680], [624, 431], [655, 431], [1058, 680]], np.float32)
    #transform_dst = np.array([[250, 680], [250, 431], [1058, 431], [1058, 680]], np.float32)
    
    
    print("Getting calibration points")
    # Collect calibration points
    if not os.path.exists(cal_save):
        # Points in the real world
        objectPoints = []
        # Points in the image taken by the camera
        imagePoints = []
        for f in os.listdir(cal_dir):
            if not f.endswith(".jpg"):
                continue
            fname = os.path.join(cal_dir, f)
            objp, corners = getCalibrationPoints(fname)       
            
            if objp is not None:
                objectPoints.append(objp)
                imagePoints.append(corners) 
        with open(cal_save, "wb") as ofh:
            pickle.dump(objectPoints, ofh)
            pickle.dump(imagePoints, ofh)
    else:
        with open(cal_save, "rb") as ifh:
            objectPoints = pickle.load(ifh)
            imagePoints = pickle.load(ifh)
            
    print("Getting transformation matrix M")
    M = cv2.getPerspectiveTransform(transform_src, transform_dst)
    
    print("Testing")
    fname = os.path.join("test_images", "straight_lines2.jpg")
    img = cv2.imread(fname)
    und = undistort(img, objectPoints, imagePoints)
    
    edges = sobel(fname)
    c_edges = np.dstack(( np.zeros_like(edges), edges, np.zeros_like(edges))) * 255
    c_edges = c_edges.astype(np.uint8)
    combined = c_edges | img
    
    img_size = (und.shape[1], und.shape[0])
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_NEAREST)
    warped_e = cv2.warpPerspective(edges, M, img_size, flags=cv2.INTER_NEAREST)
    
    # Test out calibration
    test_before_after(c_edges, warped, "test.jpg")

    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching
    
    window_centroids = find_window_centroids(warped_e, window_width, window_height, margin)

    output = map_windows_to_centroids(window_centroids, warped_e)
 
    left_x = [c[0] for c in window_centroids]
    left_y = [i*window_height + window_height/2 for i in range(len(left_x))]
    
    right_x = [c[1] for c in window_centroids]
    right_y = left_y
    
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    
    print (left_fit)
 
    # Display the final results
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(warped_e)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(output)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig("test_2.jpg")
    