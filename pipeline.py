import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.image as mpimg

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
    
    color = np.dstack(( np.zeros_like(combined), combined, np.zeros_like(combined))) * 255
    color = color.astype(np.uint8)
    
    print(np.amin(color))
    print(np.amax(color))
    
    
    return color

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
    combined = edges | img
    img_size = (und.shape[1], und.shape[0])
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_NEAREST)
    
    # Test out calibration
    test_before_after(combined, warped, "test.jpg")
    