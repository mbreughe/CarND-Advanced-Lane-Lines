import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

curvatures = list()


def window_mask(width, height, img_ref, center):
    output = np.zeros_like(img_ref)
    output[int(center[1]- height/2):int(center[1] + height/2),max(0,int(center[0] - width/2)):min(int(center[0] + width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin, minpix = 10):

    N = int(image.shape[0]/window_height) # number of windows in y direction
    
    window_centroids_left = [] # Store the left, window centroid positions per level
    window_centroids_right = [] # Store the right window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)

    np.convolve(window,l_sum)
    l_center_init = np.argmax(np.convolve(window,l_sum))-window_width/2
    l_center = l_center_init

    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center_init = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    r_center = r_center_init
    
    l_shift = 0
    r_shift = 0
    
    miss_thresh = 10
    l_miss = 0
    r_miss = 0
    
    # Go through each layer looking for max pixel locations
    for level in range(0, N):
        y_val = (N-(level+1))*window_height + window_height/2
        
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int((N-(level+1))*window_height):int((N-level)*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # It was recommend to use window_width/2 as offset because convolution signal reference is at right side of window, not center of window.
        # However, I see better results for offset = 0
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_conv = conv_signal[l_min_index:l_max_index]
        # Only change the new center if we found enough pixels
        if np.max(l_conv) >= minpix and l_miss < miss_thresh:
            prev_l_center = l_center
            l_center = np.argmax(l_conv)+l_min_index-offset
            l_shift = l_center - prev_l_center
            window_centroids_left.append((l_center, y_val))
            l_miss = 0
        else:      
            l_center = int(min(max(l_center + l_shift, 0), image.shape[1]))
            l_miss += 1
            
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_conv = conv_signal[r_min_index:r_max_index]
        # Only change the new center if we found enough pixels
        if np.max(r_conv) >= minpix and r_miss < miss_thresh:
            prev_r_center = r_center
            r_center = np.argmax(r_conv)+r_min_index-offset
            r_shift = r_center - prev_r_center
            window_centroids_right.append((r_center, y_val))
            r_miss = 0
        else:
            r_center = int(min(max(r_center + r_shift, 0), image.shape[1]))
            r_miss += 1
            
    detected = True 
    if len(window_centroids_left) == 0:
        window_centroids_left = [(l_center_init, (N - (level + 1)) * window_height + window_height/2) for level in range(0, N)]
        detected = False
     
    if len(window_centroids_right) == 0:
        window_centroids_right = [(r_center_init, (N - (level + 1)) * window_height + window_height/2) for level in range(0, N)]
        detected = False

    return detected, window_centroids_left, window_centroids_right

def map_windows_to_centroids(window_centroids_left, window_centroids_right, window_width, window_height, warped_e):    
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped_e)
    r_points = np.zeros_like(warped_e)

    # Go through each level and draw the windows     
    for left_center in window_centroids_left:
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,warped_e,left_center)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    
    for right_center in window_centroids_right:
            r_mask = window_mask(window_width,window_height,warped_e,right_center)
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    #template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(r_points) # create a zero color channel
    #template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    
    template = np.array(cv2.merge((zero_channel, np.array(r_points), np.array(l_points))), np.uint8)
    warpage= np.dstack((warped_e, warped_e, warped_e))*255 # making the original road pixels 3 color channels
    warpage = warpage.astype(np.uint8)
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    
    return output
    
def markLanesAndCalcCenter(img, warped_e, Minv, left_fit, right_fit, height_frac = 0.7):
    ploty = np.linspace(int((1-height_frac) * warped_e.shape[0]), warped_e.shape[0]-1, int(height_frac * warped_e.shape[0]) )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]      # (720,)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]  # (720,2)
    warp_zero = np.zeros_like(warped_e).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]) # (720,) and (720,) --vstack--> (2,720) --transpose--> (720,2) --np.array--> (1,720,2)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))  # (1, 1440, 2)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    
    lane_center = (left_fitx[-1] + right_fitx[-1])/2
    center_offset = (warped_e.shape[1]/2 - lane_center)
    
    return center_offset, newwarp
    

def sobel(image):
     
    # Choose a Sobel kernel size
    ksize = 31 # Choose a larger odd number to smooth gradient measurements
    x_thresh = (40, 255)
    y_thresh = (20, 100)
    mag_t = (70, 255)
    dir_thresh = (0.2, 1.7)

    # Apply each of the thresholding functions
   # gradx = sobel_abs_axis(image, orient='x', sobel_kernel=ksize, thresh=x_thresh)
   # grady = sobel_abs_axis(image, orient='y', sobel_kernel=ksize, thresh=y_thresh)
   # mag_binary = sobel_magnitude(image, sobel_kernel=ksize, mag_thresh=mag_t)
    dir_binary = sobel_dir(image, sobel_kernel=ksize, thresh=dir_thresh)
    color_grad = sobel_color(image)
    
    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[(((mag_binary == 1) & (dir_binary == 1)) | (color_grad == 1))] = 1
    
    #combined[(color_grad == 1) | (gradx == 1)] = 1
    combined[(color_grad == 1)] = 1
    
    
    #return color_grad
    return combined
    
def sobel_color(image, s_thresh=(100, 255), h_thresh = (10, 25), rgb_min = 200):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]) 
    & (h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    
    is_white = np.zeros_like(s_channel)
    
    is_white[(image[:,:,0] > rgb_min) & (image[:,:,1] > rgb_min) & (image[:,:,2] > rgb_min)] = 1
    
    color_combo = np.zeros_like(s_channel)
    color_combo [(is_white == 1) | (s_binary == 1)] = 1

    return color_combo
    

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
    
def test_before_after(img_before, img_after, ofname, convert = False):
    if convert:
        img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
        img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_before)
    ax1.set_title('Before', fontsize=50)
    ax2.imshow(img_after)
    ax2.set_title('After', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(ofname)
    
# DEBUG: potential issue here!
def calcCurvature(left_fit_real, right_fit_real, y_eval):
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_real[0]*y_eval + left_fit_real[1])**2)**1.5) / np.absolute(2*left_fit_real[0])
    right_curverad = ((1 + (2*right_fit_real[0]*y_eval + right_fit_real[1])**2)**1.5) / np.absolute(2*right_fit_real[0])
    
    return(left_curverad, right_curverad)

class LaneDetector:
    road_width = 3.7
    ym_per_pix = 3/50 # meters per pixel in y dimension
    xm_per_pix = road_width/800 # meters per pixel in x dimension
    
    # Perspective transform data
    transform_src = np.array([[256, 678], [564, 472], [722, 472], [1054, 678]], np.float32)
    transform_dst = np.array([[256, 678], [256, 472], [1054, 472], [1054, 678]], np.float32)
    
    # Search window settings
    window_width = 60 
    window_height = 35 
    margin = 100 # How much to slide left and right for searching
    
    
    avg_wind = 30
    
    cal_save = "cal_data.p"
    
    dbg_output = "detector_dbg"
    
    def __init__(self, max_frames, cal_dir):
        self.left_curvatures = []
        self.right_curvatures = []
        self.curvatures = []
        self.running_avgs = []
        self.running_tot = 0
        self.left_fit = []
        self.right_fit = []
        self.detected = [] 
        self.max_frames = max_frames
        (self.objectPoints, self.imagePoints, self.M, self.Minv) = self.get_calibration_data(cal_dir, self.cal_save, self.transform_src, self.transform_dst)
        
        if not os.path.exists(self.dbg_output):
            os.makedirs(self.dbg_output)
    
    def getCalibrationPoints(self, fname):
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
        
    def get_calibration_data(self, cal_dir, cal_save, transform_src, transform_dst):
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
                objp, corners = self.getCalibrationPoints(fname)       
                
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
        Minv = cv2.getPerspectiveTransform(transform_dst, transform_src)
        
        return (objectPoints, imagePoints, M, Minv)
    
    def fitPolynomial(self, window_centroids_left, window_centroids_right, window_height):
        left_x = [c[0] for c in window_centroids_left]
        left_y = [c[1] for c in window_centroids_left]
        
        right_x = [c[0] for c in window_centroids_right]
        right_y = [c[1] for c in window_centroids_right]
        
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
        
        left_x_real = [i * self.xm_per_pix for i in left_x]
        right_x_real = [i * self.xm_per_pix for i in right_x]
        left_y_real = [i * self.ym_per_pix for i in left_y]
        right_y_real = [i * self.ym_per_pix for i in right_y]
        
        left_fit_real = np.polyfit(left_y_real, left_x_real, 2)
        right_fit_real = np.polyfit(right_y_real, right_x_real, 2)
        
        return (left_fit, right_fit, left_fit_real, right_fit_real, left_y_real, right_y_real) 

    def plot_polynomials_in_warp_space(self, image, warped_e, window_centroids_left, window_centroids_right, left_fit, right_fit, ofname):
        img_size = (image.shape[1], image.shape[0])
        
        # Plot warped road
        warped = cv2.warpPerspective(image, self.M, img_size, flags=cv2.INTER_NEAREST)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(warped)
        
        # Plot polynomials
        ploty = np.linspace(0, warped_e.shape[0]-1, warped_e.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]     
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        ax1.set_title('Original Image', fontsize=50)
        
        # Plot detected lanes
        output = map_windows_to_centroids(window_centroids_left, window_centroids_right, self.window_width, self.window_height, warped_e)
        ax2.imshow(output)
        ax2.set_title('After', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(ofname)        

    def addHUD(self, result, center_offset_pix):
        f, ax = plt.subplots(1, figsize=(8,1.5))
        #ax = axs[0]
        ax.figsize=(8,1.5)
        ax.set_xlim([0, self.max_frames])
        ax.set_ylim([0, 3000])
        
        plt.plot(self.curvatures, label="Instant curvature")
        plt.plot(self.running_avgs, label="Running avg of curvature")
        plt.legend(bbox_to_anchor=(0., -0.2, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        plt.tick_params( 
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        ax.figure.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(ax.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
        
        result[0:data.shape[0], 0:data.shape[1], :] = data
        
        result[data.shape[0]:data.shape[0]+100, : data.shape[1],:] = 255 
        
        center = center_offset_pix * self.xm_per_pix
        if center < 0:
            center = "Vehicle is {0:.3f}m left of center".format(abs(center))
        else:
            center = "Vehicle is {0:.3f}m left of center".format(center)
        
        cv2.putText(result, "Curvature: {} m".format(int(self.cur_running_avg)), (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 80, 0), lineType=cv2.LINE_AA)
        cv2.putText(result, center, (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 80, 0), lineType=cv2.LINE_AA)
        
        return result
        
    def updateFrames(self, left_curverad, right_curverad, left_fit, right_fit):
        cur_curvature = (left_curverad + right_curverad)/2
        self.left_curvatures.append(left_curverad)
        self.right_curvatures.append(right_curverad)
        self.curvatures.append(cur_curvature)
        
        self.left_fit.append(left_fit)
        self.right_fit.append(right_fit)
      
        num_frames = len(self.left_curvatures)
        
        self.running_tot += cur_curvature
        
        if num_frames > self.avg_wind:       
            self.running_tot -= (self.left_curvatures[-(self.avg_wind+1)] + self.right_curvatures[-(self.avg_wind+1)])/2
        
        self.cur_running_avg = self.running_tot/min(self.avg_wind, num_frames)
        self.running_avgs.append(self.cur_running_avg)
        
        
    
    def sanity_check(self, detected, left_fit_real, right_fit_real, min_y, max_y, left_curverad, right_curverad):
        # Skip checking if we didn't see enough frames
        if len(self.left_curvatures) < self.avg_wind:
            return (True, True)
            
        if not detected:
            return (False, False)
        
        width_thresh = 0.4      # Road width mismatch in meters 
        curv_lr_thresh = 0.7   # Curvature mismatch between left and right as a fraction
        curve_t_thresh = 0.7        # Curvature mismatch in time as a fraction
        
        curv_lr_mismatch = abs((left_curverad - right_curverad)/max([left_curverad, right_curverad]))
        
        x_l_min = left_fit_real[0] * min_y**2 + left_fit_real[1] * min_y + left_fit_real[2]
        x_l_max = left_fit_real[0] * max_y**2 + left_fit_real[1] * max_y + left_fit_real[2]
        
        x_r_min = right_fit_real[0] * min_y**2 + right_fit_real[1] * min_y + right_fit_real[2]
        x_r_max = right_fit_real[0] * max_y**2 + right_fit_real[1] * max_y + right_fit_real[2]
        
        width_top = x_r_min - x_l_min
        width_bottom = x_r_max - x_l_max
        
        width_check = (abs(width_top-self.road_width) + abs(width_bottom-self.road_width)) < 2 * width_thresh
        curve_lr_check = curv_lr_mismatch < curv_lr_thresh
              
        if width_check and curve_lr_check:
            return (True, True)
            
        if (not width_check) and (not curve_lr_check):
            return (False, False)
        
        L_curve_t_check = (left_curverad - self.cur_running_avg)/self.cur_running_avg < curve_t_thresh
        R_curve_t_check = (right_curverad - self.cur_running_avg)/self.cur_running_avg < curve_t_thresh
        
        if (not curve_lr_check) and L_curve_t_check:
            return (True, False)
            
        if (not curve_lr_check) and R_curve_t_check :
            return (False, True)
        
        return (False, False)
        
        
        
    def run_pipeline(self, image, verbose=False, ofname_prefix=None, dump_raw_range=None):
        if verbose and ofname_prefix is None:
            print("Warning: asking for verbose output, but no prefix is given. Skipping verbose mode...")
            verbose = False
        
        prefix = ofname_prefix

        # Undistort
        und = undistort(image, self.objectPoints, self.imagePoints)
        img_size = (und.shape[1], und.shape[0])
        
        if verbose:
            test_before_after(image, und, prefix + "undistort.jpg")
        
        # Sobel
        edges = sobel(und)
        
        if verbose:
            c_edges = np.dstack(( np.zeros_like(edges), edges, np.zeros_like(edges))) * 255
            c_edges = c_edges.astype(np.uint8)
            test_before_after(image, c_edges, prefix + "sobel.jpg")
        
        # warp perspective
        if verbose:
            warped = cv2.warpPerspective(und, self.M, img_size, flags=cv2.INTER_NEAREST)
            test_before_after(und, warped, prefix + "transform.jpg")
            
        warped_e = cv2.warpPerspective(edges, self.M, img_size, flags=cv2.INTER_NEAREST)
        
        # detect lanes in warped space
        detected, window_centroids_left, window_centroids_right = find_window_centroids(warped_e, self.window_width, self.window_height, self.margin)
        left_fit, right_fit, left_fit_real, right_fit_real, left_y_real, right_y_real = self.fitPolynomial(window_centroids_left, window_centroids_right, self.window_height)
        max_y = min(max(left_y_real), max(right_y_real))
        min_y = max(min(left_y_real), min(right_y_real))
        
        
        left_curverad, right_curverad = calcCurvature(left_fit_real, right_fit_real, max_y)
        
        left_pass, right_pass = self.sanity_check(detected, left_fit_real, right_fit_real, min_y, max_y, left_curverad, right_curverad)
        
        if left_pass and right_pass:
            self.detected.append(".")
        elif not (left_pass or right_pass):
            self.detected.append("X")
        else:
            self.detected.append("P")
        
        if not left_pass:
            left_fit = self.left_fit[-1]
            left_curverad = self.left_curvatures[-1]
            
        if not right_pass:
            right_fit = self.right_fit[-1]
            right_curverad = self.right_curvatures[-1]
        
        self.updateFrames(left_curverad, right_curverad, left_fit, right_fit)
        
        # Plots the warped road, along with detected lanes and the mapped polynomials along them
        if verbose:
            self.plot_polynomials_in_warp_space(image, warped_e, window_centroids_left, window_centroids_right, left_fit, right_fit, prefix + "detect.jpg")

        # mark lanes
        center_offset_pix, warped_marking = markLanesAndCalcCenter(image, warped_e, self.Minv, left_fit, right_fit)
        
        # Combine the result with the original image
        result = cv2.addWeighted(und, 1, warped_marking, 0.3, 0)

   
        result = self.addHUD(result, center_offset_pix)
        
        if (dump_raw_range is not None) and len(self.detected) > dump_raw_range[0] and len(self.detected) < dump_raw_range[1]:
            dbg_fname = "raw_image_f{}_{}.jpg".format(len(self.detected), self.detected[-1])
            cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(self.dbg_output, dbg_fname), cvt_image)

        if verbose:
            test_before_after(image, result, prefix + "invplt.jpg")          
        
        plt.close('all')
        
        return result
        
def dumpVideoAtRanges(video_fname, sec_ranges, odir):
    if not os.path.exists(odir):
        os.makedirs(odir)
        
    set_no = 0
    for (low, high) in sec_ranges:
        clip = VideoFileClip(video_fname)
        clip = clip.cutout(high, clip.duration)
        clip = clip.cutout(0, low)
        fname = os.path.join(odir, "frame_s_{}_%03d.jpg".format(set_no))
        clip.write_images_sequence(fname)
        set_no += 1
        

# Test out following snippets:        
# 22-25
# 38 - 40    
def test_pipeline(cal_dir, test_dir = "test_images"):
    detector = LaneDetector(1, cal_dir)
    
    for f in os.listdir(test_dir):
        if not f.endswith("jpg"):
            continue
        
        print ("Detecting lanes in " + f)
        fname = os.path.join(test_dir, f)
        image = mpimg.imread(fname)
        prefix = os.path.basename(fname).replace('.jpg', '') + "_"
        result = detector.run_pipeline(image, True, prefix)
    
def process_image(image):
    global g_detector
    global g_dump_range
    return g_detector.run_pipeline(image, verbose=False, dump_raw_range=g_dump_range)
      
def process_video(cal_dir, video_fname):
    # Need to define globals as VideoFileClip.fl_image only accepts one parameter
    global g_detector
    global g_dump_range

    clip = VideoFileClip(video_fname)
    max_frames = clip.fps * (clip.duration+1)
    #g_dump_range = (38 * clip.fps, 40* clip.fps)
    g_dump_range = None
    print (g_dump_range)
    g_detector = LaneDetector(max_frames, cal_dir)
    out_clip = clip.fl_image(process_image)
    out_clip.write_videofile('result.mp4', audio=False)
    print(g_detector.detected)
    
    
    #for i in np.linspace(1,2,30):
    #    clip.save_frame("additional/frame_{}.jpg".format(i), t='00:00:{}'.format(i))

if __name__ == "__main__":
    cal_dir = "camera_cal"    
    
    video_fname = "project_video.mp4"
    
    test_pipeline(cal_dir, "test_images")
    #process_video(cal_dir, video_fname)
    
    #dumpVideoAtRanges(video_fname, [(39,42)], "frame_dump_2")
    #test_pipeline(cal_dir, "frame_dump_2")
    
    
    