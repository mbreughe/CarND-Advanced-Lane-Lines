import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

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
    
    #transform_src = np.array([[250, 680], [593, 450], [687, 450], [1058, 680]], np.float32)
    #transform_dst = np.array([[250, 680], [250, 450], [1058, 450], [1058, 680]], np.float32)
    
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
    img_size = (und.shape[1], und.shape[0])
    warped = cv2.warpPerspective(und, M, img_size, flags=cv2.INTER_NEAREST)
    
    # Test out calibration
    test_before_after(img, warped, "test.jpg")
    