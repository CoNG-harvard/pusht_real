import cv2

def chessboard(img):
    global captureId
    rendered = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    foundCheckboard, corners = cv2.findChessboardCorners(gray, (7,6),
                                                        cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                        cv2.CALIB_CB_FAST_CHECK +
                                                        cv2.CALIB_CB_NORMALIZE_IMAGE)
    # cv2.imwrite('calibInput'+str(captureId)+'.png', img)
    # ret = False
    # If found, add object points, image points (after refining them)
    if foundCheckboard == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        
        rendered = img.copy()
        cv2.drawChessboardCorners(rendered, (7,6), corners2, foundCheckboard)
        
        # Calibration
        ret, cameraMatrix, distortionCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                            gray.shape[::-1],
                                                                            None, None)
        
        # Undistortion
        h, w = img.shape[:2]
        newcameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionCoeffs,
                                                            (w,h), 1, (w,h))
        dst = cv2.undistort(rendered, cameraMatrix, distortionCoeffs, None, newcameraMatrix)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        if dst.shape[0] > 0 and dst.shape[1] > 0:
            rendered = dst
            cv2.imwrite('calibResult'+str(captureId)+'.png', rendered)
            captureId+=1