#!/usr/bin/env python3

import cv2
import cv2.aruco
import numpy as np
import os.path as osp
import math

ARUCO_DICT = {
                "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
                "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
                "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
                "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
                "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
                "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
                "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
                "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
                "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
                "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
                "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
                "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
                "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
                "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
                "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
                "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
                "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
                "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
                "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
                "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
                "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
            }

arucoParams = cv2.aruco.DetectorParameters_create()

def get_yaw_from_rvec(rvec):
    R, _ = cv2.Rodrigues(rvec)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return yaw

def draw_arrow(img, pos, angle_rad, length=50, color=(255, 255, 255), thickness=2):
    x, y = int(pos[0]), int(pos[1])
    end_x = int(x + length * math.cos(angle_rad))
    end_y = int(y + length * math.sin(angle_rad))
    cv2.arrowedLine(img, (x, y), (end_x, end_y), color, thickness, tipLength=0.3)

class Marker():
    
    def __init__(self, corners=np.zeros((0), float), id=-1,
                 rvec=np.zeros((1, 2), float), tvec=np.zeros((1, 2), float)): # , objPoint=np.zeros((0), float)
        self.corners = corners
        self.id = id
        self.rvec = rvec
        self.tvec = tvec
        # self.objPoint = objPoint
        self.distance = cv2.norm(tvec[0], cv2.NORM_L2)

 
        
    @property
    def center(self):
        return np.average(self.corners, axis=0)
    
    @property
    def center3d(self):
        return self.tvec



class MarkerReader():
    
    def __init__(self, markerId, 
                 markerDictionary, 
                 markerSize, 
                 cameraMatrix, 
                 distortionCoeffs):
        self.markerId = markerId
        self.cameraMatrix = cameraMatrix # needs to be calculate with a chessboard
        self.distortionCoeffs = distortionCoeffs
        
        self.markerDictionary = markerDictionary
        self.markerSize = markerSize
        self.marker = Marker()
        self.nbMarkers=0
        
        self.noDistortion=np.zeros((5), dtype=np.float32)
        
        
        
        # self.withKalmanFilter = withKalmanFilter
        # parameters = cv2.aruco.DetectorParameters()
        # markerDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_50"])
        # self.detector = cv2.aruco.ArucoDetector(markerDict, parameters)
    
    def detectMarkers(self, img, dictionary):
        image = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = dictionary
        parameters = cv2.aruco.DetectorParameters_create()
        # global arucoParams
        
        (allCorners, ids, rejected) = cv2.aruco.detectMarkers(gray, dictionary, parameters=arucoParams,
                                                                    # cameraMatrix=self.cameraMatrix,
                                                                    # distCoeff=self.distortionCoeffs
                                                                    )
        # allCorners, ids, _ = self.detector.detectMarkers(img)
        
        if len(allCorners) > 0:
            self.marker = Marker()
            self.nbMarkers=0
            for i in range(0, len(ids)):
                
                (topLeft, topRight, bottomRight, bottomLeft) = allCorners[i].reshape((4, 2))
                
                rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(allCorners,
                                                                              self.markerSize,
                                                                              cameraMatrix=self.cameraMatrix,
                                                                              distCoeffs=self.distortionCoeffs)
                
                
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(image, allCorners) 
                
                if ids[i] == self.markerId:
                    self.nbMarkers+=1
                    self.marker = Marker(allCorners, self.markerId, rvecs[i], tvecs[i]) # , objPoints[i]
                    # cv2.aruco.drawAxis(image, self.cameraMatrix, self.distortionCoeffs, rvecs[i], tvecs[i], 51)
                    # if self.withKalmanFilter:
                    #     tvec = self.marker.tvec[0]  # (x, y, z)
                    #     rvec = self.marker.rvec[0]
                    #     yaw = get_yaw_from_rvec(rvec)
                        
                    #     measurement = np.array([tvec[0], tvec[1], yaw]).astype(np.float32)
                    #     self.kf.correct(measurement)
                    #     predicted = self.kf.predict()
                        
                    #     print(f"Kalman predction: {predicted[0]}, {predicted[1]}, {predicted[2]}")
                        
                    #     # self.drawKalmanPredictions(image, self.cameraMatrix, self.distortionCoeffs)
                    
                    cv2.drawFrameAxes(image, self.cameraMatrix, self.distortionCoeffs, rvecs[i], tvecs[i], 40)
                    return True, image, self.marker
        
        return False, image, self.marker
    
    def drawMarkers(self, img, aruco_dict_type, matrix_coefficients, distortion_coefficients):

        frame = img.copy()
        '''
        frame - Frame from the video stream
        matrix_coefficients - Intrinsic matrix of the calibrated camera
        distortion_coefficients - Distortion coefficients associated with your camera
        
        return:-
        frame - The frame with the axis drawn on it
        '''
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters_create()
        
        
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,
                                                                    parameters=parameters,
                                                                    cameraMatrix=matrix_coefficients,
                                                                    distCoeff=distortion_coefficients)

        # If markers are detected
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                               distortion_coefficients)
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(frame, corners) 

                if ids[i] == self.markerId:
                    # Draw Axis
                    cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  
                    # if self.withKalmanFilter:
                    #     self.drawKalmanPredictions(img, matrix_coefficients, distortion_coefficients)
                    
                    
        return frame
    
    def drawKalmanPredictions(self, img, matrix_coefficients, distortion_coefficients):
        
            # Kalman prediction
        predicted = self.kf.predict()
        pred_x, pred_y, pred_theta = predicted[0], predicted[1], predicted[2]

        # Project Kalman prediction into image coordinates for drawing
        print(pred_x, pred_y)
        world_point = np.array([[pred_x, pred_y, 0.0]], dtype=np.float32)
        rvec_cam = np.zeros((3, 1))  # camera facing forward
        tvec_cam = np.zeros((3, 1))
        imgpts_pred, _ = cv2.projectPoints(world_point, rvec_cam, tvec_cam, matrix_coefficients, distortion_coefficients)
        pred_2d = tuple(imgpts_pred[0][0].astype(int))

        # Draw predicted position and orientation (red)
        cv2.circle(img, pred_2d, 5, (0, 0, 255), -1)
        draw_arrow(img, pred_2d, pred_theta, length=50, color=(0, 0, 255), thickness=2)
