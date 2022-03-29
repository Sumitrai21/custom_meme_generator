import numpy as np
import cv2
import dlib 



def get_convex_hull(image):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #mask = np.zeros_like(img_gray)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img_gray)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    for face in faces:
        landmarks = predictor(img_gray, face)
        
        landmarks_points = []
        for n in range(0,68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x,y))


            #cv2.circle(img,(x,y),3,(0,0,255),-1)

        points = np.array(landmarks_points,np.int32)
        hull = cv2.convexHull(points)

        return hull