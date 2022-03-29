import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib 
import time


def get_landmark_points(detector,predictor,image)->np.array:
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    faces = detector(img_gray)
    ##TODO: Right now would work with single face only
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

    return np.array(landmarks_points,np.int32)


def create_convex_hull(points):
    return cv2.convexHull(points)


def create_traingles(converhull):
    




class FaceSwapper():
    def __init__(self,dst_image=None,meme_image=None):
        self.dst_image = dst_image
        self.meme_image = meme_image
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.final_output = None


    def add_meme_template(self,path):
        self.meme_image = cv2.imread(path)
        ##TODO: add NoneTypeError
        

    def display_image(self,if_dst=True,if_meme=True):
        if if_dst and if_meme:
            f, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
            ax1.imshow(cv2.cvtColor(self.dst_image,cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(self.meme_image,cv2.COLOR_BGR2RGB))
            plt.show()

    def input_image(self,path):
        self.dst_image = cv2.imread(path)
        ##TODO: add NoneType error


    def create_swap(self):
        landmark_points = get_landmark_points(self.detector,self.predictor,self.dst_image)
        convex_hull = create_convex_hull(landmark_points)
        print(convex_hull)



    
