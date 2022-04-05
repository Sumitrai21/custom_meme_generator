import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib 
import time


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def get_landmark_points(detector,predictor,image)->np.array:
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


def create_traingles(image,convexhull,landmark_points):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image_gray)
    cv2.fillConvexPoly(mask,convexhull,255)
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points.tolist())
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles,np.int32)

    indexes_triangle = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])


        index_pt1 = np.where((landmark_points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((landmark_points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((landmark_points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangle.append(triangle)


    return indexes_triangle


def get_triangulation_points(img,landmarks_points,triangle_index):
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = img[y: y + h, x: x + w]
    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    return points, cropped_triangle, (x,y,w,h)



def get_new_face(img,img2,indexes_triangles,landmarks_points,landmarks_points2):
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        points, cropped_triangle,_ = get_triangulation_points(img,landmarks_points,triangle_index)

        # Triangulation of second face
        points2, cropped_triangle_2,(x,y,w,h) = get_triangulation_points(img2,landmarks_points2,triangle_index)

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area


    return img2_new_face


def swap_faces(img,convexhull2,img2_new_face):
    img2_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)

    img2_head_noface = cv2.bitwise_and(img, img, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

    return seamlessclone


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
        #destination image landmarks extraction
        landmark_points = get_landmark_points(self.detector,self.predictor,self.dst_image)
        convex_hull = create_convex_hull(landmark_points)
        
        indexes_triangle = create_traingles(self.dst_image,convex_hull,landmark_points)
        #print(indexes_triangle)

        #meme image feature extraction
        landmark_points2 = get_landmark_points(self.detector,self.predictor,self.meme_image)
        convex_hull_2 = create_convex_hull(landmark_points2)
        img2_new_face = get_new_face(self.dst_image,self.meme_image,indexes_triangle,landmark_points,landmark_points2)
        self.final_output = swap_faces(self.meme_image,convex_hull_2,img2_new_face)


    def display_output(self):
        if self.final_output is not None:
            image = cv2.cvtColor(self.final_output,cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.show()




    
