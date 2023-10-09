from rembg import remove,new_session
from cvzone.FaceDetectionModule import FaceDetector
import mediapipe as mp
import cv2
import cvzone
import os
import math as math
import numpy as np
import sys
import pprint
from mediapipe.python._framework_bindings import image
from mediapipe.python._framework_bindings import image_frame
from mediapipe.tasks.python import vision
from mediapipe import tasks
from cvzone.ClassificationModule import Classifier
pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(threshold=sys.maxsize)
model_path="D:\Python\VTON\\Models\selfie_multiclass_256x256.tflite"
human_path = 'D:\\Python\\VTON\\overlay\\human_image2.jpg'
input_path = 'D:\\Python\\VTON\\overlay\\necklace4.jpg'
BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white
model="isnet-general-use"
session=new_session(model)

#mediapipe initialization
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = BaseOptions(model_asset_path=model_path)
options = vision.ImageSegmenterOptions(base_options=base_options,running_mode=VisionRunningMode.IMAGE,
                                              output_category_mask=1)

#neckalce image
img = cv2.imread(input_path,cv2.IMREAD_UNCHANGED)
#human image
human_image=cv2.imread(human_path,cv2.IMREAD_UNCHANGED)

#initialization of mediapipe selfie_multiclass
human_image_tf = mp.Image(image_format=mp.ImageFormat.SRGB, data=human_image)
with vision.ImageSegmenter.create_from_options(options) as segmenter:
    segmentation_result = segmenter.segment(human_image_tf)
human_image=human_image_tf.numpy_view().copy()

face_cascade = cv2.CascadeClassifier('D:\\Python\\VTON\\data\\haarcascades\\haarcascade_frontalface_default.xml')
# smile_cascade = cv2.CascadeClassifier('D:\\Python\\VTON\\Models\selfie_multiclass_256x256\\OpenCV\\data\\haarcascades\\haarcascade_smile.xml')

# Initialize the FaceDetector object
# minDetectionCon: Minimum detection confidence threshold
# modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=1)



def detect_reapply_face(frame,nose_slope,nose_thorax_scale):
    
    print ("in detect_reapply_face")

    # cv2.imshow("Frame",frame)

    print(frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray",gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    img, bboxs = detector.findFaces(frame, draw=False)
    
    x, y, w, h = bboxs[0]['bbox']
    
    nose_slope_degrees=math.degrees(math.atan(nose_slope))
    if (nose_slope_degrees)>0 and (nose_slope_degrees<=60):
        print("0<slope<60")
        y=y
        x=x-round(40/(0.4)*nose_thorax_scale)
        h=round(h)-round(30/(0.4)*nose_thorax_scale)
        w=round(w)-round(20/(0.4)*nose_thorax_scale)
        
    elif (nose_slope_degrees)>60 and (nose_slope_degrees<90):
        print("60<slope<90")
        y=y
        x=x-round(40/(0.4)*nose_thorax_scale)
        h=round(h)-round(30/(0.4)*nose_thorax_scale)
        w=round(w)
        
    elif ((nose_slope_degrees)>90 or (nose_slope_degrees<=0 and nose_slope_degrees<-70)):
        print("-70<slope<=0 ")
        y=y
        x=x-round(40/(0.4)*nose_thorax_scale)
        h=round(h)-round(30/(0.4)*nose_thorax_scale)
        w=round(w)+round(40/(0.4)*nose_thorax_scale)
       
    elif (nose_slope_degrees<=0 and nose_slope_degrees>=-70):
        print("slope>= -70 ")
        y=y
        x=x+round(10/(0.4)*nose_thorax_scale)
        h=round(h)-round(30/(0.4)*nose_thorax_scale)
        w=round(w)
        
    print(nose_slope_degrees)
    print(nose_thorax_scale)
    
    
    # if (w>h):
        # delta=w-h
        # w=w-delta
    
    # if (h>w):
        # delta=h-w
        # h=h-delta
    
    
    print (x,y,w,h)   
    img=frame[y:y+h,x:x+w,:]
    cv2.imshow("CVDetecting",img)
    print (x,y,w,h)
    print(img.shape)
    return y,x,img
    

def detect_reapply_face_multiscale(imgOverlay,human_image_copy,segmentation_result):
    condition_hair = np.stack((segmentation_result.confidence_masks[1].numpy_view(),) * 3, axis=-1) > 0.2
    condition_background = np.stack((segmentation_result.confidence_masks[0].numpy_view(),) * 3, axis=-1) > 0.2
    condition_face_skin=np.stack((segmentation_result.confidence_masks[3].numpy_view(),) * 3, axis=-1) > 0.2
    combined_condition=(condition_hair| condition_background| condition_face_skin) 
    output_combined_image=np.where(combined_condition,human_image_copy, imgOverlay)
    return output_combined_image
     
     
def rotate_image(image,rotation_point, angle):
  rot_mat = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result      
   

def create_mask_img(img):
    global session
    '''
        Arguments:

    img - image of object in CV2 RGB format

    
    Function returns background with added object in CV2 RGB format
    
    CV2 RGB format is a np array with dimensions width x height x 3
    '''
    # img = segmentor.removeBG(img, imgBg=(255, 255, 255), cutThreshold=0.01)
    # cv2.imshow("segmented",img)
    #we first prepare the mask from img
   
    # im=remove(
    # img,
    # alpha_matting=True,
    # alpha_matting_foreground_threshold=240,
    # alpha_matting_background_threshold=10,
    # alpha_matting_erode_structure_size=1,
    # alpha_matting_base_size=100,
    # )
    
    
    bg = remove(img,session=session)
    img_black_bg=bg.copy()
    
    # cv2.imshow("bgremove",bg)
    img_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    
    ret, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(mask) #we need to set those regions to remove as black
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) #mask to rgb mask to get True/False
    h_bg, w_bg = bg.shape[0], bg.shape[1]
    mask_black = np.all(img_black_bg == [0, 0, 0, 0], axis=-1)
    img_black_bg[mask_black]=[255,255,255,0]
    return mask,img_black_bg



def rescale_coordinates(positions,scale):
    rescaled_positions={}
    for key in positions:
        # print(type(positions[key]))
        # print(positions[key])
        if isinstance(positions[key], list):
            rescaled_positions[key]=[round(positions[key][0]*scale),round(positions[key][1]*scale)]
        else:
            rescaled_positions[key]=round(positions[key]*scale)
    # print(real_positions)    
    return rescaled_positions 
    


def xy_coordinate_positions(positions):
    # print("xy_coordinate_positions")
    real_positions = {}
    for key in positions:
        # print(type(positions[key]))
        # print(positions[key])
        if isinstance(positions[key], list):
            real_positions[key]=[positions[key][0],-1*positions[key][1]]
        else:
            real_positions[key]=positions[key]
    # print(real_positions)    
    return real_positions    

def img_position_from_xy_coordinate_positions(positions):
    # print("img_position_from_xy_coordinate_positions")
    img_positions = {}
    for key in positions:
        # print(type(positions[key]))
        # print(positions[key])
        if isinstance(positions[key], list):
            img_positions[key]=[positions[key][0],-1*positions[key][1]]
        else:
            img_positions[key]=positions[key]
        
    return img_positions
    
def offset_coordinates(positions,x,y):
    # print("img_position_from_xy_coordinate_positions")
    offset_positions = {}
    for key in positions:
        # print(type(positions[key]))
        # print(positions[key])
        if isinstance(positions[key], list):
            offset_positions[key]=[positions[key][0]-x,positions[key][1]-y]
        
        
    return offset_positions
    
def slope_intercept(p1,p2):
    # print(p1,p2)
    slope=(p2[1]-p1[1])/(p2[0]-p1[0])
    # print(math.degrees(math.atan(slope)))
    intercept=p1[1]-slope*p1[0]
    return slope,intercept

cv2.namedWindow("Masked Image")
cv2.moveWindow("Masked Image", 10,10)
human_image_copy=human_image.copy()
mask,masked_image = create_mask_img(img)
# cv2.imwrite(masked_path,masked_image)
# sys.exit()
# print(masked_image.shape)
# print(masked_image[0][0])

# print(masked_image.shape)
# print(masked_image[0][0])




#necklace1.jpg
# jewellery_position={
# 'thorax_top':[404,270],
# 'thorax_bottom':[404,690],
# 'thorax_midpoint':[0,0],
# 'left_shoulder_pivot':[794,456],
# 'right_shoulder_pivot':[5,456]
# }

# # #necklace2.jpg
# jewellery_position={
# 'thorax_top':[184,165],
# 'thorax_bottom':[184,403],
# 'thorax_midpoint':[0,0],
# 'left_shoulder_pivot':[306,304],
# 'right_shoulder_pivot':[84,304]
# }


# # #necklace3.jpg
# jewellery_position={
# 'thorax_top':[203,307],
# 'thorax_bottom':[203,481],
# 'thorax_midpoint':[0,0],
# 'left_shoulder_pivot':[385,392],
# 'right_shoulder_pivot':[25,392]
# }


##necklace4.jpg


jewellery_position={
'thorax_top':[225,298],
'thorax_bottom':[225,525],
'thorax_midpoint':[0,0],
'left_shoulder_pivot':[385,392],
'right_shoulder_pivot':[25,392]
}

jewellery_xy_position={}
jewellery_xy_position=xy_coordinate_positions(jewellery_position)
jewellery_xy_position["thorax_midpoint"]=[round((jewellery_xy_position["thorax_top"][0]+jewellery_xy_position["thorax_bottom"][0])/2),round((jewellery_xy_position["thorax_top"][1]+jewellery_xy_position["thorax_bottom"][1])/2)]



jewellery_position['thorax_top_bottom_distance']=math.dist(jewellery_xy_position['thorax_top'],jewellery_xy_position['thorax_bottom'])
jewellery_xy_position['right_shoulder_pivot'][0]=round((jewellery_xy_position["thorax_midpoint"][0]-jewellery_position['thorax_top_bottom_distance']/2))
jewellery_xy_position['right_shoulder_pivot'][1]=jewellery_xy_position["thorax_midpoint"][1]
jewellery_xy_position['left_shoulder_pivot'][0]=round((jewellery_xy_position["thorax_midpoint"][0]+jewellery_position['thorax_top_bottom_distance']/2))
jewellery_xy_position['left_shoulder_pivot'][1]=jewellery_xy_position["thorax_midpoint"][1]


jewellery_position['right_left_shoulder_pivot_distance']=math.dist(jewellery_xy_position['right_shoulder_pivot'],jewellery_xy_position['left_shoulder_pivot'])
jewellery_xy_position['thorax_top_bottom_distance']=jewellery_position['thorax_top_bottom_distance']
jewellery_xy_position['right_left_shoulder_pivot_distance']=jewellery_position['right_left_shoulder_pivot_distance']
jewellery_position=img_position_from_xy_coordinate_positions(jewellery_xy_position)
print("----Jewellery Position----")
print(jewellery_position)

# # human_image.jpg
# face_position={   
    # 'eye_midpoint': [868, 304],
    # 'left_eye': [920, 333],
    # 'left_shoulder': [1020, 587],
    # 'nose': [852, 346],
    # 'right_eye': [816, 275],
    # 'right_shoulder': [537, 481],
    # 'thorax_midpoint': [778, 534],
    # 'left_shoulder_pivot':[0,0],
    # 'right_shoulder_pivot':[0,0]
# }

# # human_image1.jpg
# face_position={   
    # 'eye_midpoint': [721, 273],
    # 'left_eye': [777, 268],
    # 'left_shoulder': [976, 517],
    # 'nose': [730, 317],
    # 'right_eye': [665, 278],
    # 'right_shoulder': [511, 522],
    # 'thorax_midpoint': [744, 520],
    # 'left_shoulder_pivot':[0,0],
    # 'right_shoulder_pivot':[0,0]
# }

# #human_image2.jpg
face_position={   
    'eye_midpoint': [236, 348],
    'left_eye': [289, 350],
    'left_shoulder': [403, 559],
    'nose': [236, 382],
    'right_eye': [182, 345],
    'right_shoulder': [64, 544],
    'thorax_midpoint': [234, 552],
    'left_shoulder_pivot':[0,0],
    'right_shoulder_pivot':[0,0]
}

# human_image3.jpg
# face_position={    
    # 'eye_midpoint': [352, 192],
    # 'left_eye': [385, 189],
    # 'left_shoulder': [488, 296],
    # 'nose': [357, 213],
    # 'right_eye': [318, 194],
    # 'right_shoulder': [236, 328],
    # 'thorax_midpoint': [362, 312],
    # 'left_shoulder_pivot':[0,0],
    # 'right_shoulder_pivot':[0,0]   
# }


# # human_image5
# face_position={ 
    # 'eye_midpoint': [724, 392],
    # 'left_eye': [784, 391],
    # 'left_shoulder': [963, 610],
    # 'nose': [728, 435],
    # 'right_eye': [664, 393],
    # 'right_shoulder': [508, 605],
    # 'thorax_midpoint': [736, 608],
    # 'left_shoulder_pivot':[0,0],
    # 'right_shoulder_pivot':[0,0]   
# }


# # human_image6
# face_position={
    # 'eye_midpoint': [278, 122],
    # 'left_eye': [346, 120],
    # 'left_shoulder': [498, 424],
    # 'nose': [276, 164],
    # 'right_eye': [210, 123],
    # 'right_shoulder': [80, 421],
    # 'thorax_midpoint': [289, 422],
    # 'left_shoulder_pivot':[0,0],
    # 'right_shoulder_pivot':[0,0] 
# }


# #human_image7
# face_position={
    # 'eye_midpoint': [276, 140],
    # 'left_eye': [325, 137],
    # 'left_shoulder': [437, 350],
    # 'nose': [275, 166],
    # 'right_eye': [228, 143],
    # 'right_shoulder': [130, 335],
    # 'thorax_midpoint': [284, 342],
    # 'left_shoulder_pivot':[0,0],
    # 'right_shoulder_pivot':[0,0]
# }


#human_image8
# face_position={
    # 'eye_midpoint': [176, 228],
    # 'left_eye': [246, 227],
    # 'left_shoulder': [368, 512],
    # 'nose': [174, 273],
    # 'right_eye': [106, 228],
    # 'right_shoulder': [-9, 513],
    # 'thorax_midpoint': [180, 512],
    # 'left_shoulder_pivot':[0,0],
    # 'right_shoulder_pivot':[0,0]
# }

#human_image9
# face_position={
    # 'eye_midpoint': [266, 135],
    # 'left_eye': [318, 138],
    # 'left_shoulder': [504, 438],
    # 'nose': [233, 167],
    # 'right_eye': [213, 132],
    # 'right_shoulder': [116, 428],
    # 'thorax_midpoint': [310, 433],
    # 'left_shoulder_pivot':[0,0],
    # 'right_shoulder_pivot':[0,0]
# }

face_xy_position=xy_coordinate_positions(face_position)
nose_thorax_scale=40/100

face_position["face_nose_thorax_distance"]=math.dist(face_xy_position["nose"],face_xy_position["thorax_midpoint"])

reduced_circle_radius=round(face_position["face_nose_thorax_distance"] * nose_thorax_scale)
face_xy_position["face_nose_thorax_distance"]=face_position["face_nose_thorax_distance"]
face_xy_position["thorax_top_bottom_distance"]=reduced_circle_radius*2



nose_slope,nose_intercept=slope_intercept(face_xy_position["nose"],face_xy_position["thorax_midpoint"])
print("----nose slope,intercept----")
print (nose_slope,nose_intercept)
print (math.degrees(math.atan(nose_slope)))

shoulder_slope,shoulder_intercept=slope_intercept(face_xy_position["left_shoulder"],face_xy_position["right_shoulder"])
print("----shoulder slope,intercept----")
print (shoulder_slope,shoulder_intercept)


face_xy_position["thorax_top"]=[0,0]
face_xy_position["thorax_bottom"]=[0,0]
# print(math.sin(math.atan(shoulder_slope)))

if (nose_slope>=0):
    face_xy_position["thorax_top"][0]=round(face_xy_position["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.atan(nose_slope)))
    face_xy_position["thorax_top"][1]=round(face_xy_position["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.atan(nose_slope)))
else:
    face_xy_position["thorax_top"][0]=round(face_xy_position["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.pi+math.atan(nose_slope)))
    face_xy_position["thorax_top"][1]=round(face_xy_position["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.pi+math.atan(nose_slope)))

if (nose_slope>=0):
    face_xy_position["thorax_bottom"][0]=round(face_xy_position["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.pi+math.atan(nose_slope)))
    face_xy_position["thorax_bottom"][1]=round(face_xy_position["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.pi+math.atan(nose_slope)))
else:
    face_xy_position["thorax_bottom"][0]=round(face_xy_position["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.atan(nose_slope)))
    face_xy_position["thorax_bottom"][1]=round(face_xy_position["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.atan(nose_slope)))    


if (shoulder_slope<0):
    face_xy_position["right_shoulder_pivot"][0]=round(face_xy_position["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.pi+math.atan(shoulder_slope)))
    face_xy_position["right_shoulder_pivot"][1]=round(face_xy_position["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.pi+math.atan(shoulder_slope)))
else:
    face_xy_position["right_shoulder_pivot"][0]=round(face_xy_position["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.atan(shoulder_slope)))
    face_xy_position["right_shoulder_pivot"][1]=round(face_xy_position["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.atan(shoulder_slope)))   



if (shoulder_slope<=0):
    face_xy_position["left_shoulder_pivot"][0]=round(face_xy_position["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.atan(shoulder_slope)))
    face_xy_position["left_shoulder_pivot"][1]=round(face_xy_position["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.atan(shoulder_slope)))
else:
    face_xy_position["left_shoulder_pivot"][0]=round(face_xy_position["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.pi+math.atan(shoulder_slope)))
    face_xy_position["left_shoulder_pivot"][1]=round(face_xy_position["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.pi+math.atan(shoulder_slope)))
   


face_position=img_position_from_xy_coordinate_positions(face_xy_position)

print("-----Face Coordinates-----")
print(face_position)

for key in face_position:
     if isinstance(face_position[key], list):
        # print(key)
        cv2.circle(human_image, (face_position[key][0],face_position[key][1]), radius=3, color=(0, 0, 0), thickness=-1)


cv2.circle(human_image,face_position["thorax_midpoint"],radius=reduced_circle_radius,color=(0,0,255),thickness=1)

# cv2.imshow("Masked Image",human_image)




jewellery_resize_scale=face_position["thorax_top_bottom_distance"]/jewellery_position["thorax_top_bottom_distance"]

print("-----Jewellery Resize Scale-----")
print(jewellery_resize_scale)


jewellery_position["image_width"] = masked_image.shape[1]
jewellery_position["image_height"] = masked_image.shape[0]
jewellery_xy_position["image_width"]=jewellery_position["image_width"]
jewellery_xy_position["image_height"]=jewellery_position["image_height"]
jewellery_position=rescale_coordinates(jewellery_position,jewellery_resize_scale)
jewellery_xy_position=rescale_coordinates(jewellery_xy_position,jewellery_resize_scale)
masked_image=cv2.resize(masked_image, (jewellery_position["image_width"],jewellery_position["image_height"]), interpolation = cv2.INTER_AREA)




for key in jewellery_position:
    if isinstance(jewellery_position[key], list):
    # print(key)
        cv2.circle(masked_image, (jewellery_position[key][0],jewellery_position[key][1]), radius=3, color=(0, 255, 0), thickness=-1)

cv2.circle(masked_image,jewellery_position["thorax_midpoint"],radius=round(jewellery_position['thorax_top_bottom_distance']/2),color=(0,0,255),thickness=1)

cv2.imshow("mask",masked_image)
print("-----Jewellery Scaled Coordinates-----")
print(jewellery_position)
jewellery_to_human_image_midpoint_offset=[0,0]
jewellery_to_human_image_midpoint_offset[0]=face_position["thorax_midpoint"][0]-jewellery_position["thorax_midpoint"][0]
jewellery_to_human_image_midpoint_offset[1]=face_position["thorax_midpoint"][1]-jewellery_position["thorax_midpoint"][1]

jewellery_transform_final_points=offset_coordinates(face_position,jewellery_to_human_image_midpoint_offset[0],jewellery_to_human_image_midpoint_offset[1])


print("-----Jewellery Transform Coordinates-----")
print(jewellery_transform_final_points)


input_pts=np.float32([jewellery_position["left_shoulder_pivot"],jewellery_position["thorax_top"],jewellery_position["right_shoulder_pivot"],jewellery_position["thorax_bottom"]])
output_pts=np.float32([jewellery_transform_final_points["left_shoulder_pivot"],jewellery_transform_final_points["thorax_top"],jewellery_transform_final_points["right_shoulder_pivot"],jewellery_transform_final_points["thorax_bottom"]])





M = cv2.getPerspectiveTransform(input_pts,output_pts)

important_points=np.float32([
[[jewellery_position["thorax_midpoint"][0],jewellery_position["thorax_midpoint"][1]]],
[[0,0]],[[masked_image.shape[1],0]],
[[masked_image.shape[1],masked_image.shape[0]]],[[0,masked_image.shape[0]]],

])
transformed_important_points=cv2.perspectiveTransform(important_points,M)
print("transformed points")
print(important_points)
print("<==>")
print(transformed_important_points)

yadd=0
xadd=0
for points in transformed_important_points:
    # print(points[0][0])
    if (points[0][0]<xadd):
        xadd=points[0][0]
    if (points[0][1]<yadd):
        yadd=points[0][1]


print(xadd,yadd)
xadd=round(abs(xadd))
yadd=round(abs(yadd))
perspective_masked_image=cv2.warpPerspective(masked_image,M,(masked_image.shape[1]+xadd, masked_image.shape[0]+yadd),flags=cv2.INTER_LINEAR)

cv2.circle(perspective_masked_image,(jewellery_position["thorax_midpoint"][0],jewellery_position["thorax_midpoint"][1]),5,color=(0,0,0),thickness=-1)

overlaypoint_x=face_position["thorax_midpoint"][0]-jewellery_position["thorax_midpoint"][0]
overlaypoint_y=face_position["thorax_midpoint"][1]-jewellery_position["thorax_midpoint"][1]
print(overlaypoint_x, overlaypoint_y)

# imgray = cv2.cvtColor(human_image, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# im2=human_image.copy()
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(im2, contours, -1, (0,255,0), 3)
# cv2.imshow("Contours",im2)

# y1,x1,face_image=detect_reapply_face(human_image_copy,nose_slope,nose_thorax_scale)
# print("after detect")
# print (y1,x1)
# print (face_image.shape)

# cv2.imshow("FI",face_image)

imgOverlay = cvzone.overlayPNG(human_image, perspective_masked_image, pos=[overlaypoint_x, overlaypoint_y])
# cv2.imshow("IO",imgOverlay)

# print (human_image_copy[372][709])
# print (human_image_copy[709,:,:])
# human_image_gray = cv2.cvtColor(human_image_copy, cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(human_image_gray, 251, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
# cv2.imshow("Gray Mask",mask)
# sys.exit()
# print(imgOverlay[y1:y1+face_image.shape[0],x1:x1+face_image.shape[1],:].shape)


# imgOverlay[y1:y1+face_image.shape[0],x1:x1+face_image.shape[1],:]=face_image[0:face_image.shape[0],0:face_image.shape[1],:]


# cv2.imshow("Perspective",perspective_masked_image)

imgOverlay=detect_reapply_face_multiscale(imgOverlay,human_image_copy,segmentation_result)


# cv2.imshow("Human Image Copy",human_image_copy)
# for b in range(248,256):
    # for g in range(248,256):
        # for r in range (248,256):
            # color=(b,g,r)
            # imgOverlay[np.all(human_image_copy == color,axis=-1)]=(255,255,255)
            
# imgOverlay[np.all(human_image == (255,255,255),axis=-1)]=(255,255,255)
# human_image_filtered_copy[np.all(human_image_copy == (255,253,252),axis=-1)]=(0,255,0)          


cv2.imshow("Masked Image",imgOverlay)
# cv2.imwrite("D:\\Python\\VTON\\overlay\\final_image.jpg",imgOverlay)

# final_image[:,:,:]=

# imgOut = segmentor.removeBG(imgOverlay, imgBg=(255, 255, 255), cutThreshold=0.4)

# cv2.moveWindow("Masked Image",10,10)



# composition_2 = add_obj(human_image, perspective_masked_image,  overlaypoint_x, overlaypoint_y)
# cv2.imshow("Add Obj",composition_2)

sys.exit()
#####################


