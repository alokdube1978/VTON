# import extractfacemaskfromimg as extract_face
import createpngmaskmultiselfie as overlay
import cv2
import sys


    #necklace1.jpg
    # jewellery_position={
    # 'thorax_top':[404,320],
    # 'thorax_bottom':[404,740],
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


    # #necklace3.jpg
    # jewellery_position={
    # 'thorax_top':[203,307],
    # 'thorax_bottom':[203,481],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }


    # ##necklace4.jpg
    # jewellery_position={
    # 'thorax_top':[225,298],
    # 'thorax_bottom':[225,525],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }


    # ##necklace5.jpg
    # jewellery_position={
    # 'thorax_top':[245,160],
    # 'thorax_bottom':[245,409],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }


    # ##necklace6.jpg
    # jewellery_position={
    # 'thorax_top':[111,40],
    # 'thorax_bottom':[111,240],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }

    # ##necklace7.png
    # jewellery_position={
    # 'thorax_top':[128,93],
    # 'thorax_bottom':[128,293],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }

    # ##necklace8.png
    # jewellery_position={
    # 'thorax_top':[180,90],
    # 'thorax_bottom':[180,275],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }

    # ##necklace9.png
    # jewellery_position={
    # 'thorax_top':[417,257],
    # 'thorax_bottom':[417,757],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }

    # ##necklace10.png
    # jewellery_position={
    # 'thorax_top':[143,111],
    # 'thorax_bottom':[143,301],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }


jewellery_position={
    'thorax_top':[180,90],
    'thorax_bottom':[180,275]
    }

def get_preview_image(jewellery_image,jewellery_position,RUN_CV_SELFIE_SEGMENTER=True):
        
    imgOut=overlay.get_sample_preview_image(jewellery_image,jewellery_position,RUN_CV_SELFIE_SEGMENTER)
    return imgOut


def get_masked_image(jewellery_image,jewellery_position, human_image,RUN_CV_SELFIE_SEGMENTER=True):
    imgOut=overlay.get_final_image(jewellery_image,jewellery_position, human_image,RUN_CV_SELFIE_SEGMENTER)
    return imgOut
    
jewellery_image=cv2.imread("D:\\VTON\\overlay\\necklace8.png",cv2.IMREAD_UNCHANGED)
human_image=cv2.imread('D:\\VTON\\overlay\\public.jpg',cv2.IMREAD_UNCHANGED)
# imgOut=get_preview_image(jewellery_image,jewellery_position)
imgOut=get_masked_image(jewellery_image,jewellery_position,human_image,RUN_CV_SELFIE_SEGMENTER=True)
cv2.namedWindow("Image")
cv2.imshow("Image", imgOut)   
cv2.waitKey(0) 
cv2.destroyAllWindows()

sys.exit()

input_path = 'D:\\VTON\\overlay\\public.jpg'
img = cv2.imread(input_path,cv2.IMREAD_UNCHANGED) 
imgOut,positions=extract_face.getSelfieImageandFaceLandMarkPoints(img)
imgOut_landmarks=extract_face.draw_points_on_image(imgOut,positions)



