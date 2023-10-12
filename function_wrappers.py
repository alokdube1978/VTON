import extractfacemaskfromimg as extract_face
import cv2

input_path = 'D:\\VTON\\overlay\\public.jpg'
img = cv2.imread(input_path,cv2.IMREAD_UNCHANGED) 
imgOut,positions=extract_face.getSelfieImageandFaceLandMarkPoints(img)
imgOut_landmarks=extract_face.draw_points_on_image(imgOut,positions)

cv2.namedWindow("Landmark points Image")
cv2.imshow("Landmark points Image", imgOut_landmarks)   
cv2.waitKey(0) 
cv2.destroyAllWindows()

