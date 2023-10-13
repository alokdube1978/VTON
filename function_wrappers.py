import createpngmaskmultiselfie as overlay
from flask import Flask, request, make_response
import base64
import cv2
import sys
import urllib
app = Flask(__name__)

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
    # 'thorax_top':[245,120],
    # 'thorax_bottom':[245,369],
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






def get_preview_image(jewellery_image,jewellery_position,RUN_CV_SELFIE_SEGMENTER=True):
        
    imgOut=overlay.get_sample_preview_image(jewellery_image,jewellery_position,RUN_CV_SELFIE_SEGMENTER)
    return imgOut


def get_masked_image(jewellery_image,jewellery_position, human_image,RUN_CV_SELFIE_SEGMENTER=True,debug=False):
    imgOut=overlay.get_final_image(jewellery_image,jewellery_position, human_image,RUN_CV_SELFIE_SEGMENTER,debug)
    return imgOut


@app.route('/', methods=['GET'])
def index():
    return 'Flask Webserver for Serving VTON'

@app.route('/overlayimage', methods=['GET'])
def overlayimage():
    #copy paste values from list of jewellery_position values given above for relvant image
    ## For exmaple, if using necklace7, use thorax_top and thorax_bottom from necklace7 above
    jewellery_position={
        'thorax_top':[245,120],
        'thorax_bottom':[245,369],
        }
        
    jewellery_image=cv2.imread("D:\\VTON\\overlay\\necklace5.jpg",cv2.IMREAD_UNCHANGED)
    human_image=cv2.imread('D:\\VTON\\overlay\\public.jpg',cv2.IMREAD_UNCHANGED)
    # imgOut=overlay.get_sample_preview_image(jewellery_image,jewellery_position,RUN_CV_SELFIE_SEGMENTER=True)
   
    imgOut=get_masked_image(jewellery_image,jewellery_position,human_image,RUN_CV_SELFIE_SEGMENTER=True,debug=False)
    # image_url = request.args.get('imageurl')
    # requested_url = urllib.urlopen(image_url)
    # image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
    # img = cv2.imdecode(image_array, -1)
    # Do some processing, get output_img

    retval, buffer = cv2.imencode('.png', imgOut)
    response=make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    return response
    




@app.route('/preview', methods=['GET'])
def preview():
    # ##necklace8.png
    jewellery_position={
    'thorax_top':[180,90],
    'thorax_bottom':[180,275]
    }
    
    jewellery_image=cv2.imread("D:\\VTON\\overlay\\necklace8.png",cv2.IMREAD_UNCHANGED)
    # human_image=cv2.imread('D:\\VTON\\overlay\\public.jpg',cv2.IMREAD_UNCHANGED)
    imgOut=overlay.get_sample_preview_image(jewellery_image,jewellery_position,RUN_CV_SELFIE_SEGMENTER=True)
   
    # imgOut=get_masked_image(jewellery_image,jewellery_position,human_image,RUN_CV_SELFIE_SEGMENTER=True,debug=False)
    # image_url = request.args.get('imageurl')
    # requested_url = urllib.urlopen(image_url)
    # image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
    # img = cv2.imdecode(image_array, -1)
    # Do some processing, get output_img

    retval, buffer = cv2.imencode('.png', imgOut)
    response=make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    return response


if __name__ == '__main__':
    app.run(debug=True)







