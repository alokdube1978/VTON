Run pip install -r requirements.txt to get all dependencies

all paths here are Windows based paths - please change them to Linux based

Refer
https://developers.google.com/mediapipe/solutions
https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
also note all image arrays are of the form [y][x] and not [x][y] for pixel points 


How to Run:
sudo python function_wrappers.py

Runs flask on port 80, please ensure it runs with sudo or change port in function_wrappers.py

***NOTE to Try different Jewellery Images***
In function_wrappers.py to try different Jewellery:
copy paste values from list of jewellery_position values given above for relvant image
For exmaple, if using necklace7, use thorax_top and thorax_bottom from necklace7 above
 


