# Learn Convolutional Neural Network for Face Anti-Spoofing using pytorch  

## requirements  

* pytorch
* cv2
* tensorflow
* [mtcnn][1]

## Step 1  

run `generate_frames_and_bbox.py`,video is sampled as a frame,also generate a file_list containing the list of files_name and the bbox of the face   

## Step 2  

run `crop_image.py`,generate face photos at different scales

To facilitate training, generate a file_list for each scale.  

## Step 3  

run `train.py`,a network will be trained and tested every n epochs
