# Sign Letter Classification using Mediapipe

This directory contains the files used for classifying ASL letters using Mediapipe Hands.

### Data collection steps
* Took multiple images for each letter.
* Located the keypoints for each image (21 each).
* Saved the data in CSV format:
  * Features: x and y coordinates of each keypoint (42 features in total)
  * Label: letter (currently, a-y due to z being a dynamic sign)
* Augmented the data using the data augmentation functions.
* Saved the result in a CSV format.

### Training
A deep neural network was trained using the 42 features as input and the one-hot encoded version of the labels as the output.

### Results
The model was accurate when the hand was stationary. However there were drops in accuracy when the hand moved. This is due to the low framerate of the camera which resulted in blurry images.
