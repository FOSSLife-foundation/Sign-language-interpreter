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
* The model was accurate when the hand was stationary. However there were drops in accuracy when the hand moved or when the hand rotated too much.
* The letter 'h' seems to be detected only when the thumb is pointed upwards. If not, it was often detected as 'g'. This may be due to an unintentional bias that was included when collecting data. More mixed data with the thumb resting may fix this issue.
* This is due to the low framerate of the camera which resulted in blurry images. The model was tested using "Stochastic Gradient Descent(SGD)" and "Adam" as optimizers. Overall, the "Adam" optimizer seemed to perform better. "SGD", too should be tested with more images and longer training time as it tends to generalize better.
