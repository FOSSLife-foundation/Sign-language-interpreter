# ASL Word classification using egde highlighed method on unique frames

## Procedure
 1. Manually gathered data. ie. Unique frames of a gesture of 7 selected ASL words,some gestures do not have a very distinct unique frame, they were split into 2 frames, start and 
end frames.

 2. Each label contained 200-300 images all pre processed under hightlighted operator.
 
 3. Trained a CNN by using sobel edge highlighting (EH for short) operator for pre-processing.

## Results
There are 8 output labels in total, 6 unique words and 2 starting and ending frame of a gesture. With the addition of more images since the previous iteration this model had a 
relatively higher accuracy than that of binary image classifier. As this is an uses direct camera input lighting conditions, camera angle, etc. effect the overall accuracy. 
