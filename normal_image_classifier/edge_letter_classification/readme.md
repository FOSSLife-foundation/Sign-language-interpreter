# Clasification using Sobel edge detection

## Results

With limited number of train images (~200 images per letter) the resulting model was able to accurately classify images with a simple background and good lighting conditions. For complex background the model performed poorly specially for similar signs like "a" and "t", for non similar signs the model performs well. The accuracy for complex backgrounds can be improved by external lighting focusing the hand and closed up images, but it is not practical for our case. As always more data with variations will also improve the accuracy. Due to the inconsistencies, we chose to use the key points for ASL letter classification. 

