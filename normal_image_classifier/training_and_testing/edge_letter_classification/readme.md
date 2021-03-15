# Clasification using Sobel edge detection

## Procedure
1) Manually gathered data. 
2) Trained a CNN by using RGB, grayscale, sobel edge highlighting (EH for short) operator and finally coverting EH image to a binary image.
3) Compared performance and chose EH for preprossing.

### Model
![alt text](https://github.com/Chamodya-ka/Sign-language-interpreter/blob/main/normal_image_classifier/edge_letter_classification/images/model.jpg)

## Results

With limited number of train images (~200 images per letter) the resulting model was able to accurately classify images with a simple background and good lighting conditions. For complex background the model performed poorly specially for similar signs like "a" and "t", for non similar signs the model performs well. The accuracy for complex backgrounds can be improved by external lighting focusing the hand and closed up images, but it is not practical for our case. As always more data with variations will also improve the accuracy. Due to the inconsistencies, we chose to use the key points for ASL letter classification. 

### Incorrect predictions and low accuracy
In both cases correct label was 'A'

![alt text](https://github.com/Chamodya-ka/Sign-language-interpreter/blob/main/normal_image_classifier/edge_letter_classification/images/False_test/A_as_T.jpg)

![alt text](https://github.com/Chamodya-ka/Sign-language-interpreter/blob/main/normal_image_classifier/edge_letter_classification/images/False_test/A_low_acc.jpg)

