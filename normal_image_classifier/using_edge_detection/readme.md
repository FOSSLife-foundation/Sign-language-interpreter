# Clasification using Sobel edge detection

## Model

![Model_Summary](images/model.jpg?raw=true "Title")

## Results

With limited number of train images (300-400 images per letter) the resulting model was able to accurately classify images with a simple background and good lighting conditions. For complex background the model performed poorly specially for similar signs like "a" and "e", for non similar signs the model performs well. The accuracy for complex backgrounds can be improved by external lighting focusing the hand and closed up images, but it is not practical for our case. As always more data with variations will also improve the accuracy. Due to the inconsistencies, we chose to use the key points for ASL letter classification. 

### False Positives
Under natural light

![false_pos_1](images/False_test/E_complex_low_false_pos.jpg?raw=true "Title")
![false_pos_1](images/False_test/G_complex_low_false.jpg?raw=true "Title")
