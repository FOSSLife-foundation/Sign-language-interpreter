-This contrains training, predicting and preprocessing functions for ASL word classificaion using unique frames.

-Since egde highlighted images contains noise, the data was converted to a binary image (Black & White) using a threshold for edge highlighed images. 
  if pixel_value > 127 then pixel_value=255 and vice versa.
-This method was effective with a few data images per label. 

-By increasing the size of the dataset, the edge highlighted classification showed higher accuracy than the binary image classifier. 
  In low lighting conditions where edges are not  distinct, the treshold required to distinguish the edge in the binary image may need to vary. Since we do not vary the treshold,   this may leads to loss of information used for classification.
