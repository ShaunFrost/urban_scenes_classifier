# Urban Scenes Classifier
Classification of Google Urban Scenes dataset images into the corresponding city.

Checkout the project report to get a detailed idea.


MATLAB version used: 2019a


First download the already augmented and resized data folder from this google drive link:


https://drive.google.com/drive/folders/1EuDbDcExF9BpdEj3J696SfcwdP2HwvLM?usp=sharing


Download the whole data folder and put in same path as these matlab files.


Main matlab files
-----------------
1>city_classifier.m -> This file has the ConvNet approach. Simply run the file.

2>surf_features.m -> This file has the Bag of Visual approach. Simply run the file.

3>bow_tester.m -> This function uses various % strong feature values to check performance of the visual words method. Based on results of this we select the value and run the surf_feature method, which in our case was 50% strong features.


Data
----
All augmented data folders and scaled down image folders are also included. No need to run the utility methods to get them. data folder has the required data.


Utility matlab files
--------------------
1> augment_function.m -> This function createds augmented image from the original images.

2> crop_function.m -> This function is used by augmentation process to center crop the final part.

3> resizer.m -> This function is used to scale down the original high definition images(I've not included them here as it was provided to me)
