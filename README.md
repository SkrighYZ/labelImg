# LabelImg with Tensorflow Models

## To Modify Model Configs
model/inference_and_save.py contains the functions to run the inference graph and save xml label files.  
Change the threshold variable to set confidence threshold.

## New Requirements
tensorflow, opencv

### Modified Files
labelImg.py in loadFile() at line 1037 and 1055.

### Added Files
model/.
test_imgs/.

