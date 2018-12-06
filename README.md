# App:  Lung MRI segmentation

Deep learning app made for proton-weighted MRI lung segmentation using ANTsRNet

## Model training notes

* Training data: Proton lung MRI of various slice thickness
* Unet model (see ``Scripts/Training/``).
* Template-based data augmentation

## Sample usage

```
#
#  Usage:
#    Rscript doBrainExtraction.R inputImage outputPrefix reorientationTemplate
#

$ Rscript Scripts/doLungSegmentation.R Data/Example/0005Proton.nii.gz 0005Output Data/Template/T_template0.nii.gz 

Reading reorientation template Data/Template/T_template0.nii.gz  (elapsed time: 0.02635503 seconds)
Loading weights file Data/Weights/lungSegmentationWeights.h52018-11-19  (elapsed time: 0.5598979 seconds)
Reading  Data/Example/0005Proton.nii.gz  (elapsed time: 0.04004812 seconds)
Normalizing to template  (elapsed time: 0.05562806 seconds)
Prediction and decoding (elapsed time: 59.35653 seconds)
Renormalize to native space and write image  (elapsed time: 0.190855 seconds)

Total elapsed time: 59.66657 seconds
```

## Sample results

![Brain extraction results](Documentation/Images/resultsLungSegmentation.png)
