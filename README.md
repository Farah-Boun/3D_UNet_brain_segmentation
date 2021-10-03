# 3D_UNet_brain_segmentation
9 layer 3D-UNET model capable of segmenting brain gliomas on multimodal 3D MRI images.
This model was a submission in the 2021 edition of the Brain Tumor Segmentation Challenge (BraTS'2021)

#How to segment an MRI image 
This implementation works on the BraTS2021 datasets, so your data needs to have the same naming convention as said dataset. Case folder needs to contain the four modality
images in nii/gz format and named as such: *flair.nii.gz, *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz.

You can start a segmentation by running this command 
```
python segment.py --input <path to folder containing the 4 modality images> --output <path to output folder>
```
