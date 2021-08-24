import os 
import numpy as np
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
import argparse
from glob import glob
import SimpleITK as sitk
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 



def show_segmented_image(orig_img, pred_img):
    '''
    Show the prediction over the original image
    INPUT:
        1)orig_img: the test image, which was used as input
        2)pred_img: the prediction output
    OUTPUT:
        segmented image rendering
    '''
    orig_img/=np.max(orig_img)
    #img_mask = np.pad(pred_img, (16,16), mode='edge')
    ones = np.argwhere(pred_img == 1)
    twos = np.argwhere(pred_img == 2)
    threes = np.argwhere(pred_img == 3)
    fours = np.argwhere(pred_img == 4)
    gray_img = img_as_float(orig_img)

    image = adjust_gamma(color.gray2rgb(gray_img), 1)
    sliced_image = image.copy()
    red_multiplier = [1, 0.2, 0.2]
    yellow_multiplier = [1,1,0.25]
    green_multiplier = [0.35,0.75,0.25]
    blue_multiplier = [0,0.25,0.9]

        # change colors of segmented classes
    for i in range(len(ones)):
        sliced_image[ones[i][0]][ones[i][1]] = red_multiplier
    for i in range(len(twos)):
        sliced_image[twos[i][0]][twos[i][1]] = green_multiplier
    for i in range(len(threes)):
        sliced_image[threes[i][0]][threes[i][1]] = blue_multiplier
    for i in range(len(fours)):
        sliced_image[fours[i][0]][fours[i][1]] = yellow_multiplier

    return sliced_image




def div_max_min (img):
    """
    save 2d image to disk in a png format
    """
    img=np.array(img).astype(np.float32)
    if np.max(img) != 0:
        img /= np.max(img)   # set values < 1                  
    if np.min(img) <= -1: # set values > -1
        img /= abs(np.min(img))

    return img


prediction_path=glob("./sample_data/predictions"+"/**")
save_path="./sample_data/screenshots"+"/"    
gt_path = path + '/'          
       


for i in range(len(prediction_path)):
    print("processing volume "+str(i))
    base=os.path.basename(prediction_path[i])[:-7]
    full_path=glob(gt_path+base)[0]
    print("full path ; " + full_path)

    t2 = glob( full_path + '/*_t2.nii.gz')
    t1 = glob( full_path + '/*_t1.nii.gz')
    t1c = glob( full_path + '/*_t1ce.nii.gz')
    flair = glob( full_path + '/*_flair.nii.gz')
    gt = glob( full_path + '/*_seg.nii.gz')


    t1s=[scan for scan in t1 if scan not in t1c]

    scans = [flair[0], t1s[0], t1c[0], t2[0], gt[0],prediction_path[i]]

    full_image = [sitk.GetArrayFromImage(sitk.ReadImage(elem)) for elem in scans]
    full_image=np.array(full_image).astype(np.float32)

    y0=29-13 #196
    x0=42
    y1=221+13  
    x1=194  #152
    full_image=full_image[:,:,y0:y1,x0:x1]
        

    slice_id=np.argmax(np.sum(full_image[-2]>0,axis=(1,2)))

    im_flair=full_image[0,slice_id]
    im_t1=full_image[1,slice_id]
    im_t1c=full_image[2,slice_id]
    im_t2=full_image[3,slice_id]
    gt = full_image[4,slice_id]
    predict=full_image[5,slice_id]


    im_pred=show_segmented_image(im_flair, predict)
    im_gt=show_segmented_image(im_flair, gt)


    im_flair=div_max_min(im_flair)
    im_flair = img_as_float(im_flair)
    im_flair = adjust_gamma(color.gray2rgb(im_flair), 1)

    im_t1=div_max_min(im_t1)
    im_t1 = img_as_float(im_t1)
    im_t1 = adjust_gamma(color.gray2rgb(im_t1), 1)

    im_t1c=div_max_min(im_t1c)
    im_t1c = img_as_float(im_t1c)
    im_t1c = adjust_gamma(color.gray2rgb(im_t1c), 1)

    im_t2=div_max_min(im_t2)
    im_t2 = img_as_float(im_t2)
    im_t2 = adjust_gamma(color.gray2rgb(im_t2), 1)

    tmp=np.array((im_t1,im_t1c,im_t2,im_flair,im_pred,im_gt))
    tmp=tmp.swapaxes(1,0)
    tmp=tmp.reshape(192+13*2,152*6,3)

    io.imsave(save_path+base+"-{}.png".format(slice_id),tmp)



in_path=glob("./sample_data/screenshots"+"/**")
out_path="./sample_data/screenshots_labeled"+"/"

font = ImageFont.truetype("./sample_data/arial.ttf", 12)
for i in range(len(in_path)):
	print('processing volume '+str(i))

	img = Image.open(in_path[i])
	draw = ImageDraw.Draw(img)

	draw.text((400, 205),os.path.basename(in_path[i])[:-4],fill=(255,255,255),font=font)
	draw.text((70+152*0, 2),"T1",fill=(255,255,255),font=font)
	draw.text((68+152*1, 2),"T1Gd",fill=(255,255,255),font=font)
	draw.text((70+152*2, 2),"T2",fill=(255,255,255),font=font)
	draw.text((62+152*3, 2),"T2Flair",fill=(255,255,255),font=font)
	draw.text((70+152*4, 2),"GT",fill=(255,255,255),font=font)
	draw.text((35+152*5, 2),"Our segmentation",fill=(255,255,255),font=font)
	img.save(out_path+os.path.basename(in_path[i]))