import numpy as np
from scipy import ndimage



def dice_coef(y_true, y_pred, epsilon=0.00001):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    
    """
    axis = (0,1,2,3)
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true*y_true, axis=axis) + K.sum(y_pred*y_pred, axis=axis) + epsilon
    return K.mean((dice_numerator)/(dice_denominator))
 
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice(y_true, y_pred):
    #computes the dice score on two tensors
 
    sum_p=K.sum(y_pred,axis=0)
    sum_r=K.sum(y_true,axis=0)
    sum_pr=K.sum(y_true * y_pred,axis=0)
    dice_numerator =2*sum_pr
    dice_denominator =sum_r+sum_p
    dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
    return dice_score
 
 
def whole(y_true, y_pred):
    #computes the dice for the whole tumor
 
    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    y_whole=K.sum(y_true_f[:,1:],axis=1)
    p_whole=K.sum(y_pred_f[:,1:],axis=1)
    dice_whole=dice(y_whole,p_whole)
    return dice_whole
 
def enhanc(y_true, y_pred):
    #computes the dice for the enhancing region
 
    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    y_enh=y_true_f[:,-1]
    p_enh=y_pred_f[:,-1]
    dice_en=dice(y_enh,p_enh)
    return dice_en
 
def core(y_true, y_pred):
    ##computes the dice for the core region
 
    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    #workaround for tf
    y_core=K.sum(tf.gather(y_true_f, [1,3],axis =1),axis=1)
    p_core=K.sum(tf.gather(y_pred_f, [1,3],axis =1),axis=1)
    
    dice_core=dice(y_core,p_core)
    return dice_core


# Evaluation metrics

def binary_dice3d(s,g):
    #dice score of two 3D volumes
    num=np.sum(np.multiply(s, g))
    denom=s.sum() + g.sum() 
    if denom==0:
        return 1
    else:
        return  2.0*num/denom


def sensitivity (seg,ground): 
    #computs false negative rate
    num=np.sum(np.multiply(ground, seg ))
    denom=np.sum(ground)
    if denom==0:
        return 1
    else:
        return  num/denom

def specificity (seg,ground): 
    #computes false positive rate
    num=np.sum(np.multiply(ground==0, seg ==0))
    denom=np.sum(ground==0)
    if denom==0:
        return 1
    else:
        return  num/denom



def border_map(binary_img,neigh):
    """
    Creates the border for a 3D image
    """
    binary_map = np.asarray(binary_img, dtype=np.uint8)
    neigh = neigh
    west = ndimage.shift(binary_map, [-1, 0,0], order=0)
    east = ndimage.shift(binary_map, [1, 0,0], order=0)
    north = ndimage.shift(binary_map, [0, 1,0], order=0)
    south = ndimage.shift(binary_map, [0, -1,0], order=0)
    top = ndimage.shift(binary_map, [0, 0, 1], order=0)
    bottom = ndimage.shift(binary_map, [0, 0, -1], order=0)
    cumulative = west + east + north + south + top + bottom
    border = ((cumulative < 6) * binary_map) == 1
    return border


def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    neigh=8
    border_ref = border_map(ref,neigh)
    border_seg = border_map(seg,neigh)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg#, border_ref, border_seg

def Hausdorff_distance(ref,seg):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a segmentation and a reference image
    :return: hausdorff distance and average symmetric distance
    """
    ref_border_dist, seg_border_dist = border_distance(ref,seg)
    hausdorff_distance = np.max(
        [np.max(ref_border_dist), np.max(seg_border_dist)])
    return hausdorff_distance



def DSC_whole(pred, orig_label):
    #computes dice for the whole tumor
    return binary_dice3d(pred>0,orig_label>0)


def DSC_en(pred, orig_label):
    #computes dice for enhancing region
    return binary_dice3d(pred==3,orig_label==3)


def DSC_core(pred, orig_label):
    #computes dice for core region
    seg_=np.copy(pred)
    ground_=np.copy(orig_label)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return binary_dice3d(seg_>0,ground_>0)



def sensitivity_whole (seg,ground):
    return sensitivity(seg>0,ground>0)

def sensitivity_en (seg,ground):
    return sensitivity(seg==3,ground==3)

def sensitivity_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return sensitivity(seg_>0,ground_>0)



def specificity_whole (seg,ground):
    return specificity(seg>0,ground>0)

def specificity_en (seg,ground):
    return specificity(seg==3,ground==3)

def specificity_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return specificity(seg_>0,ground_>0)
    

def hausdorff_whole (seg,ground):
    return Hausdorff_distance(seg==0,ground==0)

def hausdorff_en (seg,ground):
    return Hausdorff_distance(seg!=3,ground!=3)

def hausdorff_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return Hausdorff_distance(seg_==0,ground_==0)


