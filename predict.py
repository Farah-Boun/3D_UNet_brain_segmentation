from metrics import *
from model import *
from extract_patches import *
import xlsxwriter

def save_results_in_excel_file(file_path,test_list,sensitivity_list,specificity_list,dsc_list,hausdorff_list ) :
  outWorkbook = xlsxwriter.Workbook(file_path+".xlsx")
  outSheet = outWorkbook.add_worksheet()

  outSheet.write("B1","DSC")
  outSheet.write("E1", "Sensitivity")
  outSheet.write("H1", "Specificity")
  outSheet.write("K1", "Hausdorff95")

  outSheet.write("A2","Label")
  outSheet.write("B2","ET")
  outSheet.write("C2","WT")
  outSheet.write("D2","TC")
  outSheet.write("E2","ET")
  outSheet.write("F2","WT")
  outSheet.write("G2","TC")
  outSheet.write("H2","ET")
  outSheet.write("I2","WT")
  outSheet.write("J2","TC")
  outSheet.write("K2","ET")
  outSheet.write("L2","WT")
  outSheet.write("M2","TC")

  for item in range(len(test_list)):
    outSheet.write(item+2,0,test_list[item])
    outSheet.write(item+2,1,dsc_list[item][0])
    outSheet.write(item+2,2,dsc_list[item][1])
    outSheet.write(item+2,3,dsc_list[item][2])
    outSheet.write(item+2,4,sensitivity_list[item][0])
    outSheet.write(item+2,5,sensitivity_list[item][1])
    outSheet.write(item+2,6,sensitivity_list[item][2])
    outSheet.write(item+2,7,specificity_list[item][0])
    outSheet.write(item+2,8,specificity_list[item][1])
    outSheet.write(item+2,9,specificity_list[item][2])
    outSheet.write(item+2,10,hausdorff_list[item][0])
    outSheet.write(item+2,11,hausdorff_list[item][1])
    outSheet.write(item+2,12,hausdorff_list[item][2])
  
  outWorkbook.close()

def get_patient_image(path,patient_list,i) :
  x = patient_list[i]
  print("Results on image :"+ x)
  folder_path = path + '/' + x;
  modalities = os.listdir(folder_path)
  modalities.sort()
  w = 0
  for j in range(len(modalities)):
    image_path = folder_path + '/' + modalities[j]
    if (image_path[-10:] == 'seg.nii.gz'):
      img = nib.load(image_path);
      image_label = img.get_fdata()
      image_label = np.asarray(image_label)
    else:
      img = nib.load(image_path);
      image_data = img.get_fdata()
      image_data = np.asarray(image_data)
      image_data = standardize(image_data)
      data[:,:,:,w] = image_data
      w = w+1
  image_label[image_label==4] = 3
  return data, image_label

def get_patient_image_validation(path,patient_list,i) :
  x = patient_list[i]
  print("Results on image :"+ x)
  folder_path = path + '/' + x;
  modalities = os.listdir(folder_path)
  modalities.sort()
  w = 0
  for j in range(len(modalities)):
    image_path = folder_path + '/' + modalities[j]
    img = nib.load(image_path);
    image_data = img.get_fdata()
    image_data = np.asarray(image_data)
    image_data = standardize(image_data)
    data[:,:,:,w] = image_data
    w = w+1
  return data

def get_patient_modalities_from_path(path) :
  data = np.zeros((240,240,155,4))
  print("Loading image "+ os.path.basename(path))
  folder_path = path
  modalities = os.listdir(folder_path)
  modalities.sort()
  w = 0
  for j in range(len(modalities)):
    image_path = folder_path + '/' + modalities[j]
    if (image_path[-12:] == 'flair.nii.gz' or image_path[-9:] == 't1.nii.gz' or image_path[-9:] == 't2.nii.gz' or image_path[-11:] == 't1ce.nii.gz'):
      img = nib.load(image_path)
      image_data = img.get_fdata()
      image_data = np.asarray(image_data)
      image_data = standardize(image_data)
      data[:,:,:,w] = image_data
      w = w+1
  print("Image Loaded")
  return data

def prediction_function (image, label, model):
    image_labeled = to_categorical(label, num_classes=4).astype(np.uint8)
    all_pred = np.zeros([ 240, 240, 155, 4])

    left_up = image[42:170,29:157,13:141,:]
    left_up = left_up.reshape(1,128,128,128,4)
    start = 0
    end = start + 16
    image_list=[]
    for i in range (0,8):
      image_list.append(left_up[:,:,:,start:end,:])
      start = start + 16
      end = start + 16
    full_prediction_left_up = model.predict(x=image_list[0])
    for i in range(1,8):
      Y_hat = model.predict(x=image_list[i])
      full_prediction_left_up = np.concatenate((full_prediction_left_up,Y_hat),axis=3)

    all_pred[42:170,29:157,13:141,:] = full_prediction_left_up[0,:,:,:,:]


    left_down = image[42:170,93:221,13:141,:]
    left_down = left_down.reshape(1,128,128,128,4)
    start = 0
    end = start + 16
    image_list=[]
    for i in range (0,8):
      image_list.append(left_down[:,:,:,start:end,:])
      start = start + 16
      end = start + 16
    full_prediction_left_down = model.predict(x=image_list[0])
    for i in range(1,8):
      Y_hat = model.predict(x=image_list[i])
      full_prediction_left_down = np.concatenate((full_prediction_left_down,Y_hat),axis=3)

    all_pred[42:170,157:221,13:141,:] = full_prediction_left_down[0,:,64:128,:,:]

    right_up = image[66:194,29:157,13:141,:]
    right_up = right_up.reshape(1,128,128,128,4)
    start = 0
    end = start + 16
    image_list=[]
    for i in range (0,8):
      image_list.append(right_up[:,:,:,start:end,:])
      start = start + 16
      end = start + 16
    full_prediction_right_up = model.predict(x=image_list[0])
    for i in range(1,8):
      Y_hat = model.predict(x=image_list[i])
      full_prediction_right_up = np.concatenate((full_prediction_right_up,Y_hat),axis=3)

    all_pred[170:194,29:157,13:141,:] = full_prediction_right_up[0,104:128,:,:,:]

    right_down = image[66:194,93:221,13:141,:]
    right_down = right_down.reshape(1,128,128,128,4)
    start = 0
    end = start + 16
    image_list=[]
    for i in range (0,8):
      image_list.append(right_down[:,:,:,start:end,:])
      start = start + 16
      end = start + 16
    full_prediction_right_down = model.predict(x=image_list[0])
    for i in range(1,8):
      Y_hat = model.predict(x=image_list[i])
      full_prediction_right_down = np.concatenate((full_prediction_right_down,Y_hat),axis=3)

    all_pred[170:194,157:221,13:141,:] = full_prediction_right_down[0,104:128,64:128,:,:]

    return all_pred,image_labeled


def prediction_function_validation (image, model):
    print("Segmenting image...")
    all_pred = np.zeros([ 240, 240, 155, 4])

    left_up = image[42:170,29:157,13:141,:]
    left_up = left_up.reshape(1,128,128,128,4)
    start = 0
    end = start + 16
    image_list=[]
    for i in range (0,8):
      image_list.append(left_up[:,:,:,start:end,:])
      start = start + 16
      end = start + 16
    full_prediction_left_up = model.predict(x=image_list[0])
    for i in range(1,8):
      Y_hat = model.predict(x=image_list[i])
      full_prediction_left_up = np.concatenate((full_prediction_left_up,Y_hat),axis=3)

    all_pred[42:170,29:157,13:141,:] = full_prediction_left_up[0,:,:,:,:]


    left_down = image[42:170,93:221,13:141,:]
    left_down = left_down.reshape(1,128,128,128,4)
    start = 0
    end = start + 16
    image_list=[]
    for i in range (0,8):
      image_list.append(left_down[:,:,:,start:end,:])
      start = start + 16
      end = start + 16
    full_prediction_left_down = model.predict(x=image_list[0])
    for i in range(1,8):
      Y_hat = model.predict(x=image_list[i])
      full_prediction_left_down = np.concatenate((full_prediction_left_down,Y_hat),axis=3)

    all_pred[42:170,157:221,13:141,:] = full_prediction_left_down[0,:,64:128,:,:]

    right_up = image[66:194,29:157,13:141,:]
    right_up = right_up.reshape(1,128,128,128,4)
    start = 0
    end = start + 16
    image_list=[]
    for i in range (0,8):
      image_list.append(right_up[:,:,:,start:end,:])
      start = start + 16
      end = start + 16
    full_prediction_right_up = model.predict(x=image_list[0])
    for i in range(1,8):
      Y_hat = model.predict(x=image_list[i])
      full_prediction_right_up = np.concatenate((full_prediction_right_up,Y_hat),axis=3)

    all_pred[170:194,29:157,13:141,:] = full_prediction_right_up[0,104:128,:,:,:]

    right_down = image[66:194,93:221,13:141,:]
    right_down = right_down.reshape(1,128,128,128,4)
    start = 0
    end = start + 16
    image_list=[]
    for i in range (0,8):
      image_list.append(right_down[:,:,:,start:end,:])
      start = start + 16
      end = start + 16
    full_prediction_right_down = model.predict(x=image_list[0])
    for i in range(1,8):
      Y_hat = model.predict(x=image_list[i])
      full_prediction_right_down = np.concatenate((full_prediction_right_down,Y_hat),axis=3)

    all_pred[170:194,157:221,13:141,:] = full_prediction_right_down[0,104:128,64:128,:,:]
    print("Segmentation Done")
    return all_pred

def load_latest_model():
  input_img = Input((128,128,16,4))
  model = Unet_3d(input_img, n_filters = 8, dropout = 0.1, batch_norm = True)
  model = load_model(".\model_save.hdf5",custom_objects = {'dice_coef_loss' : dice_coef_loss , 'dice_coef' : dice_coef, 'whole' : whole, 'core' : core, 'enhanc' : enhanc})
  lr = 0.00001
  decay_rate = 0.001
  model.compile(optimizer=Adam(learning_rate=lr, decay = decay_rate), loss=dice_coef_loss, metrics=[dice_coef,whole,core,enhanc], sample_weight_mode="temporal")

  return model

def predict_from_path (path,model):
  image = get_patient_modalities_from_path(path)
  all_pred = prediction_function_validation(image,model)
  return all_pred

'''
HOME_DIR = "drive/MyDrive/BraTS_dataset/BraTS2021/128_16_patches"
path = 'drive/MyDrive/BraTS_dataset/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData'
data = np.zeros((240,240,155,4))
image_label=np.zeros((240,240,155))

path_to_config_file = "drive/MyDrive/BraTS_dataset/BraTS2021/128_16_patches/config_file.json"

with open(path_to_config_file) as json_file:
        patient_list = json.load(json_file)

train_patient = patient_list['train_list']
valid_patient = patient_list['valid_list']

all_images_valid = valid_patient


input_img = Input((128,128,16,4))
model = Unet_3d(input_img, n_filters = 8, dropout = 0.1, batch_norm = True)
model = load_model(HOME_DIR+ "/models/saved_model_brats2021_128_16_48.hdf5",custom_objects = {'dice_coef_loss' : dice_coef_loss , 'dice_coef' : dice_coef, 'whole' : whole, 'core' : core, 'enhanc' : enhanc})
lr = 0.00001
decay_rate = 0.001
model.compile(optimizer=Adam(learning_rate=lr, decay = decay_rate), loss=dice_coef_loss, metrics=[dice_coef,whole,core,enhanc], sample_weight_mode="temporal")
#model.summary()

sensitivity_list = []
specificity_list = []
dsc_list = []
hausdorff_list = []

import nibabel as nib

for image_num in range(0,250):
    #print(epochs)
    print("Entering Image" , image_num)
    image_data, image_label_all = get_patient_image(path,all_images_valid,image_num)
    pred_label, image_label = prediction_function(image_data, image_label_all, model)
    Y_hat = np.argmax(pred_label,axis=-1)

  
    #Evaluation metrics
    sensitivity_list.append([ sensitivity_en(Y_hat[:,:,:],image_label_all[:,:,:]),
                              sensitivity_whole(Y_hat[:,:,:],image_label_all[:,:,:]),
                              sensitivity_core(Y_hat[:,:,:],image_label_all[:,:,:])   ])
    
    print("Sensitivity : whole=" ,sensitivity_list[image_num][1],
                      "   core=" , sensitivity_list[image_num][2],
                    "   enhanc=" , sensitivity_list[image_num][0])
    

    specificity_list.append([   specificity_en(Y_hat[:,:,:],image_label_all[:,:,:]),
                                specificity_whole(Y_hat[:,:,:],image_label_all[:,:,:]),
                                specificity_core(Y_hat[:,:,:],image_label_all[:,:,:])   ])


    print("Specificity : whole=",specificity_list[image_num][1],
          "   core=", specificity_list[image_num][2],
          "   enhanc=", specificity_list[image_num][0])
    
    dsc_list.append([ DSC_en(Y_hat[:,:,:],image_label_all[:,:,:]),
                      DSC_whole(Y_hat[:,:,:],image_label_all[:,:,:]),
                      DSC_core(Y_hat[:,:,:],image_label_all[:,:,:])   ])
    
    print("DSC : whole=", dsc_list[image_num][1],
          "   core=", dsc_list[image_num][2],
          "   enhanc=", dsc_list[image_num][0])
    
    hausdorff_list.append([ hausdorff_en(Y_hat[:,:,:],image_label_all[:,:,:]),
                            hausdorff_whole(Y_hat[:,:,:],image_label_all[:,:,:]),
                            hausdorff_core(Y_hat[:,:,:],image_label_all[:,:,:])   ])
    
    print("Hausdorff : whole=", hausdorff_list[image_num][1],
          "   core=", hausdorff_list[image_num][2],
          "   enhanc=", hausdorff_list[image_num][0])
    
    Y_hat = keras.utils.to_categorical(Y_hat , num_classes = 4)
 
path_to_excel_file = HOME_DIR + "/models/evaluation_predfun_model_128_128_16_41_results"
save_results_in_excel_file(path_to_excel_file,all_images_valid,sensitivity_list,specificity_list,dsc_list,hausdorff_list)
'''