from model import *
from metrics import *


# set home directory and data directory
HOME_DIR = "drive/MyDrive/BraTS_dataset/BraTS2021/128_16_patches"

PATCH_DIR = "drive/MyDrive/BraTS_dataset/BraTS2021/128_16_patches"
TRAIN_PATCH_DIR = PATCH_DIR + "/train/"
VALID_PATCH_DIR = PATCH_DIR + "/valid/"

path_to_config_file = "drive/MyDrive/BraTS_dataset/BraTS2021/128_16_patches/config_file.json"


with open(path_to_config_file) as json_file:
        patient_list = json.load(json_file)

train_patient = patient_list['train_list']
valid_patient = patient_list['valid_list']

input_img = Input((128,128,16,4))
model = Unet_3d(input_img, n_filters = 8, dropout = 0.1, batch_norm = True)
model = load_model(HOME_DIR+ "/models/saved_model_brats2021_128_16_48.hdf5",custom_objects = {'dice_coef_loss' : dice_coef_loss , 'dice_coef' : dice_coef, 'whole' : whole, 'core' : core, 'enhanc' : enhanc})
lr = 0.00001
decay_rate = 0.001
model.compile(optimizer=Adam(learning_rate=lr, decay = decay_rate), loss=dice_coef_loss, metrics=[dice_coef,whole,core,enhanc], sample_weight_mode="temporal")
#model.summary()


callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

filepath=HOME_DIR + "/models/saved_model_brats2021_128_16_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
 
csvlog=CSVLogger(HOME_DIR + '/model_quick_brats2021_128_16.csv', separator=',', append=True)

log_dir = "drive/MyDrive/BraTS_dataset/BraTS2021/tensorboad_brats2021_128_16"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
 
callbacks_list = [checkpoint,csvlog,tensorboard_callback,callback_scheduler]

batch_size = 6


train_generator = VolumeDataGenerator(train_patient, TRAIN_PATCH_DIR, batch_size= batch_size, dim=(128, 128, 16), verbose=0)
valid_generator = VolumeDataGenerator(valid_patient, VALID_PATCH_DIR, batch_size= batch_size, dim=(128, 128, 16), verbose=0)
train_steps=len(train_patient)// batch_size
valid_steps=len(valid_patient) // batch_size

##training
nb_epoch = 50
init_epoch = 48
model.fit(train_generator,
          steps_per_epoch=train_steps,
          workers=8,
          initial_epoch = init_epoch,
          epochs=nb_epoch,
          validation_data=valid_generator,
          validation_steps=valid_steps,
          verbose=1, 
          use_multiprocessing = True,
          callbacks = callbacks_list)