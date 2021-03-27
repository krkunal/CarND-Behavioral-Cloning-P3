# Import statements
import tensorflow as tf
from tensorflow.keras import __version__ as keras_version
tf.random.set_seed(21)
print("TF version: {}\nKeras version: {}".format(tf.__version__, keras_version))
import csv
# import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Conv2D, Dropout, MaxPool2D, Lambda, Cropping2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.metrics import mape, mae
from sklearn.model_selection import train_test_split


# Parser for CSV file
def parse_data_log(log_path, has_header):
    lines = []
    with open(log_path) as logfile:
        reader = csv.reader(logfile)
        if has_header:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

# Loading data from two sources - 
# Directory 'data/driving_log.csv' contains the data collected manually for 2 laps of the track - 1 CW, 1 CCW
# Directory 'data_old/data/driving_log.csv' contains the data provided as part of the Project repo
def load_data():
	log_new_df = pd.DataFrame(data=lines, columns=['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed'])
	lines = parse_data_log('data/driving_log.csv', False)
	log_new_df['center'] = log_new_df['center'].apply(lambda x : '/'.join(x.split('/')[-3:]))
	log_new_df['left'] = log_new_df['left'].apply(lambda x : '/'.join(x.split('/')[-3:]))
	log_new_df['right'] = log_new_df['right'].apply(lambda x : '/'.join(x.split('/')[-3:]))
	log_new_df['angle'] = log_new_df['angle'].apply(lambda x : round(float(x), 3))
	lines_old = parse_data_log('data_old/data/driving_log.csv', True)
	log_old_df = pd.DataFrame(data=lines_old, columns=['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed'])
	log_old_df['center'] = log_old_df['center'].apply(lambda x : '/'.join(['data_old/data', x.strip()]))
	log_old_df['left'] = log_old_df['left'].apply(lambda x : '/'.join(['data_old/data', x.strip()]))
	log_old_df['right'] = log_old_df['right'].apply(lambda x : '/'.join(['data_old/data', x.strip()]))
	log_old_df['angle'] = log_old_df['angle'].apply(lambda x : round(float(x), 3))
	log_df = log_new_df.append(log_old_df, ignore_index=True)
	print(log_new_df.shape, log_old_df.shape, log_df.shape)
	retun log_df

# Process the data - Train/Val split + using left & right camera images with corrected steering values
def process_data(log_df):
	# Splitting into train & validation set - 80-20 split
	X_train, X_val, y_train, y_val = train_test_split(log_df['center'], log_df['angle'], test_size=0.2, random_state = 21) 
	train_df = pd.DataFrame({'image': X_train.values, 
		                     'label': y_train.values})
	val_df = pd.DataFrame({'image': X_val.values, 
		                   'label': y_val.values})

	# Correction parameter for left & right cameras
	correction = 0.2
	train_left_images = log_df.iloc[X_train.index.values, 1]
	# Add the correction parameter for left camera images
	train_left_labels = [train_label + correction for train_label in y_train.values]
	train_right_images = log_df.iloc[X_train.index.values, 2]
	# Subtract the correction parameter for right camera images
	train_right_labels = [train_label - correction for train_label in y_train.values]
	train_df = train_df.append(pd.DataFrame({'image': train_left_images.values, 
		                                    'label': np.array(train_left_labels)}), 
		                                    ignore_index=True)
	train_df = train_df.append(pd.DataFrame({'image': train_right_images.values, 
		                        'label': np.array(train_right_labels)}), 
		                        ignore_index=True)
	val_left_images = log_df.iloc[X_val.index.values, 1]
	# Add the correction parameter for left camera images
	val_left_labels = [val_label + correction for val_label in y_val.values]
	val_right_images = log_df.iloc[X_val.index.values, 2]
	# Subtract the correction parameter for right camera images
	val_right_labels = [val_label - correction for val_label in y_val.values]
	val_df = val_df.append(pd.DataFrame({'image': val_left_images.values, 
		                    'label': np.array(val_left_labels)}), 
		                    ignore_index=True)
	val_df = val_df.append(pd.DataFrame({'image': val_right_images.values, 
		                    'label': np.array(val_right_labels)}), 
		                    ignore_index=True)

	return train_df, val_df



# Build the model
def build_model():
	model = Sequential()
	# model.add(InputLayer(input_shape=(160, 320, 3)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
	model.add(Conv2D(kernel_size=7, filters=16, activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(kernel_size=7, filters=32, activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(kernel_size=5, filters=64, activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(kernel_size=3, filters=128, activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(kernel_size=1, filters=256, activation='relu')) # Equivalent to an FC layer.
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(256, activation='relu'))
	# model.add(Dropout(0.1))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(1))

	return model


# Run model
def run_model(model, train_df, val_df):
	# Setup Hyperparams
	EPOCHS = 13
	BATCH_SIZE = 32

	# Compile model
	model.compile(optimizer=Adam(lr=0.001), 
		          loss = mean_squared_error, 
		          metrics = [mae])

	# Prep train & val Generators with batch size = BAATCH_SIZE
	img_datagen = ImageDataGenerator() # rescale=1./255
	train_generator = img_datagen.flow_from_dataframe(train_df, x_col="image", y_col="label", 
		                                              target_size=(160, 320), 
		                                              class_mode="raw", 
		                                              batch_size=BATCH_SIZE, 
		                                              seed=21)
	val_generator = img_datagen.flow_from_dataframe(val_df, x_col="image", y_col="label", 
		                                              target_size=(160, 320), 
		                                              class_mode="raw", 
		                                              batch_size=BATCH_SIZE, 
		                                              seed=21)

	STEPS_TRAIN=train_generator.n//train_generator.batch_size
	STEPS_VAL=val_generator.n//val_generator.batch_size

	# Train the model
	history = model.fit(train_generator, 
		                steps_per_epoch = STEPS_TRAIN, 
		                epochs = EPOCHS,
		                validation_data = val_generator, 
		                validation_steps = STEPS_VAL, 
		                workers=3, use_multiprocessing=True)

	

# save model
def save_model(model):
	model.save('model3.h5')

if __name__ == '__main__':
	log_df = load_data()
	train_df, val_df = process_data(log_df)
	model = build_model()
	run_model(model, train_df, val_df)
	save_model(model)
