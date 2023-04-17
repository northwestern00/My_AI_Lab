#Data Processing Libraries
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#CNN Libraries with Keras and Tensorflow
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import load_model
from keras.optimizers import Adam

#Define the metadata to our algorythm
skin_dataframe = pd.read_csv("HAM10000_metadata.csv")

#Control Variables in metadata
skin_dataframe.head()
skin_dataframe.info()
sns.countplot(x = "dx", data = skin_dataframe)

#There is no errors to our metadata, we can get preprocess metadata for use in DeepLearning
#Define a variable name for our image folder which we use
data_folder_name = "HAM10000_images_part_1/"
ext = ".jpg"
skin_dataframe["path"] = [data_folder_name + i + ext for i in skin_dataframe["image_id"]]
skin_dataframe["image"] = skin_dataframe["path"].map( lambda x: np.asarray(Image.open(x).resize((100,75))))
skin_dataframe["dx_idx"] = pd.Categorical(skin_dataframe["dx"]).codes

#Let's standardize to our data frame with OneHotEncoding
x_train = np.asarray(skin_dataframe["image"].tolist())
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = (x_train - x_train_mean)/x_train_std

y_train = to_categorical(skin_dataframe["dx_idx"], num_classes = 7)

#Convulutional Neural Network

input_shape = (75,100,3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", padding = "Same", input_shape = input_shape))
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = "softmax"))

optimizer = Adam(lr = 0.0001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

epochs = 5
batch_size = 25

history = model.fit(x = x_train, y = y_train, batch_size = batch_size, epochs = epochs, verbose = 1, shuffle = True)

model.save("deep_model2.h5")
