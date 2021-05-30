# To deliver
#
#      A document describing the process followed to build your application
#         Describe the architecture of your neural network
#         Describe the results obtained in the training process of the neural network
#         Include a sample run of your code

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Import data- path to .npy file
train = np.load(
    'C:/Users/lindb/PycharmProjects/archive/CompleteDataSet_training_tuples.npy', allow_pickle=True)
test = np.load(
    'C:/Users/lindb/PycharmProjects/archive/CompleteDataSet_testing_tuples.npy', allow_pickle=True)

train = pd.DataFrame(train, columns=['image', 'sign'])
test = pd.DataFrame(test, columns=['image', 'sign'])

X_train = train['image']
y_train = train['sign']

X_test = test['image']
y_test = test['sign']

class_names =["-", "%", "[", "]","+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


# Determine the architecture of my NN

model = tf.keras.models.Sequential()
#Add the input layer
model.add(tf.keras.layers.Flatten(input_shape=[28,28])) #get shape by x_Train.shape
#Add one hidden layer
model.add(tf.keras.layers.Dense(300, activation="relu")) # 300 -> guess
#Add the output layer
model.add (tf.keras.layers.Dense(15, activation="softmax")) #softmax = probability to belonging to each of the category

#Set the parameters for our model

model.compile(
    loss ="sparse_categorical_crossentropy",  #mse/cost function
    optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.005), #update the weight on the thresholdds
    metrics = ["accuracy"] # results on every iteration
    )

#start training the model
history = model.fit(X_train, y_train, epochs= 10, validation_data = (X_test, y_test))

#5. Display the results
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.subplot(2,2,2)
plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epochs')

plt.subplot(2,2,3)
plt.plot(range(len(history.history['val_accuracy'])), history.history['val_accuracy'])
plt.ylabel('validation accuracy')
plt.xlabel('epochs')
plt.subplot(2,2,4)
plt.plot(range(len(history.history['val_accuracy'])), history.history['val_accuracy'])
plt.ylabel('validation loss')
plt.xlabel('epochs')

plt.show()
