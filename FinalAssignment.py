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

X_train = np.stack(train['image'], 0)
y_train = np.stack(train['sign'], 0)

X_test = np.stack(test['image'], 0)
y_test = np.stack(test['sign'], 0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


y_train_cats = pd.Categorical(y_train)
y_test_cats = pd.Categorical(y_test)
y_train = y_train_cats.codes
y_test = y_test_cats.codes


# Show image
print(y_train_cats.categories[y_train[5]])
plt.imshow(X_train[5], cmap="gray")
plt.show()


# Model
model = tf.keras.models.Sequential()
# Add the input layer
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
# Add one hidden layer
model.add(tf.keras.layers.Dense(300, activation='tanh'))
# Add the output layer
model.add(tf.keras.layers.Dense(16, activation='softmax'))
# Set parameters for our model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])
# Start training
history = model.fit(X_train, y_train, epochs=10,
                    batch_size=64, validation_data=(X_test, y_test))

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
plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'])
plt.ylabel('validation loss')
plt.xlabel('epochs')

plt.show()
