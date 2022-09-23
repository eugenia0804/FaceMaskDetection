from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
import plot_model


path, batch_size = '/Users/caoyujia/Desktop/face-mask-dataset', 16

train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,
                                  shear_range=0.2)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(path+'/Train', target_size=(128, 128),
                                               batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(path+'/Test', target_size=(128, 128),
                                             batch_size=batch_size, class_mode='categorical')

path = '/Users/caoyujia/Desktop/face-mask-dataset'
fig, axes = plt.subplots(1, 2, figsize=(15, 9))

model_histories = []
models = [InceptionV3(include_top=False, input_shape=(128, 128, 3)),
          MobileNet(include_top=False, input_shape=(128, 128, 3)),
          DenseNet201(include_top=False, input_shape=(128, 128, 3)),
          VGG19(include_top=False, input_shape=(128, 128, 3))]
names = ['ConvNet', 'InceptionV3', 'MobileNet', 'DenseNet', 'VGG19']

for layer in [Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3))]:
    model = Sequential()
    model.add(layer)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
    model_histories.append(model.fit_generator(generator=train_generator,
                                               validation_data=test_generator,
                                               steps_per_epoch=len(train_generator) // 3,
                                               validation_steps=len(test_generator) // 3,
                                               epochs=10))

for functional in models:

    for layer in functional.layers:
        layer.trainable = False

    model = Sequential()
    model.add(functional)
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
    model_histories.append(model.fit_generator(generator=train_generator,
                                               validation_data=test_generator,
                                               steps_per_epoch=len(train_generator) // 3,
                                               validation_steps=len(test_generator) // 3, epochs=10))
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.subplots_adjust(hspace=0.3)
for metric in model_histories[0].history:
    index = list(model_histories[0].history).index(metric)
    ax = axes.flatten()[index]
    name_index = 0
    for history in model_histories:
        ax.plot(history.history[metric], label=names[name_index])
        name_index += 1
    ax.set_title(metric+' over epochs', size=15)
    ax.set_xlabel('epochs')
    ax.set_ylabel(metric)
    ax.legend()
plt.show()
