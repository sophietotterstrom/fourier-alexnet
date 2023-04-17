"""
Program for implementing slightly modified AlexNet for image classification
of 32x32 RGB and FFT magnitude and phase spectrum images from Cifar-10/Cifar-100 datasets.

Edit boolean swiches in main depending on what functionality is desired. 
Program can either build and train a model, and display information about the training and performance.
Otherwise program can load pretrained weights and run perform classification and report performance.

After editing main function to have proper switches, run with following command:
python fourier_alexnet.py

@author Sophie Tötterström
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, fftshift
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.datasets import cifar10, cifar100


def get_filename(cifar100_swi, fft_swi, new=False):
    """ Function for filename automatization.

    Args:
        cifar100_swi (bool): swich for what data is being used. CIFAR-100 (True) or CIFAR-10 (False)
        fft_swi (bool): swich for image type. Fourier magnitude and phase spectra (True) or RGB images (False)
        new (bool, optional): If user is training new models from scratch, add this information to filename. 
                                Defaults to False.

    Returns:
        str: filename
    """

    dataset = 'cifar10'
    im_type = 'rgb'
    if cifar100_swi:
        dataset = 'cifar100'
    if fft_swi:
        im_type = 'fft'
    
    if new:
        return f"models/new_{dataset}_{im_type}.h5"
    return f"models/{dataset}_{im_type}.h5"

def get_fft(ims):
    """ Function for fetching the Fourier magnitude and phase spectra of all input images.

    Args:
        ims (NumPy Array): all RGB images to process

    Returns:
        NumPy Array: fft magnitude and phase spectra of the input images in 6 channels
    """

    fft_ims = []
    for im in ims:
        # initialize fft magnitude-phase spectrum variable to have 6 channels
        fft_im = np.dstack((im,im))

        # calculate fft of original image
        fft_r = fftshift(fft2(im[:,:,0]))
        fft_g = fftshift(fft2(im[:,:,1]))
        fft_b = fftshift(fft2(im[:,:,2]))

        # place fft magnitude and phase results
        fft_im[:, :, 0] = np.log(abs(fft_r), where=fft_r!=0)
        fft_im[:, :, 1] = np.log(abs(fft_g), where=fft_g!=0)
        fft_im[:, :, 2] = np.log(abs(fft_b), where=fft_b!=0)

        fft_im[:, :, 3] = np.angle(fft_r)
        fft_im[:, :, 4] = np.angle(fft_g)
        fft_im[:, :, 5] = np.angle(fft_b)

        fft_ims.append(fft_im)
    return fft_ims

def preprocess_img(img, label):
    """ Function for standardizing and resizing an input image

    Args:
        img (NumPy Array): image matrix being 
        label (string): class label

    Returns:
        image, label: Return the standardized and resized image and the class label it belongs in.
    """

    # standardize each image to have mean of 0 and variance of 1
    img = tf.image.per_image_standardization(img)

    # resize images from 32x32 to 227x227 due to baseline model structure
    img = tf.image.resize(img, (227,227))

    return img, label

def load_data(cifar100_swich, fft):
    """ Function loads the training, testing and validation data specified with the flags

    Args:
        cifar100_swich (bool): swich for what data is being used. CIFAR-100 (True) or CIFAR-10 (False)
        fft (bool): swich for image type. Fourier magnitude and phase spectra (True) or RGB images (False)

    Returns:
        TensorFlow Datasets: training, validation, and test data in processed and split datasets
    """

    # load selected data (either CIFAR-10/CIFAR-100)
    if cifar100_swich: (tr_imgs, tr_labels),(te_imgs, te_labels) = cifar100.load_data()
    else: (tr_imgs, tr_labels),(te_imgs, te_labels) = cifar10.load_data()

    # split dataset to have training and validation data
    val_imgs, val_labels = tr_imgs[:5000], tr_labels[:5000]
    tr_imgs, tr_labels = tr_imgs[5000:], tr_labels[5000:]

    if fft: # get fft versions instead
        tr_imgs = get_fft(tr_imgs)
        te_imgs = get_fft(te_imgs)
        val_imgs = get_fft(val_imgs)

    ## start data pipeline ##
    # create tensorflow dataset from filenames and labels
    tr_ds = tf.data.Dataset.from_tensor_slices((tr_imgs, tr_labels))
    te_ds = tf.data.Dataset.from_tensor_slices((te_imgs, te_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))
    
    # shuffle images
    # standardize and resize images
    # and batch them for easier processing
    tr_ds = tr_ds.shuffle(buffer_size=len(tr_ds)).map(preprocess_img).batch(batch_size=32, drop_remainder=False)
    te_ds = te_ds.shuffle(buffer_size=len(te_ds)).map(preprocess_img).batch(batch_size=32, drop_remainder=False)
    val_ds = val_ds.shuffle(buffer_size=len(val_ds)).map(preprocess_img).batch(batch_size=32, drop_remainder=False)
    
    return tr_ds, val_ds, te_ds

def define_and_train_model(tr_ds, val_ds, cifar100_swich, fft=True):
    """ Defines model structure following baseline AlexNet architecture and trains it. 

    Args:
        tr_ds (TensorFlow Dataset): training dataset.
        val_ds (TensorFlow Dataset): validation dataset.
        cifar100_swich (bool):  switch for determining if CIFAR-10 or CIFAR-100 data is being used.
        fft (bool, optional):   switch for if RGB or Fourier spectra model is being builts. 
                                Defaults to True (Fourier spectra model).

    Returns:
        Keras Model history, Keras Model: the trained model and related history
    """

    no_epochs = 25

    # define model parameters which depend on different factors
    # number of classes
    if cifar100_swich: no_classes = 100
    else: no_classes = 10

    # input shape
    # RGB images have 3 channels, Fourier magnitude and phase spectra for RGB images has 6 channels
    if fft: input_shape = (227,227,6)
    else: input_shape = (227,227,3)

    # build model
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),

        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"), 
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),

        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(), 

        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"), 
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),

        keras.layers.Flatten(),

        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(no_classes, activation='softmax')
    ])
    
    # compile the model
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=tf.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # show summary of layers
    model.summary()

    # train the model
    history = model.fit(
        tr_ds,
        epochs=no_epochs,
        validation_data=val_ds,
        validation_freq=1
    )
    
    # save model weights
    model.save(get_filename(cifar100_swi=cifar100_swich, fft_swi=fft), new=True)

    return history, model

def plot_results_from_history(model, history):
    """ Function for plotting training and validation data loss and accuracy after training is completed.

    Args:
        model (Keras Model): model object
        history (Keras Model history): model history object containing information during training
    """

    try:
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy value')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'], loc='lower right')
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss value')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.grid()
        plt.show()

    except Exception as e:
        print("print method failed")


def main():

    # switch variables for changing script functionality
    cifar100_sw = True      # Data: CIFAR-100 (True) or CIFAR-10 (False)
    fft_sw = False          # Fourier magnitude and phase spectra (True) or RGB images (False)
    pretrained = True       # use pretrained models (True) or train from scratch (False)

    # load training, validation and test datasets
    tr_ds, val_ds, te_ds = load_data(cifar100_swich=cifar100_sw, fft=fft_sw)

    # either use pretrained model weights, or train model with data 
    if pretrained:
        filename = get_filename(cifar100_swi=cifar100_sw, fft_swi=fft_sw)
        model = load_model(filename)
        print(f"\nModel built from file: {filename}\n")
    else:
        history, model = define_and_train_model(tr_ds, val_ds, cifar100_swich=cifar100_sw, fft=fft_sw)

        # Show information about training
        plot_results_from_history(model, history)

    # evaluate model
    score = model.evaluate(te_ds)
    print(f"\nTest accuracy | {score[1]}\nTest loss     | {score[0]}")

if __name__ == "__main__":
    main()
