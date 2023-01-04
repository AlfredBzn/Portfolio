# Computer vision for dog breeds classification

This project is focused on computer vision thanks to ML algorithms and especially convolutional neural networks (CNN). The aim of this project is to classify pictures of dogs depending on their corresponding breed. The data used in this project have been obtained from the Stanford Dogs Dataset available <a href=http://vision.stanford.edu/aditya86/ImageNetDogs/>here</a>. First the data format and distribution have been studied through a rapid EDA. Then the data have been imported in a tensorflow dataset in order to be fed to deep neural networks with high performances. Indeed, such image classifications are often performed using deep neural networks and more specifically convolutional neural networks (<a href=https://en.wikipedia.org/wiki/Convolutional_neural_network>see</a>).

![image.png](attachment:ce13238e-f3ce-4798-b938-4236599437b3.png)

**Figure 1: Illustration of a convolutional neural network**

Several technics of image transformation (augmentation, whitening...) have been tested. These technics allow either to exacerbate the features or can reduce the model overfitting by generating some diversity in the dataset.

Then strategies of "Transfer Learning" have been assayed with popular models (<a href=https://keras.io/api/applications/>see</a>). Two methods have been tested : training only the last dense layers or training the first and last dense layers.

Finally, the prediction resulting from the model with the higher performances have been studied in detail in order to better understand the possible flaws of the model.

# EDA

The dataset is composed of **20580** dog pictures separated between **120** breeds. In figure 3, two samples from the dataset hava been displayed as examples. It can be observed that the pictures can contain a complex background (with objects, vegetation, humans…).

Figure 3: Examples of pictures present in the dataset

The average picture number per breed is represented graphically in figure 4.

Figure 4: Average picture number per breeds.

Breeds are represented by a different number of pictures (with an average number of 171). However, this number seems balanced with a standard deviation of 23.

The picture size (definition) is also an important parameter. The average pictures definitions per breeds is displayed in figure 5.

Figure 5: Average picture size per breed

Breeds are represented with pictures of different sizes. Indeed, some breeds folders contain larger pictures as for the Saint-Bernard or Irish water spaniel.

# Data augmentation and preprocessing

The dataset have been imported with tensorflow (TF) as a TF dataset. This allow to feed the data by batch to the ML model and thus allow better performances.

## Data augmentation

Data augmentation is a fast and easy way to add variance to a dataset. This augmentation can take various forms (rotation, flip, distorsion, noise...) and is perfomed in the hope of obtaining a model with a higher level of generalisation.

The processes of data augmentation descibred in this part have been developped based on the TensorFlow documentation (see [here](https://www.tensorflow.org/tutorials/images/data_augmentation))

Data augementation is performed as layers with 3 steps :
-	Flip
-	Rotation
-	Zoom

Examples of pictures after data augmentation are displayed in figure 6.

Figure 6: Examples of images transformation after processing through augmentation layers

The augmentation layers could be applied to the dataset beforehand. However, it can also be incorporated to the model (as input layers) and employed only during model.fit. This way, at each epoch, the train dataset is processed randomly by the augmentation layers.

## Preprocessing

Image preprocessing (as contour detection, equalization ...) can improve model performances by reducing noise in the input data (potentially reducing overfitting). To perform these operations, a tool called "ImageDataGenerator" from "keras.preprocessing" was generally employed. However, the use of this tool according to Keras documentation is deprecated. Instead, the use of skimage as a preprocessing tool for equalization and whitening have been employed.

### Pixel value centering

A comon preprocessing step in deep learning applied to images is to center the images pixel value (as it is the case is vgg16 as an example). Thus this preprocessing step have been essayed. The mean pixel value have been taken from the training set. The centering of the data have been done as a input layer. A pixel mean value of 111.592834 have been found.

### ZCA whitening

The ZCA Whitening procedure allow to exacerbate images features. The process was inspired from https://www.kdnuggets.com/2018/10/preprocessing-deep-learning-covariance-matrix-image-whitening.html/3 and keras code : https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/image_data_generator.py

Examples of images whitening are displayed in figure 7. It is worth mentioning that ZHC whitening requires extensive calculations and thus a reduction of the image resolution (leading to possibly pixelated images).

Figure 7: Example of images whitening

### Equalisation

Image equalization is a method that allows to automatically adjust the contrast of a given picture. Several methods of equalizations exist and thus three are demonstrated here (see figure 8). The equalization is performed thanks to the library Skimage, the image processing library of SciKit-Learn. More info and part of the code can be found at https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py

Figure 8: Equalisation of pictures from the dataset

## Contour detection

A common image preprocessing tool is contour detection. It allows highlighting object contours in an image. Previously employed for feature detection is image recognition (without deep learning) it can now be used to reduce the noise in the input data and thus improve the neural network generalization. Several technics of contour detections have been developed. Here as a demonstration, the Prewitt, Sobel and Scharr contour detection technics have been essayed (see figure 9.

Figure 9: Example of contour detection on pictures form the dataset

## The superimposition of features highlighting technics

The previously cited preprocessing methods can be superimposed in order to even more exacerbate pictures features. As an example, equalisation and contour detection have been applied to an image of the dataset in figure 10. In figure 11, ZCA whitening have also been added to the process. However, it appears clearly that the dimension reduction induced by the ZCA whitening leads to a highly pixelated image that is difficult to interpret.

Figure 10: Example of Equalisation and contour detection on a sample image

Figure 11: Example of equalisation, contour detection and ZCA whitening on a sample image.

# Training a CNN from scratch

A first model is built from scratch. The architecture of the model is inspired from VGG-16 (see figure 12). However, training VGG-16 would result in too long processing time, thus, some convolutional layers have been removed in order to ease the model learning. In the same way, the final dense layers have been lightened in order to same computational resources.

Figure 12: Schematic representation of the CNN model trained from scratch

The performances of the models have been recorded with Tensorboard and are displayed in figure 13.

Figure 13: Performances of the CNN trained from scratch

A max accuracy of about 0.1 is obtained on the validation set with a clear presence of overfitting. Indeed, the model is complex enough to memorise the train dataset leading to high train accuracy.

## With dropout layers

A method to solve this issue is to use droupout layers (see figure 14).

Figure 14: Schematic representation of the model trained form scratch including dropout layers

A similar model to the previous one including a dropout layer between dense layers (more prone to memorization) have been assessed. The obtained performances are displayed in figure 15.

Figure 15: Performances of the CNN trained from scratch with dropout layers.

## With data augmentation

Another method for avoiding overfitting is to add data augmentation layers (as described earlier) to generate diversity in the dataset and thus avoid data memorisation. The previously described augmentation layers have been added as an input to the model (see figure 16).

Figure 16: Schematic representation of the model trained from scratch including an augmentation layer

The performances obtained with this model are given in figure 17:

Figure 17: performances of the CNN trained from scratch with an augmentation layer.

It appears that the dropout layer limited overfitting in the first part of the training. However, after 4 epochs clear overfitting can be observed again. Similarly, data augmentation does not allow to completely avoid overfitting in this case. To completely avoid overfitting several technics could be investigated:
-	Adding more dropout layers
-	Increasing the dropout proportion
-	Performing even more data augmentation (through translation and cropping as an example)
-	Adding batch normalization layers that allow faster training with lower overfitting
-	Finally, increase the number of samples available in the dataset as ~171 pictures per breed is a relatively low number for training a CNN.

## With prior data pre-processing

Prior data preprocessing (equalisation and contour detection) has been applied to the dataset before being fed to the CNN. The performances of the model are displayed in figure 18.

Figure 18: Performances of the CNN with pre-processed data.

Again, the difference between the train and validation accuracy and loss suggest an overfitting of the model. Even though the data are augmented before getting fed to the model (with random rotation, zoom...) the model achieves to memorize the training data. This is explained by the low number of samples compared to the large number of targets (i.e., breeds). Trying to predict a lower number of breeds could allow higher performances and reduce overfitting.

# Transfer learning

A common solution when only a limited amount of data is available is to use a model that has already been trained on similar data. This way, data are preprocessed by the already trained model for relevant features extraction. Then, dense decision layers need to be trained in order to perform the classification task. Another strategy (but more costly in processing time) is to train the dense layers as well as the first convolutional layers of the pretrained model. This strategy allows fine-tuning the model to the data studied. Finally, a model can be imported and fully trained on the data in order to have a fully data-tuned model. As a low amount of data and computational resources are available in this project, the first two strategies have been assessed.

## Transfer learning for feature extraction

The Strategy of transfer learning have been assessed with three popular models: VGG16, ResNet50V2 and Xception (<a href=https://keras.io/api/applications/>see</a>). The process for importing a model and using it in a strategy of transfer learning is described in keras documentation (<a href= https://keras.io/guides/transfer_learning/>source</a>).

The model have been evaluated after freezing the first layers of the models and only training the last dense layers. The performances obtained with the different model are given in figure 19, 20 and 21.

Figure 19: Performances obtained with frozen VGG16

Figure 20: Performances obtained with frozen ResNet50V2

Figure 21: Performances obtained with frozen Xception

# Transfer learning with fine-tuning

As described earlier, a strategy of transfer learning with a fine-tuning of the first layers of the model is assessed. The model ResNet50V2 has been assessed as it showed poor performances on our dataset. In comparison, the model VGG16 (as it is the simpler in the selection) have also been tested through this strategy.

The performances of the models are given in figure 22 and 23.

Figure 22: Performances of ResNet50V2 model after training of the dense and first layers of the model

This time, thanks to the fine-tuning of the model first layers, higher performances were obtained with validation losses and accuracy value equivalent to the Xception model. Only traces of overfitting can be observed at the end of the model training.

Figure 23: Performances of VGG16 model after training of the dense and first layers of the model

Finetuning of the model VGG16 did not improve the model performances as the model seems to not be able to learn at all in this configuration.

# Final model assessment

The model with the higher performances was the model Xception after 3 epochs. This model is loaded, and its performances are studied in detail.

## Accuracy

First the accuracy of the model has been measured with the metrics of the accuracy (see equation 1).
Equation 1
As depicted in equation 1, the true negatives and thus 0 values are present in the numerator. However, as the target is one hot encoded, most of the values are 0 and thus can easily be true negatives. Thus, the number of true negatives is prevalent, leading to an accuracy above 95% for every predicted breeds. The measured accuracy for each breed is given in figure 24.

Figure 24: Accuracy measure for the predicting each breed (with a zoom)
As the accuracy seems to not be the ideal metrics two other metrics have been employed, the precision and recall (see equation 2 and 3).

## Precision

The precision is measuring the proportion of correct predictions among the positive prediction only. In our case, it indicates for a given breed the proportion that have been attributed falsely to dogs from other breeds. The measured precision for each breed is given in figure 25.
Figure 25: Precision calculated on the prediction for each breed
The Rhodesian_ridgeback corresponds to the class where the proportion of correct positive predictions is the lowest. The redbone seems to be the one that is the most mistaken with Rhodesian_ridgeback. Indeed, as shown in figure 26, the two breeds share a lot of physical similarities.

Figure 26: Comparison between the Rhodesian Ridgeback and Redbone

## Recall

Recall is a metric that indicates the proportion of true positives that were was identified correctly. In our case, it indicates for a given breed what is the proportion of other breeds that are falsely attributed by the model to picture of dogs of the first breed. The recall calculated for each breed is given in figure 27.
Figure 27: Recall calculated on the prediction for each breed
The breeds falsely attributed to Chihuahuas are
-	Cardigan
-	Brabancon_griffon
-	Ibizan_hound
-	English_setter

The confusion of the model with these breeds is more mysterious and could be investigated further.

# Conclusion

Conclusion

The aim of this project was to explore the use of images as input for the training of deep neural networks. For this, a dataset containing 20 508 images of dogs of 120 different breeds have been collected.

First an exploratory data analysis has been performed. From this study, it appeared that the dog breeds were represented by an unequal but balanced number of pictures with an average of 171 pictures and a standard deviation of 23. A rapid look randomly drawn picture indicated that the pictures could present multiple dogs, their owner and possibly a complex background, making the classification evermore difficult. Finally, the picture definition has been studied. The pictures present a large range of dimensions between 0.4 and 0.8 Mpixels. Moreover certain dog breeds the Saint Bernard or Irish water spaniel are represented by pictures with, on an average, a higher definition.

During this project, the images have been imported following two strategies. First data have been imported with the tool "image_dataset_from_directory" from keras. This tool allows to rapidly create an efficient dataset with the images directories. Then, a second method starting from ".Dataset.from_tensor_slices" and building the dataset "by hand" was also assessed. This method was more complex but allowed to add a parameter for choosing a list of breeds to be imported in the dataset.

After that the data were imported, several methods of images preprocessing have been studied. Such methods can amplify features in the images for allowing faster processing. Functions have been built for pixel value centering, ZCA whitening, equalization and contour detection. The superimposition of methods have also been essayed.

Then a model of deep learning inspired by the popular model VGG16 has been built from scratch. After assessing the model performances, it has been found that the model was largely overfitting. Adding dropout and augmentation layers allowed to reduce the overfitting but the model showed poor performances.

Lower the number of targets (from 120 to 10 and then to 2) had a significant impact on the model performances and learning rate. The simple model built from scratch showed the better performances for a binary classification.

Techniques of transfer learning have then been assayed. Three popular models have been tested: VGG16, ResNet50V2 and Xception. When training only the final dense layers, VGG16 showed good performances (val accuracy = 0.53) but the high performances were obtained with Xception (val accuracy = 0.80). In the contrary ResNet50 showed poor performances, possibly indicating the high specificity of the model.

Train the first and last layers of VGG16 and ResNet50 was essayed. This methodology allows adapting the first layers of the model to the data considered in our project. Training the first layers of ResNet50 greatly improved the model performances. Inversely, VGG16 showed poor performances when train the first layers, showing that this method should be used on a case-by-case basis.

Finally, the productions of the model showing the higher performances were studied in detail with appropriates metrics. When metrics were applied on a specific class, Precision and Recall showed interesting trends and allowed to better understand the in-depth working of the model. As an example, the dogs that were the most mistaken indeed showed physical similarities.
