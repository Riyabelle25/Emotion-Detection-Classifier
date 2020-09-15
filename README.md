# Submission for Round-2 in Spotle AIthon

Although the submission is done(12th September 2020), I will continue working on this model, and try to improve accuracy.


`The Objective:`

The objective of this study is to classify mood of the person from facial expressions Images are categorized in three classes namely sadness, fear and happiness based on the emotion shown in the facial expressions .

 
`The Dataset:`

The data consists of `48x48 pixel grayscale images of faces`. The pixel values are stored in 2304 (48*48) columns. These column names start with pixel. Along with pixel values, there is emotion column that say about mood of the image.

The task is to categorize each face based on the emotion shown in the facial expression in to one of three categories.

Along with pixel values, aithon2020_level2_traning.csv dataset contains another column emotion that say about mood that is present in the image. This is the dataset you will use to train your model.

 
`What we did:`

After reading blogs and some relevant research papers- We’ve started working on a `Deep Convolutional Neural Network (DCNN)`. 

To overcome the somewhat small size(10,817) of data from the training dataset, we've implemented Data Augmentation using Keras. 

BatchNormalization has been used for better results.

Two callbacks being used- `Early stopping` to avoid overfitting training data and `ReduceLROnPlateau` for the learning rate.
