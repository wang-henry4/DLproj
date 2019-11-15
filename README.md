# DLproj
Code for class project for ECS 289G Fall 2019, Deep Learning. We are using the [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification)
and leveraging ResNet50V2 to identify specific proteins in images of human cells. We are training multiple models, each using a different layer in the ResNet50V2 as the feature extractor,
in order to find the best hidden layer feature extractor for this given task. Given that the input images are very different from the Imagenet training set that ResNet is trained on, we suspect 
that using an earlier hidden layer will provide better results.
