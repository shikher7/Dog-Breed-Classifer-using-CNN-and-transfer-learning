# CNN Project: Dog Breed Classifier

### Project Overview

The Dog breed classifier is a well-known problem in Machine Learning. The problem is to
identify a breed of dog if dog image is given as input. 
The objective of the project is to build a machine learning model that can be used within
web app or any operating system app to process real-world, user-supplied images. I used
CNNs and pretrained models in solving this problem.

Machine Learning/Deep Learning and Computer Vision helps you build machine learning
models where you train a model to recognize the dog breed. CNN was a huge invention in
deep learning, this gave a boost to lots of applications in visual recognition. I built a CNN
from scratch using Pytorch and use some of the popular CNN architectures for our image
classification task.

#### Metrics


Considering the accuracy paradox which states-

“Accuracy Paradox for Predictive Analytics states that Predictive Models with a given level of
Accuracy **may** have **greater** Predictive Power than Models with **higher** Accuracy.”

As in this case, as the training set of multiple classes contains slightly imbalanced classes,
which means along with accuracy as a metric of performance we will need to use multi-class
**log loss** which will punish the classifier if the predicted probability leads to a different label
than the actual and cause higher accuracy.

In my implementation, I used **CrossEntropyLoss** as the loss function along with accuracy as
the metric of performance.

torch.nn.CrossEntropyLoss()

#### Dataset Used

For this project, the input format is of image type, we input an image and identify the breed
of the dog.

The following datasets is used:

1. Dog Images: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-
    project/dogImages.zip
2. LFW: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
    

### Algorithms and Techniques

In this project, firstly, we detected the image was human or dog. We used openCV’s
Haar feature-based cascade classifiers for human detection and pre-trained VGG- 16
model for dog breeds classifying.

CNNs are a kind of deep learning model that can learn to do things like image
classification and object recognition. They keep track of spatial information and learn
to extract features like the edges of objects in something called a convolutional layer.
The convolutional layer is produced by applying a series of many different image
filters called convolutional kernels to the input image.

Other than the CNNs, OpenCV’s implementation of Haar feature cascade classifier
was used to perform face detection of the human images.

On tasks involving computer vision CNNs outperforms other Neural Networks like
MLP by a far. This project requires two CNNs ( model from scratch and model from
transfer learning ) mainly for classifying the breed and one for detecting dogs

It is very efficient to use pre-trained networks to solve challenging problems in
computer vision. Once trained, these models work very well as feature detectors for
images they weren’t trained on. Here we’ll use transfer learning to train a network
that can classify our dog photos.

Here, I chose the VGG16 model. I thought the VGG16 is suitable for the current
problem. Because it already trained large dataset. And it performed really well (came

2nd in ImageNet classification competition) The fully connected layer was trained on
the ImageNet dataset, so it won’t work for the dog classification specific problem.
That means we need to replace the classifier (133 classes), but the features will work
perfectly on their own.

Hence, I selected VGG16 pre-trained model for transfer learning. I froze all feature
parameters. I just changed the last fully connected layer output as 133 and trained
classifier again. In this model, cross entropy loss criteria was selected because of
making classification and Adam optimizer was selected.


### Result and Improvement

- My model reached 73% testing accuracy.

In real use case user will want a more accurate model. I can try different epoch number
and architectures but this would require more computing power especially more powerful
GPUs. For the transfer learning model, I could experiment with other pre-trained models
and see if they perform better.

For further making this solution as a usable product and improving user experience, I would
like to integrate this within an website by using flask and aws, to host the model as endpoint
and using it on the website. On the Webpage user will have the options to upload pic or
maybe take on picture on camera based device and it will give result about the breed or dog,
or similar looking breed in case of human.


##### References

1. https://docs.opencv.org/trunk/db/d28/tutorial_cascade_classifier.html
2. https://pytorch.org/docs/master/torchvision/models.html
3. https://cs231n.github.io/convolutional-networks/
4. https://github.com/udacity/deep-learningv2pytorch/tree/master/project-dog-
    classification
5. https://arxiv.org/abs/1512.
6. https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-
    21c14f7a716e
7. LFW Dataset, Udacity https://s3-us-west-1.amazonaws.com/udacity-aind/dog-
    project/lfw.zip
8. Dog Image Dataset, Udacity https://s3-us-west-1.amazonaws.com/udacity-
    aind/dogproject/dogImages.zip
