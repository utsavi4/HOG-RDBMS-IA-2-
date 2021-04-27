# HOG-RDBMS-IA-2-

This repository contains code for facial recognition based on histogram of gradients method using openCV, dlib, face-recognition, scikit-image, numpy and other libraries in python. If you want to test the code then run recognition.py file which is the main file that recognizes your face.

# Technology used : 
-openCV (Opensource Computer Vision) 
-Python libraries(Numpy, Scikit-image, Scikit-learn, Face-recognition, Cmake, Dlib)

# Environment:
-OS: MacOS Catalina
-Platform: Python 3
-Librarys: OpenCV 3

Here I am working on Face recognition using Histogram of Gradients method by using OpenCV(Python). The system recognizes the face by simply training the persons image and finding the encodings.

# How it run it :
Run it in python.
When we run hog.py a window is opened which shows how the HOG Descriptor Image looks like.
And by running the Basics.py, we detect the faces and get the encodings.
By running the recognition.py, we get our face recognized.
Here are the most important aspects of HOG:

# Explanation of the Process:
HOG focuses on the structure of the object. It extracts the information of the edges magnitude as well as the orientation of the edges.
It uses a detection window of 64x128 pixels, so the image is first converted into (64, 128) shape. The image is then further divided into small parts, and then the gradient and orientation of each part is calculated. It is divided into 8x16 cells into blocks with 50% overlap, so there are going to be 7x15 = 105 blocks in total, and each block consists of 2x2 cells with 8x8 pixels. We take the 64 gradient vectors of each block (8x8 pixel cell) and put them into a 9-bin histogram.The hog() function takes 6 parameters as input:
image: The target image you want to apply HOG feature extraction.
orientations: Number of bins in the histogram we want to create, the original research paper used 9 bins so we will pass 9 as orientations.
pixels_per_cell: Determines the size of the cell, as we mentioned earlier, it is 8x8.
cells_per_block: Number of cells per block, will be 2x2 as mentioned previously.
visualize: A boolean whether to return the image of the HOG, we set it to True so we can show the image.
multichannel: We set it to True to tell the function that the last dimension is considered as a color channel, instead of spatial.

![image](https://user-images.githubusercontent.com/61929937/116241147-36170f00-a782-11eb-9b82-8023f3849f69.png)

Moving on to the first part of facial recognition module, detecting and training of facial images. The Basics.py file basically implements facial detection of images. HoG Face Detector in Dlib is a widely used face detection model, based on HoG features and SVM. The dataset used for training, consists of 3 images.
STEP 1: The first step in our pipeline is face detection. Obviously we need to locate the faces in a photograph before we can try to tell them apart! To find faces in an image, we’ll start by making our image black and white because we don’t need color data to find faces. Then we’ll look at every single pixel in our image one at a time. For every single pixel, we want to look at the pixels that directly surrounding it. If you repeat that process for every single pixel in the image, you end up with every pixel being replaced by an arrow. These arrows are called gradients and they show the flow from light to dark across the entire image. This might seem like a random thing to do, but there’s a really good reason for replacing the pixels with gradients. If we analyze pixels directly, really dark images and really light images of the same person will have totally different pixel values. But by only considering the direction that brightness changes, both really dark images and really bright images will end up with the same exact representation. That makes the problem a lot easier to solve! To do this, we’ll break up the image into small squares of 16x16 pixels each. In each square, we’ll count up how many gradients point in each major direction (how many point up, point up-right, point right, etc…). Then we’ll replace that square in the image with the arrow directions that were the strongest. The end result is we turn the original image into a very simple representation that captures the basic structure of a face in a simple way.

![image](https://user-images.githubusercontent.com/61929937/116242893-023ce900-a784-11eb-8ed4-c6b8de8ca6f1.png)
STEP 2: Encoding Faces! What we need is a way to extract a few basic measurements from each face. Then we could measure our unknown face the same way and find the known face with the closest measurements. After repeating this step millions of times for millions of images of thousands of different people, the neural network learns to reliably generate 128 measurements for each person. Any ten different pictures of the same person should give roughly the same measurements.
Machine learning people call the 128 measurements of each face an embedding.

![image](https://user-images.githubusercontent.com/61929937/116243255-66f84380-a784-11eb-9336-4c534913191b.png)

STEP 3: Finding the person’s name from the encoding.This last step is actually the easiest step in the whole process. All we have to do is find the person in our database of known people who has the closest measurements to our test image. You can do that by using any basic machine learning classification algorithm. No fancy deep learning tricks are needed. We’ll use a simple linear SVM classifier.

![image](https://user-images.githubusercontent.com/61929937/116243798-f271d480-a784-11eb-87c7-0979e067d0ff.png)









