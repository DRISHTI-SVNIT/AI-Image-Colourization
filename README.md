# ML-alpha-colorising
PROJECT DESCRIPTION 	 
 
 Coloi as the name suggests something related to colors . But what is the project about . Lets say if you want to color any black and white image : Ok thats an easy task if the picture would be colored in any sense . But if we tell you tell you to color the picture as it would have been if we would have taken a colored picture then it might seem tough and also its tough to color without knowing the situation or the moment in which the picture was taken. Now think how wonderful it be if you have a model that automatically fills the color in the photo . May seem awesome right .  We at Drishti have build a model based on Machine Learning that could colorize images without any human effort . Let me take you to a ride how we have build this model
1. LEARNING PHASE
Phases	    A. COURSES
    B. MATERIALS 
2. DATASET PHASE 
3. MAKING OF DATASET
4. NN PHASE 
5. ALPHA VERSION
6 . BETA VERSION
7. NEED OF GPU
8. TUNING PHASE
 
Learning phase:
 
 Courses:
 
As the project required deep knowledge of Machine Learning we were required to do Deep Learning Courses and a Framework Course:
  
   
	Courses 	 Followed are :
1 . Deep Learning by Andrew NG (This basically included 	 5 courses and itself is a Specialization 	 Course on Deep Learning) 
2 . Medium Courses on Neural Networks and its 	 applications
3 . Frameworks we followed was TensorFlow 	(Googles Deep Learning Framework 	)
 
 
Materials used
 
We used Google Colab whenever during any course we required as its best suited for doing Deep Learning works and also got familier with 
GOOGLE CLOUD PLATFORM which provides 
GPU access for a limited time for an account
 
 
 
 Dataset phase
IMPORTANCE OF DATASET-   As we know that Machine Learning is purely based on inputs of data and learning from the Data itself the Dataset phase cannot be neglected as it forms the whole and heart of the project and the whole model depends on the Quality of Dataset 
 
Q. What do we mean by the quality of Dataset and why is it important:
Dataset is the collection of important informations that we feed into the Machine Learning Model and thus we train the model to predict on the unknown data
 
Our Dataset: 
As creation of Dataset is not an easy task as it includes lots of information and various processes But as a try we created a Dataset in which we took a colored picture and converted it to a black and white image using OPENCV library of python in which we can convert an normal image to a black and white image
But as we know Dataset is not so easy to be created as it depends on various factors such as :
 
 
1.	Data Augmentation 
2.	Data Cleaning
3.	Data Collection (MOST INITIAL)
4.	Data Compression Techniques
 
 
 
So what we did inorder to get a Clean and Quality Data :
Our created Dataset contained around 25k images but it was not enough to train a well accurate model so we search for an open source Dataset and we found one on the Medium website 
 
 
 
NN PHASE : 
 
Now as we knew that after we have a dataset created we would have to have a proper model on which dataset can be fed and we get a proper and accurate output
 
NN PHASE - ( Neural Network Phase)
Different Algorithms work differently and have their own benifits . But training a model with the help of Neural Network can increase efficiency of model ( here efficiency co-relates to accuracy ) and helps in fine tuning the model . 
 
 CNN - the neural brother
  
CNN also known as Convolutional Neural Network is a kind of Neural Networks for Images thus is also considered as Computer Vision by Andrew Ng. Now as we are working with images we will be using Neural Network for the purpose. 
 
 
Now what the CNN does is treats the image as a RGB channel and applies Neural Network accordingly on different blocks of channel also knows pooling process (max_pooling).
 
Not covering the deep aspects how the process goes on in the CNN we will now discuss the different phases that we undergo during the whole project
 
ALPHA PHASE
DESCRIPTION -
 
 
Consider any black and white image so if we want to represent it as an array or matrix we can do it by considering pixel intensity.
Same logic goes for the colored images which can considered as a 3d Tensor as in case of Tensorflow of RGB colored surface of varying pixel intensity
 
COLOR - SPACE :
For considering the 3d Tensor might be tough but by using the algorithm to convert the color space from RGB to LAB we will have a 2d Tensor of green-red and blue-yellow
 
 
a Lab encoded image has one layer for grayscale, and has packed three color layers into two. This means that we can use the original grayscale image in our final prediction. Also, we only have two channels to predict.
 
To turn one layer into two layers, we use convolutional filters. Think of them as the blue/red filters in 3D glasses. Each filter determines what we see in a picture. They can highlight or remove something to extract information out of the picture. The network can either create a new image from a filter or combine several filters into one image.


