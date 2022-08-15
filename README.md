# Computer Vision app

This is a repository for building  a multi-class image classification using Deep Learning and python scripts (with PEP8 standards). Currently supported image classes are airport, bridge, building, car, pipe, power plant and railway.

## Technologies used

Pytorch for deep learning training and streamlit for the UI
 

## Getting Started 

Training images were collected from  various free online resources by using web scraping. 
The starting point was the transformations for training and validation dataset. Some  randomized cropping (224*224), horizontal flipping and also normalization were also applied then to make our data similar to the Imagenet database. For the test set however, only the center cropping was applied while building the pipeline.
The next step was to create the training and validation datasets and dataloaders with the batch size of 8. It turns out that this batch size was ideal for the size of the dataset. The loss function used was Cross entropy and the chosen optimization algorithm was Adam optimizer.
The final model was the pre-trained model ResNet-50 which became the state-of-the-art model for the problem. 
After training the model we can see that the model performed better in the production compared to the training set. It’s also pretty impressive that with only 10 number of epochs, the model’s best accuracy in the production dataset was actually 100% and loss was below 5% at all epochs.


## The steps for installation:

1. Clone this repo to your local machine
2. $ pip install streamlit
3. $ streamlit run main.py
   
## Analysis and visualizations: [![My Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_wrxpN34Th08xIvTmI7SFIK0-ZlkUhoD?usp=sharing) 

## Quick GIF demo:
![classification](https://user-images.githubusercontent.com/53462948/184576795-897d1963-7347-4503-9156-5fd8d474fb97.gif)
