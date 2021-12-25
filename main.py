# Get labels
import os
import random

import cv2
import imutils
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class OriginImages:
    def __init__(self, data, org_image):
        self.x_data = data
        self.org_image = org_image

def save_image(directory, img) :
    cv2.imwrite(directory, img)

def convert_to_gray(img):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def edge_detecting(img):
    denoised_image = cv2.Canny(image, 300, 310)
    return denoised_image

def rotate(img, angle):
    '''angle is going anti-clockwise direction'''
    rotate_img = imutils.rotate(img, angle)
    return rotate_img

def denoising(img):
    '''the higher h will be , the more noise will be removed as well as
    some details be removed from the image so don't make it so high!!'''
    converted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        cv2.fastNlMeansDenoisingColored(converted_img, None, 10,10,7,21)
    except:
        print("couldn't make edge detecting..")
    return converted_img


print("Processing Labels...")
labels = os.listdir('Animals/')
print("Labels Done!")
print(labels)


# Get images and resize them
x_data = []
y_data = []
org_images = []
cnt = 0

print("Processing Images...")
for label in labels:
    path = 'Animals/' + label
    cnt = 1
    images_paths = os.listdir(path)
    for image_path in images_paths:
        image = cv2.imread(path + '/' + image_path)
        '''
        for same image ratio..
        scale_percent = 60  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        '''
        image_resized = cv2.resize(image, (100, 100))
        image_resized = convert_to_gray(image_resized)
        image_resized = denoising(image_resized)
        image_resized = edge_detecting(image_resized)
        x_data.append(np.array(image_resized))
        y_data.append(label)
        org_images.append(image)
        '''
        for printing images after resizing
        '''


        if(cnt <= 1):
            cv2.imshow(path + '/' + image_path , image_resized)
            cv2.waitKey(0)
            print(path + '/' + image_path)
            cnt = cnt + 1

    break

print("Images Done!")


# Normalizing images
x_data = np.array(x_data, dtype="object")
y_data = np.array(y_data, dtype="object")
org_images = np.array(org_images, dtype="object")
x_data = x_data / 255.0



# Encoding labels
y_encoded = preprocessing.LabelEncoder().fit_transform(y_data)

# Shuffling Datasets
rand = np.arange(x_data.shape[0])
random.seed(22)
random.shuffle(rand)
x_rand_data = x_data[rand]
y_rand_data = y_encoded[rand]
org_images = org_images[rand]

images = []
for i in range(len(x_rand_data)):
    A = OriginImages(x_rand_data[i], org_images[i])
    images.append(A)

# Setting data_images to train and test
X_train, X_temp, Y_train, Y_temp = train_test_split(x_rand_data, y_rand_data,
                                                    test_size=.30)
X_test, X_vald, Y_test, Y_vald = train_test_split(X_temp, Y_temp, test_size=.50)
