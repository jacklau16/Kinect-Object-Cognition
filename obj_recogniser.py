###################################################################################################
# File       : obj_recogniser.py
# Description: Main class for implementation of object recognition model
# Usage      : <Class to be instantiated and used by other Python routine
###################################################################################################
import concurrent
import time
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


class ObjRecogniser:
    ###################################################################################################
    # Class constructor method, called when the an instance of ObjRecogniser is created
    ###################################################################################################
    def __init__(self):
        # Tuning parameters
        self.imagetteGridSize = 30
        self.imagetteStepSize = 15
        self.depthThreshold = 75
        self.imgDistThreshold = 6000
        self.weightDistThreshold = 500
        self.nnDistThreshold = 500
        self.confLevelThreshold = 0.2
        self.eigenVectorUsed = 20
        self.resizeImage = True
        self.equaliseHistogram = True
        self.useCannyDetection = True

        # PCA & model related variables
        self.W_train = []
        self.U_train = []
        self.imgMean_train = []
        self.objIdx_train = []
        self.W_trainNNLearner = []

        # File path and training label file
        self.trainImgPath = './train_img/'
        self.trainLabelFile = './Set1Labels.txt'

        # Initialise the training labels
        labelFile = open(self.trainLabelFile, 'r')
        self.objName_train = labelFile.read().splitlines()

    ###################################################################################################
    # Method to slice a single image to multiple imagettes, and return the set of imagettes sliced
    ###################################################################################################
    def slice_image_to_imagettes_rgb(self, imgInput):
        grid_size = self.imagetteGridSize
        step_size = self.imagetteStepSize
        imagette_set = []
        rows = int((imgInput.shape[0] - grid_size) / step_size) + 1
        cols = int((imgInput.shape[1] - grid_size) / step_size) + 1

        for row in range(rows):
            for col in range(cols):
                # Cut out the imagette from the original image
                imgGrid = imgInput[row*step_size:row*step_size+grid_size, col*step_size:col*step_size+grid_size, :]

                # Detect if there is any edge in the imagette, discard it if none detected
                if self.useCannyDetection:
                    edgeDetection = cv2.Canny(imgGrid, 100, 200)
                    # Any pixel on the detected edge(s) will have 255 (white) in value
                    if np.max(edgeDetection) == 255:
                        imagette_set.append(imgGrid.flatten())
                else:
                    imagette_set.append(imgGrid.flatten())

        return np.array(imagette_set).T

    ###################################################################################################
    # Method to read the image file, and check if its height & width is not smaller than a single imagette size
    ###################################################################################################
    def read_and_check_img(self, imgFile):
        grid_size = self.imagetteGridSize
        img = cv2.imread(imgFile)
        h = img.shape[0]
        w = img.shape[1]

        if (h >= grid_size) & (w >= grid_size):
            return img
        else:
            return None

    ###################################################################################################
    # Method to do pre-processing of an image, including resizing and histogram equalisation (if enabled)
    ###################################################################################################
    def preprocess_img(self, imgInput):
        # Resize the image if the resizeImage flag is turned on
        if self.resizeImage:
            resized_w = 200
            resized_h = 200
            # resized_h = int(resized_w / imgTrain.shape[1] * imgTrain.shape[0])
            imgInput = cv2.resize(imgInput, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

        # Equalise the histogram for R, G, B channels if the equaliseHistogram flag is turned on
        if self.equaliseHistogram:
            imgInput[:, :, 0] = cv2.equalizeHist(imgInput[:, :, 0])
            imgInput[:, :, 1] = cv2.equalizeHist(imgInput[:, :, 1])
            imgInput[:, :, 2] = cv2.equalizeHist(imgInput[:, :, 2])
        return imgInput

    ###################################################################################################
    # Method to perform recognition of a single imagette, return the object class ID if recognised, otherwise return -1
    ###################################################################################################
    def recognise_imagette(self, imgInput):
        imgDistThreshold = self.imgDistThreshold
        eigen_vector_used = self.eigenVectorUsed
        imgMean = self.imgMean_train
        U = self.U_train
        nnLearner = self.W_trainNNLearner

        X_input = imgInput - imgMean
        W_input = U.T @ X_input
        imgRecon = U @ W_input
        imgRecon = imgRecon + imgMean

        # Compute the Euclidean distance of two matrices using matrix norm
        dist = np.linalg.norm(imgInput - imgRecon)

        if dist > imgDistThreshold:
            print("Reconstructed Image has too large distance from original:", dist)
            return -1
        dist, ind = nnLearner.kneighbors([W_input.T[0:eigen_vector_used]], 2, return_distance=True)
        return self.objIdx_train[ind[0][0]]

    ###################################################################################################
    # Method to perform image recognition, returning the recognition result as list of matching scores for each class
    ###################################################################################################
    def recognise_img(self, imgInput):
        matchedCount = 0
        t0 = time.time()
        imagettes = self.slice_image_to_imagettes_rgb(imgInput)
        regResult = np.zeros((len(self.objName_train),))

        # if no meaningful imagette is found, simply return zero scores
        if imagettes.shape[0] == 0:
            return regResult

        # Fork multiple worker threads, to execute recognise_imagette() for each imagette concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(imagettes.shape[1]):
                futures.append(executor.submit(self.recognise_imagette, imagettes[:, i]))

            # Wait for the thread to complete, and retrieve the result
            for future in concurrent.futures.as_completed(futures):
                regIdx = future.result()
                if regIdx != -1:
                    # Add the score of the matched object class by 1 for each matched imagette
                    regResult[regIdx] += 1
        return regResult

    ###################################################################################################
    # Method to perform training of the model, given the training image sets are captured and saved
    ###################################################################################################
    def train(self, exportWeights=False):
        # Get the path for training images
        trainImgPath = self.trainImgPath
        # Reading all the images and storing into an array
        trainImgFiles = os.listdir(self.trainImgPath)
        eigenVectorUsed = self.eigenVectorUsed
        trainSet = None
        objIdx = []

        # If there is no image in the training image path, exit the routine
        if len(trainImgFiles) == 0:
            print("ERROR: no training image present in", trainImgPath)
            exit(-1)

        # Loop through the training images
        for trainImgFile in trainImgFiles:

            imgTrain = self.read_and_check_img(f'{trainImgPath}{trainImgFile}')

            if imgTrain is not None:
                # Get the class ID from the training image file name, e.g. 2-9.jpg, "2" - 1 is the class ID
                classID = int(trainImgFile.split('-')[0])-1

                # Perform pre-processing of the training image
                imgTrain = self.preprocess_img(imgTrain)

                # Slice the training image into small imagettes
                newTrainSet = self.slice_image_to_imagettes_rgb(imgTrain)
                if trainSet is None:
                    trainSet = newTrainSet
                else:
                    trainSet = np.append(trainSet, newTrainSet, axis=1)

                # Assign the same class ID to all the newly generate imagettes, as they are from the same image
                objIdx.extend([classID] * newTrainSet.shape[1])

        print("No. of training samples:", len(trainSet[0]))
        print("Normalise the training samples...")
        # Compute average image of all the imagettes
        imgMean = np.mean(trainSet, axis=1)

        # Normalise images by subtracting with their mean
        X = trainSet - np.tile(imgMean, (trainSet.shape[1], 1)).T

        # Perform SVD and obtain eigenvectors
        print("Performing SVD...")
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)

        # Project the train set onto the first N principal components (eigenvectors) using
        U_sub = U[:, 0:eigenVectorUsed]
        W = U_sub.T @ X

        self.W_train = W
        self.U_train = U
        self.imgMean_train = imgMean
        self.objIdx_train = objIdx

        print("W:", W.shape, "U:", U.shape)

        # Train the Nearest Neighbour Learner with the weights of the imagettes
        nnLearner = NearestNeighbors(algorithm='auto')
        nnLearner.fit(W.T, objIdx)
        self.W_trainNNLearner = nnLearner

        # Save the weights to file if the flag exportWeights is turned on
        if exportWeights:
            dfResult = pd.DataFrame(W.T)
            dfResult['class'] = self.objIdx_train
            dfResult.to_csv("weights.csv")
