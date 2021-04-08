###################################################################################################
# File       : obj_recogniser.py
# Description: Main class for implementation of object recognition model
# Usage      : python obj_recogniser.py <Training video dataset path>
###################################################################################################
import concurrent
import math
import time
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class ObjRecogniser:

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

    def slice_image_to_imagettes_rgb(self, imgInput):
        grid_size = self.imagetteGridSize
        step_size = self.imagetteStepSize
        train_set = []
        train_row = int((imgInput.shape[0] - grid_size) / step_size) + 1
        train_col = int((imgInput.shape[1] - grid_size) / step_size) + 1

        for row in range(train_row):
            for col in range(train_col):
                # Cut out the imagette from the original image
                ImgGrid = imgInput[row*step_size:row*step_size+grid_size, col*step_size:col*step_size+grid_size, :]

                # Detect if there is any edge in the imagette, discard it if none detected
                if self.useCannyDetection:
                    edgeDetection = cv2.Canny(ImgGrid, 100, 200)
                    # Any pixel on the detected edge(s) will have 255 (white) in value
                    if np.max(edgeDetection) == 255:
                        train_set.append(ImgGrid.flatten())
                else:
                    train_set.append(ImgGrid.flatten())

        return np.array(train_set).T

    def read_and_trunc_img_rgb(self, imgFile):
        grid_size = self.imagetteGridSize
        img = cv2.imread(imgFile)
        h = img.shape[0]
        w = img.shape[1]

        if (h >= grid_size) & (w >= grid_size):
            return img
        else:
            return None

    def preprocess_img(self, imgTrain):
        # Resize the image if the resizeImage flag is turned on
        if self.resizeImage:
            resized_w = 200
            resized_h = 200
            # resized_h = int(resized_w / imgTrain.shape[1] * imgTrain.shape[0])
            imgTrain = cv2.resize(imgTrain, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

        # Equalise the histogram for R, G, B channels if the equaliseHistogram flag is turned on
        if self.equaliseHistogram:
            imgTrain[:, :, 0] = cv2.equalizeHist(imgTrain[:, :, 0])
            imgTrain[:, :, 1] = cv2.equalizeHist(imgTrain[:, :, 1])
            imgTrain[:, :, 2] = cv2.equalizeHist(imgTrain[:, :, 2])
        return imgTrain

    def recognise_imagette(self, imgInput, imgMean, W, U, nnLearner):

        imgDistThreshold = self.imgDistThreshold
        weightDistThreshold = self.weightDistThreshold
        nnDistThreshold = self.nnDistThreshold
        eigen_vector_used = self.eigenVectorUsed

        X_input = imgInput - imgMean
        W_input = U.T @ X_input
        imgRecon = U @ W_input
        imgRecon = imgRecon + imgMean

        # Compute the Euclidean distance of two matrices using matrix norm
        dist = np.linalg.norm(imgInput - imgRecon)
        #dist = self.euclidean_distance(imgInput, imgRecon)

        if dist > imgDistThreshold:
            print("Reconstructed Image has too large distance from original:", dist)
            return -1
        dist, ind = nnLearner.kneighbors([W_input.T[0:eigen_vector_used]], 2, return_distance=True)
        return self.objIdx_train[ind[0][0]]

    def recognise_img(self, imgInput):
        matchedCount = 0
        t0 = time.time()
        imagettes = self.slice_image_to_imagettes_rgb(imgInput)
        regResult = np.zeros((len(self.objName_train),))

        # if no meaningful imagette is found, simply return zero scores
        if imagettes.shape[0] == 0:
            return regResult
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(imagettes.shape[1]):
                futures.append(executor.submit(self.recognise_imagette, imagettes[:, i], self.imgMean_train,
                                               self.W_train, self.U_train, self.W_trainNNLearner))

            for future in concurrent.futures.as_completed(futures):
                regIdx = future.result()
                if regIdx != -1:
                    regResult[regIdx] += 1
        print("recognise_imagette Time:", time.time()-t0)
        return regResult

    def euclidean_distance(self, vector1, vector2):
        dist = [(a - b) ** 2 for a, b in zip(vector1, vector2)]
        return math.sqrt(sum(dist))

    def train(self, exportWeights=False):
        # Get the path for training images
        trainImgPath = self.trainImgPath
        # Reading all the images and storing into an array
        trainImgFiles = os.listdir(self.trainImgPath)
        eigenVectorUsed = self.eigenVectorUsed
        trainSet = None
        objIdx = []

        if len(trainImgFiles) == 0:
            print("ERROR: no training image present in", trainImgPath)
            exit(-1)

        for trainImgFile in trainImgFiles:

            imgTrain = self.read_and_trunc_img_rgb(f'{trainImgPath}{trainImgFile}')

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

        print("W:",W.shape, "U:", U.shape)

        # Train the Nearest Neighbour Learner with the weights of the imagettes
        nnLearner = NearestNeighbors(algorithm='auto')
        nnLearner.fit(W.T, objIdx)
        self.W_trainNNLearner = nnLearner

        # Save the weights to file if the flag exportWeights is turned on
        if exportWeights:
            dfResult = pd.DataFrame(W.T)
            dfResult['class'] = self.objIdx_train
            dfResult.to_csv("weights.csv")

def main():
    objRecogniser = ObjRecogniser()
    objRecogniser.train()


if __name__ == "__main__":
    exit(main())
