import concurrent
import math
import time
import timeit

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

class ObjDetection:

    def __init__(self):
        self.W_train = []
        self.U_train = []
        self.imgMean_train = []
        self.objName_train = []
        self.objIdx_train = []
        self.W_trainNNLearner = []
        self.imagette_grid_size = 20
        self.imagette_step_size = 10
        self.imgDistThreshold = 6000
        self.weightDistThreshold = 500
        self.nnDistThreshold = 500
        self.eigen_vector_used = 3

    def slice_image_to_imagettes(self, imgInput):
        grid_size = self.imagette_grid_size
        step_size = self.imagette_step_size
        train_set = []
        #train_row = int(imgInput.shape[0] / grid_size)
        #train_col = int(imgInput.shape[1] / grid_size)
        train_row = int((imgInput.shape[0] - grid_size) / step_size) + 1
        train_col = int((imgInput.shape[1] - grid_size) / step_size) + 1
        print(imgInput.shape, train_row, train_col)
        for row in range(train_row):
            for col in range(train_col):
                #grid_img = imgInput[row * grid_size:row * grid_size + grid_size, col * grid_size:col * grid_size + grid_size]
                grid_img = imgInput[row*step_size:row*step_size+grid_size, col*step_size:col*step_size+grid_size]
                #print(row*step_size, row*step_size+grid_size, col*step_size, col*step_size+grid_size)
                #plt.imshow(grid_img, 'gray')
                #plt.show()
                #train_set[:,row*train_row+col] = grid_img.reshape(grid_size*grid_size)
                train_set.append(grid_img.flatten())

        return np.array(train_set).T

    def slice_image_to_imagettes_rgb(self, imgInput):
        grid_size = self.imagette_grid_size
        step_size = self.imagette_step_size
        train_set = []
        #train_row = int(imgInput.shape[0] / grid_size)
        #train_col = int(imgInput.shape[1] / grid_size)
        train_row = int((imgInput.shape[0] - grid_size) / step_size) + 1
        train_col = int((imgInput.shape[1] - grid_size) / step_size) + 1
        for row in range(train_row):
            for col in range(train_col):
                #grid_img = imgInput[row * grid_size:row * grid_size + grid_size, col * grid_size:col * grid_size + grid_size]
                grid_img = imgInput[row*step_size:row*step_size+grid_size, col*step_size:col*step_size+grid_size, :]
                #print(row*step_size, row*step_size+grid_size, col*step_size, col*step_size+grid_size)
                # Detect if there is any edge in the imagette, discard it if non detected
                edges = cv2.Canny(grid_img, 100, 200)
                if np.max(edges)==255:
                    #print("EDGE DETECTED!")
                    #train_set[:,row*train_row+col] = grid_img.reshape(grid_size*grid_size)
                    train_set.append(grid_img.flatten())
        #print(f"Input shape: {imgInput.shape}, {train_row}, {train_col}, output size: {len(train_set)}")

        return np.array(train_set).T

    def read_and_trunc_img(self, imgFile):
        grid_size = self.imagette_grid_size
        thresVal = 75
        img = cv2.imread(imgFile, 0)
        # imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(img, thresVal, 255, cv2.THRESH_BINARY_INV)
        x, y, w, h = cv2.boundingRect(threshold)
        #plt.imshow(threshold[y:y + h, x:x + w], 'gray')
        #plt.show()
        if (h >= grid_size) & (w >= grid_size):
            imgTrain = threshold[y:y + h, x:x + w]
            #plt.imshow(imgTrain, 'gray')
            #plt.show()
            #print("h,w,grid_size=", h, w, grid_size)
            return imgTrain
        else:
            return None

    def read_and_trunc_img_rgb(self, imgFile):
        grid_size = self.imagette_grid_size
        thresVal = 75
        img = cv2.imread(imgFile)
        # imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret, threshold = cv2.threshold(img, thresVal, 255, cv2.THRESH_BINARY_INV)
        #x, y, w, h = cv2.boundingRect(threshold)
        #plt.imshow(threshold[y:y + h, x:x + w], 'gray')
        #plt.show()
        h = img.shape[0]
        w = img.shape[1]

        if (h >= grid_size) & (w >= grid_size):
            #imgTrain = threshold[y:y + h, x:x + w]
            #plt.imshow(imgTrain, 'gray')
            #plt.show()
            #print("h,w,grid_size=", h, w, grid_size)
            return img
        else:
            return None


    def train(self, imgFile, labelTrain):

        eigen_vector_used = self.eigen_vector_used

        imgTrain = self.read_and_trunc_img(imgFile)

        if imgTrain is not None:
            train_set = self.slice_image_to_imagettes(imgTrain)

            # Compute average image
            imgMean = np.mean(train_set, axis=1)

            # Normalise images by subtracting with their mean
            X = train_set - np.tile(imgMean, (train_set.shape[1], 1)).T

            # Perform SVD and obtain eigenvectors
            U, sigma, VT = np.linalg.svd(X, full_matrices=False)

            # Project the train set onto the first N principal components (eigenvectors)
            U_sub = U[:,0:eigen_vector_used]

            W = U_sub.T @ X
        #    W_trained = W
        else:
            print(labelTrain,"imgTrain is None")
            W = None
            U_sub = None
            imgMean = None

        return W, U_sub, imgMean

    def train_rgb(self, imgFile, labelTrain):

        eigen_vector_used = 10

        imgTrain = self.read_and_trunc_img_rgb(imgFile)

        if imgTrain is not None:
            train_set = self.slice_image_to_imagettes_rgb(imgTrain)

            # Compute average image
            imgMean = np.mean(train_set, axis=1)

            # Normalise images by subtracting with their mean
            X = train_set - np.tile(imgMean, (train_set.shape[1], 1)).T

            # Perform SVD and obtain eigenvectors
            U, sigma, VT = np.linalg.svd(X, full_matrices=False)

            # Project the train set onto the first N principal components (eigenvectors)
            U_sub = U[:,0:eigen_vector_used]

            W = U_sub.T @ X
        #    W_trained = W
        else:
            print(labelTrain,"imgTrain is None")
            W = None
            U_sub = None
            imgMean = None

        return W, U_sub, imgMean

    def recognise_imagette(self, imgInput, imgMean, W, U, nnLearner):

        imgDistThreshold = self.imgDistThreshold
        weightDistThreshold = self.weightDistThreshold
        nnDistThreshold = self.nnDistThreshold
        X_input = imgInput - imgMean
        W_input = U.T @ X_input
        imgRecon = U @ W_input
        imgRecon = imgRecon + imgMean
        # Compute the Euclidean distance of two matrices using matrix norm
        #dist = np.linalg.norm(imgInput - imgRecon)
        dist = self.euclidean_distance(imgInput, imgRecon)

        if dist > imgDistThreshold:
            print("Reconstructed Image has too large distance from original:", dist)
            return False
     #   print("Image Distance: ", dist)

        matchedCount = 0

        # Method 1: search nearest neighbour from nnLearner
        #print(W_input.T.shape, W.T.shape)
        dist, ind = nnLearner.kneighbors([W_input.T], 1, return_distance=True)
        #print("nearest neighbour, dist =",dist[0][0])
        if dist[0][0] < nnDistThreshold:
            return True
        else:
            return False

        # Compute Euclidean distance of the input image weight against all others
        # e_k = || W_k - W_r ||
        #t0 = time.time()
        for i in range(W.shape[1]):
            #e_k = np.linalg.norm(W_input - W[:, i])
            e_k = self.euclidean_distance(W_input, W[:, i])
            if e_k < weightDistThreshold:
     #           print(f'{i}: Euclidean Distance of Weights = {e_k}')
                matchedCount += 1
                # exit the FOR loop if any matching found
                break
        #print("Matrix operation time:", time.time()-t0)
        if matchedCount > 0:
            return True
        else:
            return False

    def recognise_imagette2(self, imgInput, imgMean, W, U, nnLearner):

        imgDistThreshold = self.imgDistThreshold
        weightDistThreshold = self.weightDistThreshold
        nnDistThreshold = self.nnDistThreshold
        X_input = imgInput - imgMean
        W_input = U.T @ X_input
        imgRecon = U @ W_input
        imgRecon = imgRecon + imgMean
        # Compute the Euclidean distance of two matrices using matrix norm
        # dist = np.linalg.norm(imgInput - imgRecon)
        dist = self.euclidean_distance(imgInput, imgRecon)

        if dist > imgDistThreshold:
            print("Reconstructed Image has too large distance from original:", dist)
            return -1


        # Method 1: search nearest neighbour from nnLearner
        #print("nnLearner.kneighbors:",W_input.T.shape, W.T.shape)
        dist, ind = nnLearner.kneighbors([W_input.T[0:10]], 1, return_distance=True)
        # print("nearest neighbour, dist =",dist[0][0])
        if dist[0][0] < nnDistThreshold:
            #print("Matched:", self.objName_train[ind[0][0]])
            return self.objIdx_train[ind[0][0]]
        else:
            return -1


    def recognise(self, imgInput, index):
        matchedCount = 0
        imagettes = self.slice_image_to_imagettes(imgInput)

        for i in range(imagettes.shape[1]):
            #print(f'Recognising # {i}...')
            if self.recognise_imagette(imagettes[:,i], self.imgMean_train[index], self.W_train[index], self.U_train[index]):
                matchedCount += 1

        return matchedCount, index
        #print(f'{matchedCount} feature(s) matched.')

    def recognise_rgb(self, imgInput, index):
        matchedCount = 0
        t0 = time.time()
        imagettes = self.slice_image_to_imagettes_rgb(imgInput)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(imagettes.shape[1]):
                futures.append(executor.submit(self.recognise_imagette, imagettes[:,i], self.imgMean_train[index],
                                               self.W_train[index], self.U_train[index], self.W_trainNNLearner[index]))

            for future in concurrent.futures.as_completed(futures):
                # print("Thread:", future.result()[1], future.result()[0])
                if future.result():
                    matchedCount += 1

        #for i in range(imagettes.shape[1]):
            #print(f'Recognising # {i}...')
        #    if self.recognise_imagette(imagettes[:,i], self.imgMean_train[index], self.W_train[index], self.U_train[index], self.W_trainNNLearner[index]):
        #        matchedCount += 1
        #print(f'Recognising #{i}: operation time: {time.time() - t0}')
        return matchedCount, index
        #print(f'{matchedCount} feature(s) matched.')

    def recognise_rgb2(self, imgInput):
        matchedCount = 0
        t0 = time.time()
        imagettes = self.slice_image_to_imagettes_rgb(imgInput)

        regResult = np.zeros((len(self.objName_train),))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(imagettes.shape[1]):
                futures.append(executor.submit(self.recognise_imagette2, imagettes[:,i], self.imgMean_train,
                                               self.W_train, self.U_train, self.W_trainNNLearner))

            for future in concurrent.futures.as_completed(futures):
                regIdx = future.result()
                if regIdx != -1:
                    regResult[regIdx] += 1
                #print("Thread:", future.result())
                #if future.result():
                #matchedCount += 1
        return regResult

    def euclidean_distance(self, vector1, vector2):
        dist = [(a - b) ** 2 for a, b in zip(vector1, vector2)]
        return math.sqrt(sum(dist))

    def test_run(self):
        # Giving path for training images
        trainPath = './train/'
        # Readling all the images and storing into an array
        trainImgFiles = os.listdir(trainPath)

        for trainImgFile in trainImgFiles:
            objName = os.path.splitext(trainImgFile)[0]
            W, U, imgMean = self.train(f'{trainPath}/{trainImgFile}', objName)
            self.W_train.append(W)
            self.U_train.append(U)
            self.imgMean_train.append(imgMean)
            self.objName_train.append(objName)
        testIdx = 0
        imgTest = self.read_and_trunc_img(f'{trainPath}/{trainImgFiles[testIdx]}')
        print("Test Object:", self.objName_train[testIdx])

        for i in range(len(self.objName_train)):
            print("Recognising", self.objName_train[i],'...')
            self.recognise(imgTest, i)

    def test_run_rgb(self, testRecognition=False):
        # Giving path for training images
        trainPath = './train_rgb/'
        # Readling all the images and storing into an array
        trainImgFiles = os.listdir(trainPath)

        for trainImgFile in trainImgFiles:
            objName = os.path.splitext(trainImgFile)[0]
            W, U, imgMean = self.train_rgb(f'{trainPath}/{trainImgFile}', objName)
            self.W_train.append(W)
            self.U_train.append(U)
            self.imgMean_train.append(imgMean)
            self.objName_train.append(objName)
            # Store weights into Nearest Neighbour Leaner
            #nnLearner = NearestNeighbors(algorithm='ball_tree', n_jobs=-1)
            nnLearner = NearestNeighbors(algorithm='auto')
            nnLearner.fit(W.T)
            self.W_trainNNLearner.append(nnLearner)
        print(self.objName_train)

        if testRecognition:
            testIdx = 2
            imgTest = self.read_and_trunc_img_rgb(f'{trainPath}/{trainImgFiles[testIdx]}')
            print("Test Object:", self.objName_train[testIdx])

            for i in range(len(self.objName_train)):
                print("Recognising", self.objName_train[i],'...')
                matchedCount, index = self.recognise_rgb(imgTest, i)
                print("Matched count = ", matchedCount)

    def train_all(self, testRecognition=False):
        # Giving path for training images
        trainPath = './train_rgb/'
        # Readling all the images and storing into an array
        trainImgFiles = os.listdir(trainPath)
        eigen_vector_used = self.eigen_vector_used
        train_set = None
        objIdx = []

        for i, trainImgFile in zip(range(len(trainImgFiles)), trainImgFiles):
            objName = os.path.splitext(trainImgFile)[0]
            #W, U, imgMean = self.train_rgb(f'{trainPath}/{trainImgFile}', objName)

            eigen_vector_used = 10
            self.objName_train.append(objName)
            imgTrain = self.read_and_trunc_img_rgb(f'{trainPath}/{trainImgFile}')

            if imgTrain is not None:
                new_train_set = self.slice_image_to_imagettes_rgb(imgTrain)
                print('New Train Set:',new_train_set.shape)
                if train_set is None:
                    train_set = new_train_set
                else:
                    train_set = np.append(train_set, new_train_set, axis=1)
                print("Training Set:", train_set.shape)
                objIdx.extend([i] * new_train_set.shape[1])

        #train_set = np.asarray(train_set)
        print("objIdx:", objIdx)
        print("TrainSet[0]:",len(train_set[0]))

        # Compute average image
        imgMean = np.mean(train_set, axis=1)
        print("imgMean:", imgMean.shape)
        # Normalise images by subtracting with their mean
        X = train_set - np.tile(imgMean, (train_set.shape[1], 1)).T

        # Perform SVD and obtain eigenvectors
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)

        # Project the train set onto the first N principal components (eigenvectors)
        U_sub = U[:, 0:eigen_vector_used]

        W = U_sub.T @ X

        self.W_train = W
        self.U_train = U
        self.imgMean_train = imgMean
        self.objIdx_train = objIdx

        print("W:",W.shape, "W.T:", W.T.shape, "U:", U.shape)
        # Store weights into Nearest Neighbour Leaner
        # nnLearner = NearestNeighbors(algorithm='ball_tree', n_jobs=-1)
        nnLearner = NearestNeighbors(algorithm='auto')
        nnLearner.fit(W.T)
        self.W_trainNNLearner = nnLearner

        if testRecognition:
            testIdx = 2
            imgTest = self.read_and_trunc_img_rgb(f'{trainPath}/{trainImgFiles[testIdx]}')
            print("Test Object:", self.objName_train[testIdx])

            #print("Recognising", self.objName_train[i], '...')
            regResult = self.recognise_rgb2(imgTest)
            print("Recognition Result:", regResult)
            #    matchedCount, index = self.recognise_rgb(imgTest, i)


def main():
    objDet = ObjDetection()
    objDet.train_all(testRecognition=True)


if __name__ == "__main__":
    exit(main())
