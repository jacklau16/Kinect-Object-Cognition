import concurrent
import math
import time
import pandas as pd

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

import os

class ObjDetection2:

    def __init__(self):
        self.W_train = []
        self.U_train = []
        self.imgMean_train = []
        self.objIdx_train = []
        self.W_trainNNLearner = []
        self.imagette_grid_size = 30
        self.imagette_step_size = 15
        self.imgDistThreshold = 6000
        self.weightDistThreshold = 500
        self.nnDistThreshold = 500
        self.eigen_vector_used = 20
        self.resizeImage = True
        self.equaliseHistogram = True

        trainLabelFile = './Set1Labels.txt'
        testLabelFile = './Set2Labels.txt'

        labelFile = open('Set1Labels.txt', 'r')
        self.objName_train = labelFile.read().splitlines()

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

        eigen_vector_used = self.eigen_vector_used

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
        dist = np.linalg.norm(imgInput - imgRecon)
        #dist = self.euclidean_distance(imgInput, imgRecon)

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
        eigen_vector_used = self.eigen_vector_used

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

        # Discard if 1st matched item is too close to 2nd matched item => not a good match
        #if (dist[0][1]-dist[0][0]) > 10:
        if (dist[0][1] * 0.7) > dist[0][0]:
            return -1
        #print("NN distance:",dist)
        #print("ind:",ind[0][0])
        return self.objIdx_train[ind[0][0]]
        #if dist[0][0] < nnDistThreshold:

        # Method 1: predict output by K-nearest neighbour learner
        return nnLearner.predict([W_input.T[0:eigen_vector_used]])[0]

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

        # if no meaningful imagette is found, simply return 0s
        if imagettes.shape[0] == 0:
            return regResult

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
            #self.objName_train.append(objName)
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

    def train_all(self, testRecognition=False, exportWeights=False):
        # Giving path for training images
        trainPath = './train_rgb_hist/'
        # Readling all the images and storing into an array
        trainImgFiles = os.listdir(trainPath)
        eigen_vector_used = self.eigen_vector_used
        train_set = None
        objIdx = []

        for trainImgFile in trainImgFiles:
            objName = os.path.splitext(trainImgFile)[0]
            #W, U, imgMean = self.train_rgb(f'{trainPath}/{trainImgFile}', objName)

            #self.objName_train.append(objName)
            imgTrain = self.read_and_trunc_img_rgb(f'{trainPath}{trainImgFile}')

            if imgTrain is not None:
                classID = int(trainImgFile.split('-')[0])-1
                if self.resizeImage:
                    imgTrain = cv2.resize(imgTrain, (200, 200), interpolation=cv2.INTER_AREA)

                new_train_set = self.slice_image_to_imagettes_rgb(imgTrain)
                #print('New Train Set:',new_train_set.shape)
                if train_set is None:
                    train_set = new_train_set
                else:
                    train_set = np.append(train_set, new_train_set, axis=1)
                #print("Training Set:", train_set.shape)
                objIdx.extend([classID] * new_train_set.shape[1])

        #train_set = np.asarray(train_set)
        #print("objIdx:", objIdx)
        print("No. of training samples:",len(train_set[0]))

        print("Normalise the training samples...")
        # Compute average image
        imgMean = np.mean(train_set, axis=1)
        #print("imgMean:", imgMean.shape)
        # Normalise images by subtracting with their mean
        X = train_set - np.tile(imgMean, (train_set.shape[1], 1)).T

        # Perform SVD and obtain eigenvectors
        print("Performing SVD...")
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
        #nnLearner = KNeighborsClassifier(n_neighbors=5)
        #nnLearner = tree.DecisionTreeClassifier()
        nnLearner.fit(W.T, objIdx)
        self.W_trainNNLearner = nnLearner
        if exportWeights:
            dfResult = pd.DataFrame(W.T)
            dfResult['class'] = self.objIdx_train
            dfResult.to_csv("weights.csv")

        if testRecognition:
            # Giving path for training images
            testPath = './train_rgb/'
            # Readling all the images and storing into an array
            testImgFiles = os.listdir(testPath)
            for testIdx, testImgFile in zip(range(len(testImgFiles)), testImgFiles):
                imgTest = self.read_and_trunc_img_rgb(f'{testPath}/{testImgFile}')
                if imgTest is None:
                    continue
                if self.equaliseHistogram:
                    imgTest[:, :, 0] = cv2.equalizeHist(imgTest[:, :, 0])
                    imgTest[:, :, 1] = cv2.equalizeHist(imgTest[:, :, 1])
                    imgTest[:, :, 2] = cv2.equalizeHist(imgTest[:, :, 2])
                if self.resizeImage:
                    imgTest = cv2.resize(imgTest, (200, 200), interpolation=cv2.INTER_AREA)
                print("Test Object:", testImgFile)
                #print("Recognising", self.objName_train[i], '...')
                scores = self.recognise_rgb2(imgTest)
                max_idx = scores.argmax()
                print("Recognition Result:", max_idx, max(scores))
            #    matchedCount, index = self.recognise_rgb(imgTest, i)

    def load_hist(self, saveWeight=True):
        trainPath = './train_rgb/'
        # Readling all the images and storing into an array
        trainImgFiles = os.listdir(trainPath)
        eigen_vector_used = self.eigen_vector_used
        train_set = None
        objIdx = []

        for trainImgFile in trainImgFiles:
            objName = os.path.splitext(trainImgFile)[0]

            # self.objName_train.append(objName)
            imgTrain = self.read_and_trunc_img_rgb(f'{trainPath}/{trainImgFile}')

            if imgTrain is not None:
                classID = int(trainImgFile.split('-')[0]) - 1
                if self.resizeImage:
                    imgTrain = cv2.resize(imgTrain, (200, 200), interpolation=cv2.INTER_AREA)

                new_train_set = self.slice_image_to_imagettes_hist(imgTrain)
                print('New Train Set:',new_train_set.shape)
                if train_set is None:
                    train_set = new_train_set
                else:
                    train_set = np.append(train_set, new_train_set, axis=1)
                print("Training Set:", train_set.shape)
                objIdx.extend([classID] * new_train_set.shape[1])

        print("No. of training samples:", len(train_set[0]))

        print("Normalise the training samples...")
        # Compute average image
        imgMean = np.mean(train_set, axis=1)
        #print("imgMean:", imgMean.shape)
        # Normalise images by subtracting with their mean
        X = train_set - np.tile(imgMean, (train_set.shape[1], 1)).T

        # Perform SVD and obtain eigenvectors
        print("Performing SVD...")
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)

        # Project the train set onto the first N principal components (eigenvectors)
        U_sub = U[:, 0:eigen_vector_used]

        W = U_sub.T @ X

        print("W:",W.shape)
        #print("Exporting weight")
        #dfWeights = pd.DataFrame(train_set[0,:])
        plt.figure(figsize=(10, 7))
        plt.scatter(W[0,:], W[1,:], c=objIdx, cmap='Spectral')
        plt.colorbar(boundaries=np.arange(-1, 15) - 0.5).set_ticks(np.arange(-1, 14))
        plt.show()
        #dfResult.columns = ['Frame#', 'True Label', 'Predicted Label', 'Score', 'Conf. Level']
        #dfResult.to_csv("result.csv")
        # Compute average image
        #imgMean = np.mean(train_set, axis=1)
        # print("imgMean:", imgMean.shape)
        # Normalise images by subtracting with their mean
        #X = train_set - np.tile(imgMean, (train_set.shape[1], 1)).T

    def slice_image_to_imagettes_hist(self, imgInput):
        grid_size = self.imagette_grid_size
        step_size = self.imagette_step_size
        train_set = []
        train_row = int((imgInput.shape[0] - grid_size) / step_size) + 1
        train_col = int((imgInput.shape[1] - grid_size) / step_size) + 1
        for row in range(train_row):
            for col in range(train_col):
                grid_img = imgInput[row*step_size:row*step_size+grid_size, col*step_size:col*step_size+grid_size, :]
                # Detect if there is any edge in the imagette, discard it if non detected
                edges = cv2.Canny(grid_img, 100, 200)
                if np.max(edges)==255:
                    #train_set.append(grid_img.flatten())
                    hist = cv2.calcHist([grid_img], [0], None, [256], [0, 256])
                    #print("Hist shape:",hist.shape)
                    train_set.extend(hist.T)
        print(f"Input shape: {imgInput.shape}, {train_row}, {train_col}, output size: {len(train_set)}")

        return np.array(train_set).T


def main():
    objDet = ObjDetection2()
    objDet.load_hist()
    #return 0

    objDet.train_all(testRecognition=False, exportWeights=True)
    weights = pd.read_csv('weights.csv')
    plt.figure(figsize=(10, 7))
    plt.scatter(weights['0'], weights['1'], c=weights['class'], cmap='Spectral')
    plt.colorbar(boundaries=np.arange(-1,15)-0.5).set_ticks(np.arange(-1,14))
    plt.show()
if __name__ == "__main__":
    exit(main())
