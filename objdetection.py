import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ObjDetection:

    def __init__(self):
        self.W_train = []
        self.U_train = []
        self.imgMean_train = []
        self.objName_train = []
        self.imagette_grid_size = 20
        self.imagette_step_size = 10
        self.imgDistThreshold = 6000
        self.weightDistThreshold = 500


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
        #print(imgInput.shape, train_row, train_col)
        for row in range(train_row):
            for col in range(train_col):
                #grid_img = imgInput[row * grid_size:row * grid_size + grid_size, col * grid_size:col * grid_size + grid_size]
                grid_img = imgInput[row*step_size:row*step_size+grid_size, col*step_size:col*step_size+grid_size, :]
                #print(row*step_size, row*step_size+grid_size, col*step_size, col*step_size+grid_size)
                #plt.imshow(grid_img, 'gray')
                #plt.show()
                #train_set[:,row*train_row+col] = grid_img.reshape(grid_size*grid_size)
                train_set.append(grid_img.flatten())

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

        eigen_vector_used = 10

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

    def recognise_imagette(self, imgInput, imgMean, W, U):

        imgDistThreshold = self.imgDistThreshold
        weightDistThreshold = self.weightDistThreshold

        X_input = imgInput - imgMean
        W_input = U.T @ X_input
        imgRecon = U @ W_input
        imgRecon = imgRecon + imgMean

        # Compute the Euclidean distance of two matrices using matrix norm
        dist = np.linalg.norm(imgInput - imgRecon)

        if dist > imgDistThreshold:
            print("Reconstructed Image has too large distance from original:", dist)
            return False
     #   print("Image Distance: ", dist)

        matchedCount = 0
        # Compute Euclidean distance of the input image weight against all others
        # e_k = || W_k - W_r ||
        for i in range(W.shape[1]):
            e_k = np.linalg.norm(W_input - W[:, i])
            if e_k < weightDistThreshold:
     #           print(f'{i}: Euclidean Distance of Weights = {e_k}')
                matchedCount += 1
                # exit the FOR loop if any matching found
                break

        if matchedCount > 0:
            return True
        else:
            return False

    def recognise_imagette2(self, imgInput, index):

        imgDistThreshold = 10000
        weightDistThreshold = 100

        #print("imgInput:", imgInput.shape)
        #print("imgMean:", self.imgMean_train)
        X_input = imgInput - self.imgMean_train[index]
        W_input = self.U_train[index].T @ X_input
        imgRecon = self.U_train[index] @ W_input
        imgRecon = imgRecon + self.imgMean_train[index]

        # Compute the Euclidean distance of two matrices using matrix norm
        dist = np.linalg.norm(imgInput - imgRecon)

        #if dist > imgDistThreshold:
        #    print("Reconstructed Image has too large distance from original:", dist)
     #   print("Image Distance: ", dist)

        matchedCount = 0
        # Compute Euclidean distance of the input image weight against all others
        # e_k = || W_k - W_r ||
        for i in range(self.W_train[index].shape[1]):
            e_k = np.linalg.norm(W_input - self.W_train[index][:, i])
            if e_k < weightDistThreshold:
     #           print(f'{i}: Euclidean Distance of Weights = {e_k}')
                matchedCount += 1

        if matchedCount > 0:
            return True
        else:
            return False


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
        imagettes = self.slice_image_to_imagettes_rgb(imgInput)

        for i in range(imagettes.shape[1]):
            #print(f'Recognising # {i}...')
            if self.recognise_imagette(imagettes[:,i], self.imgMean_train[index], self.W_train[index], self.U_train[index]):
                matchedCount += 1

        return matchedCount, index
        #print(f'{matchedCount} feature(s) matched.')

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

    def test_run_rgb(self):
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

        testIdx = 2
        imgTest = self.read_and_trunc_img_rgb(f'{trainPath}/{trainImgFiles[testIdx]}')
        print("Test Object:", self.objName_train[testIdx])

        for i in range(len(self.objName_train)):
            print("Recognising", self.objName_train[i],'...')
            matchedCount, index = self.recognise_rgb(imgTest, i)
            print("Matched count = ", matchedCount)



    #if __name__ == "__main__":
    #    exit(main())
