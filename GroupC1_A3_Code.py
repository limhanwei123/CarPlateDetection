"""
This Python program consists of the training code for the classification model, and the code for classifying
the license plate. The outline is as follows:

1. Training of Model 
- input: file path for TrainingData, consists of 200 images
- output: accuracy score for validation, i.e., testing of 40 images splitted from TrainingData to test model

2. Classifying license plate
- input: file path for TestingData, consists of 10 license plate image
- output: accuracy score for classification
"""

# import libraries and initialize variables
import cv2
import numpy as np
import os 
GLOBAL_ERROR = 1
np.random.seed(31108504)

##############################################################################################################
# TRAINING OF MODEL
##############################################################################################################

def Weight_Initialization():
    """
    Initializing of the Weights.
    Random float number between -0.5 to 0.5.
    """
    wji= np.random.uniform(-0.5, 0.5, size=(128, 2450))
    wkj = np.random.uniform(-0.5, 0.5, size=(20, 128))
    bias_j = np.random.uniform(0, 1, size=(128, 1))
    bias_k = np.random.uniform(0, 1, size=(20, 1))

    return wji, wkj, bias_j, bias_k

def Read_Files(path):
    """
    Input: path for TrainingData
    Output: train_X, test_X, train_y, test_y
    This function reads the images, performs a series of transformation, and split the dataset into train 
    and test 
    """
    input_layer = []
    count, index = 0, 0
    one_hot = np.zeros((200,20))
    
    # read all images, perform transformation, and append into a list 
    for i in range(200):
        
        # read images and resize 
        file = path + "IMG_" + str(i+1) + ".png"
        image = cv2.imread(file)
        image = cv2.resize(image, (35,70))

        # convert to greyscale then binary images 
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # perform erosion 
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=2)
        thresh = thresh.flatten()
        
        # append transformed image into input layer         
        input_layer.append(thresh)

        # get the target class of the image 
        if count == 10:
            count = 0
            index += 1
        one_hot[i, index] += 1
        count += 1

    # convert to numpy array, and clip the values to (0, 1)
    input_layer = np.array(input_layer)
    input_layer = input_layer/255    

    # split into train and test set using 80-20 ratio 
    train_X, test_X, train_y, test_y = [], [], [], []
    count = 0
    for i in range(len(input_layer)):

        # reset index as this is image for next class 
        if count == 10:
            count = 0

        # use first 8 images for train
        if count < 8:
            train_X.append(input_layer[i])
            train_y.append(one_hot[i])
        
        # use last 2 images fr test 
        else:
            test_X.append(input_layer[i])
            test_y.append(one_hot[i])
        count += 1

    # convert list into numpy array 
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    return train_X, train_y, test_X, test_y

def Forward_Input_Hidden(input, wj, bj):
    """
    Forward Propagate input to hidden layer
    """
    net = wj.dot(np.transpose(input)) + bj
    net = net.astype('float64')
    out = 1 / (1 + np.exp(-net))
    return out 

def Forward_Hidden_Output(input, wk, bk):
    """
    Forward Propage hidden to output 
    """
    net = wk.dot(input) + bk
    net = net.astype('float64')
    out = 1 / (1 + np.exp(-net))
    return np.transpose(out) 

def Weight_Bias_Correction_Output(outk, outj, target):
    """
    Backward propagate output to hidden
    """
    delta1 = outk - target
    delta2 = outk * (1 - outk)
    delta3 = outj
    delta_WK = np.dot(np.transpose(delta1*delta2), np.transpose(delta3))
    delta_biasK = np.sum(np.transpose((delta1*delta2)), axis=1)  
    delta_biasK = np.expand_dims(delta_biasK, axis=1)
    return  delta_WK, delta_biasK    

def Weight_Bias_Correction_Hidden(x, outj, outk, target, wk):
    """
    Backward propagate hidden to input 
    """
    delta1 = np.transpose(x)
    delta2 = outj * (1 - outj)
    delta3 = (outk - target) * (outk * (1-outk))
    delta4 = wk
    temp = delta3.dot(delta4)
    temp = np.transpose(delta2) * temp
    delta_WJ = np.transpose(delta1.dot(temp))
    delta_biasJ = np.sum(np.transpose(temp), axis = 1)
    delta_biasJ = np.expand_dims(delta_biasJ, axis=1)
    return delta_WJ, delta_biasJ     

def Error_Correction(target, outk):
    """
    Calculate the error 
    """
    E = np.square(np.subtract(target, outk))
    total_error = sum(E) * 1/2
    total_error = np.expand_dims(total_error, axis=1)
    return sum(total_error)

def Weight_Bias_Update(wj, wk, bj, bk, dwj, dwk, dbj, dbk, lr):
    """
    Update all the paramaters with the learning rate 
    """
    nwj = wj - dwj * lr
    nwk = wk - dwk * lr
    nbj = bj - dbj * lr
    nbk = bk - dbk * lr

    return nwj, nwk, nbj, nbk

def Check_for_End(G_Error, error):
    """
    Check if training continues 
    """
    if error < G_Error:
        return True
    else:
        return False

def Saving_Weights_Bias(wji, wkj, bias_j, bias_k):
    """
    Save model parameter
    """
    return [wji, wkj, bias_j, bias_k]

def train_model(train_X, train_y, n_times, learning_rate):
    """
    Train the model based on error, however, to avoid model from training too late, we also limit by number
    of iterations. 
    The output is the saved model parameters 
    """
    
    # randomly assign values to paramaters 
    wji, wkj, bias_j, bias_k = Weight_Initialization()

    print("Model is training ...")
    # start training, based on either number of iteration or check for error 
    for _ in range(n_times): 

        # forward propagation 
        first_layer = Forward_Input_Hidden(train_X, wji, bias_j)
        output_layer = Forward_Hidden_Output(first_layer, wkj, bias_k)
        error = Error_Correction(train_y, output_layer)

        # check if continue training, only break if error is less than global error 
        if Check_for_End(GLOBAL_ERROR, error): 
            #return Saving_Weights_Bias(wji, wkj, bias_j, bias_k)
            break
        
        # backward propagation and update parameters 
        new_wk, new_bk = Weight_Bias_Correction_Output(output_layer, first_layer, train_y)
        new_wj, new_bj = Weight_Bias_Correction_Hidden(train_X, first_layer, output_layer, train_y, wkj)
        wji, wkj, bias_j, bias_k = Weight_Bias_Update(wji, wkj, bias_j, bias_k, new_wj, new_wk, new_bj, new_bk, learning_rate)
 
    
    print("Testing train_y")
    output_dic = {}
    for i in range(output_layer.shape[0]):
        true = train_y[i, :]
        true_idx = np.argmax(true)
        value = round(output_layer[i, true_idx], 2)
        if value not in output_dic.keys():
            output_dic[value] = 1
        else:
            output_dic[value] += 1

    myKeys = list(output_dic.keys())    
    myKeys.sort()
    output = {i: output_dic[i] for i in myKeys}
    print("Probability of target output is ")
    print(output)
    
    # in the case where model error is still greater than global error, forcefully stop based on iteration and return
    return Saving_Weights_Bias(wji, wkj, bias_j, bias_k)

def validating_model(valid_X, valid_y, param):
    """
    Test the model by reporting accurracy using test dataset 
    """
    # assignment paramater value based on the trained values 
    wji, wkj, bias_j, bias_k = param[0], param[1], param[2], param[3]

    print("\nTesting model ... ")
    # perform classification 
    first_layer = Forward_Input_Hidden(valid_X, wji, bias_j)
    output_layer = Forward_Hidden_Output(first_layer, wkj, bias_k)

    # calculate accuracy 
    total = len(output_layer)
    correct = 0
    for i in range(total):
        for j in range(len(output_layer[i])):
            if valid_y[i,j] == 1 and output_layer[i,j] == max(output_layer[i]):
                correct += 1            

    acc = correct/total
    return acc

##############################################################################################################
# CLASSIFICATION OF LICENSE PLATE
##############################################################################################################
def preprocess(path):
    """
    Reads the testing images and preprocess the images before segmentation 
    """
    
    # read image and resize 
    image = cv2.imread(path)
    image = cv2.resize(image, (300,60))

    # convert to greyscale and then binary image 
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # perform erosion
    kernel = np.ones((2, 2), np.uint8)
    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    
    return img_erosion

def segmentation(img):
    """
    Code for perform segmentation using vertical projection 
    """
    vpp = np.sum(img, axis=0)
    w_s = []
    w_e = []

    for i in range(2, len(vpp)-2):
        if vpp[i] < 550 and vpp[i+1] >= 550:
            w_s.append(i)
        if vpp[i] >= 550 and vpp[i+1] < 550:
            w_e.append(i)

    roi = []
    for i in range(len(w_s)):
        roi.append(img[:, w_s[i]:w_e[i]])
    
    return roi

def output_img(filepath, addpath):
    """
    Read license plate images and perform segmentation, save the segmented character into SegmentData folder
    """
    
    # read image from file and segment into characters 
    for idx, files in enumerate(os.listdir(filepath)):
        files_path = filepath + files
        img = preprocess(files_path)
        roi = segmentation(img)

        for idx2, r in enumerate(roi):
            outpath = str(idx) + str(idx2) + ".jpg"
            cv2.imwrite(os.path.join(addpath, outpath), r)

def predict_train(filepath):
    """
    Read segmented character and preprocess it to be ready for classification
    """
    input_layer = []

    for files in os.listdir(filepath):

        # read file 
        files_path = filepath + "/" + files
        image = cv2.imread(files_path)
        
        # trim the top and bottom pixels 
        image = image[3:56,]
        image = cv2.resize(image, (29, 70))
        
        # convert to greyscale and then binary image 
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # add in pixels to the left and right of image 
        new_array = np.zeros((70,35))
        for i in range(thresh.shape[0]):
            for j in range(thresh.shape[1]):
                new_array[i, j+3] = thresh[i,j]

        # remove noise 
        new_array = cv2.GaussianBlur(new_array,(7,7),0)
        
        # perform erosion 
        kernel = np.ones((3, 3), np.uint8)
        new_array = cv2.erode(new_array, kernel, iterations=1)

        # flatten and add to input layer 
        thresh = new_array.flatten()
        input_layer.append(thresh)

    # convert to numpy and clip to (0, 1)
    input_layer = np.array(input_layer)
    input_layer = input_layer/255
    test_X = np.array(input_layer)

    return test_X

def predict_test(target):
    """
    This is to prepare test_y for comparison
    """
    
    # a char to index dictionary to convert the character to their corresponding index 
    convert = {
        '0' : 0,
        '1' : 1,
        '2' : 2,
        '3' : 3,
        '4' : 4,
        '5' : 5,
        '6' : 6,
        '7' : 7,
        '8' : 8,
        '9' : 9,
        'B' : 10,
        'F' : 11,
        'L' : 12,
        'M' : 13,
        'P' : 14,
        'Q' : 15,
        'T' : 16,
        'U' : 17,
        'V' : 18,
        'W' : 19
    }

    test_y = []
    
    # read the character and convert to corresponding onehot array 
    for plates in target:
        for char in plates:
            onehot = np.zeros(20)
            idx = convert[char]
            onehot[idx] = 1  
            test_y.append(onehot)
    
    test_y = np.array(test_y)

    return test_y

def prediction(test_X, test_y, param):
    """
    Test the model by reporting accurracy using test dataset 
    """
    # assignment paramater value based on the trained values 
    wji, wkj, bias_j, bias_k = param[0], param[1], param[2], param[3]

    print("\nTesting license plate ... ")
    # perform classification 
    first_layer = Forward_Input_Hidden(test_X, wji, bias_j)
    output_layer = Forward_Hidden_Output(first_layer, wkj, bias_k)

    total = len(output_layer)
    correct = 0

    # calculate accuracy 
    predicted = []
    for i in range(total):
        for j in range(len(output_layer[i])):
            if output_layer[i,j] == max(output_layer[i]):
                predicted.append(j)
            if test_y[i,j] == 1 and output_layer[i,j] == max(output_layer[i]):
                correct += 1

    acc = correct/total
    return acc, predicted

def convert(num):
    # a dictionary for idx to character 
    ls = {
        0 : "0",
        1 : "1",
        2 : "2",
        3 : "3",
        4 : "4",
        5 : "5",
        6 : "6",
        7 : "7",
        8 : "8",
        9 : "9",
        10 : "B",
        11 : "F",
        12 : "L",
        13 : "M",
        14 : "P",
        15 : "Q",
        16 : "T",
        17 : "U",
        18 : "V",
        19 : "W"
    }

    return ls[num]


if __name__ == "__main__":
    
    train_path = "./TrainingData/"
    n = 700
    learning_rate = 0.01

    # perform training of model 
    train_X, train_y, valid_X, valid_y = Read_Files(train_path)
    param = train_model(train_X, train_y, n, learning_rate)

    # perform validation of model
    accuracy = validating_model(valid_X, valid_y, param)
    print("Validation accuracy is {}".format(round(accuracy, 2)))

    # segment license plate to corresponding characters
    output_img("./TestingData/", "./SegmentData")

    # prepare test_X and test_y to classify license plate
    plates = ["VBU3878", "WUM207", "VBT2597", "WTF6868", "PLW7969", "BPU9859", "BMT8628", "BMB8262", "PPV7422", "BQP8189"]
    test_y = predict_test(plates)
    test_X = predict_train("./SegmentData")

    # perform testing of license plate 
    accuracy,pred = prediction(test_X, test_y, param)    
    print("Testing accuracy is {}".format(round(accuracy, 2)))
    index = 0
    final = []
    for plate in plates:
        car_plate = ""
        for i in range(len(plate)):
            alpha = convert(pred[index])
            car_plate += str(alpha)
            index += 1
        final.append(car_plate)
    print("Predicted license plate is {}".format(final))




