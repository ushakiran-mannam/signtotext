# %%
"""
# Data collection
"""

# %%
import cv2
import os
import numpy as np 

'''################################# GLOBAL VARIABLES #################################'''

#all directories
og_path = 'self_made_data/image_train/'
test_path = 'self_made_data/image_test/'
processed_path = 'self_made_data/image_train_processed/'
processed_test_path = 'self_made_data/image_test_processed/'

# directory list 
dir_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','0']


# file count in each directory 

file_count = {
    '1' : 0,
    '2' : 0,
    '3' : 0,
    '4' : 0,
    '5' : 0,
    '6' : 0,
    '7' : 0,
    '8' : 0,
    '9' : 0,
    'A' : 0,
    'B' : 0,
    'C' : 0,
    'D' : 0,
    'E' : 0,
    'F' : 0,
    'G' : 0,
    'H' : 0,
    'I' : 0,
    'J' : 0,
    'K' : 0,
    'L' : 0,
    'M' : 0,
    'N' : 0,
    'O' : 0,
    'P' : 0,
    'Q' : 0,
    'R' : 0,
    'S' : 0,
    'T' : 0,
    'U' : 0,
    'V' : 0,
    'W' : 0,
    'X' : 0,
    'Y' : 0,
    'Z' : 0,
    '0':0
}

'''################################# FUNCTIONS #################################'''

#creating all folders in directory if doesnot exist       
def create_all_required_directories(path):

    for i in dir_list:
        if i in os.listdir(path):
            print('yes')
        else:
            mode = 0o666
            os.mkdir(path+i, mode)
            
            
#  processing image 
def processing(img):

    minValue = 70
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    return res

def mask_processing(roi):

    hsvim = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    
    # mask = processing(roi)
    masked = cv2.bitwise_and(roi, roi, mask=thresh)
    # cv2.imshow('mask',mask)
    # cv2.imshow('masked',masked)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',gray)

    return thresh,masked,gray

# counts files in all directories and store in file_count dictionary
def count_files_dir(arg):
    path = og_path+str(arg)+'\\'
    for files in os.listdir(path):
        file_count[arg] = file_count[arg] + 1
        

def give_contours(roi):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(roi, [contours], -1, (255,255,0), 2)
    cv2.imshow("contours", roi)

    hull = cv2.convexHull(contours)
    cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)
    cv2.imshow("hull", roi)  

# takes input such as entered key and roi of image and write the image in specific directory      
def show_info(k,roi):
    detail = " "
    for i in dir_list:
        if k%256 == ord(i):
            detail = '{}_{}.png'.format(i,file_count[i])
            cv2.imwrite(og_path+i+'\\'+detail,roi)
            file_count[i] = file_count[i]+1
            print(detail)
            break
            
    return i,detail


'''################################# MAIN #################################'''

        
# calling count_files_dir()
for i in dir_list:
    count_files_dir(i)

print(file_count)
    
    
# calling function for creating directories
create_all_required_directories(og_path)
create_all_required_directories(test_path)
create_all_required_directories(processed_path)
create_all_required_directories(processed_test_path)


cam=cv2.VideoCapture(0)
button_pressed='+'
upper_left=(0,100)
bottom_right=(250,350)




            


while True:
    ret,frame=cam.read()
    cv2.rectangle(img=frame,pt1=upper_left,pt2=bottom_right,color=(255,0,0),thickness=1)
    
    
    roi=frame[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

        
   
    if not ret:
        break
    
    k=cv2.waitKey(1)
    if k%256 ==27:
        print('Exiting the setup....')
        break

    button_pressed,show_text = show_info(k,roi)
    fonts=cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(img=frame,text=show_text,org=(10,370),fontFace=fonts,fontScale=1,color=(255,0,0),thickness=1)
    
    cv2.imshow('Webcam',frame)
    cv2.imshow('roi',roi)
        
    mask = processing(roi)
    thresh,masked,gray = mask_processing(roi)
        
    cv2.imshow('mask',mask)
    cv2.imshow('masked',masked)
    cv2.imshow('thresh',thresh)
    cv2.imshow('gray',gray)
    # cv2.imshow('mask2',mask2)
    try:
        give_contours(roi)
    except:
        pass



cam.release()
cv2.destroyAllWindows()

# %%
"""
# image processing
"""

# %%
import cv2
import os
import numpy as np 

'''################################# GLOBAL VARIABLES #################################'''

#all directories
og_path = 'self_made_data/image_train/'
test_path = 'self_made_data/image_test/'
processed_path = 'self_made_data/image_train_processed/'
processed_test_path = 'self_made_data/image_test_processed/'

# directory list 
dir_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','0']




#  processing image 
def processing(img):

    minValue = 70
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    return res

#  processing image 
def processing2(img):

    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    
    return thresh


for folders in os.listdir(og_path):
    print(folders)
    for images in os.listdir(og_path+'/'+folders):
        img = cv2.imread(og_path+'/'+folders+'/'+images)
        img = processing(img)
        cv2.imwrite('train'+'/'+folders+'/'+images,img)
        

# %%


# %%
"""
# training
"""

# %%


# %%
#importing useful libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten,MaxPooling2D
from keras.preprocessing import image
# from keras.preprocessing.image import img_to_array
from keras.utils import image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import cv2
from keras.utils import np_utils
from sklearn.utils import shuffle
import glob
import pandas as pd
import os
import csv
from keras.optimizers import Adam


# creating the model
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(64,64,3)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu"))
model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(27,activation="softmax"))
model.summary()


checkpoint = ModelCheckpoint("signs.h5", monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics = ['acc'])


train_data_dir = 'train'
batch_size = 32
nb_epochs = 7
img_height = 64
img_width = 64

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs)


# Plots of Training and validation accuracy/loss during training of model
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.figure()
plt.plot(epochs, acc, 'b', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('Accuracy.jpg')
plt.figure()
plt.plot(epochs, loss, 'b', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Loss.jpg')


dir_list = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

result = {}

count = 0
for i in dir_list:
    result[count] = i
    count = count +1
    
print(result)

# Showing image and label from training set
x,y = train_generator.next()
for i in range(0,1):
    image = x[i]
    plt.imshow(image)
    plt.title(result[np.argmax(y[i])])
    plt.show()
    
# serialize model to JSON
model_json = model.to_json()
with open("model_info/signs.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_info/signs.h5")
print("Saved model to disk")

# %%
"""
# testing
"""

# %%
# load json and create model
import numpy as np
from keras.models import model_from_json
import pandas as pd

json_file = open('model_info/signs.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
sign_classifier = model_from_json(loaded_model_json)
# load weights into new model
sign_classifier.load_weights("model_info/signs.h5")
print("Loaded model from disk")



#  processing image 
def processing(img):

    minValue = 70
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    return res

def prediction(img):
    dataset = []
#     img = cv2.imread('processed/B/0.jpg')
    img = cv2.resize(img,(64,64))
    img_array=image_utils.img_to_array(img)
    dataset.append(img_array)
    dataset_array=np.array(dataset)
    dataset_array=dataset_array/255
    y_pred = sign_classifier.predict(dataset_array)
    results_pred = y_pred.argmax(axis=1)
#     print(result_dic[results_pred[0]])
    return result[results_pred[0]]

dir_list = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

result = {}

count = 0
for i in dir_list:
    result[count] = i
    count = count +1
    
print(result)


vid = cv2.VideoCapture(0)
  
while(True):

    ret, frame = vid.read()
  
    
    
    frame = cv2.rectangle(frame,(0,0),(300,300),(0,255,0),2)
    
    roi = frame[0:300,0:300]
    mask = processing(roi)
    cv2.imwrite('test.jpg',mask)
    im = cv2.imread('test.jpg')
    pred = prediction(im)
    frame = cv2.putText(frame,pred,(400,400),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.imshow('mask',mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

# %%
