
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras import backend as K  # needed for mixing TensorFlow and Keras commands
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
sess = tf.Session(config=config)
K.set_session(sess)


# In[2]:


import tensorflow as tf
import numpy as np
import cv2
import time
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import losses


# ## The code above confirmed frames are real images

# In[3]:


total_data_length = 20520
each_length = np.array([4187, 2267, 4570, 5036, 4460])
frameWidth = 240
frameHeight = 240

frames = np.empty((total_data_length, frameWidth,
                   frameHeight, 3), np.dtype('uint8'))
print(frames.shape)


index = 0

for k in range(5):
    ret = True
    video_index = k + 1
    path = './Trimmed%d/trim%d_resized.mp4' % (video_index, video_index)
    print(path)
    capture = cv2.VideoCapture(path)
    current_length = each_length[k]
    print(current_length)
    for j in range(current_length):
        ret, frame = capture.read()
        frames[index, :, :, :] = frame
        index += 1


# In[4]:


name_string = ['E8D2', 'E84F', 'E91B', 'E863', 'E887', 'E906', 'E912']

for a in range(5):
    dir_index = a + 1
#     videoDF
    for b in range(7):
        file_name = name_string[b]
        path = './Data%d/%s.csv' % (dir_index, file_name)
        print(path)
        df = pd.read_csv(path, nrows=each_length[a])
#         print(type(df))
        if(b == 0):
            videoDF = df
        else:
            videoDF = pd.concat([videoDF, df], axis=1)
#         print(videoDF.shape)

    if(a == 0):
        totalDF = videoDF
    else:
        totalDF = pd.concat([totalDF, videoDF], axis=0)
    print(totalDF.shape)


# In[5]:


# print(frames.shape)
print(totalDF.keys())
del totalDF['Time Stamp']


# In[6]:


del totalDF['Time Stamp Unix']
print(totalDF.keys())


# In[7]:


data_output = totalDF.values
data_output.shape
all_output = np.delete(data_output, (0), axis=0)
#20520, 42


# 4460,42


# In[8]:


data_input = frames
data_input


# In[9]:


current = np.delete(data_input, (0), axis=0)
previous = np.delete(data_input, (20519), axis=0)


# In[10]:


all_frames = np.array([current, previous])


# In[ ]:

all_frames = all_frames.swapaxes(0, 1)
all_frames.shape



# In[ ]:


# need to create xTrain, yTrain, xTest, yTest
#x is input
#y is output
x_train = all_frames[0:15000]
x_test = all_frames[15000:20520]
y_train = all_output[0:15000]
y_test = all_output[15000:20520]
# print(data_input.shape)
print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)


# ## Here is the end of data processing.

# In[ ]:


class Hyperparameters(object):

    def __init__(self):
        self.depth1 = 0
        self.depth2 = 0
        self.filters = 0
        self.nodes = 0
        self.stopping = 0
        self.dropout = 0


# In[ ]:


def GenHyperparameters():
    hp = Hyperparameters()
#     hp.depth1 = np.random.choice([2, 3, 4, 5])
#     hp.depth2 = np.random.choice([1, 2, 3])
#     hp.filters = np.random.choice([16, 32, 64])
#     hp.nodes = np.random.choice([16, 32, 64, 128])
#     hp.stopping = np.random.choice([10, 15])
#     hp.dropout = np.random.choice([0.1, 0.5])
    hp.depth1 = np.random.choice([3])
    hp.depth2 = np.random.choice([2])
    hp.filters = np.random.choice([64])
    hp.nodes = np.random.choice([128])
    hp.stopping = np.random.choice([10])
    hp.dropout = np.random.choice([0.1])
    return hp


# In[ ]:


def BuildModel(hp):

    def BuildModule(model, depth=1, filters=16, input_flag=True):
        # < code ommitted >
        for j in np.arange(depth) + 1:
            if input_flag:
                model.add(Conv3D(filters, 3, strides=2,
                                 padding='same', input_shape=(2, 240, 240, 3)))
                input_flag = False
            elif j == depth:
                filters = filters * 2
                model.add(Conv3D(filters, 3, strides=4, padding='same'))
            else:
                model.add(Conv3D(filters, 3, strides=2, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        return (model, filters, input_flag)

    filters = hp.filters
    input_flag = True
    model = Sequential()
    for k in np.arange(hp.depth1) + 1:
        (model, filters, input_flag) = BuildModule(model, depth=hp.depth2,
                                                   filters=filters, input_flag=input_flag)
        print(filters)

    # FC Module
    model.add(Flatten())
    model.add(Dense(hp.nodes))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(hp.dropout))
    model.add(Dense(hp.nodes))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(42))
    return model


# In[ ]:


N = 20519  # need to adjust this to fit data
EPOCHS = 10
BATCH = 100
TRIALS = 1  # need to increase trials eventually

cols = ['depth1', 'depth2', 'filters', 'nodes', 'parameters', 'stopping', 'dropout', 'epochs',
        'time (min)', 'train error', 'val error', 'test error']
df = pd.DataFrame(np.zeros((TRIALS, len(cols))).fill(np.nan), columns=cols)
for trial in range(TRIALS):
    print('trial = %d/%d' % (trial + 1, TRIALS), flush=True)
    hp = GenHyperparameters()
#     try:
    model = BuildModel(hp)

    model.compile(loss='mean_squared_error',
                  optimizer='Adam', metrics=['accuracy'])
    time_start = time.time()
    hist = model.fit(x_train[:N, :], y_train[:N, :],
                     batch_size=BATCH,
                     epochs=EPOCHS,
                     validation_split=0.2,
                     verbose=0,
                     callbacks=[EarlyStopping(patience=hp.stopping)])
    time_stop = time.time()
    time_elapsed = (time_stop - time_start) / 60
    print("keys")
    print(hist.history.keys())
    train_err = hist.history['loss'][-1]
    val_err = hist.history['val_loss'][-1]
    test_mse = model.evaluate(x_test, y_test, batch_size=BATCH, verbose=0)[0]
    df.loc[trial, 'parameters'] = model.count_params()
    df.loc[trial, 'epochs'] = hist.epoch[-1]
    df.loc[trial, 'time (min)'] = time_elapsed
    df.loc[trial, 'train error'] = train_err
    df.loc[trial, 'val error'] = val_err
    df.loc[trial, 'test error'] = test_mse
    print('train_err', train_err)
    print('val_err', val_err)
    print('test_err', test_mse)
#     except:
    # print('warning --> exception occured')
    # df = df.sort_values('val error')
    # print(df.head().round(2),flush=True)
    # print()


# In[ ]:


# In[ ]:
