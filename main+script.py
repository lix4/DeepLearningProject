
# coding: utf-8

# In[ ]:


import tensorflow as tf         
from keras import backend as K  # needed for mixing TensorFlow and Keras commands 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0 
sess = tf.Session(config=config)
K.set_session(sess)


# In[1]:


import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# In[ ]:


# path = './Trimmed%d/trim%d_resized.mp4'%(3,3)
# capture = cv2.VideoCapture(path)
# r, frame = capture.read()


# In[ ]:


# cv2.imshow('test',frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ## The code above confirmed frames are real images 

# In[7]:


total_data_length = 20520
each_length = np.array([4187, 2267, 4570, 5036, 4460])
frameWidth = 240
frameHeight = 240
# for k in range(5):
#     video_index = k + 1
#     path = './Trimmed%d/trim%d_resized.mp4'%(video_index,video_index)
#     print(path)
#     capture = cv2.VideoCapture(path)
#     frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames += frameCount
#     print(frameCount)
#     print(frameHeight)
#     print(frameWidth)
    


frames = np.empty((total_data_length, frameWidth, frameHeight, 3), np.dtype('uint8'))
print(frames.shape)


index = 0

for k in range(5):
    ret = True
    video_index = k + 1
    path = './Trimmed%d/trim%d_resized.mp4'%(video_index,video_index)
    print(path)
    capture = cv2.VideoCapture(path)
    current_length = each_length[k]
    print(current_length)
    for j in range(current_length):
        ret, frame = capture.read()
        frames[index, :, :, :] = frame
        index += 1


# In[9]:


name_string = ['E8D2', 'E84F', 'E91B', 'E863', 'E887', 'E906', 'E912']

for a in range(5):
    dir_index = a + 1
    for b in range(7):
        file_name = name_string[b]
        path = './Data%d/%s.csv'%(dir_index, file_name)
        print(path)
        df = pd.read_csv(path, nrows=each_length[a])


# In[13]:


# print(frames.shape)
print(df.keys())
del df['Time Stamp']


# In[16]:


del df['Time Stamp Unix']


# In[22]:


data_output = df.values
data_output


# In[25]:


data_input = frames
data_input


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
    
    def BuildModule(model,depth=1,filters=16,input_flag=True):
        # < code ommitted >
        for j in np.arange(depth)+1:
            if input_flag:
                model.add(Conv2D(filters, 3, strides = 2, padding='same', input_shape=(240, 240, 3)))
                input_flag = False
            elif j == depth:
                filters = filters * 2
                model.add(Conv2D(filters, 3, strides = 4, padding='same'))
            else:
                model.add(Conv2D(filters, 3, strides = 2, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        return (model,filters,input_flag)

    filters = hp.filters
    input_flag = True
    model = Sequential()
    for k in np.arange(hp.depth1) + 1:
        (model,filters,input_flag) = BuildModule(model,depth=hp.depth2,
                filters=filters,input_flag=input_flag)
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
    model.add(Dense(54))
    return model


# In[ ]:


N      = 1000
EPOCHS = 10
BATCH  = 1000
TRIALS = 1 

cols = ['depth1','depth2','filters','nodes','parameters','stopping','dropout','epochs',
    'time (min)','train error','val error','test error'2]
df = pd.DataFrame(np.zeros((TRIALS,len(cols))).fill(np.nan),columns=cols)
for trail in range(TRIALS):
    print('trial = %d/%d' % (trial+1,TRIALS),flush=True)
    try:
        model = BuildModel(hp)

        model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        time_start = time.time()
        hist = model.fit(x_train[:N,:],y_train[:N,:],
            batch_size=BATCH,
              epochs=EPOCHS,
                validation_split=0.2,
                verbose=0,
                callbacks=[EarlyStopping(patience=hp.stopping)])
        time_stop = time.time()
        time_elapsed = (time_stop - time_start)/60
        train_err = 1 - hist.history['acc'][-1]
        val_err = 1 - hist.history['val_acc'][-1]
        test_acc = model.evaluate(x_test,y_test,batch_size=BATCH,verbose=0)
        test_err = 1 - test_acc[1]
        df.loc[trial,'parameters']  = model.count_params()
        df.loc[trial,'epochs']      = hist.epoch[-1]
        df.loc[trial,'time (min)']  = time_elapsed
        df.loc[trial,'train error'] = train_err 
        df.loc[trial,'val error']   = val_err 
        df.loc[trial,'test error']  = test_err
    except:
        print('warning --> exception occured')
        df = df.sort_values('val error')
        print(df.head().round(2),flush=True)
        print()


# In[ ]:




