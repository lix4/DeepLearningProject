
# coding: utf-8

# # Library Import

# In[ ]:

import tensorflow as tf         
from keras import backend as K  # needed for mixing TensorFlow and Keras commands 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0 
sess = tf.Session(config=config)
K.set_session(sess)


# In[ ]:

import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# # Data Pre-Processing

# In[ ]:

import cv2


# In[ ]:

# capture videos
total_frames = 0
for k in range(5):
    video_index = k + 1
    path = './Trimmed%d/trim%d_resized.mp4'%(video_index,video_index)
    print(path)
    capture = cv2.VideoCapture(path)
    frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames += frameCount
    print(frameCount)
    print(frameHeight)
    print(frameWidth)


# In[ ]:

frames = np.empty((total_frames, frameWidth, frameHeight, 3), np.dtype('uint8'))
print(frames.shape)


# In[ ]:

frames[9, :, :, :].shape


# In[ ]:

index = 0

for k in range(5):
    ret = True
    video_index = k + 1
    path = './Trimmed%d/trim%d_resized.mp4'%(video_index,video_index)
    print(path)
    capture = cv2.VideoCapture(path)
    while True:
        ret, frame = capture.read()
        if (ret == False):
            break
        frames[index, :, :, :] = frame
        index += 1


# In[ ]:

N      = 1000
EPOCHS = 10
BATCH  = 1000
TRIALS = 1 

cols = ['depth1','depth2','filters','nodes','parameters','stopping','dropout','epochs',
    'time (min)','train error','val error','test error']
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

