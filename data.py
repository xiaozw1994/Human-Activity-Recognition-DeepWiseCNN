import numpy as np
import matplotlib.pyplot as plt
import time 
import pandas as pd
import os 
from scipy import stats
from sklearn.metrics import classification_report
from keras.utils import np_utils
from sklearn import metrics
import seaborn as sns

####
#  Read Dataset and Show the dataset 
#
init_raw_data_file = './raw_data/'
### dir that saves initial raw dataset 
save_dir = './data/'
###  save dataset that is processed where is decedied by our needs
save_train_90_size =os.path.join(save_dir, 'train_90_size.npz.npy')
save_train_label_90 = os.path.join(save_dir,'train_label_90.npz.npy')
save_test_90_size =os.path.join(save_dir, 'test_90_size.npz.npy')
save_test_label_90 = os.path.join(save_dir,'test_label_90.npz.npy')
####  90 
save_train_96_size = os.path.join(save_dir, 'train_96_size.npz.npy')
save_train_label_96 = os.path.join(save_dir,'train_label_96.npz.npy')
save_test_96_size = os.path.join(save_dir,'test_96_size.npz.npy')
save_test_label_96 = os.path.join(save_dir,'test_label_96.npz.npy')
####  96
save_train_64_size = os.path.join(save_dir,'train_64_size.npz.npy')
save_train_label_64 = os.path.join(save_dir,'train_label_64.npz.npy')
save_test_64_size = os.path.join(save_dir,'test_64_size.npz.npy')
save_test_label_64 = os.path.join(save_dir,'test_label_64.npz.npy')
#### 64
save_train_48_size = os.path.join(save_dir,'train_48_size.npz.npy')
save_train_label_48 = os.path.join(save_dir,'train_label_48.npz.npy')
save_test_48_size = os.path.join(save_dir,'test_48_size.npz.npy')
save_test_label_48 = os.path.join(save_dir,'test_label_48.npz.npy')
#########48
"""
|           
|            different size :     96   90 64  48
|
|
"""
LABELS = ["Downstairs",
          "Jogging",
          "Sitting",
          "Standing",
          "Upstairs",
          "Walking"]
def read_data(file_path):
    column_names = ['user-id','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path,header = None, names = column_names)
    #f = opem(file_path,'r')
    return data
###########
def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    #print (mu)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma
###########
def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
###############
def plot_activity(activity,data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (15, 10), sharex = True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()
##
def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)

def segment_signal(data,window_size = 90):
    segments = np.empty((0,window_size,3))

    labels = []
    #print (labels)
    for (start, end) in windows(data['timestamp'], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if(len(dataset['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x,y,z])])
            #print (segments)
            labels = np.append(labels,stats.mode(data["activity"][start:end])[0][0])
            #print (labels)
            #labels.append(data["activity"][start:end])
            #print (labels)
    return segments, labels

def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
path_data = os.path.join(init_raw_data_file,"WISDM_ar_v1.1_raw.txt")
dataset = read_data( path_data)

dataset['activity'].value_counts().plot(kind='bar',
                                   title='Training Examples by Activity Type')
plt.show()

dataset['user-id'].value_counts().plot(kind='bar',
                                  title='Training Examples by User')
plt.show()

dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
dataset['z-axis'] = feature_normalize(dataset['z-axis'])
for activity in np.unique(dataset["activity"]):
    subset = dataset[dataset["activity"] == activity][:180]
    plot_activity(activity,subset)
    print(subset,subset.shape)
## Save Files Function
def Save_dataset(dataset,window_size,save_train,save_train_label,save_test,save_test_label,shapes=[1,90,3]):
    segments, labels = segment_signal(dataset,window_size)
    labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    reshaped_segments = segments.reshape(len(segments), shapes[0],shapes[1],shapes[2])
    train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
    train_x = reshaped_segments[train_test_split]
    train_y = labels[train_test_split]
    test_x = reshaped_segments[~train_test_split]
    test_y = labels[~train_test_split]
    np.save(save_train,train_x)
    np.save(save_train_label,train_y)
    np.save(save_test,test_x)
    np.save(save_test_label,test_y)
###
##
'''
window_size_90 = 90
shape = [1,90,3]
Save_dataset(dataset,window_size_90,save_train_90_size,save_train_label_90,save_test_90_size,save_test_label_90,shape)
window_size_96 = 96
shape = [96,3,1]
Save_dataset(dataset,window_size_96,save_train_96_size,save_train_label_96,save_test_96_size,save_test_label_96,shape)
###
window_size_64 = 64
shape = [64,3,1]
Save_dataset(dataset,window_size_64,save_train_64_size,save_train_label_64,save_test_64_size,save_test_label_64,shape)
###
window_size_48 = 48
shape = [48,3,1]
Save_dataset(dataset,window_size_48,save_train_48_size,save_train_label_48,save_test_48_size,save_test_label_48,shape)

##
### Test Datasets that are saved in my documents


train_x = np.load(save_train_96_size)
train_y = np.load(save_train_label_96)
test_x = np.load(save_test_96_size)
test_y = np.load(save_test_label_96)
'''




