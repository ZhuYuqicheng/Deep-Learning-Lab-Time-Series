#%% Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

#%%
def ShowSegmentation():
    # load an example time series
    raw_acc_data = pd.read_table("./HAPT Dataset/RawData/acc_exp01_user01.txt", delim_whitespace=True, header=None)
    raw_gyro_data = pd.read_table("./HAPT Dataset/RawData/gyro_exp01_user01.txt", delim_whitespace=True, header=None)

    # visualize the segmentation
    fig, ax = plt.subplots(1,1,figsize=(14,6))
    ax.plot(raw_acc_data.iloc[7000:8000,0])
    vlines = np.linspace(7000, 7000+int(1000/64)*64, int(1000/64))
    for vline in vlines:
        ax.axvline(x=vline, linestyle="--", color="#ff7f0e", alpha=.8)
    ax.set_title("Visualization of the signal segmentation for HAR dataset")
    ax.set_xlabel("time (50 Hz)")
    ax.set_ylabel("Acceleration (m/s^2)")

def ShowPosturalTransitions():
    # load an example time series
    raw_acc_data = pd.read_table("./HAPT Dataset/RawData/acc_exp01_user01.txt", delim_whitespace=True, header=None)
    raw_gyro_data = pd.read_table("./HAPT Dataset/RawData/gyro_exp01_user01.txt", delim_whitespace=True, header=None)
    label_info = pd.read_table("./HAPT Dataset/RawData/labels.txt", delim_whitespace=True, header=None)
    label_info.columns = ["exp_num", "user_num", "act_num", "label_start", "label_end"]
    label = label_info[label_info["exp_num"]==1]

    # visualization
    fig, ax = plt.subplots(1,1,figsize=(14,6))
    ax.plot(raw_acc_data.iloc[:10000,0])
    for start in label["label_start"]:
        if start>10000:
            break
        ax.axvline(x=start, linestyle="--", color="#ff7f0e", alpha=.8)
        
    for end in label["label_end"]:
        if end>10000:
            break
        ax.axvline(x=end, linestyle="--", color="#ff7f0e", alpha=.8)
        
    ax.set_title("Visualization of the postural transitions for HAPT dataset")
    ax.set_xlabel("time (50 Hz)")
    ax.set_ylabel("Acceleration (m/s^2)")

def PlotTimeSeries(index):
    # load all data
    train_paths = ["./UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt",\
                "./UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt",\
                "./UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt",\
                "./UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt",\
                "./UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt",\
                "./UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt",\
                "./UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt",\
                "./UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt",\
                "./UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt"]
    body_acc_x_train = pd.read_table(train_paths[0], delim_whitespace=True, header=None)
    body_acc_y_train = pd.read_table(train_paths[1], delim_whitespace=True, header=None)
    body_acc_z_train = pd.read_table(train_paths[2], delim_whitespace=True, header=None)
    body_gyro_x_train = pd.read_table(train_paths[3], delim_whitespace=True, header=None)
    body_gyro_y_train = pd.read_table(train_paths[4], delim_whitespace=True, header=None)
    body_gyro_z_train = pd.read_table(train_paths[5], delim_whitespace=True, header=None)
    total_acc_x_train = pd.read_table(train_paths[6], delim_whitespace=True, header=None)
    total_acc_y_train = pd.read_table(train_paths[7], delim_whitespace=True, header=None)
    total_acc_z_train = pd.read_table(train_paths[8], delim_whitespace=True, header=None)
    train_y = pd.read_table("./UCI HAR Dataset/train/y_train.txt", delim_whitespace=True, header=None)

    # Visualization
    print(train_y.iloc[index, 0])
    fig, ax = plt.subplots(3,1,figsize=(10,12))
    total_acc_x_train.iloc[index,:].plot(ax=ax[0])
    total_acc_y_train.iloc[index,:].plot(ax=ax[0])
    total_acc_z_train.iloc[index,:].plot(ax=ax[0])
    ax[0].legend(["total_acc_x_train","total_acc_y_train","total_acc_z_train"])
    ax[0].set_title("Triaxial acceleration from the accelerometer")
    ax[0].set_ylabel("acceleration (m/s^2)")
    ax[0].set_xlabel("samples (50Hz)")
    body_acc_x_train.iloc[index,:].plot(ax=ax[1])
    body_acc_y_train.iloc[index,:].plot(ax=ax[1])
    body_acc_z_train.iloc[index,:].plot(ax=ax[1])
    ax[1].legend(["body_acc_x_train","body_acc_y_train","body_acc_z_train"])
    ax[1].set_title("The estimated body acceleration")
    ax[1].set_ylabel("acceleration (m/s^2)")
    ax[1].set_xlabel("samples (50Hz)")
    body_gyro_x_train.iloc[index,:].plot(ax=ax[2])
    body_gyro_y_train.iloc[index,:].plot(ax=ax[2])
    body_gyro_z_train.iloc[index,:].plot(ax=ax[2])
    ax[2].legend(["body_gyro_x_train","body_gyro_y_train","body_gyro_z_train"])
    ax[2].set_title("Triaxial angular velocity from the gyroscope")
    ax[2].set_ylabel("angular velocity (RPS)")
    ax[2].set_xlabel("samples (50Hz)")
    plt.tight_layout()

def PlotLabelDist():
    # get feature matrix
    X_train_path = "./UCI HAR Dataset/train/X_train.txt"
    col_names = pd.read_table("./UCI HAR Dataset/features_new.txt", delim_whitespace=True, header=None)
    X_train = pd.read_table(X_train_path, delim_whitespace=True, names=col_names.iloc[:,1])

    # get labels
    y_train_path = "./UCI HAR Dataset/train/y_train.txt"
    y_train = pd.read_table(y_train_path, delim_whitespace=True, header=None)

    # Visualize label distribution
    map_dict = {1:"WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS", 4:"SITTING", 5:"STANDING", 6:"LAYING"}
    y_cate = y_train[0].map(map_dict)
    sns.histplot(y_cate, discrete=True)
    plt.xticks(rotation=90)
    plt.title("Distribution of activity label")
    plt.xlabel("")

def FeatureSelection():
    # get feature matrix
    X_train_path = "./UCI HAR Dataset/train/X_train.txt"
    col_names = pd.read_table("./UCI HAR Dataset/features_new.txt", delim_whitespace=True, header=None)
    X_train = pd.read_table(X_train_path, delim_whitespace=True, names=col_names.iloc[:,1])

    # get labels
    y_train_path = "./UCI HAR Dataset/train/y_train.txt"
    y_train = pd.read_table(y_train_path, delim_whitespace=True, header=None)

    # Removing Features with Low Variance
    thres = 0.9
    selector = VarianceThreshold(threshold=(thres*(1-thres))).fit(np.array(X_train))
    X_train_thres = selector.transform(np.array(X_train))
    # get new column names
    new_col_names = col_names[selector.get_support()].iloc[:,1]

    # L1 based feature selection
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train_thres, np.ravel(np.array(y_train)))
    importance = pd.DataFrame(data=lsvc.coef_, columns=new_col_names)
    # importance.loc[:, (importance != 0).any(axis=0)]

    # Get the importance weight for the top 10 features for each label
    label_str = ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"]
    fig, ax = plt.subplots(3,2,figsize=(12,10))
    count = 0
    for i in range(3):
        for j in range(2):
            sorted_weight = importance.iloc[count,:].abs().sort_values(ascending=True)
            sorted_weight[-10:].plot.barh(ax=ax[i,j])
            ax[i,j].set_title(label_str[count])
            ax[i,j].set_ylabel("feature name")
            ax[i,j].set_xlabel("imporatance weight")
            ax[i,j].set_xlim([0,2])
            count += 1
    plt.tight_layout()