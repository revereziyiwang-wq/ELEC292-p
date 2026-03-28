import numpy as np
import pandas as pd
import os
import h5py
import random
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# rawdataset.h5
# ├── raw/                        ← original data from all 9 CSV files
# │   ├── Tony/
# │   │   ├── Back
# │   │   ├── Front
# │   │   └── Right
# │   ├── Thomas/
# │   └── William/
# │
# ├── pre_processed/              ← smoothed data using MA filter (window=5)
# │   ├── Tony/
# │   │   ├── Back
# │   │   ├── Front
# │   │   └── Right
# │   ├── Thomas/
# │   └── William/
# │
# └── segmented/                  ← 5-second windows, shuffled, split 90/10
#     ├── train/
#     │   ├── windows 
#     │   └── labels  
#     └── test/
#         ├── windows  
#         └── labels   

# df the oridinal pandas dataframs

WINDOW_MA = 17 # take the 4 neighbor around and replace the current value with the average of those 5 values

members = ["Tony", "Thomas", "William"]
actions = ["Back", "Front", "Right"]

# creat the segmented data and save it in the hdf5 file
windows_all = [] # for 5-sec 
labels_all = [] #0 for walking 1 for jumping

# sampling_rate = 200 # 200Hz
window_size = 1000 # 5 seconds 0.005s per line


# creat HDF5 file and write data w 
with h5py.File("rawdataset.h5", "w") as f:
    #df = pd.read_csv("rawdata/Tony/Back/Raw Data.csv")
    raw_group = f.create_group("raw") # the main group for raw data = Rawdata
    
    # 2 D loop first loop for members, second loop for actions
    for member in members:
        member_group = raw_group.create_group(member) # make subgroup for each member,ex raw/Tony

        for action in actions:
            # build the path to the csv file for each member and action, ex rawdata/Tony/Back/Raw Data.csv
            file_path = os.path.join("rawdata", member, action, "Raw Data.csv")
            df = pd.read_csv(file_path) # read the csv file into a pandas dataframe

            # removes text columns vkeep onr number and convert to numpy array. (from the internet)
            data_raw = df.select_dtypes(include=[np.number]).to_numpy()
            #Create a dataset inside the member's group using the action name  /raw/Tony/Back
            dset = member_group.create_dataset(action, data=data_raw)
            #print(f"[RAW] {member}/{action} → {data_raw.shape}") # print the member, action and shape of the raw data
            # dset.attrs["columns"] = df.select_dtypes(include=[np.number]).columns.tolist()

# ── pre processed data  (Moving Average filter + High Pass Filter) ────────────
    def highpass_filter(data, cutoff=0.5, fs=200, order=4):
        # cutoff = remove anything slower than 0.5Hz (the drift)
        # fs = sampling rate 200Hz
        # order = how strong the filter is
        nyq = fs / 2                    # nyquist frequency = half of sampling rate
        normal_cutoff = cutoff / nyq    # normalize the cutoff frequency
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data, axis=0)  # apply filter to all columns

    pre_processed_group = f.create_group("pre_processed") # the main group for pre-processed data = pre_processed

    for member in members:
        member_group = pre_processed_group.create_group(member)

        for action in actions:
            file_path = os.path.join("rawdata", member, action, "Raw Data.csv")
            df_pre = pd.read_csv(file_path)

            # fill all the missing value, use the back method,fill the enpty place the the data after it. 
            df_pre = df_pre.bfill()
            # the moving average, pick only number from the file creat the window and take the average. 
            df_smoothed = df_pre.select_dtypes(include=[np.number]).rolling(window=WINDOW_MA, center=True).mean()
            df_smoothed = df_smoothed.bfill().ffill()

            data_pre = df_smoothed.to_numpy() 
            df_pre = df_pre.bfill()

            # the high pass filter, remove the drift, use the function we creat before
            # step 1 - moving average to smooth noise
            df_smoothed = df_pre.select_dtypes(include=[np.number]).rolling(
                window=WINDOW_MA, center=True).mean()
            df_smoothed = df_smoothed.bfill().ffill()

            # step 2 - high pass filter to remove drift
            data_pre = highpass_filter(df_smoothed.to_numpy())

            dset_pre = member_group.create_dataset(action, data=data_pre)# save the pre-processed data to the hdf5 file 
            #  text, member, action, shape of the pre-processed data
            print(f"[PREPROCESSED] {member}/{action} -> {data_pre.shape}")
            
    #print("done pre-processing and load the raw data and saving to rawdataset.h5")
    #print()

# --------------segmentation class----------------------
with h5py.File("rawdataset.h5", "a") as f: # open the file in append mode to add the segmented data
    for member in members:
        for action in actions:
            # read the pre-processed data from the hdf5 file
            data_pre_window = f[f"pre_processed/{member}/{action}"][:]

            half = len(data_pre_window) // 2 # get the half of the data length
            w_data = data_pre_window[:half] # take the first half for walking
            j_data = data_pre_window[half:] # take the second half for jumping

            print(f"Segmenting {member}/{action} -> Walking: {w_data.shape},     Jumping: {j_data.shape}")

            # loop through the walking data for 5 second window
            # .append add new item to the end of the list
            i = 0
            while i + window_size <= len(w_data):
                windows_all.append(w_data[i:i+window_size]) # start at i and add 5 sec of data AKA 5000 rows of data to the windows_all list
                labels_all.append(0) # make the lable = 0 for walking 
                i += window_size # move to the next window since the last wind size is saved in i

            # loop through the jumping data for 5 second window same thing
            i = 0   
            while i + window_size <= len(j_data):
                window_all = j_data[i:i+window_size]
                windows_all.append(window_all) 
                labels_all.append(1) 
                i += window_size

    # convert lists to numpy arrays
    windows_all = np.array(windows_all)  # shape: (total_windows, 1000, 5)
    labels_all = np.array(labels_all)    
    print(f"Total windows: {len(windows_all)}, Total labels: {len(labels_all)}")
    print()

#-------segmented data math ---------------------------------------------------------------------------------------
    x = random.randint(1, 10)
    # shuffle the windows 
    pairs = list(zip(windows_all, labels_all)) # create pairs of windows and labels zip make the 2 array to one array of pairs w-w1 l - 0 to p -w1, 0
    random.seed(18) # replace 18 to x to make it really random
    random.shuffle(pairs)
    windows_all, labels_all = zip(*pairs) # unzip the pairs back to windows and labels
    windows_all = np.array(windows_all) # back to 2 array 
    labels_all = np.array(labels_all) # convert back to numpy array

    # 90:10 split
    split_index = int(0.9 * len(windows_all)) # 90% for the training 

    train_window = windows_all[:split_index] # take the first 90% of the windows for training
    train_labels = labels_all[:split_index] # take the first 90% of the

    test_window = windows_all[split_index:] # take the last 10% of the windows for testing
    test_labels = labels_all[split_index:] # take the last 10% of the

    print(f"train: {len(train_window)}, test: {len(test_window)}")
    print()

# save to HDF5_file / creat.group is making the folder and create dataset is making the file and save the data in it.
#with h5py.File("rawdataset.h5", "a") as f: # open the file in append mode to add the segmented data
    seg_group = f.create_group("segmented")#the main group for seg. rawdataset.h5/segmented

    train_group = seg_group.create_group("train") # subgroup for seg
    test_group = seg_group.create_group("test") # rawdataset.h5/segmented/test

    # save the train windows to the hdf5 file      rawdataset.h5/segmented/train/windows
    train_group_window = train_group.create_dataset("windows", data=train_window) 
    train_group_labels = train_group.create_dataset("labels", data=train_labels) 

    test_group_window = test_group.create_dataset("windows", data=test_window) 
    test_group_labels = test_group.create_dataset("labels", data=test_labels) 

    print("Segmented data saved to rawdataset.h5")
    print()

    #check for the data set
    print("Checking the contents of rawdataset.h5:") # ok to remove this print statement, just for checking the data set

with h5py.File("rawdataset.h5", "r") as f:
    raw = f["raw/Tony/Back"][:]
    pre = f["pre_processed/Tony/Back"][:]
    print("Raw mean X:", np.mean(raw[:, 0]))
    print("Pre mean X:", np.mean(pre[:, 0]))
# with h5py.File("rawdataset.h5", "r") as f:
#     for member in ["Tony", "Thomas", "William"]:
#         for action in ["Back", "Front", "Right"]:
#             # check raw, just check every file exist or not. for what is in side the file no idear
#             raw_exists = f"raw/{member}/{action}" in f
#             # check pre_processed
#             pre_exists = f"pre_processed/{member}/{action}" in f
#             print(f"{member}/{action} -> raw: {raw_exists}, pre_processed: {pre_exists}")
#     #print(f"\nsegmented/train/windows : {f['segmented/train/windows'].shape}")
#     #print(f"segmented/train/labels  : {f['segmented/train/labels'].shape}")
#     #print(f"segmented/test/windows  : {f['segmented/test/windows'].shape}")
#     #print(f"segmented/test/labels   : {f['segmented/test/labels'].shape}")
#     seg_exists = "segmented/train/windows" in f and "segmented/train/labels" in f and "segmented/test/windows" in f and "segmented/test/labels" in f
#     print(f" segmented: {seg_exists}")

#----step 3 visulization------------------------------
import matplotlib.pyplot as plt

with h5py.File("rawdataset.h5", "r") as f:
    train_windows = f["segmented/train/windows"][:]
    train_labels  = f["segmented/train/labels"][:]
    
    for member in members:
        plt.figure(figsize=(12,10))
        plt.suptitle(f"All Acceleration Data for {member}", fontsize=16)
        for i, action in enumerate(actions):
            raw_sample = f[f"raw/{member}/{action}"][30000:35000] # take the first 1000 rows of the raw data for each action for visualization
            pre_sample = f[f"pre_processed/{member}/{action}"][30000:35000]

            #------Raw--------
            plt.subplot(3, 2, 2*i + 1)
            plt.plot(raw_sample[:, 1], label="X")
            plt.plot(raw_sample[:, 2], label="Y")
            plt.plot(raw_sample[:, 3], label="Z")
            plt.title(f"{action} - Raw")
            plt.xlabel("Time")
            plt.ylabel("Acceleration")
            plt.legend()
            plt.grid(True)

            #------Preprocessed--------
            plt.subplot(3, 2, 2*i + 2)
            plt.plot(pre_sample[:, 1], label="X (Filtered)")
            plt.plot(pre_sample[:, 2], label="Y (Filtered)")
            plt.plot(pre_sample[:, 3], label="Z (Filtered)")
            plt.title(f"{action} - Preprocessed")
            plt.xlabel("Time")
            plt.ylabel("Acceleration")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()


walk_windows = train_windows[train_labels == 0] # take the windows with label 0 for walking
jump_windows = train_windows[train_labels == 1] # take the windows with label

walking_mean_x = np.mean(walk_windows[:, :, 1], axis=1)  # column 1 = x
jumping_mean_x = np.mean(jump_windows[:, :, 1], axis=1)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Feature Distributions: Walking vs Jumping")

axis_names = ["X", "Y", "Z"]
feature_names = ["Mean", "Std"]

for i, axis in enumerate(range(1, 4)): # loop through x, y, z
    A = i + 1
    # mean feature
    walking_vals = np.mean(walk_windows[:, :, A], axis=1)
    jumping_vals  = np.mean(jump_windows[:, :, A], axis=1)
    # mean
    axes[0, i].hist(walk_windows[:, :, axis].mean(axis=1), bins=20, alpha=0.5, label="Walking")
    axes[0, i].hist(jump_windows[:, :, axis].mean(axis=1), bins=20, alpha=0.5, label="Jumping")
    axes[0, i].set_title(f"{feature_names[0]} {axis_names[i]}")
    axes[0, i].set_xlabel("Mean Acceleration (m/s²)") 
    axes[0, i].set_ylabel("Number of Windows") 
    axes[0, i].legend()
    axes[0, i].grid(True)

    walking_vals = np.std(walk_windows[:, :, A], axis=1)
    jumping_vals  = np.std(jump_windows[:, :, A], axis=1)
    # std
    axes[1, i].hist(walk_windows[:, :, axis].std(axis=1), bins=20, alpha=0.5, label="Walking")
    axes[1, i].hist(jump_windows[:, :, axis].std(axis=1), bins=20, alpha=0.5, label="Jumping")
    axes[1, i].set_title(f"{feature_names[1]} {axis_names[i]}")
    axes[1, i].set_xlabel("Std Acceleration (m/s²)")   # ← x axis label
    axes[1, i].set_ylabel("Number of Windows")
    axes[1, i].legend()
    axes[1, i].grid(True)

plt.tight_layout()
plt.show()


#------machine learning-------------

#load segmented data from HDF5
with h5py.File("rawdataset.h5", "r") as f:
    train_window = f["segmented/train/windows"][:]
    train_labels = f["segmented/train/labels"][:]
    test_window = f["segmented/test/windows"][:]
    test_labels = f["segmented/test/labels"][:]


# -------- Feature Extraction --------
def extract_features(window):
    window = window[:, :3] # take only the first 3 columns for x, y, z acceleration

    features = []
    
    for axis in range(window.shape[1]): # loop through each axis x, y, z
        data = window[:, axis] 
        
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.min(data))
        features.append(np.max(data))
        features.append(np.ptp(data))

    #magnitude
    mag = np.sqrt(window[:,0]**2 + window[:,1]**2 + window[:,2]**2) #mag = √(x² + y² + z²)
    features.append(np.mean(mag))
    features.append(np.std(mag))

    return np.array(features) # return a 16 dimensional feature vector for each window. 


#feature extraction
X_train = np.array([extract_features(w) for w in train_window])
X_test = np.array([extract_features(w) for w in test_window])

#normalization
from sklearn.preprocessing import StandardScaler # standard scaler is a common method for normalizing features

scaler = StandardScaler() # create an instance of the StandardScaler
X_train = scaler.fit_transform(X_train) # fit the scaler to tarin data and transform it
X_test = scaler.transform(X_test)


#model training
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000) # create a logistic regression model max_iter is the maximum number of iterations for the solver to converge
model.fit(X_train, train_labels) 
y_pred = model.predict(X_test) # use trained model to predict labels for test windows
print("Accuracy:", accuracy_score(test_labels, y_pred))# print the accuracy of the model on the test set



#learning curve 
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, test_scores = learning_curve(model, X_train, train_labels, cv=5, scoring='accuracy') # tests the model with different amounts of training data

train_mean = train_scores.mean(axis=1) #  average the scores across the 5 splits
test_mean = test_scores.mean(axis=1)
# graph the 2 line for training and validation accuracy as the training size increases.
plt.plot(train_sizes, train_mean, label="Training Accuracy")
plt.plot(train_sizes, test_mean, label="Validation Accuracy")

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Learning Curve")
plt.show()









