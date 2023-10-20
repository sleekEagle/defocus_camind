
****************************************************************************************************
# training the NYU depth models
code : train_nyu.py

Assume your blurred images are in 
D:\data\nyu_depth_v2\refocused_f_25_fdist_2\
your depth ground truth files are in 
D:\data\nyu_depth_v2\rawDepth\

1. set data_path in nyu_options.py to D:\\data\\       (double back slash because windows)
2. set rgb_dir to refocused_f_25_fdist_2
Note : the data must be in a directory called "nyu_depth_v2". Otherwise this will not work 
3. set depth_dir to rawDepth
4. set resultspth in base_options.py to where you need the models and log files to save
5. create a conda environment and install the required packages
6. run the training code
python train_nyu.py
Note above options can also be provided as terminal arguments like
python train_nyu.py --data_path D:\\data\\

Optional :
if you need to change the batch size, change 
batch_size parameter of base_options.py


****************************************************************************************************
#testing trained NYU models
code: test_nyu.py
1. set the blurred directory to be tested in --eval_test_rgb_dir option in nyu_options.py
2. set the path to trained model --trained_model
The code will automatically read the camera parameters from the directory name
e.g : refocused_f_40_fdist_2 will have a focal length of 40mm and focal distance of 2m 
and calculate kcam using these values and other parameters are hardcoded.
3. run the code 
python test_nyu.py

Note above options can also be provided as terminal arguments like
python train_nyu.py --trained_model D:\\data\\f_50_fdist_2_f_25_fdist_2.tar







