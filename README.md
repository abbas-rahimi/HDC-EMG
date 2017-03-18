We provide Matlab implementation of hyperdimensional (HD) computing 
for electromyography (EMG)-based hand gesture recognition. 
We compare its effectiveness (recognition accuracy, speed of learning, robustness, etc.) 
with a multi-class Support Vector Machine (SVM) as the state-of-the-art method for EMG classification. 

This program is licensed as GNU GPLv3.

The files are organized as follows.

1. "ICRC.m": All functions used for HD encoding/decoding for EMG signals
2. "generatePaperFigures.m": Generate Figures (5, 7, 8, 9, 10) used in the paper
3. "dataset.mat": EMG full dataset for 5 subjects
4. "svmtrain.mexa64": SVM train function from LIBSVM v3.21 (a Library for Support Vector Machines available in https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
5. "svmpredict.mexa64": SVM predict function from LIBSVM v3.21, too
6. "errorbar_groups.m" : a grouped bar plot with error bars available in Matlab fileexchange/29702.  
7. "binaryCode.m": We also provided another version of the EMG encoder using the binary seed hypervectors as opposed to the bipolar codes used in our paper. This MATLAB file can be used instead of "ICRC.m". To use this encoder in "generatePaperFigures.m", we should call binaryCode instead of ICRC in line 21. In addition, the following lines should be added:

global codingMode;
codingMode = 'dense_binary';
global encoderMode;
encoderMode = 'N_feature_perm';

For more information, please refer to our paper and cite it if you use the code:
Abbas Rahimi, Simone Benatti, Pentti Kanerva, Luca Benini, and Jan M. Rabaey, 
“Hyperdimensional Biosignal Processing: A Case Study for EMG-based Hand Gesture Recognition,” 
in Proceeding IEEE International Conference on Rebooting Computing (ICRC), October 2016.

Thanks!

Abbas Rahimi
email: abbas@eecs.berkeley.edu
