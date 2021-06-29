clear,clc
addpath('Test2')

% Set a false alarm rate as you like,
P_fa2 = 0.05;                         % Set it before using this code. 

% 1 Load the pre-trained ensemble classifier
load Test2/model_51_ori_mixed_PGDresave.mat

% 2 Load the threshold as a function of P_fa2, learned from a number of
% benign images
load Test2/Threshold_fpr2PGDresave.mat
threshold = (Threshold_fpr2(P_fa2/0.01)+Threshold_fpr2(P_fa2/0.01+1))/2;

% 3 read the image and extract SRM features
sourcefile = 'sample/PGD01/demo.png';
% sourcefile = 'sample/PGD04/demo.png';
% sourcefile = 'sample/ORI/demo.png';
fea       = SRM34671(sourcefile);

% 4 Classify the features with a pre-trained ensemble classifier
test_results = ensemble_testing(fea,trained_ensemble)
%% test_results: the classification results of N sub classifiers, here N=51 
%% We do not use the 'prediction' variable directly as that done in steganalysis
if test_results.votes <= threshold
    disp('The image is benign'),
else
    disp('The image is adversarial'),
end



