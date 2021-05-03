clc; clear; close all;

%%
img1 = imread('daisy1.png');
img2 = imread('daisy2.png');

img = cell(2,1);
img = {img1 img2};
%% size
r = size(img,1); 
c = size(img,2);

%% feature extraction use inceptýonv3
% deepNetworkDesigner
net = inceptionv3;

inputSize = net.Layers(1).InputSize; %makalede belirttiði gibi giriþ 299x299 olucak
% analyzeNetwork(net); %deep learning network analyzer de mimariyi görebilirim , 316 layers .

%verilerimiz farklý boyutlarda bunlarýn hepsini mimariye
%uygunlaþtýrmalýyýz,299x299 olmalý
augimg_Train = cell(r,c);
for i = 1 : c
augimg_Train {1,i}=imresize(img{r,i},[299 299]);
end

%% 
featuresTrain = cell(r,c);
layer = net.Layers(313); %only feature extraction , böyle çok uzun %3ten baþlýyorum size ve scaling iþini halletmiþtim
% link : https://www.mathworks.com/matlabcentral/answers/440373-creating-a-convolutional-neural-network-that-outputs-an-image
layer_r = size(layer,1);
layer_c = size(layer,2);

%%
for i = 1 : c
        featuresTrain {1,i}  = activations(net,augimg_Train{r,i},layer.Name);
end
%% 1x1x2048 i 1x2048 yapayým diyorum :) 
%  train kýsmý
featuresTrain_new = cell (r,c);
for d = 1 : c
featuresTrain_new {1,d} =  reshape(featuresTrain{1,d},1,2048);
end
featuresTrain_new =featuresTrain_new'; %transpozunu al svm fonksiyonu için
featuresTrain_new =  cell2mat(featuresTrain_new);
%%
% featuresTrain_new =featuresTrain_new';
%%
tic
load classifier.mat
YPred = predict(classifier,featuresTrain_new);
toc








