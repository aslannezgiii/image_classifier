clc; clear; close all;
%%
% load all image of flowers
folder='C:\Masaüstü\okul\4. sınıf\1. dönem\makine öğrenmesine giriş\proje\kodlar\17flowers'  ;
I=dir(fullfile(folder,'*.jpg'));
for k=1:numel(I)
  filename=fullfile(folder,I(k).name);
  images{k}=imread(filename);
end
% figure; imshow(images{1}); %deneme

% load imagelabels
imagelabels = load('imagelabels17.mat');
imagelabels = imagelabels.labels;

%% size
r = size(images,1); 
c = size(images,2);
%% Split the datasets randomly into a training (80 %) and a test set (20 %)
rng('default'); %be same index
PD = 0.80 ;  % percentage 80
%% 
%train
rng(1)% For reproducibility
idx = randperm(c);
images_train = images(:,idx(1:round(c*PD))); 
img_labels_train = imagelabels(:,idx(1:round(c*PD))); 
%test
rng(1)% For reproducibility
images_test = images(:,idx(round(c*PD)+1:end));
img_labels_test = imagelabels(idx(:,round(c*PD)+1:end));
% %link : https://www.mathworks.com/matlabcentral/answers/388385-how-can-i-do-a-80-20-split-on-datasets-to-obtain-training-and-test-datasets

%% data augmentation - vertically flipped
% train için
train_r = size(images_train,1); 
train_c = size(images_train,2); 

img_train_dataaug = cell(train_r,2*train_c); %new flowers matris, data augmentation
img_train_dataaug_labels = zeros(train_r,2*train_c); %new labels matris, data augmentation

for i=1:train_c
    flip_image_train = flip(images_train{i},2); % vertically flipping
    img_train_dataaug{1,i} = images_train{i}; %
    img_train_dataaug{1,i+train_c} = flip_image_train; 
    img_train_dataaug_labels(train_r,i+train_c) = img_labels_train(train_r,i);
    img_train_dataaug_labels(train_r,i) = img_labels_train(train_r,i); 
end
% test için
test_r = size(images_test,1); 
test_c = size(images_test,2); 

img_test_dataaug = cell(test_r,2*test_c); %new flowers matris, data augmentation
img_test_dataaug_labels = zeros(test_r,2*test_c); %new labels matris, data augmentation

for i=1:test_c
    flip_image_test = flip(images_test{i},2); % vertically flipping
    img_test_dataaug{1,i} = images_test{i}; 
    img_test_dataaug{1,i+test_c} = flip_image_test; 
    img_test_dataaug_labels(test_r,i+test_c) = img_labels_test(test_r,i);
    img_test_dataaug_labels(test_r,i) = img_labels_test(test_r,i); 
end
%% şimdi ezgi sen bunları da bi karıştır ki ezberleme işlemi olmasın
rng(1)% For reproducibility
idx = randperm(train_c*2); %matris boyutu 
img_train_dataaug = img_train_dataaug(:,idx(1:round(train_c*2*1))); 
img_train_dataaug_labels = img_train_dataaug_labels(:,idx(1:round(train_c*2*1))); 
%%
rng(1)% For reproducibility
idx = randperm(test_c*2); %matris boyutu 
img_test_dataaug = img_test_dataaug(:,idx(1:round(test_c*2*1))); 
img_test_dataaug_labels = img_test_dataaug_labels(:,idx(1:round(test_c*2*1))); 
%% feature extraction use inceptıonv3
% deepNetworkDesigner
net = inceptionv3;

inputSize = net.Layers(1).InputSize; %makalede belirttiği gibi giriş 299x299 olucak
% analyzeNetwork(net); %deep learning network analyzer de mimariyi görebilirim , 316 layers .

%verilerimiz farklı boyutlarda bunların hepsini mimariye
%uygunlaştırmalıyız,299x299 olmalı
augimg_Train = cell(train_r,2*train_c);
augimg_Test = cell(test_r,2*test_c);
for i = 1 : 2*train_c
% augimg_Train {1,i}  = augmentedImageDatastore(inputSize(1:2),img_train_dataaug{train_r,i});
augimg_Train {1,i}=imresize(img_train_dataaug{train_r,i},[299 299]);

end
for i = 1 : 2*test_c
% augimg_Test {1,i}  = augmentedImageDatastore(inputSize(1:2),img_test_dataaug{test_r,i});
augimg_Test {1,i}=imresize(img_test_dataaug{test_r,i},[299 299]);

end

%% 
featuresTrain = cell(train_r,2*train_c);
featuresTest = cell(test_r,2*test_c);

layer = net.Layers(313); %only feature extraction 
% link : https://www.mathworks.com/matlabcentral/answers/440373-creating-a-convolutional-neural-network-that-outputs-an-image
layer_r = size(layer,1);
layer_c = size(layer,2);

%  layer = 'average_pooling2d_1'; %layer array'i yapmalısın ezgi:)
% layer(1,1).Name

% layers = [ ...
%     imageInputLayer([299 299 3])
%     convolution2dLayer(5,20)
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)]; 
% % link : https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.relulayer.html
%%
for i = 1 : 2*train_c
%     for k = 1 : layer_r
        featuresTrain {1,i}  = activations(net,augimg_Train{train_r,i},layer.Name);
%     end
end
for i = 1 : 2*test_c
%     for k = 1 : layer_r
        featuresTest {1,i}  = activations(net,augimg_Test{test_r,i},layer.Name);
%     end
end
%% 1x1x2048 i 1x2048 yapayım diyorum :) 
%  train kısmı
featuresTrain_new = cell (train_r,2*train_c);
for d = 1 : 2*train_c
featuresTrain_new {1,d} =  reshape(featuresTrain{1,d},1,2048);
end
featuresTrain_new =featuresTrain_new'; %transpozunu al svm fonksiyonu için
featuresTrain_new =  cell2mat(featuresTrain_new);
%% 1x1x2048 i 1x2048 yapayım diyorum :)
%  test kısmı
featuresTest_new = cell (test_r,2*test_c);
for d = 1 : 2*test_c
featuresTest_new {1,d} =  reshape(featuresTest{1,d},1,2048);
end
featuresTest_new =featuresTest_new'; %transpozunu al predict fonksiyonu için
featuresTest_new =  cell2mat(featuresTest_new);
%%
img_train_dataaug_labels = img_train_dataaug_labels';

%% fonksiyonlara yollayalım
 
acc_svm = func_svm (test_c,featuresTrain_new , img_train_dataaug_labels ,featuresTest_new,img_test_dataaug_labels,test_r);
acc_knn = func_knn (test_c,featuresTrain_new , img_train_dataaug_labels ,featuresTest_new,img_test_dataaug_labels,test_r);
acc_rf = func_rf (test_c,featuresTrain_new , img_train_dataaug_labels ,featuresTest_new,img_test_dataaug_labels,test_r);





