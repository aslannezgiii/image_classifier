function[acc_svm] = func_svm (test_c,featuresTrain_new , img_train_dataaug_labels ,featuresTest_new,img_test_dataaug_labels,test_r)
%%  classification with svm
% featuresTrain_new2 = featuresTrain_new{1,:};
%denemem lazým 3 farklý kernel için
tic
t = templateSVM('Standardize',true,'KernelFunction','linear');
% t = templateSVM('KernelFunction','gaussian'); %linear , polynomial , gaussian
classifier = fitcecoc(featuresTrain_new,img_train_dataaug_labels,'Learners',t);
save ( 'classifier.mat' , 'classifier' );
% saveLearnerForCoder(classifier,'classifier');
YPred = predict(classifier,featuresTest_new);
toc
%%
img_test_dataaug_labels = img_test_dataaug_labels';
%% accuracy

acc=0;
for m = 1 : test_c*2
   if( YPred(m,1) == img_test_dataaug_labels(m,1))
       acc = acc + 1;
   end 
end 

acc_svm = (100 / (test_c*2)) * acc;

%%
a= 'svm';
plot_roc (test_c,test_r , YPred ,img_test_dataaug_labels,a);
%% confusion matrix
figure;
cm = confusionchart(img_test_dataaug_labels,YPred);
cm.Title = 'confusion matrix - svm';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

