function[acc_knn] = func_knn (test_c,featuresTrain_new , img_train_dataaug_labels ,featuresTest_new,img_test_dataaug_labels,test_r)
%%  classification with knn
tic
classifier = fitcknn(featuresTrain_new,img_train_dataaug_labels,'NumNeighbors',7,...
    'NSMethod','exhaustive','Distance','cityblock',...
    'Standardize',1); % cityblock , euclidean , minkowski

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

acc_knn = (100 / (test_c*2)) * acc;
%% roc
a= 'knn';
plot_roc (test_c,test_r , YPred ,img_test_dataaug_labels,a);
%% confusion matrix
figure;
cm = confusionchart(img_test_dataaug_labels,YPred);
cm.Title = 'confusion matrix - knn';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

