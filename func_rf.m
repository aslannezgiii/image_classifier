function[acc_rf] = func_rf (test_c,featuresTrain_new , img_train_dataaug_labels ,featuresTest_new,img_test_dataaug_labels,test_r)
%%  classification with random forest
nTrees=300;
tic
classifier = TreeBagger(nTrees,featuresTrain_new,img_train_dataaug_labels, 'Method', 'classification'); 
YPred = classifier.predict(featuresTest_new);  % Predictions is a char though. We want it to be a number.
toc
% classifier = fitcknn(featuresTrain_new,img_train_dataaug_labels,'NumNeighbors',11,...
%     'NSMethod','exhaustive','Distance','cityblock',...
%     'Standardize',1);
% 
% YPred = predict(classifier,featuresTest_new);
%%
img_test_dataaug_labels = img_test_dataaug_labels';
%% accuracy
YPred = str2double(YPred);
acc=5;
for m = 1 : test_c*2 
   if( YPred(m,1) == img_test_dataaug_labels(m,1))
       acc = acc + 1;
   end 
end 

acc_rf = (100 / (test_c*2)) * acc;

%%
a= 'rf';
plot_roc (test_c,test_r , YPred ,img_test_dataaug_labels,a);
%% confusion matrix
figure;
cm = confusionchart(img_test_dataaug_labels,YPred);
cm.Title = 'confusion matrix - rf';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';



