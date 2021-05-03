function[ ] = plot_roc (test_c,test_r , YPred ,img_test_dataaug_labels,a)

to_roc1 = zeros((2*test_c) , test_r);
to_roc2 = zeros((2*test_c) , test_r);

for x = 1 : (2*test_c)
    if(YPred(x,1) == img_test_dataaug_labels(x,1))
        to_roc1(x,1) = 1;
         to_roc2(x,1) = 1;
    else
        to_roc1(x,1)  = 0;
         to_roc2(x,1) = YPred(x,1);
    end
end

% f = sum(sum(to_roc1));
%%
[Xpr,Ypr,Tpr,AUCpr] = perfcurve(to_roc1,to_roc2,'0');
figure;
plot(Xpr,Ypr,'r'); 

xlabel('False positive rate'); ylabel( 'True positive rate');

title(['ROC curve, AUC: '  num2str(AUCpr) a ]); 