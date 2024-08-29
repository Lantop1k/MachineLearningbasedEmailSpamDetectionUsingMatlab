clc
clear all


%read train data 
datatrain=xlsread('spam-train.csv');

%split data into input X and target Y
X=datatrain(:,1:end-1); %input, X
Y=datatrain(:,end);  %target, Y

%train logistical regression matlab
b = glmfit(X,Y,'binomial','link','probit'); %coefficents of the logistic regression b

scores = glmval(b,X,'probit');

[x,y,~,~] = perfcurve(Y,scores,1);

figure(1)
plot(x,y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')

%read test data into matlab
datatest=xlsread('spam-test.csv');

yfit = glmval(b,datatest,'probit');

n=length(yfit);
predictions=zeros(1,n);
predictions(yfit>0.5)=1;


predictions= categorical(predictions,[0 1],{'Spam' 'Non-Spam'}); 

figure(2)
pie(predictions)

%compute the accuracy of the model developed by developed
yfit = glmval(b,X,'probit');

%accuracy 
n=length(yfit);
predictions=zeros(1,n);
predictions(yfit>0.5)=1;

%compute confusion matrix
C=confusionmat(predictions,Y);

%compute accuracy
accuracy=sum(diag(C))/sum(C(:));

fprintf('Accuracy of the model = %2.4f\n',accuracy)