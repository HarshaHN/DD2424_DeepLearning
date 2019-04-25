load('confusion.mat');
confMat = C;
for i =1:size(confMat,1)
    precision(i)=confMat(i,i)/sum(confMat(:,i));
end

for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(i,:));
end

Precision = mean(precision);
Recall = mean(recall);

conf = plotconfusion(T, C); save('conf.mat', 'conf');