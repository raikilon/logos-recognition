% Extract data
fileID = fopen("train/data.out","r");
train_data = fscanf(fileID,"%f",[10000 ,Inf]);
fclose(fileID);
train_data = train_data';
fileID = fopen("test/data.out","r");
test_data = fscanf(fileID,"%f",[10000 ,Inf]);
fclose(fileID);
test_data = test_data';
data = cat(1,train_data,test_data);

% Extract results
fileID = fopen("train/classes.out","r");
train_results = fscanf(fileID,"%f",642);
fclose(fileID);
fileID = fopen("test/classes.out","r");
test_results = fscanf(fileID,"%f",240);
fclose(fileID);
actual = cat(1,train_results,test_results);
for i=1:length(actual)
    actual(i) = actual(i) + 1;
end
% https://ch.mathworks.com/help/nnet/ref/ind2vec.html
actual = full(ind2vec(actual'));
actual = actual';

