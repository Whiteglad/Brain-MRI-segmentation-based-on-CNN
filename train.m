
miniBatchSize=100;
class_size=4;
stride=input('请输入步长：');
patch_size=input('请输入区块的大小:');
[label,patch]=MRI_input(stride,patch_size);
distribute=[7,10,17];  %训练集、验证集和测试集的分配

% label=categorical(label');         
train_patch=patch(:,:,:,1:(distribute(1)*(ceil(181/stride))*(ceil(181/stride))));
validation_patch=patch(:,:,:,(distribute(1)*(ceil(181/stride))*(ceil(181/stride))+1):(distribute(2)*(ceil(181/stride))*(ceil(181/stride))));
test_patch=patch(:,:,:,(distribute(2)*(ceil(181/stride))*(ceil(181/stride))+1):(distribute(3)*(ceil(181/stride))*(ceil(181/stride))));
train_label=label(1:(distribute(1)*(ceil(181/stride))*(ceil(181/stride))));
train_label=categorical(train_label');
validation_label=label((distribute(1)*(ceil(181/stride))*(ceil(181/stride))+1):(distribute(2)*(ceil(181/stride))*(ceil(181/stride))));
validation_label=categorical(validation_label');
test_label=label((distribute(2)*(ceil(181/stride))*(ceil(181/stride))+1):(distribute(3)*(ceil(181/stride))*(ceil(181/stride))));

layers = [ ...
    imageInputLayer([patch_size patch_size 1], 'Name','input')
    convolution2dLayer(5,48, 'Name','conv_1')
%     batchNormalizationLayer
    reluLayer('Name','Relu_1')
    maxPooling2dLayer(2,'Stride',2, 'Name','pool_1')
    convolution2dLayer(5,96, 'Name','conv_2')
%     batchNormalizationLayer
    reluLayer('Name','Relu_2')
    maxPooling2dLayer(2,'Stride',2, 'Name','pool_2')
    convolution2dLayer(4,700, 'Name','conv_3')
%     batchNormalizationLayer
    reluLayer( 'Name','Relu_3')
    maxPooling2dLayer(2,'Stride',2, 'Name','pool_3')
    dropoutLayer('Name','dropout')
    fullyConnectedLayer(4, 'Name','Fully_con')
    softmaxLayer( 'Name','softmax')
    classificationLayer('Name','classify')];


layers1 = [ ...
    imageInputLayer([patch_size patch_size 1], 'Name','input')
    convolution2dLayer(5,64, 'Name','conv_1')
%     batchNormalizationLayer
    reluLayer('Name','Relu_1')
    maxPooling2dLayer(2,'Stride',2, 'Name','pool_1')
    convolution2dLayer(5,128,'Name','conv_2')
%     batchNormalizationLayer
    reluLayer('Name','Relu_2')
    maxPooling2dLayer(2,'Stride',2, 'Name','pool_2')
    convolution2dLayer(4,512, 'Name','conv_3')
%     batchNormalizationLayer
    reluLayer( 'Name','Relu_3')
    convolution2dLayer(2,1024,'Name','conv_4')
%     batchNormalizationLayer
    reluLayer('Name','Relu_4')
    dropoutLayer('Name','dropout')
    fullyConnectedLayer(4, 'Name','Fully_con')
    softmaxLayer( 'Name','softmax')
    classificationLayer('Name','classify')];

layers2 = [ ...
    imageInputLayer([patch_size patch_size 1],'Name','input')
    convolution2dLayer(5,64,'Name','conv_1')
%     batchNormalizationLayer
    reluLayer('Name','Relu_1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool_1')
    convolution2dLayer(5,128,'Name','conv_2')
%     batchNormalizationLayer
    reluLayer('Name','Relu_2')
%     maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,512,'Name','conv_3')
%     batchNormalizationLayer
    reluLayer('Name','Relu_3')
    convolution2dLayer(2,1024,'Name','conv_4')
%     batchNormalizationLayer
    reluLayer('Name','Relu_4')
    dropoutLayer('Name','dropout')
    fullyConnectedLayer(4, 'Name','Fully_con')
    softmaxLayer( 'Name','softmax')
    classificationLayer('Name','classify')];

layers3 = [ ...
    imageInputLayer([patch_size patch_size 1],'Name','input')
    convolution2dLayer(5,64,'Name','conv_1')
%     batchNormalizationLayer
    reluLayer('Name','Relu_1')
%     maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,128,'Name','conv_2')
%     batchNormalizationLayer
    reluLayer('Name','Relu_2')
%     maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,512,'Name','conv_3')
%     batchNormalizationLayer
    reluLayer('Name','Relu_3')
    convolution2dLayer(2,1024,'Name','conv_4')
     batchNormalizationLayer
    reluLayer('Name','Relu_4')
    dropoutLayer('Name','dropout')
    fullyConnectedLayer(4, 'Name','Fully_con')
    softmaxLayer( 'Name','softmax')
    classificationLayer('Name','classify')];

layers4 = [ ...
    imageInputLayer([patch_size patch_size 1],'Name','input')
    convolution2dLayer(5,64,'Name','conv_1')
%     batchNormalizationLayer
    reluLayer('Name','Relu_1')
%     maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,256,'Name','conv_2')
%     batchNormalizationLayer
    reluLayer('Name','Relu_2')
%     maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(4,768,'Name','conv_3')
%     batchNormalizationLayer
%      reluLayer
%     convolution2dLayer(2,1024)
%     batchNormalizationLayer
    reluLayer('Name','Relu_3')
    dropoutLayer('Name','dropout')
    fullyConnectedLayer(4, 'Name','Fully_con')
    softmaxLayer( 'Name','softmax')
    classificationLayer('Name','classify')];

options = trainingOptions('sgdm',...
    'InitialLearnRate',0.001,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.05,...
    'LearnRateDropPeriod',2,...
    'Momentum',0.9,...
    'MaxEpochs',5,...
    'MiniBatchSize',miniBatchSize,...
    'L2Regularization',0.001,...
    'ValidationData',{validation_patch,validation_label},...
    'ValidationPatience',30,...
    'Plots','training-progress');

%     'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

net = trainNetwork(train_patch,train_label,layers3,options);

predictedTestAngles = predict(net,test_patch);
[x,predictedTestAngles]=max(predictedTestAngles' );
predictedTestAngles=(predictedTestAngles==2 ).*1+(predictedTestAngles==3).*2+(predictedTestAngles==4)*3;
predictionError = test_label- predictedTestAngles;

thr = 1;
numCorrect = sum(abs(predictionError) <thr);
numTestImages = size(test_label,2);
accuracy = numCorrect/numTestImages;
disp('测试集准确率为：');
disp(accuracy);

seg_img=reshape(predictedTestAngles,ceil(181/stride),ceil(181/stride),distribute(3)-distribute(2));
seg_img=(seg_img==1 ).*50+(seg_img==2).*100+(seg_img==3)*150;
test_label=reshape(test_label,ceil(181/stride),ceil(181/stride),distribute(3)-distribute(2));
test_label=(test_label==1 ).*50+(test_label==2).*100+(test_label==3)*150;
%%

for i=1:1:7
    figure(1);
    subplot(3,7,7+i);
    title(['人工',num2str(i)]);
    imshow(uint8(seg_img(:,:,i)));
    subplot(3,7,14+i);
    imshow(uint8(test_label(:,:,i)));
    title(['网络',num2str(i)]);
end

% mask1=(test_label==50)*1;
% mask2=(seg_img==50)*1;
% accuracy1=sum(sum(sum(mask1.*mask2)))/sum(sum(sum(mask1)));
% disp('准确率1为：')
% disp(accuracy1)
% 
% mask1=(test_label==100)*1;
% mask2=(seg_img==100)*1;
% accuracy2=sum(sum(sum(mask1.*mask2)))/sum(sum(sum(mask1)));
% disp('准确率2为：')
% disp(accuracy2)
% 
% mask1=(test_label==150)*1;
% mask2=(seg_img==150)*1;
% accuracy3=sum(sum(sum(mask1.*mask2)))/sum(sum(sum(mask1)));
% disp('准确率3为：')
% disp(accuracy3)

correct_pred_num=zeros(3);
sum_pred_num=zeros(3);
% accuracy=zeros(3);
dice_ratio=zeros(3);
for i=50:50:150
    mask1=(test_label==i)*1;
    mask2=(seg_img==i)*1;
    figure();
    correct_pred_num(i/50)=sum(sum(sum(mask1.*mask2)));
    sum_pred_num(i/50)=sum(sum(sum(mask1)));
%     accuracy(i/50)=correct_pred_num(i/50)/sum_pred_num(i/50);
    dice_ratio(i/50)=2*correct_pred_num(i/50)/(sum_pred_num(i/50)+sum(sum(sum(mask2))));
    disp(['dice_ratio',num2str(i/50),'为：']);
%     disp(accuracy(i/50));
    disp(dice_ratio(i/50));
end
accuracy_total=sum(correct_pred_num)/sum(sum_pred_num);
disp('总准确率为：');
disp(accuracy_total);
%     accuracy1=sum(sum(sum(mask1.*mask2)))/sum(sum(sum(mask1)));
% residuals = testAngles - predictedTestAngles;
% residualMatrix = reshape(residuals,,500,10);   %The test data groups images by digit classes 0–9 with 500 examples of each
% 
% figure
% boxplot(residualMatrix, ...
%     'Labels',{'0','1','2','3','4','5','6','7','8','9'})
% 
% xlabel('Digit Class')
% ylabel('Degrees Error')
% title('Residuals')