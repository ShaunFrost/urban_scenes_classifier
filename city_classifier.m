dataDir= './data/';
checkpointDir = 'modelCheckpoints';
rng(1) % For reproducibility
city = {'NYC', 'ROME', 'SF'};
cat = categorical(city);

data_folder = 'Aerial_Final_aug';

original_data_folder = 'Aerial_Final_scaled';

fprintf('Loading Train, Test and Validation Filenames and Label Data...'); t = tic;
imgs = imageDatastore(fullfile(dataDir,data_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
imgs.Labels = reordercats(imgs.Labels,city);
fprintf('Done in %.02f seconds\n', toc(t));



fprintf('Loading original image Filenames and Label Data...'); t = tic;
imgs1 = imageDatastore(fullfile(dataDir,original_data_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
imgs1.Labels = reordercats(imgs1.Labels,city);
% Split data
fprintf('Done in %.02f seconds\n', toc(t));

%%
% imageSize = [5616 3744 3];
% imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[0,360], ...
%     'RandScale',[1.5 2]);
% augimds = augmentedImageDatastore(imageSize,train,train.Labels,'DataAugmentation',imageAugmenter);

%%
rng('default');
randimgs = imgs;
layers=[
    imageInputLayer([234 351 3]); % Input to the network is a 128x128x1 sized image
    convolution2dLayer(5,40,'Padding',[2 2],'Stride', [1,1], 'Name', 'conv1');  % convolution layer with 40, 5x5 filters
    batchNormalizationLayer();
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    convolution2dLayer(5,40,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 40, 5x5 filters
    batchNormalizationLayer();
    reluLayer();
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    convolution2dLayer(5,80,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 80, 5x5 filters
    batchNormalizationLayer();
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    convolution2dLayer(5,80,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 80, 5x5 filters
    batchNormalizationLayer();
    reluLayer();
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    fullyConnectedLayer(100); % Fullly connected layer with 100 activations
    dropoutLayer(.25); % Dropout layer
    fullyConnectedLayer(3, 'Name', 'fclast'); % Fully connected with 17 layers
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];

val_acc = 0;
test_acc = 0;
train_acc = 0;
orig_acc = 0;
orig_ConfMat = zeros(3);
train_ConfMat = zeros(3);
for i = 1:1:10
    % Split data
    [train, val, test] = splitEachLabel(imgs,.7,.2,.1);
    imgs = shuffle(imgs);
    numEpochs = 15; % 5 for both learning rates
    batchSize = 10;
    nTraining = length(train.Labels);
    if ~exist(checkpointDir,'dir'); mkdir(checkpointDir); end
    % Set the training options
    options = trainingOptions('sgdm','MaxEpochs',20,...
        'InitialLearnRate',5e-4,...% learning rate
        'CheckpointPath', checkpointDir,...
        'MiniBatchSize', batchSize, ...
        'MaxEpochs',numEpochs, ...
        'Plots','training-progress');
    t = tic;
    [net1,info1] = trainNetwork(train,layers,options);
    fprintf('Trained in in %.02f seconds\n', toc(t));
    % Test on the validation data
    YVal = classify(net1,val);
    val_acc = val_acc + mean(YVal==val.Labels);
    % Test on the Testing data
    YTest = classify(net1,test);
    test_acc = test_acc + mean(YTest==test.Labels);
    % Test on train data
    YTrain = classify(net1, train);
    train_acc = train_acc + mean(YTrain==train.Labels);
    train_ConfMat = train_ConfMat + confusionmat(train.Labels, YTrain);
    
    %Test on original images
    YOrig = classify(net1, imgs1);
    orig_acc = orig_acc + mean(YOrig==imgs1.Labels);
    orig_ConfMat = orig_ConfMat + confusionmat(imgs1.Labels, YOrig);
end

%Uncomment this to save the network
%rash_convnet = net1;
%save rash_convnet;

test_acc = test_acc/10
train_acc = train_acc/10
val_acc = val_acc/10
orig_acc = orig_acc/10
orig_ClassMat = orig_ConfMat./(meshgrid(transpose(groupcounts((imgs1.Labels)))));
train_ClassMat = train_ConfMat./(meshgrid(transpose(groupcounts((train.Labels)))));

figure(1);
heatmap(cat, cat, round(train_ClassMat/10,2));
title('Classification Matrix : Original');

figure(12);
heatmap(cat, cat, round(orig_ClassMat/10,2));
title('Classification Matrix : Training');

sample_image = imread(imgs1.Files(1));
act1 = activations(net1,im,'conv1');

sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
figure(2);
I = imtile(mat2gray(act1),'GridSize',[8 8]);
imshow(I);
title('Visualize 1st Conv layer');

[maxValue,maxValueIndex] = max(max(max(act1)));
act1chMax = act1(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);
act1chMax = imresize(act1chMax,imgSize);
figure(3);
I = imtile({im,act1chMax});
imshow(I);
title('Visualize Strongest activation channel from 1st Conv layer');



%Visualize last FC layer
Y1 = tsne(activations(net1,...
    train,'fclast',"OutputAs","rows"));
figure(4);
gscatter(Y1(:,1), Y1(:,2), train.Labels, ...
    [], '.', 7, 'on');
title("Final Fully Connected layer activations: Train");
Y2 = tsne(activations(net1,...
    imgs1,'fclast',"OutputAs","rows"));
figure(5);
gscatter(Y2(:,1), Y2(:,2), imgs1.Labels, ...
    [], '.', 7, 'on');
title("Final Fully Connected layer activations: Test(Unaugmented)");
