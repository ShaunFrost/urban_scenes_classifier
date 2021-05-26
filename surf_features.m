dataDir= './data/';
checkpointDir = 'modelCheckpoints';
rng(1) % For reproducibility
city = {'NYC', 'ROME', 'SF'};
cat = categorical(city);

data_folder = 'Aerial_Final_scaled_aug';
orig_folder = 'Aerial_Final';
orig_scaled = 'Aerial_Final_scaled';

fprintf('Loading Train, Test and Validation Filenames and Label Data...'); t = tic;
imgs = imageDatastore(fullfile(dataDir,data_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
imgs.Labels = reordercats(imgs.Labels,city);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Gray Train, Test and Validation Filenames and Label Data...'); t = tic;
imgs1 = imageDatastore(fullfile(dataDir,orig_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
imgs1.Labels = reordercats(imgs1.Labels,city);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Gray Train, Test and Validation Filenames and Label Data...'); t = tic;
imgs2 = imageDatastore(fullfile(dataDir,orig_scaled),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
imgs2.Labels = reordercats(imgs2.Labels,city);
fprintf('Done in %.02f seconds\n', toc(t));

[trainingSet, validationSet] = splitEachLabel(imgs,.7, 'randomize');

%strong_feature_percentages = [0.3, 0.5, 0.8, 0.9];

N= 1;
confMatrix1 = zeros(3,3);
confMatrix2 = zeros(3,3);
confMatrix3 = zeros(3,3);
for i = 1:1:N
    [trainingSet, validationSet] = splitEachLabel(imgs,.7, 'randomize');
    bag = bagOfFeatures(trainingSet, 'StrongestFeatures', 0.5);
    img = readimage(imgs, 1);
    featureVector = encode(bag, img);
    categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
    confMatrix1 = confMatrix1 + evaluate(categoryClassifier, trainingSet);
    confMatrix2 = confMatrix2 + evaluate(categoryClassifier, validationSet);
    confMatrix3 = confMatrix3 + evaluate(categoryClassifier, imgs2);
end

confMatrix1 = confMatrix1/N;
confMatrix2 = confMatrix2/N;
confMatrix3 = confMatrix3/N;


figure(1);
heatmap(cat, cat, confMatrix1);
title('Classification Matrix : Training');

figure(2);
heatmap(cat, cat, confMatrix3);
title('Classification Matrix : Original');

I = imread(char(imgs2.Files(1)));
points = detectSURFFeatures(rgb2gray(I));
figure(3);
imshow(I); hold on;
plot(points.selectStrongest(50));
hold off;
