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

strong_feature_percentages = [30, 50, 80, 90];

confMatrix1 = zeros(3,3);
confMatrix2 = zeros(3,3);
confMatrix3 = zeros(3,3);
img = readimage(imgs, 1);
figure(1);
for i = 1:1:size(strong_feature_percentages,2)
    bag = bagOfFeatures(trainingSet, 'StrongestFeatures', strong_feature_percentages(i)/100);
    featureVector = encode(bag, img);
    subplot(2,2,i);
    bar(featureVector);
    title(strcat(strcat('Visual word occurrences ', num2str(strong_feature_percentages(i))),' % strong features'));
    xlabel('Visual word index');
    ylabel('Frequency of occurrence');
    categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
    %confMatrix1 = confMatrix1 + evaluate(categoryClassifier, trainingSet);

    %confMatrix2 = confMatrix2 + evaluate(categoryClassifier, validationSet);

    confMatrix3 = evaluate(categoryClassifier, imgs2);
end
