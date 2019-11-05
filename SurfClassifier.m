rootfolder = fullfile('101_ObjectCategories');
 filefolder = fullfile(rootfolder,'mugs');

 imgset = [ imageSet(fullfile(rootfolder,'mugs')),imageSet(fullfile(rootfolder,'laptop')),imageSet(fullfile(rootfolder,'stapler')) ];
[trainingSet,testSet] = partition(imgset,0.6,'randomize');

mug = readimage(trainingSet(1),5);
laptop = readimage(trainingSet(2),5);
stapler = readimage(trainingSet(3),5);

figure
subplot(1,3,1); imshow(mug);
subplot(1,3,2); imshow(laptop);
subplot(1,3,3);imshow(stapler);

bag = bagOfFeatures(trainingSet);

categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);

confMatrix = evaluate(categoryClassifier, trainingSets);

% confMatrix = evaluate(categoryClassifier, validationSets);

img = imread(fullfile(rootfolder, 'mugs', '130.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);

categoryClassifier.Labels(labelIdx)
