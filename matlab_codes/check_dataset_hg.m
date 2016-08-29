
% stacked hourglass
%
hg_train = h5read('/home/yumin/codes/pose-hg-train/data/mpii/annot/train.h5','/imgname');
hg_train_unique = unique(hg_train);
%
hg_test = h5read('/home/yumin/codes/pose-hg-train/data/mpii/annot/test.h5','/imgname');
hg_test_unique = unique(hg_test);


% original dataset
% load('/home/yumin/dataset/MPII/mpii_human_pose_v1_u12_1.mat')
ltrain = find(RELEASE.img_train==1);
train = RELEASE.annolist(ltrain);
nTrain = length(train);
trainnames = cell(nTrain,1);
for i = 1 : nTrain
    trainnames{i} = train(i).image.name;
end
trainnames_unique = unique(trainnames);

ltest = find(RELEASE.img_train==0);
test = RELEASE.annolist(ltest);
nTest = length(test);
testnames = cell(nTest,1);
for i = 1 : nTest
    testnames{i} = test(i).image.name;
end
testnames_unique = unique(testnames);
sum(cellfun(@length,RELEASE.single_person(ltest)))