function mpii_multi()

annoPath = '/home/yumin/dataset/MPII/mpii_human_pose_v1_u12_1.mat';
load(annoPath);

% % train single
% sel = RELEASE.img_train==1;
% savePath = 'train_single.h5';
% create_dataset(RELEASE, sel, 'single', savePath);
% val single
sel = RELEASE.img_train==1;
savePath = 'val_single.h5';
create_dataset(RELEASE, sel, 'single', savePath);
% test single
% sel = RELEASE.img_train==0;
% savePath = 'test_single.h5';
% create_dataset(RELEASE, sel, 'single', savePath);

% % train multi
% sel = RELEASE.img_train==1;
% savePath = 'train_multi.h5';
% create_dataset(RELEASE, sel, 'multi', savePath);
% % val multi
% sel = RELEASE.img_train==1;
% savePath = 'val_multi.h5';
% create_dataset(RELEASE, sel, 'multi', savePath);
% % test multi
% sel = RELEASE.img_train==0;
% savePath = 'test_multi.h5';
% create_dataset(RELEASE, sel, 'multi', savePath);

end

function create_dataset(RELEASE, sel, mode, savePath)

imgDir = '/home/yumin/dataset/MPII/images';
annolist = RELEASE.annolist(sel);
single = RELEASE.single_person(sel);

% Get dataset
cnt = 0;
imageNames = {};
bbs = [];
for iImg = 1 : 10 %numel(annolist)
    imageName = annolist(iImg).image.name;
    nPerson = numel(annolist(iImg).annorect);
    switch mode
        case 'single'
            lPerson = single{iImg};
        case 'multi'
            lPerson = setdiff(1:nPerson,single{iImg});
        otherwise
            error('wrong mode')
    end
    for iPerson = lPerson(:)'
        xs = [annolist(iImg).annorect(iPerson).annopoints.point.x];
        ys = [annolist(iImg).annorect(iPerson).annopoints.point.y];
        xs(isnan(xs))=[];  ys(isnan(ys))=[];
        xmin = min(xs);
        xmax = max(xs);
        ymin = min(ys);
        ymax = max(ys);
        
        %
        cnt = cnt + 1;
        imageNames{end+1} = imageName;
        bbs(end+1,:) = [xmin, ymin, xmax, ymax];
%         bid(end+1) = iPerson;
        %         joints(end+1,:) = []; % nan to -1?
        
        % debug. display
        if 1
            img = imread(fullfile(imgDir, imageName));
            figure(1); clf;
            imshow(img);
            rectangle('Position',[xmin,ymin,xmax-xmin+1,ymax-ymin+1],'EdgeColor','r','LineWidth',2);
        end
    end
end
assert(numel(imageNames)==cnt);
assert(size(bbs,1)==cnt);

items(1).name = 'imageNames';
items(1).data = imageNames;
items(2).name = 'bbs';
items(2).data = bbs;

% Save to hdf5 file
fid = H5F.create(savePath);
fileattrib(savePath,'+w');
plist = 'H5P_DEFAULT';
fid = H5F.open(savePath,'H5F_ACC_RDWR',plist);
for iItem = 1 : numel(items)
    dset_id = H5D.open(fid, ['/',items(iItem).name]);
    H5D.write(dset_id,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist,items(iItem).data);
    H5D.close(dset_id);
end
H5F.close(fid);

end