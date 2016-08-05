expID = 'A01';
cropID = 'full';
cropDir = fullfile('data','FLIC', cropID);

webDir = 'webpages';
htmlPath = fullfile(webDir,[expID,'.html']);
imgDir = sprintf('result_%s',expID);
relImgDir = fullfile('..',imgDir);
nJoint = 10;
iwidth_fig = 200;
cropDir = fullfile('data','FLIC');
info = load(fullfile(cropDir,cropID,'info.mat'));

% =============================================
fprintf('Writing html documents\n');
fout = fopen(strrep(fullfile(htmlPath),'\','/'), 'w');

fprintf(fout, ['<html><head><title>Pose Estimatoion Results</title></head>\n']);
fprintf(fout, ['<h1>Qualitative results in FLIC</h1>\n']);
fprintf(fout, ['<h2>', expID '</h2>\n']);
% fprintf(fout, ['<h3>', train '</h3>\n']);
fprintf(fout, ['<h3>', 'test set' '</h3>\n']);
fprintf(fout, '<br><br>\n');

% >>>>> table start
fprintf(fout, '<table border="0">\n');


imlist = dir(fullfile(imgDir, '*_est.png'));
for iJoint = 1 : nJoint
    jointName = info.jointNames{iJoint};
    heatmaplist{iJoint} = dir(fullfile(imgDir, sprintf('*_%s.png',jointName)));
end

ncol = 6;
nrow = ceil((nJoint+1)/ncol);
for iImg = 1 : length(imlist)
    paths = cell(nJoint,1);
    paths{1} = strrep(fullfile(relImgDir,imlist(iImg).name),'\','/');
    for iJoint = 1 : nJoint
        paths{iJoint+1} = strrep(fullfile(relImgDir,heatmaplist{iJoint}(iImg).name),'\','/');
    end
    
    for r = 1 : nrow
        fprintf(fout, '<tr>\n');
        for c = 1 : ncol
            if(c+(r-1)*ncol > nJoint+1) break;  end
            fprintf(fout, '<td>');
            fprintf(fout, ['<img src="', paths{c+(r-1)*ncol}, '" width="', num2str(iwidth_fig), '" border="1"></a>']);
            fprintf(fout, '</td>');
        end
        fprintf(fout, '<tr>\n');
    end
    fprintf(fout, '</tr>\n');
end
fprintf(fout, '</table>\n');
% >>>>> table end
fclose(fout);
