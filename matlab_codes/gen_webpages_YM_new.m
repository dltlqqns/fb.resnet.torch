webDir = 'webpages';
htmlPath = fullfile(webDir,['compare.html']);
localInputImgDir = fullfile('..','..','deepcut','data','mpii-multiperson','images','test');
localResultDirs = {fullfile('..','..','deepcut','results_deepercut'),fullfile('..','..','faster_rcnn','results')};
hostDir = 'https://dl.dropboxusercontent.com/u/14617521/webpage_images';
repoResultDirs = {'results_deepercut','faster_rcnn'};
iwidth_fig = 500;

% =============================================
fprintf('Writing html documents\n');
fout = fopen(strrep(fullfile(htmlPath),'\','/'), 'w');

fprintf(fout, ['<html><head><title>Pose Estimatoion Results</title></head>\n']);
fprintf(fout, ['<h1>Qualitative results in MPII multi</h1>\n']);
fprintf(fout, '<br><br>\n');

% >>>>> table start
fprintf(fout, '<table border="0">\n');

imlist = dir(fullfile(localInputImgDir, '*.png'));
resultlist = cell(length(localResultDirs),1);
for iMethod = 1 : length(localResultDirs)
    resultlist{iMethod} = dir(fullfile(localResultDirs{iMethod},'*.png'));
end
for iImg = 1 : 80 %length(imlist)
    fprintf(fout, '<tr>\n');
    relTestImgPath = strrep(fullfile(hostDir,'mpii_multi','test',imlist(iImg).name),'\','/');
    
    % input images
    fprintf(fout, '<td>');
    fprintf(fout, ['<img src="', relTestImgPath, '" width="', num2str(iwidth_fig), '" border="1"></a>']);
    fprintf(fout, '</td>');
    
    % results
    for iMethod = 1 : length(repoResultDirs)
%         resultImagePath = fullfile('..',resultDirs{iMethod},resultlist{iMethod}(iImg).name);
        resultImagePath = fullfile(hostDir,repoResultDirs{iMethod},resultlist{iMethod}(iImg).name);
        fprintf(fout, '<td>');
        fprintf(fout, ['<img src="', resultImagePath, '" width="', num2str(iwidth_fig), '" border="1"></a>']);
        fprintf(fout, '</td>');
    end
    
    fprintf(fout, '</tr>\n');
end
fprintf(fout, '</table>\n');
% >>>>> table end
fclose(fout);
