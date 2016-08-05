expID = 'A01';
webDir = 'webpages';
htmlPath = fullfile(webDir,[expID,'.html']);
imgDir = expID;

iwidth_fig = 200;

% =============================================
fprintf('Writing html documents\n');
fout = fopen(strrep(fullfile(htmlPath),'\','/'), 'w');

fprintf(fout, ['<html><head><title>Pose Estimatoion Results</title></head>\n']);
fprintf(fout, ['<h1>Qualitative results in FLIC</h1>\n']);
fprintf(fout, ['<h2>', expID </h2>\n']);
fprintf(fout, '<br><br>\n');

% >>>>> table start
fprintf(fout, '<table border="0">\n');

imlist = dir(fullfile(imgDir, '*_img.png'));
for iJoint = 1 : nJoint
    jointName = info.jointNames{iJoint};
    heatmaplist{iJoint} = dir(fullfile(imgDir, sprintf('*_%s.png',jointName)));
end

for iImg = 1 : length(imlist)
    fprintf(fout, '<tr>\n');
    imgPath = strrep(fullfile(imgDir,sprintf('%04d_img.png',iImg)),'\','/');
    
    fprintf(fout, '<td>');
    fprintf(fout, ['<img src="', imgPath, '" width="', num2str(iwidth_fig), '" border="1"></a>']);
    fprintf(fout, '</td>');
    fprintf(fout, '<td>');
    fprintf(fout, ['<img src="', lshoPath, '" width="', num2str(iwidth_fig), '" border="1"></a>']);
    fprintf(fout, '</td>');
    
    fprintf(fout, '</tr>\n');
end
fprintf(fout, '</table>\n');
% >>>>> table end
fclose(fout);
