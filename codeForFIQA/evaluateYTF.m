run('/home/caoyushe/opensource/vlfeat-0.9.20/toolbox/vl_setup')

splits = fopen('/media/cupwater/software1/YTF/YTFdata/splits.txt', 'r');
index = 1;
while feof(splits) ~= 1
    line = fgetl(splits);
    line = regexp(line, ',', 'split');
    namePairs{index, 1} = line{3}(2:end);
    namePairs{index, 2} = line{4}(2:end);
    compairLabels(index) = str2num(line{5});
    idxPairs(index, 1) = 0;
    idxPairs(index, 2) = 0;
    index = index + 1;
end

compairLabels(find(compairLabels == 0)) = -1;

YTFlist = fopen('/media/cupwater/software1/YTF/YTFdata/landmark_YTF_inf.txt', 'r');
line = fgetl(YTFlist);
line = regexp(line, '\t', 'split');
lastName = regexp(line{1}, '/', 'split');
lastName = [lastName{1} '/' lastName{2}];

lastLabel = 1;


for i=1:length(namePairs)
    if strcmpi(namePairs{i,1},lastName) == 1
        idxPairs(i,1) = lastLabel;
    end
    if strcmpi(namePairs{i,2},lastName) == 1
        idxPairs(i,2) = lastLabel;
    end
end

index  = 1;
class(index) = lastLabel;
index = index+1;
while feof(YTFlist) ~= 1
    line = fgetl(YTFlist);
    line = regexp(line, '\t', 'split');
    
    currentName = regexp(line{1}, '/', 'split');
    currentName = [currentName{1} '/' currentName{2}];
    
    if ~strcmp(currentName, lastName)
        lastName = currentName;
        lastLabel = lastLabel + 1;
        for i=1:length(namePairs)
            if strcmpi(namePairs{i,1},lastName) == 1
                idxPairs(i,1) = lastLabel;
            end
            if strcmpi(namePairs{i,2},lastName) == 1
                idxPairs(i,2) = lastLabel;
            end
        end
            
    end
    class(index) = lastLabel;
    index = index+1;
end


for i = 1:4
    if i == 1
        load('/media/cupwater/software1/YTF/YTFFeas_caffeface.mat');
    elseif i == 2
        load('/media/cupwater/software1/YTF/YTFFeas_insightface.mat');
    elseif i ==3
        load('/media/cupwater/software1/YTF/YTFFeas_sphereface.mat');
    elseif i == 4
        load('/media/cupwater/software1/YTF/YTFFeas_dlib.mat');
    end
    
    [ap, roc] = YTF_evaluation(feas, class, compairLabels, idxPairs);

    fprintf('ap:                 %f\n', ap.measure);
    fprintf('eer:               %f\n', roc.measure);
    fprintf('auc:               %f\n', roc.extra.auc);

end