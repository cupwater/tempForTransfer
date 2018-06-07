run('/home/cupwater/opensource/vlfeat-0.9.20/toolbox/vl_setup')
addpath(genpath('/media/cupwater/software1/YTF'))

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

resAccuracies = zeros(5, 8);
% 
% color = ['b-'; 'g-'; 'k:'; 'c-'; 'k-'; 'y-'; 'm-'; 'r-'];

for modelIndex = 4:4
    if modelIndex == 1
        %load('/media/cupwater/software1/YTF/YTFFeas_caffeface.mat');
    elseif modelIndex == 2
        %load('/media/cupwater/software1/YTF/YTFFeas_sphereface.mat');
    elseif modelIndex ==3
        %load('/media/cupwater/software1/YTF/YTFFeas_insightface.mat');
    elseif modelIndex == 5
        %load('/media/cupwater/software1/YTF/YTFFeas_dlib.mat');
    elseif modelIndex == 4
       load('/media/cupwater/software1/YTF/YTFFeas_normface.mat');
    end
    
    feas = feas ./ repmat(sqrt(sum(feas'.^2))', 1, size(feas, 2));
    
    
    for methodIndex = 4:4
    
        if methodIndex == 5
            allScoreArray = 'CNNmodel/smallBN.txt';
        elseif methodIndex == 2
            allScoreArray =  'learning2rank/RQS.txt';
        elseif methodIndex == 1
            allScoreArray = 'noQuality/random.txt';
        elseif methodIndex == 5
            allScoreArray = 'caffeface/caffeface.txt';
        elseif methodIndex == 5
            allScoreArray =  'feasLength/feasLength.txt';
        elseif methodIndex == 5
            allScoreArray = 'learning2rank/RQS.txt';
        elseif methodIndex == 4
            allScoreArray =  'YTFscore_caffeface.txt';
        elseif methodIndex == 3
            allScoreArray = 'YTFscore_fineLastLayer.txt';
        end
        
    
        scoreFin = fopen(allScoreArray, 'r');
        score = textscan(scoreFin, '%f');
        score = score{1};
        
        if methodIndex == 4
            score = score / 10.0;
        end

        modelIndex
        methodIndex
        
        
        for sidx=1:10
            
            currentCompairLabels = compairLabels((sidx-1)*500+1:sidx*500);
            currentIdxPairs = idxPairs((sidx-1)*500+1:sidx*500,:);
            
%             
%             [scores, accuracy] = YTF_evaluation_selectedNum(feas, class, currentCompairLabels, currentIdxPairs, score, 0, 1);
%             res((modelIndex-1)*40 + (methodIndex-1)*10 + sidx).scores   = scores;
%             res((modelIndex-1)*40 + (methodIndex-1)*10 + sidx).accuracy = accuracy;
            
            
%         
            for choice=5:5
                for selectNum=1:1
                    
                    for lambdaIndex=1:10
                        lambda = (lambdaIndex-1) * 1
                        
                        if selectNum <3
                            [scores, accuracy] = YTF_evaluation_selectedNum(feas, class, currentCompairLabels, currentIdxPairs, score+0.10, 5*selectNum-1, choice, 0.2*lambda, 1);
                        else
                            [scores, accuracy] = YTF_evaluation_selectedNum(feas, class, currentCompairLabels, currentIdxPairs, score+0.10, 10*selectNum-1, choice, 0.2*lambda, 1);
                        end

                        res((sidx-1)*30 + (selectNum-1)*10 + lambdaIndex).scores   = scores;
                        res((sidx-1)*30 + (selectNum-1)*10 + lambdaIndex).accuracy = accuracy;
                        
                        % res( (sidx-1)*10 + (selectNum-1)*1 + lambdaIndex).scores   = scores;
                        % res( (sidx-1)*10 + (selectNum-1)*1 + lambdaIndex).accuracy = accuracy;
                    end
                end
            end
        end
        
%         for choiceIndex=1:1
%             for k=0:6
%                 [scores, accuracy] = YTF_evaluation_selectedNum(feas, class, compairLabels, idxPairs, score, 5*k-1, choiceIndex);
%                 res( (choiceIndex-1)*30+ k).scores   = scores;
%                 res( (choiceIndex-1)*30+ k).accuracy = accuracy;
%             end
%         end
        
        
        %[ap, roc, accuracy] = YTF_evaluation1(feas, class, compairLabels, idxPairs, score, selectNumber);
        %res((i-1)*8 + methodIndex+1).ap = ap;
        %res((i-1)*8 + methodIndex+1).roc = roc;
        %res((i-1)*8 + methodIndex+1).accuracy = accuracy;
        
    end

end
%  save('ytf_quality_randomselect_normface.mat', 'res');

% save('ytf_quality_normface.mat', 'res');