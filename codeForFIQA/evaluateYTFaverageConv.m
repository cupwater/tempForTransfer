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


for modelIndex = 1:1
    for methodIndex = 4:4
        for sidx=1:10   
            currentCompairLabels = compairLabels((sidx-1)*500+1:sidx*500);
            currentNamePairs = namePairs((sidx-1)*500+1:sidx*500, :);
            [scores, accuracy] = YTF_evaluation_averageConv(currentCompairLabels, currentNamePairs);
            res(sidx).scores   = scores;
            res(sidx).accuracy = accuracy;
        end
        
    end

end

%save('YTFres_allmodelsFIQAs_solveMatrix_lambda.mat', 'res');