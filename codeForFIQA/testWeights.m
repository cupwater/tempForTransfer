% get the feas 
load('/media/cupwater/software1/IJBA/IJBAfeatures/IJBA11FeasNormface.mat');
scoreFin = fopen('/media/cupwater/software1/IJBA/IJBAfeatures/IJBA11FeasScores.txt');
scores = textscan(scoreFin, '%f %f');
scores = (scores{1} + scores{2}) / 2;


feas = feas ./ repmat(sqrt(sum(feas'.^2))', 1, size(feas, 2));

startIndex = 1;
endIndex = 10;

for i=1:10
    weights = showWeights(feas(startIndex:endIndex, :), scores(startIndex:endIndex), 0.1*i)
end


for i=1:10
    weights = showWeights(feas(startIndex:endIndex, :), scores(startIndex:endIndex)+0.05, 0.1*i)
end


for i=1:10
    weights = showWeights(feas(startIndex:endIndex, :), scores(startIndex:endIndex)-0.05, 0.1*i)
end



for i=1:10
    weights = showWeights(feas(startIndex:endIndex, :), scores(startIndex:endIndex)+0.1, 0.1*i)
end



for i=1:10
    weights = showWeights(feas(startIndex:endIndex, :), scores(startIndex:endIndex)+0.2, 0.1*i)
end