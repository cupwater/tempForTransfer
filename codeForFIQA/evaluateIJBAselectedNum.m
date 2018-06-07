run('/home/cupwater/opensource/vlfeat-0.9.20/toolbox/vl_setup')


addpath(genpath('/media/cupwater/software1/YTF'))
% read the whole index 
csvFin = fopen('/media/cupwater/software1/IJBA/IJBAdata/IJBA11allOne.csv', 'r');
index = 1;
while feof(csvFin) ~= 1
    line = fgetl(csvFin);
    line = regexp(line, ',', 'split');
    class(index) = str2num(line{2}) ;
    personClass(index) = str2num(line{1});
    index = index+1;
end
fclose(csvFin);
color = ['b-'; 'g-'; 'k:'; 'c-'; 'k-'; 'y-'; 'm-'; 'r-'];

resAccuracies = zeros(1,8);
prefix = '/media/cupwater/software1/IJBA/';
for modelIndex = 4:4
    if modelIndex == 1
        load([prefix 'IJBAfeatures/IJBA11FeasCaffeface.mat']);
    elseif modelIndex == 2
        load([prefix 'IJBAfeatures/IJBA11FeasSphereface.mat']);
    elseif modelIndex ==3
        load([prefix 'IJBAfeatures/IJBA11FeasInsightface.mat']);
    elseif modelIndex == 5
        load([prefix 'IJBAfeatures/IJBA11Feasdlib.mat']);
    elseif modelIndex == 4
        load([prefix 'IJBAfeatures/IJBA11FeasNormface.mat']);
    end
    
    feas = feas ./ repmat(sqrt(sum(feas'.^2))', 1, size(feas, 2));
    
   
    
    
    for methodIndex = 4:4
        
        if methodIndex == 5
            allScoreArray = 'CNNmodel/IJBAverificationScore.txt';
        elseif methodIndex == 2
            allScoreArray = 'learning2rank/IJBAverificationScore.txt';
        elseif methodIndex == 1
            allScoreArray = 'noQuality/IJBAverificationScore.txt';
        elseif methodIndex == 5
            allScoreArray = 'caffeface/IJBAverificationScore.txt';
        elseif methodIndex == 5
            allScoreArray = 'feasLength/IJBAverificationScore.txt';
        elseif methodIndex == 5
            allScoreArray = '/IJBAfeatures/IJBA11FeasScores.txt';
        elseif methodIndex == 4
            allScoreArray =  'IJBAfeatures/IJBA11CaffefaceScores.txt';
        elseif methodIndex == 3
            allScoreArray = 'IJBAfeatures/IJBA11CaffefacefineLastLayerScores.txt';
        end
        
        scoreFin = fopen([prefix allScoreArray], 'r');
        if methodIndex >= 4 || methodIndex == 3
            score = textscan(scoreFin, '%f %f');
            score = (score{1} + score{2}) / 2;
        else 
            score = textscan(scoreFin, '%f');
            score = score{1};
        end
        
        if methodIndex == 4
            score = score / 10.0;
        end
        
        %zerosIndex = find(score==0);
        %score(zerosIndex) = 0.01;
        
        modelIndex
        methodIndex
        
       

        for sidx=1:10
            fprintf('split index %d \n', sidx);
            
            testClass = [];
            testPersonClass = [];
            metadataIndex = [];
            comparePairsClass = [];
            compareLabels = [];

            %read the current index 
            csvFin = fopen([prefix 'IJBAdata/IJBA11/split' num2str(sidx) '/verify_metadata_' num2str(sidx) '.csv'], 'r');
            index = 1;
            while feof(csvFin) ~= 1
                line = fgetl(csvFin);
                line = regexp(line, ',', 'split');
                testClass(index) = str2num(line{1});
                testPersonClass(index) = str2num(line{2}); % the testPersonClass only used for computing compareLabels
                index = index + 1;
            end
            fclose(csvFin);
            
            metadataIndexFin = fopen([prefix 'IJBAdata/IJBA11/metadata' num2str(sidx) '.txt'], 'r');
            metadataIndex = textscan(metadataIndexFin, '%d');
            metadataIndex = metadataIndex{1};
            metadataIndex = metadataIndex + 1;
            
            csvFin = fopen([prefix 'IJBAdata//IJBA11/split' num2str(sidx) '/verify_comparisons_' num2str(sidx) '.csv'], 'r');
            comparePairsClass = textscan(csvFin, '%d, %d');
            fclose(csvFin);
            
            % compute the labels for each compare pair
            for j = 1:length(comparePairsClass{1})
                p = find(testClass == comparePairsClass{1}(j));
                p1 = testPersonClass(p(1));
                
                p = find(testClass == comparePairsClass{2}(j));
                p2 = testPersonClass(p(1));
                
                if p1 == p2
                    compareLabels(j) = 1;
                else
                    compareLabels(j) = -1;
                end
            end
            
            
            

%             [ap, roc, accuracy, similarity, gt] = IJBA_evaluation(feas, class, compareLabels, comparePairsClass, testClass, metadataIndex, score, 0, 1);
%             res((modelIndex-1)*40 + (methodIndex-1)*10 + sidx).ap = ap;
%             res((modelIndex-1)*40 + (methodIndex-1)*10 + sidx).roc = roc;
%             res((modelIndex-1)*40 + (methodIndex-1)*10 + sidx).accuracy = accuracy;
%             res((modelIndex-1)*40 + (methodIndex-1)*10 + sidx).similarity = similarity;
%             res((modelIndex-1)*40 + (methodIndex-1)*10 + sidx).gt = gt;
            
            frameNumber = 2;
            lambdaNumber = 4;
            choiceNumber = 1;
            
            for choiceIndex=0:0
                choice = choiceIndex+5;
                for selectIndex=0:(frameNumber-1)
                    for lambdaIndex=1:2:2*lambdaNumber
                        lambda = lambdaIndex;
                        [ap, roc, accuracy, similarity, gt] = IJBA_evaluation1(feas, class, compareLabels, comparePairsClass, testClass, metadataIndex, score, 5*(selectIndex+1)-1, choice, 0.1*lambda);
                        
                        res((sidx-1)*80+selectIndex*lambdaNumber+lambdaIndex).ap = ap;
                        res((sidx-1)*80+selectIndex*lambdaNumber+lambdaIndex).roc = roc;
                        res((sidx-1)*80+selectIndex*lambdaNumber+lambdaIndex).accuracy = accuracy;
                        res((sidx-1)*80+selectIndex*lambdaNumber+lambdaIndex).similarity = similarity;
                        res((sidx-1)*80+selectIndex*lambdaNumber+lambdaIndex).gt = gt;

%                         res((sidx-1)*frameNumber*choiceNumber*lambdaNumber+choiceIndex*frameNumber*lambdaNumber+selectIndex*lambdaNumber+lambdaIndex).ap = ap;
%                         res((sidx-1)*frameNumber*choiceNumber*lambdaNumber+choiceIndex*frameNumber*lambdaNumber+selectIndex*lambdaNumber+lambdaIndex).roc = roc;
%                         res((sidx-1)*frameNumber*choiceNumber*lambdaNumber+choiceIndex*frameNumber*lambdaNumber+selectIndex*lambdaNumber+lambdaIndex).accuracy = accuracy;
%                         res((sidx-1)*frameNumber*choiceNumber*lambdaNumber+choiceIndex*frameNumber*lambdaNumber+selectIndex*lambdaNumber+lambdaIndex).similarity = similarity;
%                         res((sidx-1)*frameNumber*choiceNumber*lambdaNumber+choiceIndex*frameNumber*lambdaNumber+selectIndex*lambdaNumber+lambdaIndex).gt = gt;
                    end
                end
             
            end
            
        end
        
        
    end
    

end
save('IJBA11res_Proposed_lambda.mat', 'res');