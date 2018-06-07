run('/home/caoyushe/opensource/vlfeat-0.9.20/toolbox/vl_setup')


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

resAccuracies = zeros(5, 8);
prefix = '/media/cupwater/software1/IJBA/';
for i = 1:5
    if i == 1
        load([prefix 'IJBAfeatures/IJBA11FeasCaffeface.mat']);
    elseif i == 2
        load([prefix 'IJBAfeatures/IJBA11FeasSphereface.mat']);
    elseif i ==3
        load([prefix 'IJBAfeatures/IJBA11FeasInsightface.mat']);
    elseif i == 4
        load([prefix 'IJBAfeatures/IJBA11Feasdlib.mat']);
    elseif i == 5
        load([prefix 'IJBAfeatures/IJBA11FeasNormface.mat']);
    end
    
   
    
    
    for methodIndex = 0:7
        
        if methodIndex == 0
            allScoreArray = 'CNNmodel/IJBAverificationScore.txt';
        elseif methodIndex == 1
            allScoreArray = 'learning2rank/IJBAverificationScore.txt';
        elseif methodIndex == 2
            allScoreArray = 'noQuality/IJBAverificationScore.txt';
        elseif methodIndex == 3
            allScoreArray = 'caffeface/IJBAverificationScore.txt';
        elseif methodIndex == 4
            allScoreArray = 'feasLength/IJBAverificationScore.txt';
        elseif methodIndex == 5
            allScoreArray = '/IJBAfeatures/IJBA11FeasScores.txt';
        elseif methodIndex == 6
            allScoreArray =  'IJBAfeatures/IJBA11CaffefaceScores.txt';
        elseif methodIndex == 7
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
        
       

        for sidx=1:10
            
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
            
            [ap, roc, accuracy, similarity, gt] = IJBA_evaluation(feas, class, compareLabels, comparePairsClass, testClass, metadataIndex, score);
            
            
            
            res((i-1)*8*10+10*methodIndex+sidx).ap = ap;
            res((i-1)*8*10+10*methodIndex+sidx).roc = roc;
            res((i-1)*8*10+10*methodIndex+sidx).accuracy = accuracy;
            res((i-1)*8*10+10*methodIndex+sidx).similarity = similarity;
            res((i-1)*8*10+10*methodIndex+sidx).gt = gt;
        end
        
%         res(methodIndex+1).ap = ap;
%         res(methodIndex+1).roc = roc;
%         res(methodIndex+1).accuracy = accuracy;
        
%         x = log(1-roc.tnr);
%         plot(x(1520:end), roc.tpr(1520:end), color(methodIndex+1, :), 'LineWidth', 2);
%         if methodIndex == 0
%             title('ROC Curve', 'FontSize', 2);
%             xlabel('False Positive Rate', 'FontSize', 2);
%             ylabel('True Positive Rate', 'FontSize', 2);
%         end
%         hold on
        
    end
    
%     title('ROC Curve', 'FontSize', 2);
%     xlabel('False Positive Rate', 'FontSize', 2);
%     ylabel('True Positive Rate', 'FontSize', 2);
%     grid on;
%     axis square;

end
save('IJBA11res_all.mat', 'res');