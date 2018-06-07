function [ap, roc, accuracy, similarity, gt] = IJBA_evaluation(features, class, compareLabels, comparePairsClass, testClass, metadataIndex, FIQAscores, selectNum, choice)
    %features = features ./ repmat(sqrt(sum(features'.^2))', 1, size(features, 2));
    
    for i = 1:length(compareLabels)
        idx1 = comparePairsClass{1}(i);
        idx2 = comparePairsClass{2}(i);
       
        temp1_idx = find(testClass == idx1);
        temp1_idx = metadataIndex(temp1_idx);
        temp2_idx = find(testClass == idx2);
        temp2_idx = metadataIndex(temp2_idx);
        
        [scores1,idx1]=sort(FIQAscores(temp1_idx));
        selectNum1 = selectNum;
        if selectNum1 >= length(idx1) || selectNum1 == -1
            selectNum1 = length(idx1)-1;
        end
        
        
        [scores2,idx2]=sort(FIQAscores(temp2_idx));
        selectNum2 = selectNum;
        if selectNum2 >= length(idx2) || selectNum2 == -1
            selectNum2 = length(idx2)-1;
        end
        
        
%         threshold = 5;
%         index = find(scores1>threshold);
%         if length(index) < selectNum1
%             if length(index) == 0
%                 selectNum1 = 0;
%             else
%                 selectNum1 = length(index) - 1;
%             end
%         end
%         index = find(scores2>threshold);
%         if length(index) < selectNum2
%             if length(index) == 0
%                 selectNum2 = 0;
%             else
%                 selectNum2 = length(index) - 1;
%             end
%         end
        
        idx1 =temp1_idx(idx1(end-selectNum1:end));
        idx2 =temp2_idx(idx2(end - selectNum2:end));
        
        
        if choice==1   %% quality weighted pooling
  
            fuse_feature1 = scores1(end-selectNum1:end)'*features(idx1,:) ;
            fuse_feature1 = fuse_feature1 / sum(scores1(end-selectNum1:end));
            fuse_feature2 = scores2(end-selectNum2:end)'*features(idx2,:);
            fuse_feature2 = fuse_feature2 / sum(scores2(end-selectNum2:end));
            scores(i) = fuse_feature1 * fuse_feature2';
            scores(i) = scores(i) / sqrt((fuse_feature1 * fuse_feature1') * (fuse_feature2 * fuse_feature2'));
        elseif choice == 2  %%average pooling
            weights = ones(selectNum1+1, 1);
            fuse_feature1 = weights'*features(idx1,:) ;
            fuse_feature1 = fuse_feature1 / (selectNum1+1) ;
            weights = ones(selectNum2+1, 1);
            fuse_feature2 = weights'*features(idx2,:) ;
            fuse_feature2 = fuse_feature2 / (selectNum2+1) ;
            scores(i) = fuse_feature1 * fuse_feature2';
            scores(i) = scores(i) / sqrt((fuse_feature1 * fuse_feature1') * (fuse_feature2 * fuse_feature2'));
        elseif choice == 3  %% max pooling
            fuse_feature1 = features(idx1,:) ;
            fuse_feature1 = fuse_feature1 ./ repmat(sqrt(sum(fuse_feature1'.^2))', 1, size(fuse_feature1, 2));
            fuse_feature2 = features(idx2,:) ;
            fuse_feature2 = fuse_feature2 ./ repmat(sqrt(sum(fuse_feature2'.^2))', 1, size(fuse_feature2, 2));
            temp_score = fuse_feature1 * fuse_feature2';
            scores(i) = max(max(temp_score));
        
        elseif choice == 4
            weihts1 = showWeights(features(idx1,:), scores1(end-selectNum1:end), 0);
            weihts2 = showWeights(features(idx2,:), scores2(end-selectNum2:end), 0);
            fuse_feature1 = weights1' * features(idx1,:);
            fuse_feature1 = weights2' * features(idx2,:);
            scores(i) = fuse_feature1 * fuse_feature2';
        end
        
        if isnan(scores(i)) 
            estr = scores(1)+1
        end
        
        
       % scores(i) = fuse_feature1 * fuse_feature2';
       % scores(i) = scores(i) / sqrt((fuse_feature1 * fuse_feature1') * (fuse_feature2 * fuse_feature2'));
       
%         temp1 = features(idx1, :);
%         temp2 = features(idx2, :);
%         temp_score = temp1*temp2';
%         scores(i) = mean(mean(temp_score));
    end
    
    % ap
    ap = evaluate('ap', scores, compareLabels);

    % roc
    roc = evaluate('roc', scores, compareLabels);
    
    bestThreshold = getThreshold(scores, compareLabels, 200);
    accuracy = getAccuracy(scores, compareLabels, bestThreshold);
    similarity = scores;
    gt = compareLabels;
end


function bestThreshold = getThreshold(scores, flags, thrNum)
    accuracys  = zeros(2*thrNum+1, 1);
    thresholds = (-thrNum:thrNum) / thrNum;
    for i = 1:2*thrNum+1
        accuracys(i) = getAccuracy(scores, flags, thresholds(i));
    end
    bestThreshold = mean(thresholds(accuracys==max(accuracys)));
end

function accuracy = getAccuracy(scores, flags, threshold)
    accuracy = (length(find(scores(flags==1)>threshold)) + ...
                length(find(scores(flags~=1)<threshold))) / length(scores);
end


function result = evaluate(config, scores, gt)

    scores = reshape(scores, 1, []);

    switch config     
        case 'ap' 
            [recall, precision, res, extra] =ap(scores, gt);
            result.recall = recall;
            result.precision = precision;
            
        case 'roc'   
            [tpr, tnr, res, extra] = roc(scores, gt);
            result.tpr = tpr;
            result.tnr = tnr;
    end
        
    % measure name
    result.meas_name = config;

    % measure value (a scalar)
    result.measure = res;

    % extra data in a struct (e.g. optimal thresh), or empty
    result.extra = extra;
    
end

function [recall, precision, res, extra] = ap(scores, gt)

    [recall, precision,info] = vl_pr(gt, scores);
    
    res = info.auc * 100;
    extra = info;
end

function [tpr, tnr, res, extra] = roc(scores, gt)
    
    [tpr,tnr, info] = vl_roc(gt, scores);
    
    % the accuracy at the ROC operating point where the error rates are equal (as in [Guillaumin et al., ICCV '09])
    res = (1 - info.eer) * 100;
    extra = info;
end