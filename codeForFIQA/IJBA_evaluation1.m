function [ap, roc, accuracy, similarity, gt] = IJBA_evaluation1(features, class, compareLabels, comparePairsClass, testClass, metadataIndex, FIQAscores, selectNum, choice, lambda)
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
        idx1 =temp1_idx(idx1(end-selectNum1:end));
        
        [scores2,idx2]=sort(FIQAscores(temp2_idx));
        selectNum2 = selectNum;
        if selectNum2 >= length(idx2) || selectNum2 == -1
            selectNum2 = length(idx2)-1;
        end
        idx2 =temp2_idx(idx2(end - selectNum2:end));        


        if choice==3
        
            A = zeros(1,selectNum1+1);
            b = 0;
            Aeq = ones(1,selectNum1+1);
            beq = 1;
            lb = zeros(1,selectNum1+1);
            ub = ones(1,selectNum1+1);
            %ub = ub;
            
            s = features(idx1,:) * features(idx1,:)';
            H = (s*s + (s*s)') / 2 + selectNum1 * eye(selectNum1+1);
            weights1 = quadprog(H, -2*scores1(end-selectNum1:end)'*s, A, b, Aeq, beq, lb, ub);
            fuse_feature1 = weights1'*features(idx1,:);

            
            A = zeros(1,selectNum2+1);
            b = 0;
            Aeq = ones(1,selectNum2+1);
            beq = 1;
            lb = zeros(1,selectNum2+1);
            ub = ones(1,selectNum2+1);
            %ub = ub-0.4;
            
            s = features(idx2,:) * features(idx2,:)';
            H = (s*s + (s*s)') / 2 + selectNum2 * eye(selectNum2+1);
            weights2 = quadprog(H, -2*scores2(end-selectNum2:end)'*s, A, b, Aeq, beq, lb, ub);
            fuse_feature2 = weights2'*features(idx2,:) ;
            
            
            
            scores(i) = fuse_feature1 * fuse_feature2';
        
        elseif choice==1   %% quality weighted pooling
  
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
        elseif choice == 4  %% max pooling
            fuse_feature1 = features(idx1,:) ;
            fuse_feature1 = fuse_feature1 ./ repmat(sqrt(sum(fuse_feature1'.^2))', 1, size(fuse_feature1, 2));
            fuse_feature2 = features(idx2,:) ;
            fuse_feature2 = fuse_feature2 ./ repmat(sqrt(sum(fuse_feature2'.^2))', 1, size(fuse_feature2, 2));
            temp_score = fuse_feature1 * fuse_feature2';
            scores(i) = max(max(temp_score));
        elseif choice == 5
            weights1 = showWeights(features(idx1,:), scores1(end-selectNum1:end), lambda);
            weights2 = showWeights(features(idx2,:), scores2(end-selectNum2:end), lambda);
            fuse_feature1 = weights1' * features(idx1,:);
            fuse_feature2 = weights2' * features(idx2,:);
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