function [ap, roc, accuracy] = YTF_evaluation1(features, class, compairLabels, idxPairs, FIQAscores, selectNumber)
    features = features ./ repmat(sqrt(sum(features'.^2))', 1, size(features, 2));
    
    for i = 1:length(compairLabels)
        temp1 = [];
        temp2 = [];
        
        idx1 = idxPairs(i, 1) ;
        idx2 = idxPairs(i, 2) ;
       
        temp1_idx = find(class == idx1);
        temp2_idx = find(class == idx2);
        
        [scores1,idx1]=sort(FIQAscores(temp1_idx));
        idx1 =temp1_idx(idx1(end-selectNumber:end));
        [scores2,idx2]=sort(FIQAscores(temp2_idx));
        idx2 =temp2_idx(idx2(end-selectNumber:end));
        
        %[score1, idx1] = max(FIQAscores(temp1_idx));
        %idx1 =temp1_idx(idx1);
        %[score2, idx2] = max(FIQAscores(temp2_idx));
        %idx2 =temp2_idx(idx2);
       
        temp1 = features(idx1, :);
        temp2 = features(idx2, :);
        temp_score = temp1*temp2';
        scores(i) = mean(mean(temp_score));
    end
    
    % ap
    ap = evaluate('ap', scores, compairLabels);

    % roc
    roc = evaluate('roc', scores, compairLabels);
    
    bestThreshold = getThreshold(scores, compairLabels, 200);
    accuracy = getAccuracy(scores, compairLabels, bestThreshold);
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