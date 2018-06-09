function [ap, roc] = YTF_evaluation(features, class, compairLabels, idxPairs)
    features = features ./ repmat(sqrt(sum(features'.^2))', 1, size(features, 2));
    
    for i = 1:length(compairLabels)
        temp1 = [];
        temp2 = [];
        
        idx1 = idxPairs(i, 1) ;
        idx2 = idxPairs(i, 2) ;
       
        temp1_idx = find(class == idx1);
        temp2_idx = find(class == idx2);
       
        temp1 = features(temp1_idx, :);
        temp2 = features(temp2_idx, :);
        temp_score = temp1*temp2';
        scores(i) = mean(mean(temp_score));
    end
    
    % ap
    ap = evaluate('ap', scores, compairLabels);

    % roc
    roc = evaluate('roc', scores, compairLabels);
    
    bestThreshold = getThreshold(scores, compairLabels, 200);
    accuracy = getAccuracy(scores, compairLabels, bestThreshold)
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
            [res, extra] =ap(scores, gt);
        case 'roc'   
            [res, extra] = roc(scores, gt);
    end
        
    % measure name
    result.meas_name = config;

    % measure value (a scalar)
    result.measure = res;

    % extra data in a struct (e.g. optimal thresh), or empty
    result.extra = extra;
end

function [res, extra] = ap(scores, gt)

    [~,~,info] = vl_pr(gt, scores);
    
    res = info.auc * 100;
    extra = info;
end

function [res, extra] = roc(scores, gt)
    
    [~,~,info] = vl_roc(gt, scores);
    
    % the accuracy at the ROC operating point where the error rates are equal (as in [Guillaumin et al., ICCV '09])
    res = (1 - info.eer) * 100;
    extra = info;
end