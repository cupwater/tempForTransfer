function [scores, accuracy] = YTF_evaluation_averageConv(compairLabels, namePairs)
    for i = 1:length(compairLabels)
        name1 = namePairs{i, 1};
        name2 = namePairs{i, 2};
        
        feas1 = load(['/media/cupwater/software1/YTF_img/single_face_patch/' name1 name1(end-1:end) '_fc5_averageFeas.mat']);
        feas2 = load(['/media/cupwater/software1/YTF_img/single_face_patch/' name2 name2(end-1:end) '_fc5_averageFeas.mat']);
        
        feas1 = reshape(feas1.averageFeas, 512*1*1, 1);
        feas2 = reshape(feas2.averageFeas, 512*1*1, 1);
        
        scores(i) = feas1' * feas2 / (norm(feas1) * norm(feas2));
    end
    
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