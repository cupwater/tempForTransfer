function weights = showWeights(features, q, lambda)
    [num, d] = size(features);
    if num==1
        weights = 1.0;
    else
        A = zeros(1,num);
        b = 0;
        Aeq = ones(1,num);
        beq = 1;
        lb = zeros(1,num);
        ub = ones(1,num);
        ub = ub * 0.9;

        temp = zeros(d,d);
        for i=1:num
            temp = temp + features(i,:)' * features(i,:);
        end

        s = features * temp * features';
        H = s + lambda * eye(num);
        H=(H+H')/2;

        f = -q' * features * features';

        weights = quadprog(H, f, A, b, Aeq, beq, lb, ub);
    end
    