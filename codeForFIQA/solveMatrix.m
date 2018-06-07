function [a,error] = solveMatrix(F,q, lambda)
    % F is the matrix of features, the dimension is d * k
    %k = length(q);  % the dimension of q(the number of features)
    [d,k] = size(F);
    
    if k ~= length(q)
        return    %% the dimension is not match
    end

    S = F' * F;
    H = S*S + lambda * eye(k);
    
    f = q' * S;
    
    quadprog(H,f,A,b,Aeq,beq,lb,ub,x0)
    
    
    
    refFeas = F*a;
    error = 0;
    % compute the error
    for i=1:k
        temp = F(:,i)'*refFeas - q(i);
        error = error + 0.5 * temp*temp + 0.5 * lambda*(a'*a);
    end
    
    
    
end 





