% creates 1-dimensional sparse difference operator matrix for
% vectors with length n
function result = diffop(n)
    if (n < 1)
        error('n must be positive');
    end
    
    result = spdiags([-ones(n-1,1) ones(n-1,1); 0 1],0:1,n,n);            
end
