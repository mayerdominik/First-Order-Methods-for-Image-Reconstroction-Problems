%Solve the TV-Denoising Problem with FISTA
function ustar = solve_rof(g, lambda, term_maxiter, term_eps)
    [dim1, dim2] = size(g);

    % construct 2-dimensional gradient
    
    grad1 = diffop(dim1);
    grad2 = diffop(dim2); 
    Grad = [kron(grad2,eye(dim1)); kron(eye(dim2),grad1)];
   
    g = g(:); % vectorize matrix
    vk = zeros(numel(g)*2,1);
    wk = zeros(numel(g)*2,1);
    tk = 1;
    
    tau = 1/8; % step size, as norm of discrete Laplacian equals 8
    
    for k = 1:term_maxiter
        vkp1 = wk - tau * Grad * (Grad' * wk - (1/lambda) * g); % reparametrization of dual variable
        vkp1 = project_D(vkp1, dim1 * dim2, 2);
        delta = norm(vk-vkp1,+inf);
        
        %extrapolation
        tkp1 = (1 + sqrt(1 + 4*(tk^2)))/2;
        wk = vkp1 + (tk - 1)/tkp1 * (vkp1-vk);

        %update variables
        vk = vkp1;
        tk = tkp1;

        %disp(['Iteration #' num2str(k) ', delta = ' num2str(delta)]);        
        
        if (delta < term_eps)
            break;
        end
    end
    
    vstar = vkp1;   
    ustar = g - lambda * Grad' * vstar;
end

function result = project_D(v, numpix, numdim)
    v = reshape(v, numpix, numdim);
    vnorms = sqrt(sum(v.^2,2));
    
    % find indices where ||v_i|| > 1
    ind = find(vnorms > 1);
    
    result = v;
    result(ind,1) = v(ind,1) ./ vnorms(ind);
    result(ind,2) = v(ind,2) ./ vnorms(ind);
    
    result = result(:);
end
