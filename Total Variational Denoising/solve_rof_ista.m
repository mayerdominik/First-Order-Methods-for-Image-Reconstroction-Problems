%solve the Total Variational Denoising Problem with Proximal Gradient
function [ustar,F,G] = solve_rof(img, g, lambda, term_maxiter, term_eps)
    [dim1, dim2] = size(g);

    % construct 2-dimensional gradient
    grad1 = diffop(dim1);
    grad2 = diffop(dim2); 
    Grad = [kron(grad2,eye(dim1)); kron(eye(dim2),grad1)];

    %arrays for plotting
    F = zeros(term_maxiter,1);
    G = zeros(term_maxiter,1);

    % calculate smooth part of objective function for plotting
    cal_f = @(u)((1/2)*norm(u-g,2)^2);

    g = g(:); % vectorize matrix

    %initialize variable for iteration
    vk = zeros(numel(g)*2,1);
    
    tau = 1/8; % step size, as norm of discrete Laplacian equals 8
    
    for k = 1:term_maxiter
        vkp1 = vk - tau * Grad * (Grad' * vk - (1/lambda) * g); % reparametrization of dual variable
        vkp1 = project_D(vkp1, dim1 * dim2, 2);
        delta = norm(vk-vkp1,+inf);

        %track SNR and function value F(x_k)
        j = reshape(g - lambda * Grad' * vkp1,[dim1,dim2]);
        F(k) = snr(img,j-img);
        G(k) = cal_f(j) + lambda*cal_TV(j);
        disp(['SNR at iteration ' num2str(k) ' = ' num2str(F(k))]);
        %disp(['F[' num2str(k) '] = ' num2str(G(k))]); 
        
        %update variables
        vk = vkp1;   
        
        if (delta < term_eps)
            break;
        end
    end
    %return primal solution
    vstar = vkp1;   
    ustar = g - lambda * Grad' * vstar;
    ustar = reshape(ustar,[dim1,dim2]);
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
