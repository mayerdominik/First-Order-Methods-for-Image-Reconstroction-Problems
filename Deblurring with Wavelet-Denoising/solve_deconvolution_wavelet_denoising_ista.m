function [ustar,F,G] = solve_deconvolution_wavelet_denoising_fista(img, g, h, lambda, filter, lvl, maxiter,std)
    [dim1, dim2] = size(g);

    %lipschitz constant of gradient
    trans = conv_operator(g,std);
    trans = trans'*trans;
    l = norm(trans(:),2);

    %initialize Arrays for plotting
    F = zeros(maxiter,1);
    G = zeros(maxiter,1);

    %gradient of smooth part of the objective function
    Grad_u = @(u)(imfilter(imfilter(u,h,"replicate")-g,h,"replicate")); 

    %smooth part of objective function for plotting the function value
    cal_f = @(u)((1/2)*norm(imfilter(u,h,"replicate")-g,2)^2);
    
    %initialize variable for iterations
    vk = zeros(dim1,dim2);

    for k = 1:maxiter
        %p_L(y_k)
        vk = vk - (1/l).*Grad_u(vk);
        vk = wavelet_soft_thresholding(vk, lambda/l, filter, lvl);
        %project onto image domain
        vk = proj_C(vk, 1,0);

        %plot SNR and value F(x_k) of the objective function
        F(k) = snr(img,vk-img);
        disp(['SNR in iteration' num2str(k) ' = ' num2str(F(k))]);
        G(k) = cal_f(vk) + lambda*norm(wavedec2(vk,lvl,filter),1);
        %disp(['F[' num2str(k) '] = ' num2str(G(k))]);  

    end
   ustar = vk;


end

    