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
    
    %initialize variables for iterations
    vk = zeros(dim1,dim2);
    wk = zeros(dim1,dim2);
    tk = 1;

    for k = 1:maxiter
        %calculate quadratic minimizer p_L(y_k)
        vkp1 = wk - (1/l).*Grad_u(wk);
        vkp1 = wavelet_soft_thresholding(vkp1, lambda/l, filter, lvl);
        %project onto image domain
        vkp1 = proj_C(vkp1, 1,0);

        %plot SNR and function value F(x_k)
        F(k) = snr(img,vkp1-img);
        disp(['SNR in iteration' num2str(k) ' = ' num2str(F(k))]);
        G(k) = cal_f(vkp1) + lambda*norm(wavedec2(vkp1,lvl,filter),1);
        %disp(['F[' num2str(k) '] = ' num2str(G(k))]); 

        %extrapolation
        tkp1 = (1 + sqrt(1 + 4*(tk^2)))/2;
        wk = vkp1 + (tk - 1)/tkp1 * (vkp1-vk);

        %update variables
        vk = vkp1;
        tk = tkp1;

    end
   ustar = vkp1;


end
    