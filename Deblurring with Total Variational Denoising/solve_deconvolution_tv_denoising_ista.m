%Solve Deblurring with TV-Denoising with Proximal Gradient
function [ustar,F,G] = solve_deconvolution_tv_denoising_ista(img, g, h, lambda, maxiter, maxiterdenoise,std)
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
    
    %value of smooth part of objective function
    cal_f = @(u)((1/2)*norm(imfilter(u,h,"replicate")-g,2)^2);

    %initialize variable for iteration
    vk = zeros(dim1,dim2);

    for k = 1:maxiter
        %p_L(y_k)
        vkp1 = vk - (2/l).*Grad_u(vk);
        vkp1 = solve_rof_fista(vkp1, 2*lambda/l, maxiterdenoise,0);
        vkp1 = reshape(vkp1,[dim1,dim2]);
        %project onto image domain
        vkp1 = proj_C(vkp1, 1,0);
       
        %plot SNR or function value F(x_k)
        F(k) = snr(img,vkp1-img);
        disp(['SNR in iteration' num2str(k) ' = ' num2str(F(k))]);
        G(k) = cal_f(vkp1) + lambda*cal_TV(vkp1);
        %disp(['F[' num2str(k) '] = ' num2str(G(k))]);  

        %update variable
        vk = vkp1;

    end
   ustar = vkp1;


end

    