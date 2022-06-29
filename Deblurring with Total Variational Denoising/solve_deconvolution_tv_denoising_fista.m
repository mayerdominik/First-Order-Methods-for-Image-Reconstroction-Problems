%Solve Deblurring with TV-Denoising with FISTA
function [ustar,F,G] = solve_deconvolution_tv_denoising_fista(img, g, h, lambda, maxiter, maxiterdenoise,std)
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

    %initialize variables for iteration
    vk = zeros(dim1,dim2);
    wk = zeros(dim1,dim2);
    tk = 1;

    for k = 1:maxiter
        %p_L(y_k)
        vkp1 = wk - (2/l).*Grad_u(wk);
        vkp1 = solve_rof_fista(vkp1, 2*lambda/l, maxiterdenoise,0);
        vkp1 = reshape(vkp1,[dim1,dim2]);
        %project onto image domain
        vkp1 = proj_C(vkp1, 1,0);
        
       %plot SNR or function value F(x_k)
        F(k) = snr(img,vkp1-img);
        disp(['SNR in iteration' num2str(k) ' = ' num2str(F(k))]);
        G(k) = cal_f(vkp1) + lambda*cal_TV(vkp1);
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

    