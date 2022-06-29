  %solve deblurring with TV-denoising with FISTA and proximal gradient
  %import image
  img = double(rgb2gray(imread('data/flower.bmp')))/255.0; 
  [dim1, dim2] = size(img);

  %parameters
  std = 7;              %size of blurring matrix
  niter = 100;          %number of iterations
  niterdenoise = 10;    %number of iterations for the subproblem
  sigma = .02;          %noise level    
  lambda = .02;         %regularization parameter

  %show original image
  figure; imshow(img); 
  imshow(img); title('Original');
  pause

  %blur image
  h = (1/std^2)*ones(std,std);
  imgc = imfilter(img,h,"replicate");
  figure; imshow(imgc);
  title('blurred' );
  pause
    
  %add noise
  imgcn = imgc + randn(dim1,dim2)*sigma; 
  figure; imshow(imgcn);
  title(['Blurred and noisy, SNR = ' num2str(round(snr(img,imgcn-img),2))]);
  pause

  %solve with FISTA
  [imgdc,F1,G1] = solve_deconvolution_tv_denoising_fista(img, imgcn,h,lambda,niter,niterdenoise,std);figure; imshow(imgdc);
  title(['FISTA, \lambda = ' num2str(lambda) ', SNR = ' num2str(round(snr(img,imgdc-img),2))]);
  pause

  %solve with ISTA
  [imgdc,F2,G2] = solve_deconvolution_tv_denoising_ista(img, imgcn,h,lambda,niter,niterdenoise,std);figure; imshow(imgdc);
  title(['Proximal Gradient, \lambda = ' num2str(lambda) ', SNR = ' num2str(round(snr(img,imgdc-img),2))]);
  pause
  
  %plot SNR over iterations
  figure; plot([F1,F2]); xlabel('Iteration #');ylabel('SNR');legend({'FISTA','Proximal Gradient'},'Location','southeast');title('SNR');
  pause
  %plot function values over iterations
  figure; plot([G1,G2]); xlabel('Iteration #');ylabel('F(x_k)');legend('FISTA','Proximal Gradient');title('Objective Function');set(gca, 'YScale', 'log');
