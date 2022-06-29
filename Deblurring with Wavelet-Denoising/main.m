  %solve deblurring with wavelet denoising with FISTA and proximal gradient
  %import image
  img = double(rgb2gray(imread('data\flower.bmp')))/255.0; 
  [dim1, dim2] = size(img);


  %parameters
  filter = 'db3';   %wavelet filter
  L = 3;            %levels of discrete wavelet transform
  sigma = .02;      %noise level
  lambda = .002;    %denoising parameter
  iter = 100;       %number of fista iterations
  std = 7;          %size of blurring matrix

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

% solve with FISTA
  [imgdc,F1,G1] = solve_deconvolution_wavelet_denoising_fista(img, imgcn,h,lambda, filter, L, iter, std);figure; imshow(imgdc);
  title(['FISTA, lambda = ' num2str(lambda) ', SNR = ' num2str(round(snr(img,imgdc-img),2))]);
  pause
  
% solve with Proximal Gradient
  [imgdc,F2,G2] = solve_deconvolution_wavelet_denoising_ista(img, imgcn,h,lambda, filter, L, iter, std);figure; imshow(imgdc);
  title(['ISTA, lambda = ' num2str(lambda) ', SNR = ' num2str(round(snr(img,imgdc-img),2))]);
  pause
  
  %plot SNR over iterations
  figure; plot([F1,F2]); xlabel('Iteration #');ylabel('SNR');legend({'FISTA','Proximal Gradient'},'Location','southeast');
  pause

  %plot function values over iterations
  figure; plot([G1,G2]); xlabel('Iteration #');ylabel('F(x_k)');legend('FISTA','Proximal Gradient');set(gca, 'YScale', 'log');
 

