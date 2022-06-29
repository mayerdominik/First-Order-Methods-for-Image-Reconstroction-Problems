
  img = double(rgb2gray(imread('data/hibiscus.bmp')))/255.0; 
  [dim1, dim2] = size(img);

  %parameters
  filter = 'db3';   %wavelet filter
  L = 2;            %levels of discrete wavelet transform
  sigma = .05;      %noise level
  %lambda = .05;    %denoising parameter
  nlambda = 20;
  F = (1:nlambda)/(100*sigma*nlambda);
  G = zeros(nlambda,1);

  %show original image
%   figure; imshow(img); 
%   imshow(img); title('input')
%   pause
%   imwrite(img,'original2.bmp');
    
  %add noise
  imgn = img + randn(dim1,dim2)*sigma; figure; imshow(imgn);
  title('Image with noise' );
  pause
  
  %perform wavelet denoising for different values of lambda

  for k = 1:nlambda
    imgdn = wavelet_soft_thresholding(imgn, F(k), filter, L);
    G(k) = snr(img,imgdn-img);
  end
  [M,I] = max(G);
  lambda = F(I);
  
  
figure; plot(F,G); xlabel('\lambda');ylabel('SNR');title('Wavelet denoising');
  pause

  %show image with highest SNR
  imgdn = wavelet_soft_thresholding(imgn, lambda, filter, L); figure;imshow(imgdn);
  title(['Maximum SNR, \lambda = ' num2str(lambda) ', SNR = ' num2str(snr(img,imgdn-img))]);
 
 

