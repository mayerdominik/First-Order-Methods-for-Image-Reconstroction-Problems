  %import images
  img = double(rgb2gray(imread('data/flower.bmp')))/255.0; 
  [dim1, dim2] = size(img);

  %parameters
  niter = 30;   %Number of iterations
  sigma = .1;  %Gaussian noise level
  lambda = .1; %Regulaization parameter

  %show original image
  figure; imshow(img); 
  imshow(img); title('input')
  pause

  %add noise
  imgn = img + randn(dim1,dim2)*sigma; figure; imshow(imgn);
  title(['With noise, SNR = ' num2str(round(snr(img,imgn-img),2))]);
  pause

  %solve with FISTA
  [imgdn,F1,G1] = solve_rof_fista(img,imgn, lambda, niter,0);figure;imshow(imgdn);
  title(['FISTA, lambda = ' num2str(lambda) ', SNR = ' num2str(round(snr(img,imgdn-img),2))]);
  pause
  
  %solve with ISTA
  [imgdn,F2,G2] = solve_rof_ista(img,imgn, lambda, niter,0);figure;imshow(imgdn);
  title(['Proximal Gradient, lambda = ' num2str(lambda) ', SNR = ' num2str(round(snr(img,imgdn-img),2))]);
  pause
  
  %plot SNR over iterations
  figure; plot([F1,F2]); xlabel('Iteration #');ylabel('SNR');legend({'FISTA','Proximal Gradient'},'Location','southeast');title('SNR');
  pause
  
  %plot function values over iterations
  figure; plot([G1,G2]); xlabel('Iteration #');ylabel('F(x_k)');legend('FISTA','Proximal Gradient');title('Objective Function');set(gca, 'YScale', 'log');
 

