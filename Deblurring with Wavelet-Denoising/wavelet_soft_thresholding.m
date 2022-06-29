function result = wavelet_soft_thresholding(x, lambda, filter, L)
    %wavelet transform
    [c,s]=wavedec2(x,L,filter);

    %apply shrinkage operator to wavelet coefficients
    c_lambda = max(1-lambda./abs(c), 0).*c;

    %inverse wavelet transform
    result = waverec2(c_lambda,s,filter);
end