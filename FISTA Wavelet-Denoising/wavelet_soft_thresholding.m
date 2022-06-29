function result = wavelet_soft_thresholding(x, lambda, filter, L)
    [c,s]=wavedec2(x,L,filter);
    c_lambda = max(1-lambda./abs(c), 0).*c;
    result = waverec2(c_lambda,s,filter);
end