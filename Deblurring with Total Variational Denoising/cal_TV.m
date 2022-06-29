%calculate value of TV-norm for a given image
function result = cal_TV(g)
    [dim1, dim2] = size(g);
    g = g(:);
    grad1 = diffop(dim1);
    grad2 = diffop(dim2); 
    Grad = [kron(grad2,eye(dim1)); kron(eye(dim2),grad1)];
    a = reshape(Grad*g,[dim1,dim2,2]);
    b = sqrt(sum(a.^2,3));
    c = sum(b(:));
    result = c;
end

