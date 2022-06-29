function result = convop(g, s)
    [dim1, dim2] = size(g);
    c = (s-1)/2;
    a = spdiags(repmat(ones(dim2,1),1,s),-c:c,dim2,dim2);
    b = spdiags(repmat(ones(dim1,1),1,s),-c:c,dim1,dim1);
    result = kron(b.*(1/(s^2)),a);            
end