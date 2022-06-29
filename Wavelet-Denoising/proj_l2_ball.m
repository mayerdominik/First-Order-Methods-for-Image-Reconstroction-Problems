% projection on the unit ball with center c
function result = proj_l2_ball(u,c)

aux = norm(u-c,2);

if  aux >= 1
    result = (u-c)/aux + c;
else
    result = u;
end

end
