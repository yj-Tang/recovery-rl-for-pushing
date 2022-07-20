% theta = 2;
% 
% rotM = [cos(theta) -sin(theta); sin(theta) cos(theta)]
% 
% F_fg = [2, 3]';

%% calculate the moment

syms x y fx fy theta real

% in the object frame
o_f = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)]*[fx, fy]';

m1 = cross([x,y,0], [o_f(1), o_f(2), 0]');

% in the world frame
o_dis = [cos(theta) -sin(theta); sin(theta) cos(theta)]*[x, y]';

m2 = cross([o_dis(1), o_dis(2), 0], [fx, fy, 0]');

m1-m2; % should be zero

%% test
func = @(x,y) x.*y.^2;
integral2(func,0,2,0,1);

%% double integrate
theta = 0;
a= -4.56;
b= 0.78;
xmin = -0.25;
xmax = 0.25;
ymin = -0.25;
ymax = 0.25;
vec = [x*cos(theta) - y*sin(theta) - a, x*sin(theta) + y*cos(theta) - b];
% syms x y
func_x = @(x,y) (x*cos(theta) - y*sin(theta) - a)./sqrt( (x*cos(theta) - y*sin(theta) - a).^2 + (x*sin(theta) + y*cos(theta) - b).^2 );
q_x = integral2(func_x,xmin,xmax,ymin,ymax);
func_y = @(x,y) (x*sin(theta) + y*cos(theta) - b)./sqrt( (x*cos(theta) - y*sin(theta) - a).^2 + (x*sin(theta) + y*cos(theta) - b).^2 );
q_y = integral2(func_y,xmin,xmax,ymin,ymax);
func_m = @Integrand_m;
q_m = integral2(func_m,xmin,xmax,ymin,ymax);

func_test = @Integrand_test;
q = integral2(func_test,xmin,xmax,ymin,ymax)

function p = Integrand_m(x, y)
    theta = 0;
    a= -4.56;
    b= 0.78;
% % %     syms x y
%     vec = [x*cos(theta) - y*sin(theta) - a; x*sin(theta) + y*cos(theta) - b];
%     norm_vec = vec ./ sqrt( (x*cos(theta) - y*sin(theta) - a).^2 + (x*sin(theta) + y*cos(theta) - b).^2 );
    norm_vec_1 = (x*cos(theta) - y*sin(theta) - a)./sqrt( (x*cos(theta) - y*sin(theta) - a).^2 + (x*sin(theta) + y*cos(theta) - b).^2 );
    norm_vec_2 = (x*sin(theta) + y*cos(theta) - b)./sqrt( (x*cos(theta) - y*sin(theta) - a).^2 + (x*sin(theta) + y*cos(theta) - b).^2 );
    W_x_1 = x.*cos(theta) - y.*sin(theta) ; 
    W_x_2 = x.*sin(theta) + y.*cos(theta);
    p = W_x_1.* norm_vec_1+ W_x_2.* norm_vec_2;
end

function p = Integrand_test(x, y)
    theta = 0;
    vec_o2rc_x= -4.56;
    vec_o2rc_y= 0.78;
    p = (x-theta)+vec_o2rc_x*y -vec_o2rc_y;
end

