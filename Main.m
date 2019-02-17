% This code calculates the reliability of a MEMS switch in its buckling
% mode. It is developed in a functional method so as one can replace the
% functions and use the algorithm for any physical problem. The
% calculations are conducted for both Hasofer-Lind and Monte Carlo methods.
% Please cite this piece of code if you use it in your research. Thank you!
%% 
clc
clear
close all
%% Initialization 
global x0 COV C k_P k_l;
x0 = [0.025;	7.80E+10;	0.03;	1000000.00;	1.04];
COV = 0.7*[0 0.1 0.1 0.1 0];
C = [1 0.8 0.7;
    0.8 1 0.6;
    0.7 0.6 1];
k_P = 15;
k_l = 15;
%% Evaluation
N = 10000;
[R_MC,gg] = MyReliability_MC(N); % Monte Carlo Reliability Calculation
[R_HL,g,beta_HL,ii,alpha_history] = MyReliability_HL(); % Hasofer-Lind Reliability Calculation

%% Post Processing
% HistogramPlot(gg);
CDFPlot(gg)
createfigure1(g(1:ii-1),'g(x)');
createfigure1(beta_HL(1:ii-1),'\beta');
createfigure2(alpha_history(1:ii-1,:));
%% Functions

function y = ExactFun() %Exact stress calculation
y = 0.71936;
end

function y = GradientFun() % Gradient calculation
y = [-28.82; 8.38e-12; 58.815; 6.5e-8; -0.96583];
end

function y = hessian() % Hessian matrix calculation
y = [2320	-3.36E-10	-2355.833333	-2.5E-06	37.91666667;
    -3.36E-10	-2.19E-23	6.36458E-10	1.5625E-18	-1.08E-11;
    -2355.833333	6.36458E-10	2.68E+03	9.16667E-06	-75.25;
    -2.5E-06	1.5625E-18	9.16667E-06	1.62E-23	2.31296E-18;
    37.91666667	-1.08E-11	-75.25	2.31296E-18	-1.51E+01];

end

function y = QuadApprox(x)% Quadratic approximation 
global x0
dx = x - x0';
sigma = ExactFun();
grad_sigma = GradientFun();
grad_2_sigma = hessian();
y = zeros(size(x,1),1);
for ii=1:1:size(x,1)
    y(ii) = sigma + dot(grad_sigma,dx(ii,:)) + 0.5 * dx(ii,:) *...
        grad_2_sigma * dx(ii,:)';
end
end

function g = StateLimit(x)% state limit by quadratic approximation
LoadFactor = QuadApprox(x);
g = 1-LoadFactor;
end

function [R,g] = MyReliability_MC(N) % reliability calculation
x = PointGenerator(N);
g = StateLimit(x);
R = length(g(g>0))/length(g);
end


function [R,g,betta_HL,i,alpha_history] = MyReliability_HL()
global COV x0 C k_l k_P
xm = x0';
k = 5;
n = 100;
x = zeros(n,k);
u = zeros(n,k);
betta_HL = zeros(n,1);
sigma = xm.*COV;
[V,~] = eig(C);
x(1,:) = xm;
% u(1,:) = RVNormalize(x(1,:),xm,sigma);
i = 1;
g = 3*ones(n,1);
% z = xm(2:4)
% % z = RVNormalize(z,xm(2:4),sigma(2:4))
% y = diag(sigma(2:4))*V*z' + xm(2:4)'
% y = V*z'




% [mu_eq_l ,sigma_eq_l] = normalEquivalent(x0(5),k_l,x(1,5));
% [mu_eq_P ,sigma_eq_P] = normalEquivalent(x0(1),k_P,x(1,1));
% sigma(1) = sigma_eq_P;
% sigma(5) = sigma_eq_l;
% xm(1) = mu_eq_P;
% xm(5) = mu_eq_l;

l = weibullGen(k_l,x0(5),100000);
P = weibullGen(k_P,x0(1),100000);

% 
yl = mle(l);
yP = mle(P);
sigma(1) = (yP(2));
sigma(5) = (yl(2));
xm(1) = yP(1);
xm(5) = yl(1);

dy = (gradient_HL(xm));
betta_HL(1) = (StateLimit(xm)-sum(gradient_HL(xm).*(xm-xm)))...
    ./sqrt(dot(dy.^2,sigma.^2));
betta_HL(1) = 2;
alpha = -dy.*sigma./...
    sqrt(dot(dy.^2,sigma.^2));

while 1
    z = x(i,2:4);
    z = RVNormalize(z,xm(2:4),sigma(2:4));
    y = diag(sigma(2:4))*V*z' + xm(2:4)';
    x(i,2:4) = y';
    x(i+1,:) = alpha.*betta_HL(i).*sigma + xm;
%     [mu_eq_l ,sigma_eq_l] = normalEquivalent(x0(5),k_l,x(i+1,5));
%     [mu_eq_P ,sigma_eq_P] = normalEquivalent(x0(1),k_P,x(i+1,1));
%     sigma(1) = sigma_eq_P;
%     sigma(5) = sigma_eq_l;
%     xm(1) = mu_eq_P;
%     xm(5) = mu_eq_l; 
    dy = (gradient_HL(x(i+1,:)));
     betta_HL(i+1) = (StateLimit(x(i+1,:))-dot(dy,(x(i+1,:)-xm)))...
    ./sqrt(dot(dy.^2,sigma.^2));
    alpha = -dy.*sigma./...
        sqrt(dot(dy.^2,sigma.^2));
    alpha_history(i,:) = alpha;
    g(i+1) = StateLimit(x(i+1,:));
    i = i+1;
    if abs(betta_HL(i-1)-betta_HL(i)) < 0.001
        break
    end
    if i > n
        break
    end
end
R = normcdf(betta_HL(i-1));
end


function x = PointGenerator(N)
global x0 COV C k_P k_l
Z = mvnrnd([0 0 0], C, N);
U = normcdf(Z,0,1);
z = [norminv(U(:,1),x0(2),COV(2)*x0(2)) norminv(U(:,2),x0(3),COV(3)*x0(3))...
    norminv(U(:,3),x0(4),COV(4)*x0(4))];
E = z(:,1);
t = z(:,2);
K = z(:,3);
l = weibullGen(k_l,x0(5),N);
P = weibullGen(k_P,x0(1),N);
x = [P E t K l];
end

function y = RVNormalize(x,xm,sigma)
y = (x - xm)./sigma;
end

function dy = gradient_HL(x)
k = 5;
dy = zeros(1,k);
Delta = .0001;
x1 = x;
for i = 1 : k
    x1(i) = x1(i) + (Delta) * x(i);
    dy(i) = (StateLimit(x1) - StateLimit(x)) / (Delta * x(i) );
    x1(i) = x(i);
end  
end

function [mu_eq ,sigma_eq] = normalEquivalent(xi,k,mean)

lambda = mean/gamma(1+1/k);
[f] = wblpdf(xi,lambda,k);
[F] = wblcdf(xi,lambda,k);
if F == 0
    F = 0.000001;       
end
if f == 0          
    f = 0.000001;
end
sigma_eq = normpdf(norminv(F))/f;
mu_eq = xi - norminv(F)*sigma_eq;
end

function x = weibullGen(k,mean,N)
if nargin < 3
N = 1;
end
lambda = mean/gamma(1+1/k);
x_l = rand(N,1);
x = wblinv(x_l,lambda,k);
end

function createfigure1(Y1,y_label)
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create plot
plot(Y1,'Marker','o','LineWidth',2,'Color',[0 0 0]);

% Create xlabel
xlabel('Iteration');

% Create ylabel
ylabel(y_label,'FontSize',30);

box(axes1,'on');
grid(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontSize',20);
end

function createfigure2(YMatrix1)

figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(YMatrix1,'Marker','o','LineWidth',2,'Parent',axes1);
set(plot1(1),'DisplayName','P');
set(plot1(2),'DisplayName','E');
set(plot1(3),'DisplayName','t');
set(plot1(4),'DisplayName','K');
set(plot1(5),'DisplayName','l');

% Create xlabel
xlabel('Iteration');

% Create ylabel
ylabel('\alpha');

box(axes1,'on');
grid(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontSize',20);
% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.85190972259475 0.737947215809923 0.035416666402792 0.180291148242598]);

end

function CDFPlot(Y1)
%CREATEFIGURE(X1, Y1)
%  X1:  vector of x data
%  Y1:  vector of y data
% Create figure
figure;

% Create axes
axes1 = axes;
hold(axes1,'on');

% Create plot
h = cdfplot(Y1);
set(h,'linewidth',2);
set(h,'Color',[0 0 0]);
% Create ylabel
ylabel('Probability');

% Create xlabel
xlabel('Limit State Function Value');

% Uncomment the following line to preserve the X-limits of the axes
xlim(axes1,[-1 1.5]);
box(axes1,'on');
grid(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontSize',20);
end

function HistogramPlot(data1)
%CREATEFIGURE(data1)
%  DATA1:  histogram data
% Create figure
figure;

% Create axes
axes1 = axes;
hold(axes1,'on');

% Create histogram
histogram(data1,'LineStyle','none','NumBins',200);

% Create ylabel
ylabel('Incident Number');

% Create xlabel
xlabel('State Limit Function Value');

% Uncomment the following line to preserve the X-limits of the axes
xlim(axes1,[-1 1.5]);
box(axes1,'on');
grid(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontSize',20);
end