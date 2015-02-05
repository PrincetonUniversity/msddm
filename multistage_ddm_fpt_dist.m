function [T,Y, Yplus, Yminus]=multistage_ddm_fpt_dist(a,s,threshold,x0,x0dist,deadlines,tfinal)


%Input:
% a = vector of drift rates
% s = vector of diffusion rates
% z = threshold
% x0= discretized initial condition support set
% x0dist= discretized pdf of the initial condition (equal to 1 if x0 is deterministic)
% deadlines = set of deadlines, first element is zero
% tfinal = support for decision time =[0 tfinal]
% Pick tfinal sufficiently high


%Output:
% T = support of decision time
% Y = cdf of decision time


stages=length(deadlines);

% probability of decision before current stage

weight=0;

weight_p=0;

weight_m=0;

Y=[];

Yplus=[];

Yminus=[];

T=[];


for stage=1:stages-1
    
    z=threshold(stage);
    
    % deadline for the stage
    
    t_end = deadlines(stage+1)-deadlines(stage);
    
    % FPT distribution for the DDM at current stage
    
    [cdf, cdf_p, cdf_m, t] = ddm_fpt(a(stage),s(stage),z,x0,x0dist,t_end);
    
    % Appending the FPT dist at current stage to previous stage
    
    Y=[Y weight+cdf*(1-weight)];
    
    Yplus=[Yplus weight_p+cdf_p*(1-weight)];
    
    Yminus=[Yminus weight_m+cdf_m*(1-weight)];
    
    T=[T t+deadlines(stage)];
    
    % Computation of the initial distribution for next stage
    
    [x0, x0dist , ~ , ~, ~, ~] = dist_deadline(a(stage),s(stage),x0, x0dist,t_end,z, 0);
    
    if length(x0)>1
        step=x0(2)-x0(1);
    else
        step=1;
    end
    
    pinst_plus=sum(x0dist(x0>=threshold(stage+1)))*step;
    
    pinst_minus=sum(x0dist(x0<=-threshold(stage+1)))*step;
    
    if threshold(stage+1)< threshold(stage)
        x0dist(x0 > threshold(stage+1))=[];
        x0dist(x0 < -threshold(stage+1))=[];
        x0(x0 > threshold(stage+1))=[];
        x0(x0 < -threshold(stage+1))=[];
        x0dist=x0dist/(sum(x0dist)*step);
    else
        x0dist=[x0dist(1:end-1) 0*(x0(end):x0(2)-x0(1): threshold(stage+1))];
        x0dist=[0*(-threshold(stage+1):x0(2)-x0(1): x0(1)) x0dist(2:end)];
        x0=[x0(1:end-1) x0(end):x0(2)-x0(1): threshold(stage+1)];
        x0=[-threshold(stage+1):x0(2)-x0(1): x0(1) x0(2:end)];
    end
    

    
    Y(end) = Y(end) + pinst_plus +pinst_minus;
    Yplus(end)=Yplus(end) + pinst_plus;
    Yminus(end) = Yminus(end) + pinst_minus;
    
    weight=Y(end);
    
    weight_p=Yplus(end);
    
    weight_m=Yminus(end); 
    
end

% Final stage


% Times for the final stage
t_end = tfinal-deadlines(stages);

z=threshold(stages);

% cdf for the final stage
[cdf, cdf_p, cdf_m,t] = ddm_fpt(a(stages),s(stages),z,x0,x0dist,t_end);

% appending the current cdf to previous cdf

Y=[Y weight+cdf*(1-weight)];

Yplus=[Yplus weight_p+cdf_p*(1-weight)];

Yminus=[Yminus weight_m+cdf_m*(1-weight)];


T=[T t+deadlines(stages)];





function [cdf,cdf_p, cdf_m, t] = ddm_fpt(a,s,z,x0,x0dist,t_end)


% Support set for the density 

step_t=0.01;

t1= linspace(0.0001, 0.01,100);

step1=t1(2)-t1(1);

t2=t1(end)+step_t:step_t:t_end;

t=[t1, t2];

% Terms in the series solution to the density

N=-50:50;


% Lebegue measure for the discretized input density

if length(x0)>1
    step_in=x0(2)-x0(1);
else
    step_in=1;
end


% Initialization

prob=zeros(1,length(t));
prob_p=zeros(1,length(t));
prob_m=zeros(1,length(t));
X0=ones(length(N),1)*x0;
N=N'*ones(size(x0));



for kk=1:length(t)
    
    T=t(kk);
    
    % computation of the density of the ddm using series sum
    prob_vec_p= sum((exp(a*(z-X0)/s^2).*ssfunc(T, (X0+z)/s, 2*z/s,N))*exp(-a^2*T/2/s^2));
    prob_vec_m= sum((exp(a*(-z-X0)/s^2).*ssfunc(T, (z-X0)/s,2*z/s, N))*exp(-a^2*T/2/s^2));
    prob_vec = prob_vec_p + prob_vec_m;
    %sum((exp(a*(-z-X0)/s^2).*ssfunc(T, (z-X0)/s,2*z/s, N) + exp(a*(z-X0)/s^2).*ssfunc(T, (X0+z)/s, 2*z/s,N))*exp(-a^2*T/2/s^2));
    prob(kk)=sum(prob_vec.*x0dist)*step_in;
    prob_p(kk)=sum(prob_vec_p.*x0dist)*step_in;
    prob_m(kk)=sum(prob_vec_m.*x0dist)*step_in;
end

% converting pdf to cdf

cdf1 = cumsum(prob(1:length(t1)))*step1;

cdf2 =cdf1(end) + cumsum(prob(length(t1)+1:end))*step_t;

cdf=max(0,[cdf1, cdf2]);

cdf1p = cumsum(prob_p(1:length(t1)))*step1;

cdf2p =cdf1p(end) + cumsum(prob_p(length(t1)+1:end))*step_t;

cdf_p =max(0,[cdf1p, cdf2p]) ;

cdf1m = cumsum(prob_m(1:length(t1)))*step1;

cdf2m =cdf1m(end) + cumsum(prob_m(length(t1)+1:end))*step_t;

cdf_m = max(0,[cdf1m, cdf2m]) ;

if cdf(end)>1
    
    cdf=cdf/cdf(end);
    
end


function y=ssfunc(t, u, v, N)

% computation of special theta function

y = (v-u+ 2*N.*v)/(sqrt(2*pi)*t^1.5).*exp(-(v-u+2*N.*v).^2/2/t);








