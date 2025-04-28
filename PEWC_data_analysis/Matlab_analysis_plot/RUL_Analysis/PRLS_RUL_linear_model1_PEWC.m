% This program is used to traking the RUL by projected RLS
% The model is choose as linear model (three order)
% y=phi+ax+b*1/(x+gamma)
clc; clear; close all;
addpath("..\Helpler_functions\");
%% load timelist 

PEWC_timelist_name="motor_time_list2_0208to0217PEWC";
file_path = fullfile('../../time_list_extraction/RUL_2/timelist_data/parquet/', PEWC_timelist_name.append('.parquet'));

warning('off','MATLAB:table:ModifiedVarnames');
PEWC_time_list = parquetread(file_path);
warning('on','MATLAB:table:ModifiedVarnames');

%% load and show HI
time = 1:length(PEWC_time_list.ElapsedTime)';
torque_timelist=PEWC_time_list.torque_time_list';

% reference health state 
health_ref= mean(torque_timelist(1:10));

EMA_HI=get_EMA(torque_timelist,0.1,health_ref);
HI_raw=torque_timelist;

figure(); hold on; grid minor;
plot(time, EMA_HI);plot(time, HI_raw);
xlabel('sample');ylabel('HI [p.u.]');


%% RUL setting 
% determine RUL threshold (drop by 20%)
RUL_thres=health_ref*0.8;

% if the data set not reach the RUL, choose the maximum HI for thres 
if max(EMA_HI)<RUL_thres
    RUL_thres=max(EMA_HI);
end 
% find failure index 
Fail_idx_MA=find(EMA_HI<=RUL_thres);
Fail_idx_MA=Fail_idx_MA(1);

% RUL trggering (MK test)
RUL_start_thres=health_ref*0.95;% satrt thres (drop by 2%)
RUL_start_idx=find(EMA_HI<=RUL_start_thres);
RUL_start_idx=RUL_start_idx(1); % index of ruls start (HI)

% initial constants 
phi0=EMA_HI(RUL_start_idx); C1=0; x_P0=0;

% find the strat index 
% tend detection by MK test 
MK_length=50;                           % MK window length 
[S,Z] = rolling_MK(EMA_HI,MK_length,0); % MK test 
MK_thres=3;                             % MK confidence value for triggering 

% two-step RUL triggering 
RUL_start_idx=find(abs(Z)>MK_thres & EMA_HI'<RUL_start_thres);
RUL_start_idx=RUL_start_idx(1);

%nonlinear  regression 
% 設定初始點 (注意：gamma 設為非零值，例如 1)
startPoints = [health_ref, 0, 0, 1];

% Linear model 1
% lin_mod = 'phi0+alpha*x+beta*exp(-(x*gamma))';
% % 設定 fitoptions 並限制參數範圍
% fo = fitoptions('Method', 'NonlinearLeastSquares', ...
%     'Lower', [-inf, -inf, -inf,    0], ...   % phi0 無限制，alpha 無下限但上限為0，beta和gamma必須非負
%     'Upper', [inf,  0,    inf,  inf], ...
%     'StartPoint', startPoints);

% Linear model 2
lin_mod = 'phi0+alpha*x+beta*log2(x+gamma)';
fo = fitoptions('Method', 'NonlinearLeastSquares', ...
    'Lower', [-inf, -inf, -inf,    0], ...   % phi0 無限制，alpha 無下限但上限為0，beta和gamma必須非負
    'Upper', [inf,  0,    0,  inf], ...
    'StartPoint', startPoints);

% fit the model
ft=fittype(lin_mod,'options',fo);
f1=fit(time(1:MK_length)',EMA_HI(RUL_start_idx-MK_length+1:RUL_start_idx)',ft)
phi0=f1.phi0;alpha=f1.alpha;beta=f1.beta;, gamma=f1.gamma;

% show triggering result 
figure();
subplot(2,1,1);hold on;
plot(time,EMA_HI);
% plot(time(RUL_start_idx-MK_length+1:RUL_start_idx), phi0+C1*exp(x_P0*(1:MK_length)),LineWidth=1.5);
plot(time(RUL_start_idx-MK_length+1:RUL_start_idx), ...
     phi0 + alpha*(1:MK_length) + beta*exp(-(1:MK_length)*gamma), ...
     'LineWidth', 1.5);
xline(time(RUL_start_idx),'r--');
xline(time(Fail_idx_MA),'k--');
yline(RUL_start_thres,'r--');
title('HI');grid minor;

subplot(2,1,2);hold on;
plot(time, Z);
xline(time(RUL_start_idx),'r--');
yline(time(MK_thres),'r--');grid minor;
xline(time(Fail_idx_MA),'k--');
title(append('MK value, window length: ',string(MK_length) ));


%% RLS tracking 
% Set RUL triggering value
N=Fail_idx_MA-RUL_start_idx+1; % sample number 
t=(RUL_start_idx):Fail_idx_MA;% time steps from RUL trigger point 
RLS_bond_mult=0.1;
% exponetial RLS tracking (begining by NRstart)


% lin_mod = 'phi0+alpha*x+beta*exp(-(x+gamma))';
%%% very important! t0 of RUL should be zero  
Beta=0.95;
h_exp=[ones(1,N)' ((1:length(t))+MK_length)' exp(-((1:length(t))+MK_length)*gamma)'];
h_exp=[ones(1,N)' ((1:length(t))+MK_length)' exp(-((1:length(t))+MK_length)*gamma)'];

%fake HI
% EMA_HI=2*exp(0.001*(1:length(EMA_HI)));
P0=1*eye(length(3));% initial covariance matrix
x_initial=[phi0;alpha;beta]; % RLS initial guess by NLR

% normal RLS
% boundary is | X1_upperbond X2_upperbond ...|
%             | X1_lowerbond X2_upperbond ...|

RLS_bond_L=[log(C1*0.1); -1];
RLS_bond_U=[log(C1*2); -x_P0*0.1];

% RUL estimation
[X_exp,y_hat_exp,P,MSE_exp]=RLS_functionsContainerPEWC.RLS_covR_bond_3( ...
    EMA_HI(t),h_exp,Beta,P0,500000,x_initial,RLS_bond_L, RLS_bond_U);

%% Comparison of EKF and RLS parameter 
% plot the result 
figure(); 

% fitted curve 

subplot(4,1,1); hold on;
plot(EMA_HI(RUL_start_idx:end)); 
plot(X_est(:,1)+phi0);
plot(y_hat_exp); 
subtitle('y/hat');

% model parameter1 
subplot(4,1,2);hold on;
plot((X_exp(1,:)));plot((X_exp(1,:)).*h_exp(:,1)');
subtitle('phi0');

% model parameter2  
subplot(4,1,3);hold on;
plot((X_exp(2,:)));plot((X_exp(2,:)).*h_exp(:,2)');
subtitle('alpha');

% model parameter3  
subplot(4,1,4);hold on;
plot((X_exp(3,:)));plot((X_exp(2,:)).*h_exp(:,2)');
subtitle('beta');

%% plot the curve fit
figure();hold on;
plot(EMA_HI);% row HI

% plot non linear regression result  
NLR_curv=phi0 + alpha*(1:MK_length) + beta*exp(-(1:MK_length)*gamma);
plot(RUL_start_idx-MK_length+1:RUL_start_idx,NLR_curv,'r',LineWidth=1.5);
% plot(RUL_start_idx-MK_length+1:RUL_start_idx,NLR_curv,'r',LineWidth=1.5);
% plot RLS fit result
RLS_fit=y_hat_exp;
RLS_fit(1)=NLR_curv(end); % connect 2 curves 
plot(t,y_hat_exp,'k--','LineWidth',1);% fitted HI
plot(t,X_est(:,1)+phi0,'r--','LineWidth',1);% estimate HI state 

xlabel('sample'); ylabel('HI');
axis([RUL_start_idx-MK_length+1 Fail_idx_MA -inf inf]);
grid minor;
legend('HI',"NLR initial",append('RLS fitted ', '\beta=',string(Beta)),'EKF');

%% RUL estimation from the RLS result 
% get true RUL 
true_RUL_MA=flip((1:Fail_idx_MA-RUL_start_idx+1));
L_RUL=length(true_RUL_MA);

% Estimate RUL of EKF
[RUL,failure_time]=help_plot_RUL_exp(X_exp(:,1:end),RUL_thres,-MK_length,phi0);

RUL_RLS=zeros(1,length(h_exp));

tic;
for i_RLS=1:length(h_exp)

% 定義符號變數
syms x
% solve the equation of linear model 
eqn = X_exp(1,i_RLS) + X_exp(2,i_RLS)*x  + X_exp(2,i_RLS)*MK_length+ X_exp(3,i_RLS)*exp(-gamma*x-gamma*MK_length) == RUL_thres;
% 使用 solve 求解
RUL_RLS(i_RLS) = vpasolve(eqn, x, 1000);
end 

figure(); 
plot(RUL_RLS);
toc

% alpha bounds of true RUL
alpha=0.2;
% Compute the alpha bounds
alphaPlus = true_RUL_MA + alpha*true_RUL_MA;
alphaMinus = true_RUL_MA - alpha*true_RUL_MA;


RUL_limit_EKF=RUL_EKF;
for i=2:length(RUL_EKF)
if RUL_EKF(i)<0
RUL_limit_EKF(i)=RUL_limit_EKF(i-1);
end 
end 

%% show result

figure();hold on;grid on;
t_RUL_plot=t(1:length(RUL));

plot(t_RUL_plot,RUL);plot(t_RUL_plot,true_RUL_MA);plot(t_RUL_plot,RUL_EKF);plot(t_RUL_plot,RUL_boundary);
fill([(t_RUL_plot) fliplr(t_RUL_plot)], [alphaPlus fliplr(alphaMinus)], ...
    'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
legend( 'P-RLS RUL', 'True RUL','EKF RUL','P-RLS boundary', '+-20% region');
xlabel('Time [min]'); ylabel('RUL [min]');

% axis('tight');
axis([RUL_start_idx Fail_idx_MA 0 max(true_RUL_MA)*5]);

smape_value_RLS=get_SMAPE (true_RUL_MA(379:end-1), RUL(379:end-1));
smape_value_EKF=get_SMAPE (true_RUL_MA, RUL_EKF);

%% zoom in 
zoom_idx1=length(true_RUL_MA)-50+1;zoom_idx2=length(true_RUL_MA);
figure();hold on;grid on;
plot(RUL);plot(RUL_EKF);plot(true_RUL_MA);
fill([(1:length(true_RUL_MA)) fliplr(1:length(true_RUL_MA))], [alphaPlus fliplr(alphaMinus)], ...
    'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
axis([max(zoom_idx1,1) zoom_idx2 0 max(true_RUL_MA)]);
title(append(' zoom in RUL estimation of ',PEWC_timelist_name));
legend( 'P-RLS RUL', 'True RUL','EKF RUL','P-RLS boundary', '+-10% confidence interval');

%% do animation 2
videoObj = VideoWriter('rul_animation1.mp4','MPEG-4');
videoObj.FrameRate = 24; % 24 fps
open(videoObj);

figure(10);
title('Real-time estimation')
% pre define plot region 
% for i=1:500
for i=1:20:length(t)
% estimated HI curve HI_curv_est
% HI_TimeLine=((1:i)+MK_length) ;
% HI_TimeLine=[HI_TimeLine HI_TimeLine(end)+RUL(i)];
HI_TimeLine=linspace(0, i+MK_length+RUL(i), 1000);
HI_curv_est=phi0+exp(X_exp(1,i)+X_exp(2,i)*HI_TimeLine);

% plot real time model 
sgtitle(['Sample: ', num2str(i)]);
subplot(2,1,1);
plot(EMA_HI(1:i),LineWidth=1);
hold on;
% plot(exp(y_hat_exp(1:i))+phi0,'r','LineWidth',1);
plot(HI_TimeLine-MK_length,HI_curv_est,LineWidth=1,LineStyle="-.");
% mark the RUL point 
plot(HI_TimeLine(end),RUL_thres,'b*');
xline(HI_TimeLine(end),'k--');yline(RUL_thres,'r--');
% text(HI_TimeLine(end),RUL_thres,'Estimated Failure time')

hold off;grid minor;
axis([0.99 i+MK_length+max(RUL) 0 0.1]);
legend('Observed Health indicator','Estimation curve','Estimated Failure time'...
    , 'Location','southeast');

% plot real time estimation 
subplot(2,1,2);
plot(RUL(1:i),'b');hold on;
% plot(RUL_limit(1:i),'b');
plot(true_RUL_MA);
fill([(1:length(true_RUL_MA)) fliplr(1:length(true_RUL_MA))], [alphaPlus fliplr(alphaMinus)], ...
    'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
hold off;grid minor;
axis([1 true_RUL_MA(1) 0 max(true_RUL_MA)*2]);
xlabel('time'); ylabel('RUL');

%write current frame into video
frame = getframe(gcf);
writeVideo(videoObj, frame);
end 
% close video object 
close(videoObj);



%% do animation 2
videoObj = VideoWriter('rul_animation.avi');
videoObj.FrameRate = 24; % 設定幀率為 10 fps
open(videoObj);

figure;

for t = 1:20:length(EMA_HI)
    hold on;
    plot(EMA_HI(1:t));
    plot(X_exp)
    axis([0 length(EMA_HI) 0 max(EMA_HI)]); % set plot axis 
    title(['Frame: ', num2str(t)]);
    
    %write current frame into video 
    frame = getframe(gcf);
    writeVideo(videoObj, frame);
    hold off; 
end

% close video object 
close(videoObj);