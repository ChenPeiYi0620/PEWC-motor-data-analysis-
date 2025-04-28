% This program is to calculate EKF result by hand 
% The estimation is limited to Exponetial model 
% f=a+b*exp(ck)
% estimation state xk=[ak+bk*exp(ckk) ak bk ck]
% Fk= [0 1 exp(ck) kbkexp(ckk); ...] 
function [y_hat,x_hat] = EKF_hand(x0,P0,zk)
% initialize first estimation 
x_hat=zeros(4,length(zk));x_hat(:,1)=x0;
Pk_1=P0;Mk=[];Kk=[];
Hk=[1 0 0 0];
y_hat=zeros(1,length(zk));

%Constant covariance Q,R
Qk=diag([0.0001;0.0001;0.0001;0.0001]);
Rk=0.0001;

% initilaize Fk_1
% Fk= [0 1 exp(ck) kbkexp(ckk); ...] 
Fk_1=[0 1 exp(x0(4)) x0(3)*exp(x0(4)); 0 1 0 0 ; 0 0 1 0 ;0 0 0 1];

% EKF start 
for i=2:length(zk)
% Predict the next step 
% ideally xk=[a+b*exp(ck) a b c] and  x(k+1)=[a+b*exp(ck)*exp(c) a b c]
x_hat(:,i)=[(x_hat(1,i-1)-x_hat(2,i-1))*exp(x_hat(4,i-1))+x_hat(2,i-1); x_hat(2,i-1) ;x_hat(3,i-1) ;x_hat(4,i-1)];
% x_hat(:,i)=[x_hat(2,i-1)+x_hat(3,i-1)*exp(x_hat(4,i-1)*(i+1)); x_hat(2,i-1) ;x_hat(3,i-1) ;x_hat(4,i-1)];
% x_hat(:,i)=x_hat(:,i)+sqrt(0.0001)*rand(4,1);
x_hat(:,i)=x_hat(:,i);
% calculate uncertainty matrix 
Mk=Fk_1*Pk_1*Fk_1+Qk;

%take in measurement zk
% calculate EKF 
Kk=Mk*Hk'/((Hk*Mk*Hk'+Rk));
x_hat(:,i)=x_hat(:,i)+Kk*(zk(i)-Hk*x_hat(:,i));

% for next step 
Pk_1=(eye(4)-Kk*Hk)*Mk;
Fk_1=[0 1 exp(x_hat(4,i)*i) i*x_hat(3,i)*exp(x_hat(4,i)*i); 0 1 0 0 ; 0 0 1 0 ;0 0 0 1];

end 
end

