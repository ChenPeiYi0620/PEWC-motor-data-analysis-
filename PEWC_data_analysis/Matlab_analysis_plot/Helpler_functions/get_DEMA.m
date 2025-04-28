% function for calculating double exponetial moving average
function [at,bt,S1t,S2t] = get_DEMA(data,alpha)
S1t=zeros(size(data));% first EMA value
S2t=zeros(size(data));% second EMA value
at=zeros(size(data));
bt=zeros(size(data));
%initialize EMA
S1t(1)=data(1);
S2t(1)=data(1);

% calculate EMA step by step
for i=2:length(data)
    if i==2
        S1t(i)=alpha*data(i)+(1-alpha)*S1t(i-1);
    else% the sencond EMA is availabe when data is more than 3
        S1t(i)=alpha*data(i)+(1-alpha)*S1t(i-1);
        S2t(i)=alpha*S1t(i)+(1-alpha)*S2t(i-1);
        at(i)=2*S1t(i)-S2t(i);bt(i)=alpha/(1-alpha)*(S1t(i)-S2t(i));
    end
end
end
