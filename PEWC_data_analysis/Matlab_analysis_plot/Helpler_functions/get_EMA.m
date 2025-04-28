% function for calculating exponetial moving average 
function St = get_EMA(data,alpha, initial_value)
St=zeros(size(data));
%initialize EMA
St(1)=initial_value;
% calculate EMA step by step
for i=2:length(data)
St(i)=alpha*data(i)+(1-alpha)*St(i-1);
end 
end
