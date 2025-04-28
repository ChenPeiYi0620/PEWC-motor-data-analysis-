% This program is aim to determine the MK value to find the trend in data 
% roll_length is the size of the rolling window
% epsilon is control term of sensitivity 

% function [S,Z] = rolling_MK(data,roll_length,epsilon)
function [S,Z] = rolling_MK(varargin)
if nargin ==3
    % MK test with original sample 
data=varargin{1};
roll_length=varargin{2};
epsilon=varargin{3};
% total MK values 
S=zeros(length(data),1);
Z=zeros(length(data),1);
L=length(data);
% MK test 
for i=1:(L-roll_length+1)
    [Si,Zi]=Get_MK_S(data(i:i+roll_length-1),epsilon);
    S(i+roll_length-1)=Si;
    Z(i+roll_length-1)=Zi;
end 
else 
% MK test with specific sample 
data=varargin{1};
roll_length=varargin{2};
epsilon=varargin{3};
n=varargin{4};
L=length(data);
% storage of the long term MK
long_MK_data_S=zeros(L,1);
long_MK_data_Z=zeros(L,1);
% resample by n  
dos_data = downsample(data,n);
% rolling MK test 
S=zeros(length(dos_data),1);
Z=zeros(length(dos_data),1);
for i=1:(length(dos_data)-roll_length+1)
    [Si,Zi]=Get_MK_S(dos_data(i:i+roll_length-1),epsilon);
    S(i+roll_length-1)=Si;
    Z(i+roll_length-1)=Zi;
end 
for i=1:length(data)
    if mod(i-1,n)==0
        long_MK_data_S(i)= S((i-1)/n+1);
        long_MK_data_Z(i)= Z((i-1)/n+1);
    else 
long_MK_data_S(i)=long_MK_data_S(i-1);
long_MK_data_Z(i)=long_MK_data_Z(i-1);
    end 
end 
% return 
S=long_MK_data_S;Z=long_MK_data_Z;
end 

end

function [Si,Zi]=Get_MK_S(roll_data,epsilon)
Si=0;
n=length(roll_data);
for k=1:n-1
    for j=k+1:n
        temp=roll_data(j)-roll_data(k);
        if temp>epsilon
            temp=1;
        elseif abs(temp)<=epsilon
            temp=0;
        else
            temp=-1;
        end
        Si=Si+temp;
    end
end
Var_S=1/18*n*(n-1)*(2*n+5);
if Si>epsilon
    Zi=(Si-1)/sqrt(Var_S);
elseif abs(Si)<=epsilon
    Zi=0;
else
    Zi=(Si+1)/sqrt(Var_S);
end
end