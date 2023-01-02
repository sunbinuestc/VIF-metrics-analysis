function coeff = myKendall(X , Y)  
% 本函数用于实现肯德尔等级相关系数的计算操作  
%  
% 输入：  
%   X：输入的数值序列  
%   Y：输入的数值序列  
%  
% 输出：  
%   coeff：两个输入数值序列X，Y的相关系数  
  
  
if length(X) ~= length(Y)  
    error('两个数值数列的维数不相等');  
    return;  
end  
  
%将X变为行序列（如果X已经是行序列则不作任何变化）  
if size(X , 1) ~= 1  
    X = X';  
end  
%将Y变为行序列（如果Y已经是行序列则不作任何变化）  
if size(Y , 1) ~= 1  
    Y = Y';  
end  
  
N = length(X); %得到序列的长度  
XY = [X ; Y]; %得到合并序列  
C = 0; %一致性的数组对数  
D = 0; %不一致性的数组对数  
N1 = 0; %集合X中相同元素总的组合对数  
N2 = 0; %集合Y中相同元素总的组合对数  
N3 = 0; %合并序列XY的总对数  
XPair = ones(1 , N); %集合X中由相同元素组成的各个子集的元素数  
YPair = ones(1 , N); %集合Y中由相同元素组成的各个子集的元素数  
cont = 0; %用于计数  
  
%计算C与D  
for i = 1 : N - 1  
    for j = i + 1 : N  
        if abs(sum(XY(: , i) ~= XY(: , j))) == 2   
            switch abs(sum(XY(: , i) > XY(: , j)))  
                case 0  
                    C = C + 1;  
                case 1  
                    D = D + 1;  
                case 2  
                    C = C + 1;  
            end  
        end  
    end  
end  
  
%计算XPair中各个元素的值  
while length(X) ~= 0  
    cont = cont + 1;  
    index = find(X == X(1));  
    XPair(cont) = length(index);  
    X(index) = [];  
end  
%计算YPair中各个元素的值  
cont = 0;  
while length(Y) ~= 0  
    cont = cont + 1;  
    index = find(Y == Y(1));  
    YPair(cont) = length(index);  
    Y(index) = [];  
end  
  
%计算N1、N2及N3的值  
N1 = sum(0.5 * (XPair .* (XPair - 1)));  
N2 = sum(0.5 * (YPair .* (YPair - 1)));  
N3 = 0.5 * N * (N - 1);  
  
coeff = (C - D) / sqrt((N3 - N1) * (N3 - N2));  
  
end %函数myKendall结束