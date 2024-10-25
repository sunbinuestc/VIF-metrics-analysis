function coeff = myKendall(X , Y)  
% ����������ʵ�ֿϵ¶��ȼ����ϵ���ļ������  
%  
% ���룺  
%   X���������ֵ����  
%   Y���������ֵ����  
%  
% �����  
%   coeff������������ֵ����X��Y�����ϵ��  
  
  
if length(X) ~= length(Y)  
    error('������ֵ���е�ά�������');  
    return;  
end  
  
%��X��Ϊ�����У����X�Ѿ��������������κα仯��  
if size(X , 1) ~= 1  
    X = X';  
end  
%��Y��Ϊ�����У����Y�Ѿ��������������κα仯��  
if size(Y , 1) ~= 1  
    Y = Y';  
end  
  
N = length(X); %�õ����еĳ���  
XY = [X ; Y]; %�õ��ϲ�����  
C = 0; %һ���Ե��������  
D = 0; %��һ���Ե��������  
N1 = 0; %����X����ͬԪ���ܵ���϶���  
N2 = 0; %����Y����ͬԪ���ܵ���϶���  
N3 = 0; %�ϲ�����XY���ܶ���  
XPair = ones(1 , N); %����X������ͬԪ����ɵĸ����Ӽ���Ԫ����  
YPair = ones(1 , N); %����Y������ͬԪ����ɵĸ����Ӽ���Ԫ����  
cont = 0; %���ڼ���  
  
%����C��D  
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
  
%����XPair�и���Ԫ�ص�ֵ  
while length(X) ~= 0  
    cont = cont + 1;  
    index = find(X == X(1));  
    XPair(cont) = length(index);  
    X(index) = [];  
end  
%����YPair�и���Ԫ�ص�ֵ  
cont = 0;  
while length(Y) ~= 0  
    cont = cont + 1;  
    index = find(Y == Y(1));  
    YPair(cont) = length(index);  
    Y(index) = [];  
end  
  
%����N1��N2��N3��ֵ  
N1 = sum(0.5 * (XPair .* (XPair - 1)));  
N2 = sum(0.5 * (YPair .* (YPair - 1)));  
N3 = 0.5 * N * (N - 1);  
  
coeff = (C - D) / sqrt((N3 - N1) * (N3 - N2));  
  
end %����myKendall����