function SSE = mySSE(X)
%����������ʵ�ּ��㷽�����������ƽ����
%���룺X��ֵ����
%�����SSEΪX�����ڵ�����ƽ����
% ���������ڼ�����������ֵ�Ĳ�����������ڱ�ʵ����Ҫ�������ݶԱȣ���˱���������Ҫ��������������һ��
OriginalData = X;
OriginalData = OriginalData(:).';   %����������Ϊ����������Ϊmapminmax����������й�һ����
MappedData = mapminmax(OriginalData, 0, 1); %��һ��
valueall = 0;
result = 0;
num = length(MappedData);   %��ֵ���г���

for i = 1 : num
    valueall = valueall + MappedData(i)/num;
end

for i = 1 : num
    result = result + (MappedData(i) - valueall)^2;
end

SSE = result;
end