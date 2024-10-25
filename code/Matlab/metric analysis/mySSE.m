function SSE = mySSE(X)
%本函数用于实现计算方差分析的组内平方和
%输入：X数值序列
%输出：SSE为X序列内的组内平方和
% 本函数用于计算数组内数值的波动情况，由于本实验需要多组数据对比，因此本函数首先要对组内数据做归一化
OriginalData = X;
OriginalData = OriginalData(:).';   %将列向量变为行向量（因为mapminmax对行数组进行归一化）
MappedData = mapminmax(OriginalData, 0, 1); %归一化
valueall = 0;
result = 0;
num = length(MappedData);   %数值序列长度

for i = 1 : num
    valueall = valueall + MappedData(i)/num;
end

for i = 1 : num
    result = result + (MappedData(i) - valueall)^2;
end

SSE = result;
end