%%本实验数据处理部分的主程序
%Date:2021.1.16
%Author：高云翔

%%数据读取部分
%%x = xlsread('.xls','sheet1','a1:a73');    % 读取某列
%A(50,:)提取的是第50行的元素，
%A(:,50)提取的是第50列的元素。
Sheet1 = xlsread('D:\论文编写\修改稿3\拓展VIFB数据(指标数量不变).xlsx', 'Sheet1');
%Sheet2 = xlsread('F:\论文编写\修改稿\Graydata.xlsx', 'Sheet1');
%fileK = fopen('F:\论文编写\实验数据\Kendall.txt','w');
%fileS = fopen('F:\论文编写\实验数据\SSE.txt','w');
%data = Sheet1(5,5) ;%%读取固定的行列
% i = 1;
% Ymax = 0;
% C = Sheet1(i,1);
% while  C ~= 0 
%     i = i + 1;
%     Ymax = i;               
%     C = Sheet1(i,1);
% end
% 数据处理部分
% for i = 1 : 16
%     currentdata = Sheet1(1:32,i);
%     for j = 1 : 16
%         otherdata = Sheet1(1:32,j);
%         Kendall = myKendall(currentdata,otherdata);
%         B(i,j) = Kendall; 
%       %  fprintf(fileK,'%.2f   ',Kendall);
%     end
%   %  fprintf(fileK,'\r\n\r\n');
% end
% B = vpa(B,2);
% b=double(B);
% xlswrite('D:\论文编写\修改稿3\Kendall.xlsx',b);

% for i = 1 : 13
%     currentdata = Sheet2(1:20,i);
%     for j = 1 : 13
%         otherdata = Sheet2(1:20,j);
%         Kendall = myKendall(currentdata,otherdata);
%         B(i,j) = Kendall; 
%       %  fprintf(fileK,'%.2f   ',Kendall);
%     end
%   %  fprintf(fileK,'\r\n\r\n');
% end
% B = vpa(B,2);
% b=double(B);
% xlswrite('F:\论文编写\修改稿\Gray Kendall.xlsx',b);
% 排序相关性
for i = 1 : 12
    currentdata = Sheet1(37:68,i);
    otherdata = Sheet1(37:68,14);
    Kendall = myKendall(currentdata,otherdata);
    B(1,i) = Kendall; 
      %  fprintf(fileK,'%.2f   ',Kendall);

  %  fprintf(fileK,'\r\n\r\n');
end
B = vpa(B,2);
b=double(B);
xlswrite('D:\论文编写\修改稿3\borda kendall（指标不变）.xlsx',b);
%%组内平方和
% for i = 1 : 20
%     currentdata = Sheet1(43:48,i);
%     SSE = mySSE(currentdata);
%     C(1,i)=  SSE;
% %    fprintf(fileS,'%.2f   ',SSE);
% end
% C= vpa(C,2);
% c=double(C);
% 
% xlswrite('F:\论文编写\实验数据\2021.5.13\SSE IV.xlsx',c,'Sheet1','A7:T7');
%%数据记录部分