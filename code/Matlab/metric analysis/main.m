%%��ʵ�����ݴ����ֵ�������
%Date:2021.1.16
%Author��������

%%���ݶ�ȡ����
%%x = xlsread('.xls','sheet1','a1:a73');    % ��ȡĳ��
%A(50,:)��ȡ���ǵ�50�е�Ԫ�أ�
%A(:,50)��ȡ���ǵ�50�е�Ԫ�ء�
Sheet1 = xlsread('D:\���ı�д\�޸ĸ�3\��չVIFB����(ָ����������).xlsx', 'Sheet1');
%Sheet2 = xlsread('F:\���ı�д\�޸ĸ�\Graydata.xlsx', 'Sheet1');
%fileK = fopen('F:\���ı�д\ʵ������\Kendall.txt','w');
%fileS = fopen('F:\���ı�д\ʵ������\SSE.txt','w');
%data = Sheet1(5,5) ;%%��ȡ�̶�������
% i = 1;
% Ymax = 0;
% C = Sheet1(i,1);
% while  C ~= 0 
%     i = i + 1;
%     Ymax = i;               
%     C = Sheet1(i,1);
% end
% ���ݴ�����
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
% xlswrite('D:\���ı�д\�޸ĸ�3\Kendall.xlsx',b);

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
% xlswrite('F:\���ı�д\�޸ĸ�\Gray Kendall.xlsx',b);
% ���������
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
xlswrite('D:\���ı�д\�޸ĸ�3\borda kendall��ָ�겻�䣩.xlsx',b);
%%����ƽ����
% for i = 1 : 20
%     currentdata = Sheet1(43:48,i);
%     SSE = mySSE(currentdata);
%     C(1,i)=  SSE;
% %    fprintf(fileS,'%.2f   ',SSE);
% end
% C= vpa(C,2);
% c=double(C);
% 
% xlswrite('F:\���ı�д\ʵ������\2021.5.13\SSE IV.xlsx',c,'Sheet1','A7:T7');
%%���ݼ�¼����