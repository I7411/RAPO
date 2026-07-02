
%%
close all
clear 
clc
SearchAgents_no=50; 
Function_name=2;     
Max_iteration=500; 
lb=-100;             
ub=100;              
dim=10;              


fobj = @(x) cec22_test_func(x', Function_name);

[Best_score2,Best_pos2,GWO_cg_curve2]=GJO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score3,Best_pos3,GWO_cg_curve3]=GA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);



%Draw objective space
 figure

semilogy(GWO_cg_curve2,'g','linewidth',2,'MarkerSize',6)
hold on
semilogy(GWO_cg_curve3,'k','linewidth',2,'MarkerSize',6)

axis tight
grid on
box on
legend('GJO','GA','IGJOGA')

display(['The best solution obtained by GJO is : ', num2str(Best_pos2)]);
display(['The best optimal value of the objective funciton found by GJO is : ', num2str(Best_score2)]);

display(['The best solution obtained by GA is : ', num2str(Best_pos3)]);
display(['The best optimal value of the objective funciton found by GA is : ', num2str(Best_score3)]);


