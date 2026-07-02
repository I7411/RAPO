
%_________________________________________________________________________%
function [Best_score,Best_pos,curve]=GA(pop,Max_iter,lb,ub,dim,fobj)

popsize=pop;             
lenchrom=dim;         
fun = fobj;  
pc=0.6;                 
pm=0.001;                  
if(max(size(ub)) == 1)
   ub = ub.*ones(dim,1);
   lb = lb.*ones(dim,1);  
end
maxgen=Max_iter;   

bound=[lb,ub];  


for i=1:popsize
   
    GApop(i,:)=Code(lenchrom,bound);      
    
    [fitness(i)]=fun(GApop(i,:));            
end


[bestfitness bestindex]=min(fitness);
zbest=GApop(bestindex,:);   
gbest=GApop;                
fitnessgbest=fitness;       
fitnesszbest=bestfitness;  

for i=1:maxgen
        GApop=Select2(GApop,fitness,popsize);

        
        GApop=Cross(pc,lenchrom,GApop,popsize,bound);

      
        GApop=Mutation(pm,lenchrom,GApop,popsize,[i maxgen],bound);

        pop=GApop;
        
      for j=1:popsize
       
        [fitness(j)]=fun(pop(j,:));
 
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
        
     end
    
    curve(i)=fitnesszbest;     
end
Best_score = fitnesszbest;
Best_pos = zbest;
end
