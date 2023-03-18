
clear


load diabetic;
% load diabetic_kernel.mat
% load diabetic_trainmat.mat
% load diabetic_testmat.mat
dim=size(diabetic,2);
dim=dim-1;



%-----------------------------------------------------
%   Learning and Learning Parameters

% c =a1(aa1);
% kerneloption=a2(aa2);


  % c =a1(aa1);
nbkernel=5;
C = 10;



epsilon = .000001;

verbose = 0;
Sigma=ones(1,nbkernel)/nbkernel;
hnodes=100;




	diabetic_train=diabetic;

	xapp=diabetic_train(1:600,1:(size(diabetic_train,2)-1));
	yapp=diabetic_train(1:600,(size(diabetic_train,2)));
	xtest=diabetic_train(601:1151,1:(size(diabetic_train,2)-1));
	ytest=diabetic_train(601:1151,(size(diabetic_train,2)));
	pos5=find(yapp==0);
	yapp(pos5)=-1;
	pos6=find(ytest==0);
	ytest(pos6)=-1;	
	[xapp,xtest]=normalizemeanstd(xapp,xtest);

for k=1:nbkernel

[kernel_train(:,:,k),H1_sum(:,:,k),H_test_sum(:,:,k)]=elmkernelgenerate(xapp,xtest,hnodes);

end;

ps=sumKbetamkl(kernel_train,Sigma);


H =ps.*(yapp*yapp'); 
 [alpha , b , pos] = monqp(H,ones(size(yapp)),yapp,0,C,epsilon,verbose,xapp,ps,[]);         

  xsup = xapp(pos,:);
 ysup = yapp(pos);



w =alpha.*ysup;

Hsup=H(:,pos);
ps1=H'*Hsup;
y=ps1*w+b;


% %------------------------Compute test data output----------------%



for j=1:nbkernel
H1=H1_sum(:,:,j);
Hsup=H1(:,pos);
H_test=H_test_sum(:,:,j);
ps3=H_test'*Hsup;
kernel_test(:,:,j)=ps3;

end;


ps2=sumKbetamkl(kernel_test,Sigma);

clear kernel_test

y1=ps2*w+b;
     err1=0;
   for l=1:size(xtest,1)
    if(ytest(l)==1)
        if(y1(l)<0)
            err1=err1+1;
        end
    else
        if(y1(l)>0)
            err1=err1+1;
        end
    end
   end
   
   accu1=((size(xtest,1)-err1)/size(xtest,1))*100
	sv1= length(pos);



