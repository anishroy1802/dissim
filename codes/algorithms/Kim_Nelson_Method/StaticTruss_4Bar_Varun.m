function [hfun]=StaticTruss_4Bar_Varun(para,n)
    % objective to minimize expectation of hfun
    % design variable = para,  para_lower=0;   para_upper=175;
    % n= number of realizations of the function to estimate expectation
    % output function evaltions  size (n,1)
    theta = randn(n,3); % three is the number of random variables
    
    len=1000;   %member length
    Area1=para;
    Area2=250-sqrt(2)*Area1;
    Coord=[0,0,0; 1,0,0; 2,0,0; 1,-1,0; 2,-1,0]*len;
    Con=[1,4; 2,4;3,4;5,4;];
    Reaction=[1,1,1;1 1 1;1 1 1;0,0,1;1 1 1;];
    Area=[Area1,Area2,Area1,Area2];
    Rho=ones(1,size(Con,1))*7860;
    
    for kk=1:n
        Load=[0,0,0;0 0 0;0 0 0;(100+20*theta(kk,1)) 0 0;0 0 0;];
        Elasticity1=200+30*theta(kk,2);
        Elasticity2=80+10*theta(kk,3);
        Elasticity=[Elasticity1,Elasticity2,Elasticity1,Elasticity2];
        D=struct('Coord',Coord','Con',Con','Re',Reaction','Load',Load','E',Elasticity','A',Area','R',Rho');
    
        w=size(D.Re);
        %Global Stiffness Matrix S
        S=zeros(3*w(2));
        %Global Mass Matrix M
        M=zeros(3*w(2));
        %Unrestrained Nodes U
        U=1-D.Re;
        %Location of unrestraind nodes f
        f=find(U);
        for i=1:size(D.Con,2)
           H=D.Con(:,i);
           C=D.Coord(:,H(2))-D.Coord(:,H(1));
           %Length of Element Le
           Le=norm(C);
           T=C/Le;
           s=T*T';
           e=[3*H(1)-2:3*H(1),3*H(2)-2:3*H(2)];
           % Stiffness for element i G=EA/Le 
           G=D.E(i)*D.A(i)/Le;
           S(e,e)=S(e,e)+G*[s -s;-s s];
           % Mass of element i
           K=(D.A(i)*D.R(i)*Le/2);
           M(e,e)=M(e,e)+K*[eye(3) zeros(3);zeros(3) eye(3)];
           Tj(:,i)=G*T;
        end
        U(f)=S(f,f)\D.Load(f);      %node displacements
        F=sum(Tj.*(U(:,D.Con(2,:))-U(:,D.Con(1,:))));   %member forces
        R=reshape(S*U(:),w);R(f)=0;         %reactions
        Stiff=S(f,f);   %stifness matrix
        Mass=M(f,f);    %mass matrix
        hfun(kk,1) = U(1,4);
    end
    end
    