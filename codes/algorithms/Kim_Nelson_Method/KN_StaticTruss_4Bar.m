%KN method application
clc;
clear;

alpha = 0.05;
delta = 0.2;
n0 = 500;
k = 174;

eta = 0.5*((2*alpha/(k - 1))^(-2/(n0 - 1)) - 1);

hsq = 2*eta*(n0 - 1);
h = sqrt(hsq);
reps = zeros(n0,k);

for i = 1:k
    rng(100+i);
    reps(:,i) = StaticTruss_4Bar_Varun(i,n0);
end

sys_mean = zeros(k,1);
sys_std = zeros(k,k);
w = zeros(k,k);

for i = 1:k
    sys_mean(i,1) = mean(reps(:,i));
end 

for i = 1:k
    for j = 1:k
        if i == j
            sys_std(i,j) = 0;
            w(i,j) = 0;
        else
            sum = 0;
            for b = 1:n0
                sum = sum + (reps(b,i) - reps(b,j) - (sys_mean(i) - sys_mean(j)))^2;
            end
            sys_std(i,j) = (1/(n0-1))*sum;
            w(i,j) = max(0, (delta/(2*n0))*((hsq*sys_std(i,j)/(delta^2)) - n0) );
        end

    end
end

%disp(w)

%Initial screening
for i = 1:k
    a = 0;
    for j = 1:k
        if i == j
            a = a;
        else 
            if sys_mean(i) - sys_mean(j) - w(i,j) > 0
                a = a + 1;
            end
        end
    end
    if a == 0
        I(i,1) = i;
    else
        I(i,1) = 0;
    end
end

disp(I)

