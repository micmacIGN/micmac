% Test ATAN approximations

figure(1000) ; clf ; hold on ;
r=(-1:.0001:1) ;
r=r(2:end) ;
plot(r,abs( (0.2146.*r.^2-1).*r      -atan((1-r)./(1+r))+pi/4),'b')
plot(r,abs( (0.1963.*r.^2-0.9817).*r -atan((1-r)./(1+r))+pi/4),'r')

%find best approx
search=linspace(-0.968,-0.967,300) ;

E=[] ;
for d=search
  A = [ 1 1 1 ; -1 1 -1 ; 1 0 0] ;
  b = [-pi/4;pi/4;d] ; 
  c = inv(A) * b ;
  
  e = pi/4 + c(1)*r + c(2)*r.^2 + c(3)*r.^3 - atan((1-r)./(1+r))  ;
  E = [E max(abs(e))] ;
  hold on ; plot(r,abs(e)) ;
end

figure(1001) ; clf ;
plot(search,E) ;

[dr,i]=min(E) ;
b=[-pi/4;pi/4;search(i)] ;
c=inv(A)*b