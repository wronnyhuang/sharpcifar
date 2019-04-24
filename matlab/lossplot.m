y1 = .5;
y2 = .5;
p1 = linspace(0,1);
p2 = linspace(0,1);
[pp2, pp1] = meshgrid(p1, p2);
loss =  - ( y1*log(pp1) + y2*log(pp2) );
contourf(p1, p2, loss'); hold on;
line([1,0],[0,1],'color','k'); hold off;
colorbar; axis image;