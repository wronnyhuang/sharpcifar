load('surface1d_clean_xent.mat');
[m, n] = size(mat);
nspan = 100;
noldspan = 30;
oldspan = linspace(-1,1,n);
span = linspace(-1,1,nspan);



bins = linspace(-5,25,50);
density = zeros(noldspan, numel(bins) - 1);
avg = zeros(1, noldspan);
for i = 1:noldspan
  density(i, :) = histcounts(log10(mat(:,i)), bins);
  avg(i) = nanmean(mat(:,i));
end

density = interp1(oldspan, density, span);
avg = interp1(oldspan, avg, span, 'pchip');
imagesc(span, bins, density'); hold on;
plot(span, log10(avg)); hold off;
set(gca, 'clim', [0 700]);
set(gca, 'ydir', 'normal')
