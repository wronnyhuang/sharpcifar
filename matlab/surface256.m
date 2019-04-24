names = {'acc_clean', 'acc_poison', 'xent_clean', 'xent_poison'};

for i = 1:4
  name = names{i}
  load(['/Users/dl367ny/repo/sharpcifar/matlab/' name '.mat'])
  data(data<0) = nan;
  if contains(name, 'clean')
    data(134:end, 6:32:end) = nan;
  end
  data = inpaint_nans(data, 1);
  save(['/Users/dl367ny/repo/sharpcifar/matlab/' name '.mat'], 'data');
  pause(1);
  if contains(name, 'xent')
    data = log10(data);
    name = [name ' (log)'];
  end
  subplot(2,2,i);
  surf(data, 'facealpha', .8, 'edgecolor', 'none');
  colorbar()
  title(strrep(name,'_','\_'))
end
saveas(gcf, ['svhn_surfaces.fig']);
saveas(gcf, ['svhn_surfaces.png']);