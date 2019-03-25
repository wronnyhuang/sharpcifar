from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim, contourf
from comet_ml import API, Experiment
import numpy as np

cleanpoison = 'poison'
xentacc = 'xent'

experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                  project_name='images', workspace="wronnyhuang")

api = API(rest_api_key='W2gBYYtc8ZbGyyNct5qYGR2Gl')
allexperiments = api.get('wronnyhuang/landscape2d')

class args:
  res = 256
  span = 2

# span
clin = args.span/2 * np.linspace(-1, 1, args.res)
cc1, cc2 = np.meshgrid(clin, clin)
cfeed = np.array(list(zip(cc1.ravel(), cc2.ravel())))
data = -1 * np.ones(len(cfeed))

for expt in allexperiments:

  # filter experiments by name
  name = api.get_experiment_other(expt, 'Name')[0]
  if cleanpoison not in name: continue

  # merge data into cfeed
  raw = api.get_experiment_metrics_raw(expt)
  for r in raw:
    if r['metricName'] != xentacc: continue
    data[r['step']] = r['metricValue']

data = data.reshape(args.res, args.res)
contourf(cc1, cc2, data)
colorbar()
print(experiment.log_figure()['web'])

from scipy.io import savemat
savemat(xentacc+'_'+cleanpoison+'.mat', {'data':data})




