from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim, contourf
from comet_ml import API, Experiment
import numpy as np

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
xent = -1 * np.ones(len(cfeed))

for expt in allexperiments:

  # filter experiments by name
  name = api.get_experiment_other(expt, 'Name')[0]
  if 'poison' not in name: continue

  # merge data into cfeed
  raw = api.get_experiment_metrics_raw(expt)
  for r in raw:
    if r['metricName'] != 'acc': continue
    xent[r['step']] = r['metricValue']

xent = xent.reshape(args.res, args.res)
contourf(cc1, cc2, xent)
colorbar()
experiment.log_figure()



