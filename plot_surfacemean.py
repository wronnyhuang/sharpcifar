from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim, contourf
from comet_ml import API, Experiment
import numpy as np

experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='images', workspace="wronnyhuang")

cleanpoison = 'poison'
xentacc = 'xent'

api = API(rest_api_key='W2gBYYtc8ZbGyyNct5qYGR2Gl')
allexperiments = api.get('wronnyhuang/surface-mean')

class args:
  res = 30
  span = 2

# span
clin = args.span/2 * np.linspace(-1, 1, args.res)
data = {}

for i, expt in enumerate(allexperiments):

  # filter experiments by name
  name = api.get_experiment_other(expt, 'Name')[0]
  if cleanpoison not in name: continue

  # merge data into cfeed
  raw = api.get_experiment_metrics_raw(expt)
  for r in raw:
    if xentacc not in r['metricName']: continue
    rollout = name+'_'+r['metricName']
    if rollout not in data:
      data[rollout] = [None for _ in range(30)]
    data[rollout][r['step']] = r['metricValue']

  print(i, 'of', len(allexperiments))

mat = [rollout for rollout in data.values()]

from scipy.io import savemat
savemat(xentacc+'_'+cleanpoison+'.mat', {'data':data})




