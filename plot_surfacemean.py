from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim, contourf
from comet_ml import API, Experiment
import numpy as np
from scipy.io import savemat

experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='images', workspace="wronnyhuang")

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
  name = expt.name
  namename = 'poison' if 'poison' in name else 'clean'

  # merge data into cfeed
  raw = expt.metrics_raw
  for r in raw:
    if 'xent' not in r['metricName'] and 'acc' not in r['metricName']: continue
    metricname= 'xent' if 'xent' in r['metricName'] else 'acc'
    rollout = name+'_'+r['metricName']
    bucket = namename + '_' + metricname
    if bucket not in data:
      data[bucket] = {}
    if rollout not in data[bucket]:
      data[bucket][rollout] = [None for _ in range(30)]
    data[bucket][rollout][r['step']] = r['metricValue']

  print(i, 'of', len(allexperiments))

for bucket in data:
  mat = [rollout for rollout in data[bucket].values()]
  mat = np.array(mat)
  mat = mat.astype(float)
  # idx = mat == None
  # mat[idx] = -1
  savemat('surface1d_' + bucket + '.mat', mdict={'mat': mat})




