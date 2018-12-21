import numpy as np
import glob
import os
import argparse

# if __name__=='__main__'
#
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--log_root', default=None, type=str, help='Directory of the checkpoint.')
#   args = parser.parse_args()
#
#   print(args.log_root)
#
#   if args.log_root == None:
#     pass
#   else:
#     _del_old_ckpt(args.log_root)

def _del_old_ckpt(log_dir):
  '''deletes the old checkpoints in log_root, leaving only the most recent one'''
  # todo update except section
  try:
    ckptstr = 'model.ckpt-'
    globstr = os.path.join(log_dir,ckptstr+'*')
    files = glob.glob(globstr)
    iteration = []
    for f in files:
      index = f.find(ckptstr)
      iteration.append(int(f[index+len(ckptstr):].split('.')[0]))
    iteration = np.array(iteration)
    files = np.array(files)
    filesToDel = files[iteration!=np.max(iteration)]
    if len(filesToDel)>0:
      for f in filesToDel:
        os.remove(f)
      print('old checkpoints removed')
    else:
      print('no old checkpoints to remove')
  except:
    pass

def _del_events(log_root):
  try:
    eventstr = 'events.out.tfevents.'
    globstr = os.path.join(log_root, eventstr+'*')
    files = glob.glob(globstr)
    if len(files) > 1: print('WARNING: more than one events file being deleted')
    for f in files:
      os.remove(f)
  except:
    print('_del_events erorr')
    pass