import scipy
from scipy import io
import h52vtp


filenames = ['/Users/tom/Downloads/xent_poison.mat',
              '/Users/tom/Downloads/xent_clean.mat',
              '/Users/tom/Downloads/acc_poison.mat',
              '/Users/tom/Downloads/acc_clean.mat'
              ]


for filename in filenames:
    mat = scipy.io.loadmat(filename)
    filename_no_extension = filename[0:-4]
    #h52vtp.h5_to_vtp(mat['data'],vtp_file=file_name[0:-4], log=True, zmax=1000)
    h52vtp.h5_to_vtp(mat['data'],vtp_file=filename_no_extension, log=True, zmax=1000, chopxmin=80, chopxmax=40, chopymin=60, chopymax=60, interp=500)

