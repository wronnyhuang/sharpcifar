import scipy
from scipy import io
import h52vtp


# filenames = ['/Users/tomg/Downloads/svhn_xent_poison.mat',
#               '/Users/tomg/Downloads/svhn_xent_clean.mat'
#               ]

# for filename in filenames:
#     mat = scipy.io.loadmat(filename)
#     filename_no_extension = filename[0:-4]
#     #h52vtp.h5_to_vtp(mat['data'],vtp_file=file_name[0:-4], log=True, zmax=1000)
#     h52vtp.h5_to_vtp(mat['data'],vtp_file=filename_no_extension, log=True, zmax=1000, chopxmin=122, chopxmax=00, chopymin=61, chopymax=61, interp=500)


filenames = ['/Users/tomg/Downloads/swiss_data_poison.pickle',
              '/Users/tomg/Downloads/swiss_data_clean.pickle'
              ]

import pickle
for filename in filenames:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    filename_no_extension = filename[0:-4]
    #h52vtp.h5_to_vtp(mat['data'],vtp_file=file_name[0:-4], log=True, zmax=1000)
    h52vtp.h5_to_vtp(data['xent'],vtp_file=filename_no_extension, log=True, zmax=1000, chopxmin=140, chopxmax=00, chopymin=70, chopymax=70, interp=500)
