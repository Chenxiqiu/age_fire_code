import numpy as np

#age spectra
VMIN = 0 #the lowest possible value for age variables
VMAX = 100 #the highest possible value for age variables except mean age or medium age
VMAX_AGE = 80 #the lowest possible value for mean age or medium age
VRES = 5 #the size of the bins for age variables
VRANGE = np.arange(VMIN, VMAX+VRES, VRES)
VRANGE_AGE = np.arange(VMIN, VMAX_AGE+VRES, VRES)
ALL_AGEV = ('AGE', 'MF_03', 'MF_06', 'MF_12', 'MF_24', 'MF_48')
ALL_AGEV_SOUTHTRAC = ('P50', 'AGE','MF_06', 'MF_24', 'AGE')

#OCS
OCSMIN_MIPAS = -200
OCSMIN_ACE = -60
OCSMAX = 600
OCSRES = 20
OCSRANGE_MIPAS = np.arange(OCSMIN_MIPAS, OCSMAX+OCSRES, OCSRES)
OCSRANGE_ACE = np.arange(OCSMIN_ACE, OCSMAX+OCSRES, OCSRES)
OCSRANGE_SOUTHTRAC = np.arange(0, OCSMAX+OCSRES, OCSRES)

#N2O
N2OMIN_MIPAS = -200
N2OMIN_ACE = -60
N2OMAX = 600
N2ORES = 20
N2ORANGE_MIPAS = np.arange(N2OMIN_MIPAS, N2OMAX+N2ORES, N2ORES)
N2ORANGE_ACE = np.arange(N2OMIN_ACE, N2OMAX+N2ORES, N2ORES)
