"""
Python script which converts snrfit.py FITS models into 
WODEN readable models.
"""

#!/usr/bin/env python -W ignore::DeprecationWarning
__author__ = "Jack Line"
__credits__ = ["Jack Line"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

from astropy.table import Table

ref_freq = 200e+6

filename = 'Cas_A_N13_200MHz_model_rescaled.fits'
path = '/home/jaiden/Documents/Diffuse/workbooks/'
out_path = '/home/jaiden/Documents/Diffuse/data/'
table_data = Table.read(path + filename)

ras = table_data['RA']
decs = table_data['DEC']
fluxes = table_data['Sint']
majors = table_data['Maj']
minors = table_data['Min']

# Definition of position angle is different for RA and DEC compared to 
# Pixel coordinate grid.
pas = 270.0 - table_data['PA']
SIs = table_data['alpha']


with open(out_path + 'srclist-woden_CasA_N13_200MHz_rescaled.txt','w') as outfile:

    outfile.write(f"SOURCE jaiden_sky P 0 G {len(ras)} S 0 0\n")

    for ind, ra in enumerate(ras):
        outfile.write(f"COMPONENT GAUSSIAN {ra/15:.10f} {decs[ind]:.10f}\n")
        outfile.write(f"LINEAR {ref_freq:.6e} {fluxes[ind]:.5f} 0 0 0 {SIs[ind]:.3f}\n")
        outfile.write(f"GPARAMS {pas[ind]:.5f} {majors[ind]:.5f} {minors[ind]:.5f}\n")
        outfile.write("ENDCOMPONENT\n")

    outfile.write("ENDSOURCE")
