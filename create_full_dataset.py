import h5py
import numpy as np
import os

"""
Run this to combine data from different run directories into a single hdf5 file
Looks in a directory with run directories named for each spin
"""

#set mode threshold

#iterate through everything and combine into one big dataset for 
#nn training and basis reduction

threshold = 1e-5

#limits on modes
n_max = 30

l_min = 2 
l_max = 10

#hdf5 file to store everything
outfile = h5py.File("Teuk_amps_a0.1-0.99_lmax_10_nmax_30.hdf5", "w")

par_list = outfile.create_dataset('grid', (10, 4), maxshape=(None, 4), dtype=float)
modes = outfile.create_group('modes')

#iterate through and create a dataset to store each mode
for l in range(l_min, l_max+1):
    for m in range(0, l+1):
        for n in range(-n_max, n_max+1):

            #hacky fix for negative numbers
            exec("varname = 'l{}m{}k0n{}'".format(l, m, n))

            if '-' in varname:
                varname = varname.replace('-', "_")

            exec("{} = modes.create_dataset('l{}m{}k0n{}', (10,), maxshape=(None,), dtype=complex)"\
                    .format(varname, l, m, n))

#keeping track of this for resizing
total_len = 0

for f in os.listdir():
    if f.startswith('a'):
        
        print("Entering directory {}".format(f))

        modedir = os.listdir("{}/YlmHDF5/".format(f))

        total_len += len(modedir)
        
        #resize
        par_list.resize((total_len, 4))
       
        for l in range(l_min, l_max+1):
            for m in range(0, l+1):
                for n in range(-n_max, n_max+1):
                    
                    #hacky fix for negative numbers
                    exec("varname = 'l{}m{}k0n{}'".format(l, m, n))
                    
                    if '-' in varname:
                        varname = varname.replace('-', "_")

                    exec("{}.resize((total_len,))".format(varname))

        idx = 0

        #iterate through mode files
        for item in modedir:
            
            print("Reading data from {}".format(item))

            with h5py.File('{}/YlmHDF5/{}'.format(f, item), 'r') as dat:

                #append parameters
                par_list[-len(modedir)+idx, :] = dat['params'][:4]

                for l in range(l_min, l_max+1):
                    for m in range(0, l+1):
                        for n in range(-n_max, n_max+1):
                                
                            #print("mode: l{} m{} k0 n{}".format(l, m, n))

                            #hacky fix for negative numbers
                            exec("varname = 'l{}m{}k0n{}'".format(l, m, n))
                            
                            if '-' in varname:
                                varname = varname.replace('-', "_")

                            #try to load and if no data, append zeros
                            try:
                                exec("val = np.array(dat['modes']['l{}m{}k0n{}'])".format(l, m, n))
                            except:
                                val = np.array([0., 0.])
      
                            #check if complex conj is below threshold
                            if ((val[0] + 1j*val[1]) * (val[0] - 1j*val[1])) < threshold:
                                val = np.array([0., 0.])

                            #store in dataset
                            exec("{}[-len(modedir)+idx] = val[0]+1j*val[1]".format(varname))
            
            #iterate by 1
            idx+=1
