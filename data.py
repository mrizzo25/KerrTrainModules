import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from few.utils.utility import get_separatrix

import greedy
import integrals

import os

class PrepareData:

    def __init__(self, data_file, prograde=True, save_tt=True, \
            rb_tolerance=1e-8, test_fraction=0.1):
        """
        assumes hdf5 file format (initially)
        once parsed, saves as ascii
        data_file: name of hdf5 file where raw data is stored
        prograde: use only prograde BH spins, or prograde & retrograde
        save_tt: save train and test data
        rb_tolerance: tolerance to use when generating reduced basis
        test_fraction: fraction of data to set aside for testing
        """

        self.data_file = data_file
       
        self.fname = self.data_file.split('/')[-1].replace('.hdf5', '')

        if '/' in self.data_file:
            self.path = self.data_file.replace(self.fname+'.hdf5', '')

        else:
            self.path = './'

        print("File path:", self.path)

        self.prograde = prograde

        if self.prograde:
            self.fname = self.fname + "_prograde"
        
        print("File name:", self.fname)

        self.save_tt = save_tt

        self.rb_tolerance = rb_tolerance
        self.test_fraction = test_fraction

        self.load_data()
        self.tt_split()
        self.generate_basis()
        self.NN_input()

    def load_data(self):

        if (self.fname + '_pars.dat' and self.fname + '_modes.dat') in os.listdir(self.path):
            
            #load from ascii file if exists
            self.pars_all = np.loadtxt(self.path + self.fname + '_pars.dat')
            print("Loading parameters from:", self.path + self.fname + '_pars.dat')

            self.modes_all = np.loadtxt(self.path + self.fname + '_modes.dat', dtype=complex)
            print("Loading modes from:", self.path + self.fname + '_modes.dat')

        else:
            #parse from hdf5 if not
            print("Loading data from:", self.data_file)
            with h5py.File(self.data_file,'r') as file:

                self.pars_all = np.array(file['grid'], dtype=file['grid'].dtype)

                self.modes_all = np.array([])
               
                for l in range(2, 10+1):
                    for m in range(0, l+1):
                        for n in range(-30, 30+1):

                            if self.modes_all.shape[0] == 0:
                                self.modes_all = np.append(self.modes_all, np.array(file['modes/l{}m{}k0n{}'.format(l,m,n)]))
                            else:
                                self.modes_all = np.vstack((self.modes_all, np.array(file['modes/l{}m{}k0n{}'.format(l,m,n)])))
                   
            self.modes_all = self.modes_all.T
            
            if self.prograde:
                
                mask_prograde = np.where(self.pars_all[:, 3] > 0.)[0]
                self.pars_all, self.modes_all = self.pars_all[mask_prograde], self.modes_all[mask_prograde]
                
            #get rid of high spins for now
            mask_hs = np.where(self.pars_all[:, 0] < 0.9)
            self.pars_all, self.modes_all = self.pars_all[mask_hs], self.modes_all[mask_hs]

            np.savetxt(self.path + self.fname + "_pars.dat", self.pars_all)
            np.savetxt(self.path + self.fname + "_modes.dat", self.modes_all)

    def u_transform(self, a, p, e, x=1.):
                
        u_vals = np.array([])
        
        for i in range(len(a)):

            p_lso = get_separatrix(a[i], e[i], x)
    
            u = 1./np.sqrt(p[i] - 0.9 * p_lso)
        
            u_vals = np.append(u_vals, u)
    
        return u_vals

    def tt_split(self):

        #split all mode data into training and test set
        self.pars_train, self.pars_test, self.modes_train, self.modes_test = \
                train_test_split(self.pars_all[:, 0:3], self.modes_all, \
                        test_size=self.test_fraction, random_state=42)

        if self.save_tt:
            np.savetxt(self.path + self.fname + "_train_pars.dat", self.pars_train)
            np.savetxt(self.path + self.fname + "_train_modes.dat", self.modes_train)
            np.savetxt(self.path + self.fname + "_test_pars.dat", self.pars_test)
            np.savetxt(self.path + self.fname +  "_test_modes.dat", self.modes_test)

    def generate_basis(self):

        #load basis if exists
        if (self.fname + "_reduced_basis.dat" and self.fname + "_rb_alpha.dat") in \
                os.listdir(self.path):
                        
                    self.red_basis = np.loadtxt(self.path + self.fname + "_reduced_basis.dat", dtype=complex)
                    self.rb_alpha = np.loadtxt(self.path + self.fname + "_rb_alpha.dat", dtype=complex)
                        
        else:
            #initial number of modes
            v_size = self.modes_train.shape[1]
            
            #make reduced order basis
            inner = integrals.Integration(interval=[-1,1], num=v_size, rule='riemann')
            rb = greedy.ReducedBasis(inner=inner, loss='L2')
            rb.make(self.modes_train, 0, tol=self.rb_tolerance, verbose=True)

            self.red_basis = rb.basis
            self.rb_alpha = rb.alpha
                
            #save basis
            np.savetxt(self.path + self.fname + "_reduced_basis.dat", rb.basis)
            np.savetxt(self.path + self.fname + "_rb_alpha.dat", rb.alpha)
                
            #save indices for plotting
            np.savetxt(self.path + self.fname + "_rb_indices.dat", rb.indices)


    def __p_cut__(self):
        """
        perform cut in param space based on e
        (i still don't really know what this does)
        """
        def min_p(e):
            return np.where(e < 0.5, 2*e + 62/10, 14*(e - 0.5) + 72/10)

        mask_test = (self.pars_test[:,1] > min_p(self.pars_test[:,2])) & (self.pars_test[:,2] < 0.8)
        self.pars_test, self.modes_test = self.pars_test[mask_test], self.modes_test[mask_test]

        self.mask_train = ~np.isin(np.arange(len(self.pars_train)), \
                    np.concatenate([np.arange(i-4, 0) - i*50 + len(self.pars_train) for i in range(4)]))
        self.pars_train, self.modes_train = self.pars_train[self.mask_train], self.modes_train[self.mask_train]

    def NN_input(self):
        """
        convert training data to nn input
        """
        #self.__p_cut__()

        #convert p to u
        u = self.u_transform(self.pars_train[:, 0], self.pars_train[:, 1], self.pars_train[:, 2])
        self.input_train = np.c_[self.pars_train[:,0], u, self.pars_train[:, 2]]
        self.input_dim = self.input_train.shape[1]

        print("Input dimensions:", self.input_dim)

        #self.output_train = (np.vstack((self.rb_alpha.real, self.rb_alpha.imag)).T*1e3)[self.mask_train]
        self.output_train = np.vstack((self.rb_alpha.real, self.rb_alpha.imag)).T*1e3
        self.output_dim = self.output_train.shape[1]

        print("Output dimensions:", self.output_dim)
