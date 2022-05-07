import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plotly.express as px

import pandas as pd

import numpy as np

import h5py

from sklearn.model_selection import train_test_split

import greedy
import integrals

from few.utils.utility import get_fundamental_frequencies, get_separatrix

#RUN THIS TWICE TO UPDATE PLOT FORMATTING
plt.rcParams.update({'font.size': 16})


class CompareRBWaveform(object):

    def __init__(self, training_pars, training_modes, \
            rb, rb_alpha, rb_indices, param_list):
        """
        all inputs should be file names, expect param list - params
        to construct wf for
        """

        #initalize variables and load relevant values
        #modes + pars used to construct basis
        self.training_pars = np.loadtxt(training_pars)
        self.training_modes = np.loadtxt(training_modes, dtype=complex)

        
        self.rb = np.loadtxt(rb, dtype=complex)
        self.rb_alpha = np.loadtxt(rb_alpha, dtype=complex)
        self.rb_indices = np.loadtxt(rb_indices)

        self.param_list = param_list

        self.__closest_pt__()
        self.__gen_mode_order__()

    def __closest_pt__(self):
        """
        Find closest point to specified grid pt
        """

        print("input parameters: a={}, p={}, e={}".format(\
                self.param_list[0], \
                self.param_list[1], \
                self.param_list[2]))

        dist = np.sqrt((self.training_pars[:, 0] - self.param_list[0])**2 +\
                        (self.training_pars[:, 1] - self.param_list[1])**2 +\
                        (self.training_pars[:, 2] - self.param_list[2])**2)


        idx = np.where(dist == min(dist))[0]

        print("selecting closest params: a={}, p={}, e={}".format(\
                self.training_pars[idx, 0], \
                self.training_pars[idx, 1], \
                self.training_pars[idx, 2]))

        self.point_index = idx

    def __gen_mode_order__(self, l_min=2, l_max=10, k=0, n_max=30):

        self.lmkn_all = np.array([])

        for l in range(l_min, l_max+1):
            for m in range(0, l+1):
                for n in range(-n_max, n_max):
                    
                    if self.lmkn_all.shape[0] == 0:
                        self.lmkn_all = np.append(self.lmkn_all, \
                                np.array([l, m, 0, n]))
                    else:
                        self.lmkn_all = np.vstack((self.lmkn_all, \
                                np.array([l, m, 0, n])))

    def construct_wf(self, modes, t_max, pts):
        """
        build time domain waveform 
        modes: mode data
        t_max: maximum time
        pts: number of points
        """
        a, p, e = self.training_pars[self.point_index, :][0]
 

        #fundamental frequencies using few utility function
        omega_phi, omega_theta, omega_r = \
                get_fundamental_frequencies(a, p, e, x=1.)


        #time range for wf
        t_range = np.linspace(0, t_max, pts)

        h_of_t = np.zeros(len(t_range), dtype=complex)

        for i in range(len(self.lmkn_all)):

            m = self.lmkn_all[i, 1]
            k = self.lmkn_all[i, 2]
            n = self.lmkn_all[i, 3]
    
            omega_mkn = m * omega_phi + k * omega_theta + n * omega_r

            if omega_mkn != 0:

                h_of_t += (-2. * modes[i] / omega_mkn**2) * np.exp(-1.j * omega_mkn * t_range)


        return t_range, h_of_t


    def plot_wfs(self, t_max=8000, pts=50000):
        
        approx_modes = np.dot(self.rb_alpha[:, self.point_index].T, self.rb)[0]

        #actual wf
        t1, h1 = self.construct_wf(self.training_modes[self.point_index, :][0], t_max, pts)

        #approx
        t2, h2 = self.construct_wf(approx_modes, t_max, pts)

        plt.figure(figsize=(8,6))
        plt.plot(t1, h1.real, alpha=0.8, label="real modes")
        plt.plot(t2, h2.real, alpha=0.8, linestyle='--', label="approx modes")
        plt.legend(loc=0)
        plt.xlabel('t')
        plt.ylabel('h_+')
        plt.show()

        plt.figure(figsize=(8,6))
        plt.plot(t1, h1.imag, alpha=0.8, label="real modes")
        plt.plot(t2, h2.imag, alpha=0.8, linestyle='--', label="approx modes")
        plt.legend(loc=0)
        plt.xlabel('t')
        plt.ylabel('h_x')
        plt.show()

    def compute_overlap(self):
        """
        Compute the overlap between real and approximate waveform
        """

        approx_modes = np.dot(self.rb_alpha[:, self.point_index].T, self.rb)[0]
        real_modes = self.training_modes[self.point_index, :][0]

        norm_approx = np.linalg.norm(approx_modes)
        norm_real = np.linalg.norm(real_modes)

        overlap = np.inner(np.conj(approx_modes/norm_approx), real_modes/norm_real)

        return overlap.real


    def basis_pt_plot(self):
        """
        Make 3D plot of points used in basis and save as html
        """
        pts = {'a':self.training_pars[self.rb_indices, 0], \
                'p':self.training_pars[self.rb_indices, 1], \
                'e':self.training_pars[self.rb_indices, 2]}

        #make 3d scatter and save as html
        fig = px.scatter_3d(pts, x='a', y='p', z='e', opacity=0.3)
        fig.write_html("interactive_3D_rob_points_kerr.html")

