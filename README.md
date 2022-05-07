## Converting data to hdf5

Assuming runs are stored in directories named according to the BH spin, run `create_full_dataset.py` in directory containing individual 'a' directories.
- edit line 18 to change n mode limit
- edit line 24 to change output file name
- edit line 47 to change run directory name criteria

## Training NN

See `run_network` script for an example of how to use each module
*Note*: change python path at top of script to run

data.py: 

**PrepareData class:**

Settings:
	
`data_file`: name of hdf5 file where raw data is stored     
`prograde`: bool, use only prograde BH spins, or prograde & retrograde     
`save_tt`: save train and test data        
`rb_tolerance`: tolerance to use when generating reduced basis     
`test_fraction`: fraction of data to set aside for testing      

Outputs:

- all parameters and modes saved as ascii files (loads faster)
- ascii files containing reduced basis, reduced basis indices, and reduced basis alpha

Future Edits:

- lines 83 & 84 set spin cutoff to 0.9, comment these out to include all spin data
- line 111 makes train-test split using only parameters a, p, and e, edit input to include x as well 
- `__p_cut__` method makes parameter cut using some criteria i honestly don't understand, edit method to change cut

network.py

**BuildNetwork class:**

Settings:

`input_dim`: number of params     
`output_dim`: number of modes (in reduced basis)     
`max_pow`: maximum power of 2 used to set layer size     
`repeat_max`: number of times to repeat the largest layer      
`learning_rate`: initial learning rate for optimizer (not adaptive)     
`metrics`: metrics to output (probably only need accuracy for now)     

Future Edits:

- can change activation functions, kernels, etc in `setup_model` method	

**TrainNetwork class**

Initialization Settings:

`model`: compiled keras model (loaded or created using BuildNetwork)     
`train_pars`: input data     
`train_modes`: output data      
`save_model`: save keras model or not (to continue training)     
`model_name`: string to identify specific model      

`train_for_epoch` Settings:

`max_epoch`: how many epochs to train for    
`valid_stop`: if loss gradient is increasing over the last 'valid_stop' epochs stop evaluating      
`batch_size`: size of each training batch     
`validation_split`: fraction of input data to set aside for validation    
`print_epoch`: print information at this interval		

## Visualizing reduced-order basis

See `run_rob_comparison` for example of how to use module
*Note*: change python path at top of script to run

compare_rob_waveforms.py:

**CompareRBWaveform class**

Settings:

`training_pars`: file containing training parameters      
`training_modes`: file containing training modes     
`rb`: file containing reduced basis      
`rb_alpha`: file containing reduced basis alpha      
`rb_indices`: file containing reduced basis indices     
`param_list`: parameters to find wf for      

Methods:

`plot_wfs(t_max=500, pts=5000)`: inputs are max time and number of pts         
`compute_overlap(t_max=500, pts=5000)`: same inputs as `plot_wfs` - use large t_max and pts to ensure enough resolution for accurate overlap        
`basis_pt_plot()`: make 3D plot of points used in basis and save as html 		

## Preliminary result

Stored in preliminary result directory. Most recent trained model is 'model_1' in the 'kerr_train_saved_model' dir. Weights can be retrieved by loading model. To resume training, load model and train on data saved in 'test' dir
