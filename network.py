import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, PReLU, LeakyReLU
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import backend
from tensorflow.keras.callbacks import CSVLogger

from sklearn.model_selection import train_test_split
import time
import os

class BuildNetwork:
    """
    for given settings, build nn
    hidden layers increase in dimensionality by powers of 2
    input + output dimensions are number of params + number of 
    modes respectively
    """

    def __init__(self, input_dim, output_dim, max_pow=11, \
            repeat_max=14, learning_rate=1e-4, metrics=['accuracy']):
        """
        input_dim: number of params
        output_dim: number of modes
        max_pow: maximum power of 2 used to set layer size
        repeat_max: number of times to repeat the largest layer
        learning_rate: initial learning rate for optimizer (not adaptive)
        metrics: metrics to output (probably only need accuracy for now)
        """


        #learning rate and evaluation metrics
        self.learning_rate = learning_rate
        self.metrics = metrics

        #input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_dims = []

        #find closest power of 2 to input dim
        min_pow = int(np.ceil(np.log(input_dim)/np.log(2.)))
     
        for i in range(min_pow, max_pow):
        
            self.layer_dims.append(2**i)


        #list containing number of nodes in hidden layers
        self.layer_dims = self.layer_dims + [2**max_pow]*repeat_max

        print("Hidden layer dimensions:", self.layer_dims)
    
        self.setup_model()
   
    def setup_model(self):
        """
        For given input params, create model
        All settings are the same as for the schwarzchild case (activation function, 
        kernels & optimizer)
        """

        #set up model using layer settings
        self.model = Sequential()
    
        #first layer
        self.model.add(Dense(self.layer_dims[0], input_dim=self.input_dim, \
                activation=LeakyReLU(),\
                kernel_initializer=TruncatedNormal(stddev=1./np.sqrt(self.layer_dims[0]))))

        #all other hidden layers
        for i in range(1, len(self.layer_dims)):
            self.model.add(Dense(self.layer_dims[i], activation=LeakyReLU(), \
            kernel_initializer = TruncatedNormal(stddev=1./np.sqrt(float(self.layer_dims[i])))))

        #last layer
        self.model.add(Dense(self.output_dim))

        #compile model
        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate), \
             metrics=self.metrics)

        print(self.model.summary())

    
class TrainNetwork:
    """
    Given model, input data, and output data, train NN 
    """

    def __init__(self, model, train_pars, train_modes, save_model=True, \
         model_name="kerr_train"):
        """
        model: compiled keras model (loaded or created using BuildNetwork)
        train_pars: input parameters
        train_modes: output parameters
        save_model: save keras model or not (to continue training)
        model_name: string to identify specific model
        """


        #initialize step number, epoch, and arrays for metrics
        self.step = 0
        self.epoch = 0

        #initialize metric arrays
        self.train_loss = []
        self.train_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []

        #inputs params, output modes, and NN model
        self.train_pars = train_pars
        self.train_modes = train_modes
        self.model = model
        self.save_model = save_model
        self.model_name = model_name


    def __chunk_batches__(self, indices, batch_size):
        """
        Given indices of input and batch size, generate 
        lists to chunk data
        """
        #generate lists of indices for each batch
        for i in range(0, len(indices), batch_size):
            yield indices[i:i+batch_size]


    def train_for_epoch(self, max_epoch, valid_stop, batch_size=500, validation_split=0.2, print_epoch=10):
        """
        Train until max epoch is reached

        max_epoch: how many epochs to train for
        valid_stop: if loss gradient is increasing over the last 'valid_stop' epochs
                    stop evaluating
        batch_size: size of each training batch
        validation_split: fraction of input data to set aside for validation
        print_epoch: print information at this interval
        """
        
        t = time.time()    

        sample_size = self.train_pars.shape[0]

        #calculate number of batches/batch size
        active_size = sample_size - np.ceil(sample_size * validation_split)
        batches = int(np.ceil(active_size/batch_size))
        batch_size = int(np.ceil(active_size/batches))
        
        for e in range(max_epoch):
        
            #split data into validation and training
            pars_train, pars_val, modes_train, modes_val = \
                    train_test_split(self.train_pars, self.train_modes, test_size=validation_split, shuffle=True)

            #make sure size is correct
            if len(pars_train) != active_size:
                print(active_size, len(pars_train))
                break

            #train on batches
            for batch in self.__chunk_batches__(range(len(pars_train)), batch_size):
                
                print("Training on batch:", min(batch), "-", max(batch))

                #train and validate on batch
                train_result = self.model.train_on_batch(pars_train[batch], y=modes_train[batch])
                val_result = self.model.test_on_batch(pars_val, modes_val)
                
                #append output
                self.train_loss.append(train_result[0])
                self.train_accuracy.append(train_result[1])
                self.validation_loss.append(val_result[0])
                self.validation_accuracy.append(val_result[1])

                self.step+=1

            self.epoch += 1

            if (e+1) % print_epoch == 0:
                print('Time:', str.format('{0:.1f}', time.time()-t), 's')
                print("epoch: {}, step:{}, train_loss: {}, train_accuracy: {},\
                        \n validation_loss: {}, validation_accuracy: {}".format(self.epoch,\
                            self.step, train_result[0], train_result[1], val_result[0], \
                            val_result[1]))

            #if we've run enough steps and our loss is increasing, stop
            if (e + 1) > valid_stop and \
                (np.all(np.diff(self.train_loss)[-valid_stop:] > 0.) or \
                np.all(np.diff(self.validation_loss)[-valid_stop:] > 0.)):
                
                print("Loss is increasing, stopping")

                break

        
        #make directory to store model history (accuracy metrics)
        if "log" not in os.listdir('./'):
            os.mkdir("log")  
            
        with open("log/{}.log".format(self.model_name), "ab") as f:
            
            np.savetxt(f, np.c_[self.train_loss, self.train_accuracy, \
                    self.validation_loss, self.validation_accuracy])

        #save keras model 
        if self.save_model:
            
            print("Saving model")
            #if dir already exists
            if self.model_name + "_saved_model" in os.listdir('./'):
                
                #check to see if any existing models in dir
                if len(os.listdir(self.model_name + "_saved_model")) > 0:
                    
            
                    print("Saving network to:", self.model_name + "_saved_model")
                    number = []

                    #number of previous models
                    for f in os.listdir(self.model_name + "_saved_model"):
                        number.append(int(f.split('_')[-1])) 
                    
                    new_num = max(number) + 1

                    #save as model n+1
                    self.model.save("{}_saved_model/model_{}".format(self.model_name, new_num), \
                            save_format='tf')

                #save as model 1
                else:
                    self.model.save("{}_saved_model/model_1".format(self.model_name), \
                            save_format='tf')

            else:

                os.mkdir(self.model_name + "_saved_model")

                print("Creating dir:", self.model_name + "_saved_model")

                self.model.save("{}_saved_model/model_1".format(self.model_name), save_format='tf')



    def update_learning_rate(self, lr):
        """
        Set learning rate to different value
        """
        #set the model learning rate to new value
        backend.set_value(self.model.optimizer.learning_rate, lr)
