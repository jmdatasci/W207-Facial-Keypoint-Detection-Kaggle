import os
import pickle
import random
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from modules.data_augmentation import AugmentData


def plot_learning_curve(history, model_lc_file, response_name='none'):
    """
    Plot the learning curve using learning history
    :return:
    """

    # Code taken entirely from Hands-on machine learning book
    df = pd.DataFrame(history)
    epoch = list(range(df.shape[0]))
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.loglog(epoch, df['loss'], linestyle='-', color='k')
    plt.loglog(epoch, df['val_loss'], linestyle='-', color='b')
    plt.legend(['Train loss', 'Validation loss'])
    plt.title('Response %s train and validation loss %.2f/%.2f' %
              (response_name, df['loss'].min(),
               df['val_loss'].min()))

    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.loglog(epoch, df['root_mean_squared_error'], linestyle='-', color='k')
    plt.loglog(epoch, df['val_root_mean_squared_error'], linestyle='-', color='b')
    plt.legend(['Train rmse', 'Validation rmse'])
    plt.title('Response %s train and validation rmse %.2f/%.2f' %
              (response_name, df['root_mean_squared_error'].min(),
               df['val_root_mean_squared_error'].min()))

    plt.grid(True)
    plt.savefig(model_lc_file)
    plt.show()


class Means:
    """

    """

    def __init__(self, model_gen_params=None, optimizer_params=None, response_name='test', prefix='', folder='means'):
        """
        :param model_gen_params: Parameters for model generation
        :param optimizer_params:
        :param response_name: Name of the response we are trying to Optimizer
        :param prefix: Prefix to add to the model files
        :param folder:
        """

        # Name of this response
        self.response_name = response_name
        self.model_param_list = []
        self.optimizer_param_list = {}

        # Set the model generator parameters
        self.model_gen_params = {} if model_gen_params is None else model_gen_params

        # Set the model optimizer parameters
        self.optimizer = None
        self.optimizer_params = {} if optimizer_params is None else optimizer_params

        # This stores the history
        self.history = None

        try:
            basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        except AttributeError:
            basepath = '/content/drive/MyDrive/Colab Notebooks/cct/'

        if prefix:
            if prefix[-1] != '_':
                prefix = prefix + '_'
        self.prefix = prefix

        self.model_file = os.path.join(basepath, 'data/models/' + folder + '/%smeans_%s.h5' %
                                       (self.prefix, response_name))
        self.model_params_file = os.path.join(basepath, 'data/models/' + folder + '/%smeans_%s.params' %
                                              (self.prefix, response_name))
        self.model_lc_file = os.path.join(basepath, 'data/models/' + folder + '/%smeans_elc_%s.png' %
                                          (self.prefix, response_name))

        # Model exists then load it, if not then need to create one
        try:
            # Load the model
            self.model = pickle.load(open(self.model_file, 'rb'))

            # Load all other parameters of this model
            all_params = pickle.load(open(self.model_params_file, 'rb'))

            # Load the history, this must exist for an already generated model
            self.history = all_params['history']

            # Load the model generator params
            self.model_gen_params = all_params['model_gen_params']

            # Load the model optimizer params
            self.optimizer_params = all_params['optimizer_params']

            # Load the optimizer
            self.optimizer = all_params['optimizer']

        except OSError:
            self.model = None

        self.metric = 0

    @staticmethod
    def _clean_x(x):
        # Make X a 4D array
        if isinstance(x, list) or isinstance(x, tuple):
            x = list(x)
            x = np.stack(x, axis=0)

            # Add a fourth dimension to the data
            x = x[:, :, :, np.newaxis]

        return x

    @staticmethod
    def _clean_y(y):
        # Take every instance of x and add an extra axis to it
        # Then concatenate it to make it a 3D array
        if isinstance(y, list) or isinstance(y, tuple):
            try:
                y = list(y)
                y = [x.flatten() for x in y]
                y = np.vstack(y)
            except AttributeError:
                y = np.array(y)

        return y

    @staticmethod
    def build(input_shape, num_of_responses):
        if input_shape:
            pass

        if num_of_responses:
            pass

    def fit(self, x, y, cv_data, epochs=100, save_model=True, sample_weight=None, x_unlabeled=None):
        """

        :param x: Train X
        :param y: Train y
        :param cv_data: cross validation X, y data as a tuple
        :param epochs: Number of epochs to train
        :param save_model: Save the model or not
        :param sample_weight: Unused
        :param x_unlabeled: This model cannot use unlabeled data so this is ignored
        :return:
        """

        if x is not None:
            pass

        if cv_data is not None:
            pass

        if sample_weight is not None:
            pass

        if x_unlabeled is not None:
            pass

        # Make x in a 2D numpy format
        y = self._clean_y(y)

        if self.model is not None:
            print('Model already fit, relocate model file and restart to fit again')
            return
        self.model = np.mean(y, axis=0)

        if save_model:
            pickle.dump(self.model, open(self.model_file, 'wb'))

        # We just need the history dictionary of history

        self.history = {'epochs': range(epochs),
                        'val_root_mean_squared_error': [random.random(), ] * epochs,
                        'root_mean_squared_error': [random.random(), ] * epochs,
                        'loss': [random.random(), ] * epochs, 'val_loss': [random.random(), ] * epochs,
                        }
        outdict = {'model_gen_params': self.model_gen_params, 'optimizer': self.optimizer, 'history': self.history,
                   'optimizer_params': self.optimizer_params}
        pickle.dump(outdict, open(self.model_params_file, 'wb'))

        return self.history

    def predict(self, predict_data, debug=False):
        """

        :param predict_data: Train X,y data
        :param debug: True of debug mode
        :return:
        """

        if debug:
            pass

        predict_data = self._clean_x(predict_data)
        y_predict = np.ones((predict_data.shape[0], self.model.size)) * self.model
        y_predict = [np.squeeze(x1) for x1 in np.split(y_predict, y_predict.shape[0])]
        return y_predict

    def plot_learning_curve(self, response_name='none'):
        plot_learning_curve(self.history, self.model_lc_file, response_name=response_name)


class SimpleNN:
    """

    """

    def __init__(self, model_gen_params=None, optimizer_params=None, response_name='test', prefix='', folder='dnn'):
        """
        :param model_gen_params: Parameters for model generation
            :dense_layers: Arrangement of the convolutional layers format ((num of nodes, num of nodes), )
            :dense_activation: Activation function for the dense layer
            :dense_dropout: Dropout rate for the dense layers
        :param optimizer_params:
            Dictionary with field 'name' as the optimizer name
            Other fields contain the other optimizer parameters
                SGD:
                    :sgd_lr: Learning rate for SGD
                    :sgd_momentum: Nesterov momentum
                AdaMax:
                    :learning_rate: Learning rate for Adam
                    :beta_1: Beta 1 value
                    :beta_2: Beta 2 value
        :param response_name: Name of the response we are trying to Optimizer
        :param prefix: Prefix to add to the model files
        :param folder: Folder to save the models
        """

        # Name of this response
        self.response_name = response_name
        self.model_param_list = ['dense_activation', 'dense_dropout',  'dense_layers']
        self.optimizer_param_list = {'sgd': ['learning_rate', 'momentum'],
                                     'adamax': ['learning_rate', 'beta_1', 'beta_2']}

        # Set the model generator parameters
        self.model_gen_params = {}
        self._get_model_gen_params(model_gen_params)

        # Set the model optimizer parameters
        self.optimizer = None
        self.optimizer_params = {}
        self._get_model_optimizer_params(optimizer_params)

        # This stores the history
        self.history = None

        try:
            basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        except AttributeError:
            basepath = '/content/drive/MyDrive/Colab Notebooks/cct/'

        if prefix:
            if prefix[-1] != '_':
                prefix = prefix + '_'
        self.prefix = prefix

        self.model_file = os.path.join(basepath, 'data/models/' + folder + '/%sdnn_%s.h5' %
                                       (self.prefix, response_name))
        self.model_params_file = os.path.join(basepath, 'data/models/' + folder + '/%sdnn_%s.params' %
                                              (self.prefix, response_name))
        self.model_lc_file = os.path.join(basepath, 'data/models/' + folder + '/%sdnn_elc_%s.png' %
                                          (self.prefix, response_name))

        # Model exists then load it, if not then need to create one
        try:
            # Load the model
            self.model = keras.models.load_model(self.model_file)

            # Load all other parameters of this model
            all_params = pickle.load(open(self.model_params_file, 'rb'))

            # Load the history, this must exist for an already generated model
            self.history = all_params['history']

            # Load the model generator params
            self.model_gen_params = all_params['model_gen_params']

            # Load the model optimizer params
            self.optimizer_params = all_params['optimizer_params']

            # Load the optimizer
            self.optimizer = all_params['optimizer']

        except OSError:
            self.model = None

        self.metric = 0

    def _get_model_gen_params(self, model_gen_params):
        """

        :param model_gen_params:
        :return:
        """

        # Default parameters
        default_params = {'dense_activation': 'relu',  'dense_dropout': 0.2, 'dense_layers': (500, 500)}

        if model_gen_params is None:
            # If the input is None then just use the default parameters
            self.model_gen_params = default_params
        else:
            # If the model parameters were provided
            # Check the model gen params to make sure it has the right parameters
            for key in model_gen_params:
                # Check if the keys in model_gen_params are the right keys
                if key not in self.model_param_list:
                    raise KeyError('Key %s is not a valid key' % key)
                else:
                    self.model_gen_params[key] = model_gen_params[key]

            # If the input was not provided then use the default value for it
            for key in self.model_param_list:
                if key not in self.model_gen_params:
                    self.model_gen_params[key] = default_params[key]

    def _get_model_optimizer_params(self, optimizer_params):
        """

        :param optimizer_params:
        :return:
        """

        # Supported optimizers
        optimizer_list = ['sgd', 'adamax']
        # Default parameters
        default_params = {'learning_rate': 0.001, 'momentum': 0.9, 'nesterov': True,
                          'beta_1': 0.9, 'beta_2': 0.999}

        # If it None or empty
        if optimizer_params is None or not optimizer_params:
            # If the input is None then just use the default parameters
            self.optimizer = 'sgd'
            # Load all the parameters related to SGD here
            self.optimizer_params = {k: default_params[k] for k in self.optimizer_param_list[self.optimizer]}
        else:
            # This is the optimizer we use
            self.optimizer = optimizer_params['name'].lower()

            if self.optimizer not in optimizer_list:
                raise TypeError('Optimizer must be one of %s' % str(optimizer_list))

            # If the optimizer parameters were provided the load them
            # Check the optimizer params to make sure it has the right parameters
            for key in optimizer_params:
                # Check if the keys in optimizer_params are the right keys
                if key not in self.optimizer_param_list[self.optimizer] + ['name', ]:
                    raise KeyError('Key %s is not a valid key' % key)
                else:
                    if key != 'name':
                        self.optimizer_params[key] = optimizer_params[key]

            # If the input was not provided then use the default value for it
            for key in self.optimizer_param_list[self.optimizer]:
                if key not in self.optimizer_params:
                    self.optimizer_params[key] = default_params[key]

    def build(self, input_shape, num_of_responses):
        """
        Now that we know the shape
        :param input_shape:
        :param num_of_responses:
        :return:
        """

        # Create an object of input to define what it will look like
        inputs = keras.Input(shape=input_shape, name="original_img")

        # Convolution layer
        hidden = inputs
        for node in self.model_gen_params['dense_layers']:
            # Add more layers to the
            hidden = keras.layers.Dense(node, activation=self.model_gen_params['dense_activation'])(hidden)
            hidden = keras.layers.BatchNormalization()(hidden)
            hidden = keras.layers.Dropout(self.model_gen_params['dense_dropout'])(hidden)

        # Add the output layer back in
        output = keras.layers.Dense(num_of_responses, activation=self.model_gen_params['dense_activation'])(hidden)
        self.model = keras.Model(inputs=inputs, outputs=output)
        self.model.summary()

        # Compile model
        # Keeping these fixed for now
        if self.optimizer == 'sgd':
            # Create the optimizer here
            optimizer = keras.optimizers.SGD(**self.optimizer_params)
        elif self.optimizer == 'adamax':
            optimizer = keras.optimizers.Adamax(**self.optimizer_params)
        else:
            raise TypeError('Unsupported Optimizer type')

        self.model.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=[keras.metrics.RootMeanSquaredError()])

        # Make predictions on testing data
        return self.model

    @staticmethod
    def _clean_x(x):
        # Make X a 4D array
        if isinstance(x, list) or isinstance(x, tuple):
            x = list(x)
            x = np.stack(x, axis=0)

            # Add a fourth dimension to the data
            x = x[:, :, :, np.newaxis]

        return x

    @staticmethod
    def _clean_y(y):
        # Take every instance of x and add an extra axis to it
        # Then concatenate it to make it a 3D array
        if isinstance(y, list) or isinstance(y, tuple):
            try:
                y = list(y)
                y = [x.flatten() for x in y]
                y = np.vstack(y)
            except AttributeError:
                y = np.array(y)

        return y

    def fit(self, x, y, cv_data, epochs=100, save_model=True, x_unlabeled=None):
        """

        :param x: Train X
        :param y: Train y
        :param cv_data: cross validation X, y data as a tuple
        :param epochs: Number of epochs to train
        :param save_model: Save the model or not
        :param x_unlabeled: This model cannot use unlabeled data so this is ignored
        :return:
        """

        if x_unlabeled is not None:
            pass

        # Make x in the format Keras is expecting
        x = self._clean_x(x)
        y = self._clean_y(y)

        cv_data[0] = self._clean_x(cv_data[0])
        cv_data[1] = self._clean_y(cv_data[1])

        if self.model is not None:
            print('Model already fit, relocate model file and restart to fit again')
            return

        # Create a model with the input shape
        self.build(x.shape[1:], y.shape[1])

        # Checkpoint the model
        if save_model:
            checkpoint = [keras.callbacks.ModelCheckpoint(
                self.model_file, save_best_only=True, monitor="val_loss", mode='min')]
        else:
            checkpoint = []

        # noinspection PyUnresolvedReferences
        self.history = self.model.fit(x, y, epochs=epochs,
                                      validation_data=tuple(cv_data),
                                      callbacks=checkpoint)

        # We just need the history dictionary of history
        self.history = self.history.history
        outdict = {'model_gen_params': self.model_gen_params, 'optimizer': self.optimizer, 'history': self.history,
                   'optimizer_params': self.optimizer_params}
        pickle.dump(outdict, open(self.model_params_file, 'wb'))

        return self.history

    def predict(self, predict_data):
        """

        :param predict_data: Train X,y data
        :return:
        """
        x = self._clean_x(predict_data)
        y_predict = self.model.predict(x)
        y_predict = [np.squeeze(x1) for x1 in np.split(y_predict, y_predict.shape[0])]

        return y_predict

    def plot_learning_curve(self, response_name='none'):
        plot_learning_curve(self.history, self.model_lc_file, response_name=response_name)


class ResNet:
    """

    """

    def __init__(self, model_gen_params=None, optimizer_params=None, response_name='test', prefix='', folder='resnet'):
        """
        :param model_gen_params: Parameters for model generation
            :conv_layers: Arrangement of the convolutional layers format ((Filters, kernel_size, stride, pool_size), )
            :dense_layers: Arrangement of the convolutional layers format ((num of nodes, num of nodes), )
            :dense_activation: Activation function for the dense layer
            :conv_activation: Activation for the convolutional layers
            :conv_dropout: Dropout rate for the convolutional layers
            :dense_dropout: Dropout rate for the dense layers
        :param optimizer_params:
            Dictionary with field 'name' as the optimizer name
            Other fields contain the other optimizer parameters
                SGD:
                    :sgd_lr: Learning rate for SGD
                    :sgd_momentum: Nesterov momentum
                AdaMax:
                    :learning_rate: Learning rate for Adam
                    :beta_1: Beta 1 value
                    :beta_2: Beta 2 value
        :param response_name: Name of the response we are trying to Optimizer
        :param prefix: Prefix to give model filenames
        :param folder:
        """

        # Name of this response
        self.response_name = response_name
        self.model_param_list = ['dense_activation', 'conv_activation', 'dense_dropout', 'conv_dropout', 'conv_layers',
                                 'dense_layers']
        self.optimizer_param_list = {'sgd': ['learning_rate', 'momentum'],
                                     'adamax': ['learning_rate', 'beta_1', 'beta_2']}

        # Set the model generator parameters
        self.model_gen_params = {}
        self._get_model_gen_params(model_gen_params)

        # Set the model optimizer parameters
        self.optimizer = None
        self.optimizer_params = {}
        self._get_model_optimizer_params(optimizer_params)

        # This stores the history
        self.history = None

        try:
            basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        except AttributeError:
            basepath = '/content/drive/MyDrive/Colab Notebooks/cct/'

        # This prefix is added to all models
        if prefix:
            if prefix[-1] != '_':
                prefix = prefix + '_'
        self.prefix = prefix

        self.model_file = os.path.join(basepath, 'data/models/' + folder + '/%sresnet_%s.h5' %
                                       (self.prefix, response_name))
        self.model_params_file = os.path.join(basepath, 'data/models/' + folder + '/%sresnet_%s.params' %
                                              (self.prefix, response_name))
        self.model_lc_file = os.path.join(basepath, 'data/models/' + folder + '/%sresnet_elc_%s.png' %
                                          (self.prefix, response_name))

        # Model exists then load it, if not then need to create one
        try:
            # Load the model
            self.model = keras.models.load_model(self.model_file)

            # Load all other parameters of this model
            all_params = pickle.load(open(self.model_params_file, 'rb'))

            # Load the history, this must exist for an already generated model
            self.history = all_params['history']

            # Load the model generator params
            self.model_gen_params = all_params['model_gen_params']

            # Load the model optimizer params
            self.optimizer_params = all_params['optimizer_params']

            # Load the optimizer
            self.optimizer = all_params['optimizer']

        except OSError:
            self.model = None

        self.metric = 0

    def _get_model_gen_params(self, model_gen_params):
        """

        :param model_gen_params:
        :return:
        """

        # Text book convolutional network
        conv_layers = ((64, (2, 2), 1, (2, 2)), )
        # A lot of dense layers
        dense_layers = (500, 500)

        # Default parameters
        default_params = {'dense_activation': 'relu', 'conv_activation': 'relu',
                          'dense_dropout': 0.2, 'conv_dropout': 0.2,
                          'conv_layers': conv_layers, 'dense_layers': dense_layers}

        if model_gen_params is None:
            # If the input is None then just use the default parameters
            self.model_gen_params = default_params
        else:
            # If the model parameters were provided
            # Check the model gen params to make sure it has the right parameters
            for key in model_gen_params:
                # Check if the keys in model_gen_params are the right keys
                if key not in self.model_param_list:
                    raise KeyError('Key %s is not a valid key' % key)
                else:
                    self.model_gen_params[key] = model_gen_params[key]

            # If the input was not provided then use the default value for it
            for key in self.model_param_list:
                if key not in self.model_gen_params:
                    self.model_gen_params[key] = default_params[key]

    def _get_model_optimizer_params(self, optimizer_params):
        """

        :param optimizer_params:
        :return:
        """

        # Supported optimizers
        optimizer_list = ['sgd', 'adamax']
        # Default parameters
        default_params = {'learning_rate': 0.001, 'momentum': 0.9, 'nesterov': True,
                          'beta_1': 0.9, 'beta_2': 0.999}

        # If it None or empty
        if optimizer_params is None or not optimizer_params:
            # If the input is None then just use the default parameters
            self.optimizer = 'sgd'
            # Load all the parameters related to SGD here
            self.optimizer_params = {k: default_params[k] for k in self.optimizer_param_list[self.optimizer]}
        else:
            # This is the optimizer we use
            self.optimizer = optimizer_params['name'].lower()

            if self.optimizer not in optimizer_list:
                raise TypeError('Optimizer must be one of %s' % str(optimizer_list))

            # If the optimizer parameters were provided the load them
            # Check the optimizer params to make sure it has the right parameters
            for key in optimizer_params:
                # Check if the keys in optimizer_params are the right keys
                if key not in self.optimizer_param_list[self.optimizer] + ['name', ]:
                    raise KeyError('Key %s is not a valid key' % key)
                else:
                    if key != 'name':
                        self.optimizer_params[key] = optimizer_params[key]

            # If the input was not provided then use the default value for it
            for key in self.optimizer_param_list[self.optimizer]:
                if key not in self.optimizer_params:
                    self.optimizer_params[key] = default_params[key]

    @staticmethod
    def _res_net(input_shape):
        """
        Create a ResNet for the input shape
        :param input_shape:
        :return:
        """
        # Create a ResNet
        res_cnn = keras.applications.ResNet101V2(
            include_top=False,  # We do not need the top Deeply connected Layer
            weights="imagenet",  # Weights to use will be the pretrained from imagenet
            input_tensor=None,  # If input is a Tensor, this is the better way of doing this
            input_shape=input_shape,  # Shape of the input ignoring the first axis (number of samples)
            pooling=None,
        )

        # Make these layers non-trainable to start with
        res_cnn.trainable = False

        return res_cnn

    def build(self, input_shape, num_of_responses):
        """
        Now that we know the shape
        :param input_shape:
        :param num_of_responses:
        :return:
        """

        # The output of this model would be a 2D pooled output of the CNN
        res_cnn = self._res_net(input_shape)

        # Convolution layer
        hidden = keras.layers.MaxPooling2D(padding="same")(res_cnn.output)
        for (filters, kernel_size, stride, pool_size) in self.model_gen_params['conv_layers']:
            # First time around it creates teh convulation layer based on the properties of the input
            # Next time aroudn it takes the output of the previous layer
            hidden = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(hidden)
            hidden = keras.layers.Activation(self.model_gen_params['conv_activation'])(hidden)

            # Create these hidden layers
            hidden = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(hidden)
            hidden = keras.layers.Activation(self.model_gen_params['conv_activation'])(hidden)
            hidden = keras.layers.BatchNormalization()(hidden)
            hidden = keras.layers.MaxPooling2D(pool_size)(hidden)
            hidden = keras.layers.Dropout(self.model_gen_params['conv_dropout'])(hidden)

        # Flatten the output of the convolution layers
        hidden = keras.layers.Flatten()(hidden)
        for node in self.model_gen_params['dense_layers']:
            # Add more layers to the
            hidden = keras.layers.Dense(node, activation=self.model_gen_params['dense_activation'])(hidden)
            hidden = keras.layers.BatchNormalization()(hidden)
            hidden = keras.layers.Dropout(self.model_gen_params['dense_dropout'])(hidden)

        # Add the output layer back in
        output = keras.layers.Dense(num_of_responses, activation=self.model_gen_params['dense_activation'])(hidden)
        self.model = keras.Model(inputs=res_cnn.input, outputs=output)
        self.model.summary()

        if self.optimizer == 'sgd':
            # Create the optimizer here
            optimizer = keras.optimizers.SGD(**self.optimizer_params)
        elif self.optimizer == 'adamax':
            optimizer = keras.optimizers.Adamax(**self.optimizer_params)
        else:
            raise TypeError('Unsupported Optimizer type')

        self.model.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=[keras.metrics.RootMeanSquaredError()])

        # Make predictions on testing data
        return self.model

    def model_params(self):

        # All the model generator parameters and the optimizer parameters including optimizer name
        return self.model_param_list + self.optimizer_param_list[self.optimizer] + ['name', ]

    @staticmethod
    def fit_params():

        return ['x', 'y', 'cv_data', 'save_model', 'epochs']

    @staticmethod
    def _clean_x(x):
        # Make X a 4D array
        if isinstance(x, list) or isinstance(x, tuple):
            x = list(x)
            x = np.stack(x, axis=0)

            # Add a fourth dimension to the data
            # This simply duplicates X three times
            x = np.stack((x, x, x), axis=-1)

        return x

    @staticmethod
    def _clean_y(y):
        # Take every instance of x and add an extra axis to it
        # Then concatenate it to make it a 3D array
        if isinstance(y, list) or isinstance(y, tuple):
            try:
                y = list(y)
                y = [x.flatten() for x in y]
                y = np.vstack(y)
            except AttributeError:
                y = np.array(y)

        return y

    def fit(self, x, y, cv_data, epochs=100, save_model=True, x_unlabeled=None):
        """

        :param x: Train X
        :param y: Train y
        :param cv_data: cross validation X, y data as a tuple
        :param epochs: Number of epochs to train
        :param save_model: Save the model or not
        :param x_unlabeled: This model cannot use unlabeled data so this is ignored
        :return:
        """

        if x_unlabeled is not None:
            pass

        # Make x in the format Keras is expecting
        x = self._clean_x(x)
        y = self._clean_y(y)

        cv_data[0] = self._clean_x(cv_data[0])
        cv_data[1] = self._clean_y(cv_data[1])

        if self.model is not None:
            print('Model already fit, relocate model file and restart to fit again')
            return

        # Create a model with the input shape
        self.build(x.shape[1:], y.shape[1])

        # Checkpoint the model
        if save_model:
            checkpoint = [keras.callbacks.ModelCheckpoint(
                self.model_file, save_best_only=True, monitor="val_loss", mode='min')]
        else:
            checkpoint = []

        # noinspection PyUnresolvedReferences
        self.history = self.model.fit(x, y, epochs=epochs,
                                      validation_data=tuple(cv_data),
                                      callbacks=checkpoint)

        # We just need the history dictionary of history
        self.history = self.history.history
        outdict = {'model_gen_params': self.model_gen_params, 'optimizer': self.optimizer, 'history': self.history,
                   'optimizer_params': self.optimizer_params}
        pickle.dump(outdict, open(self.model_params_file, 'wb'))

        return self.history

    def predict(self, predict_data):
        """

        :param predict_data: Train X,y data
        :return:
        """
        x = self._clean_x(predict_data)
        y_predict = self.model.predict(x)
        y_predict = [np.squeeze(x1) for x1 in np.split(y_predict, y_predict.shape[0])]

        return y_predict

    def plot_learning_curve(self, response_name='none'):
        plot_learning_curve(self.history, self.model_lc_file, response_name=response_name)


class CNN:
    """

    """

    def __init__(self, model_gen_params=None, optimizer_params=None, response_name='test', prefix='', folder='cnn'):
        """
        :param model_gen_params: Parameters for model generation
            :conv_layers: Arrangement of the convolutional layers format ((Filters, kernel_size, stride, pool_size), )
            :dense_layers: Arrangement of the convolutional layers format ((num of nodes, num of nodes), )
            :dense_activation: Activation function for the dense layer
            :conv_activation: Activation for the convolutional layers
            :conv_dropout: Dropout rate for the convolutional layers
            :dense_dropout: Dropout rate for the dense layers
        :param optimizer_params:
            Dictionary with field 'name' as the optimizer name
            Other fields contain the other optimizer parameters
                SGD:
                    :sgd_lr: Learning rate for SGD
                    :sgd_momentum: Nesterov momentum
                AdaMax:
                    :learning_rate: Learning rate for Adam
                    :beta_1: Beta 1 value
                    :beta_2: Beta 2 value
        :param response_name: Name of the response we are trying to Optimizer
        :param prefix: Prefix for the models
        """

        # Name of this response
        self.response_name = response_name
        self.model_param_list = ['dense_activation', 'conv_activation', 'dense_dropout', 'conv_dropout', 'conv_layers',
                                 'dense_layers']
        self.optimizer_param_list = {'sgd': ['learning_rate', 'momentum'],
                                     'adamax': ['learning_rate', 'beta_1', 'beta_2']}

        # Set the model generator parameters
        self.model_gen_params = {}
        self._get_model_gen_params(model_gen_params)

        # Set the model optimizer parameters
        self.optimizer = None
        self.optimizer_params = {}
        self._get_model_optimizer_params(optimizer_params)

        # This stores the history
        self.history = None

        try:
            basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        except AttributeError:
            basepath = '/content/drive/MyDrive/Colab Notebooks/cct/'

        # Prefix for the models

        if prefix:
            if prefix[-1] != '_':
                prefix = prefix + '_'
        self.prefix = prefix

        self.model_file = os.path.join(basepath, 'data/models/' + folder + '/%scnn_%s.h5' %
                                       (self.prefix, response_name))
        self.model_params_file = os.path.join(basepath, 'data/models/' + folder + '/%scnn_%s.params' %
                                              (self.prefix, response_name))
        self.model_lc_file = os.path.join(basepath, 'data/models/' + folder + '/%scnn_elc_%s.png' %
                                          (self.prefix, response_name))

        # Model exists then load it, if not then need to create one
        try:
            # Load the model
            self.model = keras.models.load_model(self.model_file)

            # Load all other parameters of this model
            all_params = pickle.load(open(self.model_params_file, 'rb'))

            # Load the history, this must exist for an already generated model
            self.history = all_params['history']

            # Load the model generator params
            self.model_gen_params = all_params['model_gen_params']

            # Load the model optimizer params
            self.optimizer_params = all_params['optimizer_params']

            # Load the optimizer
            self.optimizer = all_params['optimizer']

            print('Found model ' + self.model_file)

        except (OSError, FileNotFoundError):
            self.model = None

        self.metric = 0

    def _get_model_gen_params(self, model_gen_params):
        """

        :param model_gen_params:
        :return:
        """

        # Text book convolutional network
        conv_layers = ((64, (7, 7), 2, (2, 2)), (128, (3, 3), 1, (2, 2)), (256, (3, 3), 1, (2, 2)))
        # A lot of dense layers
        dense_layers = (500, 500)

        # Default parameters
        default_params = {'dense_activation': 'relu', 'conv_activation': 'relu',
                          'dense_dropout': 0.2, 'conv_dropout': 0.2,
                          'conv_layers': conv_layers, 'dense_layers': dense_layers}

        if model_gen_params is None:
            # If the input is None then just use the default parameters
            self.model_gen_params = default_params
        else:
            # If the model parameters were provided
            # Check the model gen params to make sure it has the right parameters
            for key in model_gen_params:
                # Check if the keys in model_gen_params are the right keys
                if key not in self.model_param_list:
                    raise KeyError('Key %s is not a valid key' % key)
                else:
                    self.model_gen_params[key] = model_gen_params[key]

            # If the input was not provided then use the default value for it
            for key in self.model_param_list:
                if key not in self.model_gen_params:
                    self.model_gen_params[key] = default_params[key]

    def _get_model_optimizer_params(self, optimizer_params):
        """

        :param optimizer_params:
        :return:
        """

        # Supported optimizers
        optimizer_list = ['sgd', 'adamax']
        # Default parameters
        default_params = {'learning_rate': 0.001, 'momentum': 0.9, 'nesterov': True,
                          'beta_1': 0.9, 'beta_2': 0.999}

        # If it None or empty
        if optimizer_params is None or not optimizer_params:
            # If the input is None then just use the default parameters
            self.optimizer = 'sgd'
            # Load all the parameters related to SGD here
            self.optimizer_params = {k: default_params[k] for k in self.optimizer_param_list[self.optimizer]}
        else:
            # This is the optimizer we use
            self.optimizer = optimizer_params['name'].lower()

            if self.optimizer not in optimizer_list:
                raise TypeError('Optimizer must be one of %s' % str(optimizer_list))

            # If the optimizer parameters were provided the load them
            # Check the optimizer params to make sure it has the right parameters
            for key in optimizer_params:
                # Check if the keys in optimizer_params are the right keys
                if key not in self.optimizer_param_list[self.optimizer] + ['name', ]:
                    raise KeyError('Key %s is not a valid key' % key)
                else:
                    if key != 'name':
                        self.optimizer_params[key] = optimizer_params[key]

            # If the input was not provided then use the default value for it
            for key in self.optimizer_param_list[self.optimizer]:
                if key not in self.optimizer_params:
                    self.optimizer_params[key] = default_params[key]

    def build(self, input_shape, num_of_responses):
        """
        Now that we know the shape
        :param input_shape:
        :param num_of_responses:
        :return:
        """

        # Create an object of input to define what it will look like
        inputs = keras.Input(shape=input_shape, name="original_img")

        # Convolution layer
        hidden = inputs
        for (filters, kernel_size, stride, pool_size) in self.model_gen_params['conv_layers']:
            # Two convolution layers
            # First one with ReLU activation
            hidden = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(hidden)
            hidden = keras.layers.Activation(self.model_gen_params['conv_activation'])(hidden)
            # Second one with batch normalization and linear activation
            # Why linear? Accidentally set it to linear and it performed well
            # Need to investigate as to why
            hidden = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(hidden)
            hidden = keras.layers.BatchNormalization()(hidden)
            hidden = keras.layers.Activation(self.model_gen_params['conv_activation'])(hidden)

            # Batch normalization, pooloing and dropout
            hidden = keras.layers.MaxPooling2D(pool_size)(hidden)
            hidden = keras.layers.Dropout(self.model_gen_params['conv_dropout'])(hidden)

        # Flatten the output of the convolution layers
        hidden = keras.layers.Flatten()(hidden)

        # Adding some dense layers here
        for node in self.model_gen_params['dense_layers']:
            hidden = keras.layers.Dense(node, activation=self.model_gen_params['dense_activation'])(hidden)
            hidden = keras.layers.BatchNormalization()(hidden)
            hidden = keras.layers.Dropout(self.model_gen_params['dense_dropout'])(hidden)

        # Add the output layer back in
        output = keras.layers.Dense(num_of_responses, activation=self.model_gen_params['dense_activation'])(hidden)
        self.model = keras.Model(inputs=inputs, outputs=output)
        self.model.summary()

        # Compile model
        # Keeping these fixed for now
        if self.optimizer == 'sgd':
            # Create the optimizer here
            optimizer = keras.optimizers.SGD(**self.optimizer_params)
        elif self.optimizer == 'adamax':
            optimizer = keras.optimizers.Adamax(**self.optimizer_params)
        else:
            raise TypeError('Unsupported Optimizer type')

        self.model.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=[keras.metrics.RootMeanSquaredError()])

        # Make predictions on testing data
        return self.model

    def model_params(self):

        # All the model generator parameters and the optimizer parameters including optimizer name
        return self.model_param_list + self.optimizer_param_list[self.optimizer] + ['name', ]

    @staticmethod
    def fit_params():

        return ['x', 'y', 'cv_data', 'save_model', 'epochs']

    @staticmethod
    def _clean_x(x):
        # Make X a 4D array
        if isinstance(x, list) or isinstance(x, tuple):
            x = list(x)
            x = np.stack(x, axis=0)

            # Add a fourth dimension to the data
            x = x[:, :, :, np.newaxis]

        return x

    @staticmethod
    def _clean_y(y):
        # Take every instance of x and add an extra axis to it
        # Then concatenate it to make it a 3D array
        if isinstance(y, list) or isinstance(y, tuple):
            try:
                y = list(y)
                y = [x.flatten() for x in y]
                y = np.vstack(y)
            except AttributeError:
                y = np.array(y)

        return y

    def fit(self, x, y, cv_data, epochs=300, save_model=True, sample_weight=None, x_unlabeled=None):
        """

        :param x: Train X
        :param y: Train y
        :param cv_data: cross validation X, y data as a tuple
        :param epochs: Number of epochs to train
        :param save_model: Save the model or not
        :param sample_weight: Weight of each sample
        :param x_unlabeled: This model cannot use unlabeled data so this is ignored
        :return:
        """

        if x_unlabeled is not None:
            pass

        # Make x in the format Keras is expecting
        x = self._clean_x(x)
        y = self._clean_y(y)

        cv_data[0] = self._clean_x(cv_data[0])
        cv_data[1] = self._clean_y(cv_data[1])

        if self.model is not None:
            print('Model already fit, relocate model file and restart to fit again')
            return

        # Create a model with the input shape
        self.build(x.shape[1:], y.shape[1])

        # Checkpoint the model
        if save_model:
            checkpoint = [keras.callbacks.ModelCheckpoint(
                self.model_file, save_best_only=True, monitor="val_loss", mode='min')]
        else:
            checkpoint = []

        if sample_weight is None:
            # noinspection PyUnresolvedReferences
            self.history = self.model.fit(x, y, epochs=epochs,
                                          validation_data=tuple(cv_data),
                                          callbacks=checkpoint)
        else:
            sample_weight = self._clean_y(sample_weight)
            # noinspection PyUnresolvedReferences
            self.history = self.model.fit(x, y, epochs=epochs,
                                          validation_data=tuple(cv_data),
                                          callbacks=checkpoint,
                                          sample_weight=sample_weight)

        # We just need the history dictionary of history
        self.history = self.history.history
        outdict = {'model_gen_params': self.model_gen_params, 'optimizer': self.optimizer, 'history': self.history,
                   'optimizer_params': self.optimizer_params}
        pickle.dump(outdict, open(self.model_params_file, 'wb'))

        return self.history

    def predict(self, predict_data):
        """

        :param predict_data: Train X,y data
        :return:
        """
        x = self._clean_x(predict_data)
        y_predict = self.model.predict(x)
        y_predict = [np.squeeze(x1) for x1 in np.split(y_predict, y_predict.shape[0])]

        return y_predict

    def plot_learning_curve(self, response_name='none'):
        plot_learning_curve(self.history, self.model_lc_file, response_name=response_name)


class SemiSuper:
    def __init__(self, model_gen_params=None, optimizer_params=None, response_name='test', prefix=''):
        self.model_class = CNN
        self.model_gen_params = model_gen_params
        self.optimizer_params = optimizer_params
        self.response_name = response_name

        # Prefix for the models
        if prefix:
            if prefix[-1] != '_':
                prefix = prefix + '_'
        self.prefix = prefix

        # Create a model for
        self.model = None
        self.history = None

        # Check if the m2 model already exists
        m2 = self.model_class(model_gen_params=self.model_gen_params, optimizer_params=self.optimizer_params,
                              response_name=self.response_name, prefix=self.prefix + 'm2', folder='semisup')

        # If the second model has already been tried then we are done
        if m2.model is not None:
            print(m2.model)
            self.model = m2
            self.history = m2.history

        self.metric = None

    @staticmethod
    def _clean_x(x):
        # Make X a 4D array
        if isinstance(x, list) or isinstance(x, tuple):
            x = list(x)
            x = np.stack(x, axis=0)

            # Add a fourth dimension to the data
            x = x[:, :, :, np.newaxis]

        return x

    @staticmethod
    def _clean_y(y):
        # Take every instance of x and add an extra axis to it
        # Then concatenate it to make it a 3D array
        if isinstance(y, list) or isinstance(y, tuple):
            try:
                y = list(y)
                y = [x.flatten() for x in y]
                y = np.vstack(y)
            except AttributeError:
                y = np.array(y)

        return y

    def fit(self, x, y, x_unlabeled, cv_data, epochs=300, unlabeled_weight_factor=1, save_model=True, num_transforms=0):
        """

        :param x: Train X
        :param y: Train y
        :param x_unlabeled: x_unlabeled
        :param cv_data: cross validation X, y data as a tuple
        :param epochs: Number of epochs to train
        :param unlabeled_weight_factor: What factor to give the RMSE of the unlabeled data?
        :param save_model: Save the model or not
        :param x_unlabeled: This model cannot use unlabeled data so this is ignored
        :param num_transforms: Number of transforms to perform on the augmented data
        :return:
        """
        if self.model is not None:
            print('Model already fit, relocate model file and restart to fit again')
            return

        # The first CNN model
        print('Working on first CNN with %d samples' % len(x))
        cnn1 = self.model_class(model_gen_params=self.model_gen_params, optimizer_params=self.optimizer_params,
                                response_name=self.response_name, prefix=self.prefix + 'm1', folder='semisup')
        cnn1.fit(x, y, cv_data=[cv_data[0], cv_data[1]], epochs=epochs, save_model=save_model)

        print("learning curve for model 1")
        cnn1.plot_learning_curve(response_name=self.response_name)

        # Make a prediction with the model on the
        print('Making prediction on unlabeled')
        y_predicted = cnn1.predict(x_unlabeled)

        # Change it into a DataFrame for augmentation
        x_unlabeled = [np.squeeze(x1) for x1 in x_unlabeled]
        df = pd.DataFrame(list(zip(x_unlabeled, y_predicted)), columns=['X', 'y'])

        # Create an Augmentor
        augment = AugmentData(num_transforms=num_transforms)
        semisup_df = augment.augment_one(data_df=df, response_name=self.response_name, force_create=True)

        # Now compare the number of samples from the original data with this and find the ratio
        # This ratio now describes the sample weight that must be given to make the new model better
        sample_ratio = len(x)/semisup_df.shape[0] * unlabeled_weight_factor
        # sample_ratio = unlabeled_weight_factor
        print('Sample size %d augment size %d unlabeled weight factor %.2f augmented data weight %.2f' %
              (len(x), semisup_df.shape[0], unlabeled_weight_factor, sample_ratio))

        # These are the sample weights that will be applied
        sample_weight = np.hstack((np.ones(len(x), ), np.ones(semisup_df.shape[0], )*sample_ratio))

        # Convert the original data into a DataFrame for stacking
        # Change it into a DataFrame for augmentation
        orig_df = pd.DataFrame(list(zip(x, y)), columns=['X', 'y'])

        # Stack the DataFrames with the original on top
        data_df = pd.concat((orig_df, semisup_df))
        data_df['sample_weight'] = sample_weight

        # Shuffle it so that batches do not get the same data
        data_df = data_df.sample(frac=1).reset_index(drop=True)
        x = data_df['X'].tolist()
        y = data_df['y'].tolist()
        sample_weight = data_df['sample_weight'].tolist()

        # Create a model with the input shape
        print('Training second CNN with %d samples' % len(x))
        m2 = self.model_class(model_gen_params=self.model_gen_params, optimizer_params=self.optimizer_params,
                              response_name=self.response_name, prefix=self.prefix + 'm2_', folder='semisup')
        self.history = m2.fit(x, y, cv_data=[cv_data[0], cv_data[1]], epochs=epochs, sample_weight=sample_weight,
                              save_model=save_model)
        print('Learning curve for model 2')
        m2.plot_learning_curve(response_name=self.response_name)

        # noinspection PyUnresolvedReferences
        self.model = m2
        return self.history

    def predict(self, predict_data):
        """

        :param predict_data: Train X,y data
        :return:
        """

        predict = self.model.predict(predict_data=predict_data)
        self.metric = self.model.metric

        return predict

    def plot_learning_curve(self, response_name='none'):
        """
        Plot the learning curve using learning history
        :return:
        """
        return self.model.plot_learning_curve(response_name=response_name)


class Corrector:
    """

    """

    def __init__(self, model_gen_params=None, optimizer_params=None, response_name='all', prefix='',
                 folder='corrector'):
        """
        :param model_gen_params: Parameters for model generation
            :conv_layers: Arrangement of the convolutional layers format ((Filters, kernel_size, stride, pool_size), )
            :dense_layers: Arrangement of the convolutional layers format ((num of nodes, num of nodes), )
            :dense_activation: Activation function for the dense layer
            :conv_activation: Activation for the convolutional layers
            :conv_dropout: Dropout rate for the convolutional layers
            :dense_dropout: Dropout rate for the dense layers
        :param optimizer_params:
            Dictionary with field 'name' as the optimizer name
            Other fields contain the other optimizer parameters
                SGD:
                    :sgd_lr: Learning rate for SGD
                    :sgd_momentum: Nesterov momentum
                AdaMax:
                    :learning_rate: Learning rate for Adam
                    :beta_1: Beta 1 value
                    :beta_2: Beta 2 value
        :param response_name: Name of the response we are trying to Optimizer
        :param prefix: Prefix for the models
        """

        # Name of this response
        self.response_name = response_name
        self.model_param_list = ['dense_activation', 'conv_activation', 'dense_dropout', 'conv_dropout', 'conv_layers',
                                 'dense_layers', 'reduced_nodes']
        self.optimizer_param_list = {'sgd': ['learning_rate', 'momentum'],
                                     'adamax': ['learning_rate', 'beta_1', 'beta_2']}

        # Set the model generator parameters
        self.model_gen_params = {}
        self._get_model_gen_params(model_gen_params)

        # Set the model optimizer parameters
        self.optimizer = None
        self.optimizer_params = {}
        self._get_model_optimizer_params(optimizer_params)

        # This stores the history
        self.history = None

        try:
            basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        except AttributeError:
            basepath = '/content/drive/MyDrive/Colab Notebooks/cct/'

        # Prefix for the models

        if prefix:
            if prefix[-1] != '_':
                prefix = prefix + '_'
        self.prefix = prefix

        self.model_file = os.path.join(basepath, 'data/models/' + folder + '/%scnn_%s.h5' %
                                       (self.prefix, response_name))
        self.model_params_file = os.path.join(basepath, 'data/models/' + folder + '/%scnn_%s.params' %
                                              (self.prefix, response_name))
        self.model_lc_file = os.path.join(basepath, 'data/models/' + folder + '/%scnn_elc_%s.png' %
                                          (self.prefix, response_name))

        # Model exists then load it, if not then need to create one
        try:
            # Load the model
            self.model = keras.models.load_model(self.model_file)

            # Load all other parameters of this model
            all_params = pickle.load(open(self.model_params_file, 'rb'))

            # Load the history, this must exist for an already generated model
            self.history = all_params['history']

            # Load the model generator params
            self.model_gen_params = all_params['model_gen_params']

            # Load the model optimizer params
            self.optimizer_params = all_params['optimizer_params']

            # Load the optimizer
            self.optimizer = all_params['optimizer']

            print('Found model ' + self.model_file)

        except (OSError, FileNotFoundError):
            self.model = None

        self.metric = 0

    def _get_model_gen_params(self, model_gen_params):
        """

        :param model_gen_params:
        :return:
        """

        # Text book convolutional network
        conv_layers = None
        # A lot of dense layers
        dense_layers = None

        # Default parameters
        default_params = {'dense_activation': 'relu', 'conv_activation': 'relu',
                          'dense_dropout': 0.2, 'conv_dropout': 0.2,
                          'conv_layers': conv_layers, 'dense_layers': dense_layers,
                          'reduced_nodes': 100}

        if model_gen_params is None:
            # If the input is None then just use the default parameters
            self.model_gen_params = default_params
        else:
            # If the model parameters were provided
            # Check the model gen params to make sure it has the right parameters
            for key in model_gen_params:
                # Check if the keys in model_gen_params are the right keys
                if key not in self.model_param_list:
                    raise KeyError('Key %s is not a valid key' % key)
                else:
                    self.model_gen_params[key] = model_gen_params[key]

            # If the input was not provided then use the default value for it
            for key in self.model_param_list:
                if key not in self.model_gen_params:
                    self.model_gen_params[key] = default_params[key]

    def _get_model_optimizer_params(self, optimizer_params):
        """

        :param optimizer_params:
        :return:
        """

        # Supported optimizers
        optimizer_list = ['sgd', 'adamax']
        # Default parameters
        default_params = {'learning_rate': 0.001, 'momentum': 0.9, 'nesterov': True,
                          'beta_1': 0.9, 'beta_2': 0.999}

        # If it None or empty
        if optimizer_params is None or not optimizer_params:
            # If the input is None then just use the default parameters
            self.optimizer = 'sgd'
            # Load all the parameters related to SGD here
            self.optimizer_params = {k: default_params[k] for k in self.optimizer_param_list[self.optimizer]}
        else:
            # This is the optimizer we use
            self.optimizer = optimizer_params['name'].lower()

            if self.optimizer not in optimizer_list:
                raise TypeError('Optimizer must be one of %s' % str(optimizer_list))

            # If the optimizer parameters were provided the load them
            # Check the optimizer params to make sure it has the right parameters
            for key in optimizer_params:
                # Check if the keys in optimizer_params are the right keys
                if key not in self.optimizer_param_list[self.optimizer] + ['name', ]:
                    raise KeyError('Key %s is not a valid key' % key)
                else:
                    if key != 'name':
                        self.optimizer_params[key] = optimizer_params[key]

            # If the input was not provided then use the default value for it
            for key in self.optimizer_param_list[self.optimizer]:
                if key not in self.optimizer_params:
                    self.optimizer_params[key] = default_params[key]

    def build(self, input_shape, num_of_responses):
        """
        Now that we know the shape
        :param input_shape:
        :param num_of_responses:
        :return:
        """

        # Create an object of input to define what it will look like
        inputs = keras.Input(shape=input_shape, name="original_img")

        # Check if the convolutional layers are

        # Convolution layer
        hidden = inputs
        for (filters, kernel_size, stride, pool_size) in self.model_gen_params['conv_layers']:
            # First time around it creates teh convulation layer based on the properties of the input
            # Next time aroudn it takes the output of the previous layer
            hidden = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding="same", activation="relu")(
                hidden)

            # Create these hidden layers
            hidden = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(hidden)
            hidden = keras.layers.BatchNormalization()(hidden)
            hidden = keras.layers.Activation(self.model)(hidden)
            hidden = keras.layers.MaxPooling2D(pool_size)(hidden)
            hidden = keras.layers.Dropout(self.model_gen_params['conv_dropout'])(hidden)

        # Flatten the output of the convolution layers
        hidden = keras.layers.Flatten()(hidden)

        for node in self.model_gen_params['dense_layers']:
            # Add more layers to the
            hidden = keras.layers.Dense(node, activation=self.model_gen_params['dense_activation'])(hidden)
            hidden = keras.layers.BatchNormalization()(hidden)
            hidden = keras.layers.Dropout(self.model_gen_params['dense_dropout'])(hidden)

        # Add the output layer back in
        output = keras.layers.Dense(num_of_responses, activation=self.model_gen_params['dense_activation'])(hidden)
        self.model = keras.Model(inputs=inputs, outputs=output)
        self.model.summary()

        # Compile model
        # Keeping these fixed for now
        if self.optimizer == 'sgd':
            # Create the optimizer here
            optimizer = keras.optimizers.SGD(**self.optimizer_params)
        elif self.optimizer == 'adamax':
            optimizer = keras.optimizers.Adamax(**self.optimizer_params)
        else:
            raise TypeError('Unsupported Optimizer type')

        self.model.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=[keras.metrics.RootMeanSquaredError()])

        # Make predictions on testing data
        return self.model

    def model_params(self):

        # All the model generator parameters and the optimizer parameters including optimizer name
        return self.model_param_list + self.optimizer_param_list[self.optimizer] + ['name', ]

    @staticmethod
    def fit_params():

        return ['x', 'y', 'cv_data', 'save_model', 'epochs']

    @staticmethod
    def _clean_x(x):
        # Make X a 4D array
        if isinstance(x, list) or isinstance(x, tuple):
            x = list(x)
            x = np.stack(x, axis=0)

            # Add a fourth dimension to the data
            x = x[:, :, :, np.newaxis]

        return x

    @staticmethod
    def _clean_y(y):
        # Take every instance of x and add an extra axis to it
        # Then concatenate it to make it a 3D array
        if isinstance(y, list) or isinstance(y, tuple):
            try:
                y = list(y)
                y = [x.flatten() for x in y]
                y = np.vstack(y)
            except AttributeError:
                y = np.array(y)

        return y

    def fit(self, x, y, cv_data, epochs=300, save_model=True, sample_weight=None, x_unlabeled=None):
        """

        :param x: Train X
        :param y: Train y
        :param cv_data: cross validation X, y data as a tuple
        :param epochs: Number of epochs to train
        :param save_model: Save the model or not
        :param sample_weight: Weight of each sample
        :param x_unlabeled: This model cannot use unlabeled data so this is ignored
        :return:
        """

        if x_unlabeled is not None:
            pass

        # Make x in the format Keras is expecting
        x = self._clean_x(x)
        y = self._clean_y(y)

        cv_data[0] = self._clean_x(cv_data[0])
        cv_data[1] = self._clean_y(cv_data[1])

        if self.model is not None:
            print('Model already fit, relocate model file and restart to fit again')
            return

        # Create a model with the input shape
        self.build(x.shape[1:], y.shape[1])

        # Checkpoint the model
        if save_model:
            checkpoint = [keras.callbacks.ModelCheckpoint(
                self.model_file, save_best_only=True, monitor="val_loss", mode='min')]
        else:
            checkpoint = []

        if sample_weight is None:
            # noinspection PyUnresolvedReferences
            self.history = self.model.fit(x, y, epochs=epochs,
                                          validation_data=tuple(cv_data),
                                          callbacks=checkpoint)
        else:
            sample_weight = self._clean_y(sample_weight)
            # noinspection PyUnresolvedReferences
            self.history = self.model.fit(x, y, epochs=epochs,
                                          validation_data=tuple(cv_data),
                                          callbacks=checkpoint,
                                          sample_weight=sample_weight)

        # We just need the history dictionary of history
        self.history = self.history.history
        outdict = {'model_gen_params': self.model_gen_params, 'optimizer': self.optimizer, 'history': self.history,
                   'optimizer_params': self.optimizer_params}
        pickle.dump(outdict, open(self.model_params_file, 'wb'))

        return self.history

    def predict(self, predict_data):
        """

        :param predict_data: Train X,y data
        :return:
        """
        x = self._clean_x(predict_data)
        y_predict = self.model.predict(x)
        y_predict = np.split(y_predict, y_predict.shape[0])

        return y_predict

    def plot_learning_curve(self, response_name='none'):
        plot_learning_curve(self.history, self.model_lc_file, response_name=response_name)


if __name__ == '__main__':

    do_lnn = False
    if do_lnn:
        # Scenario 1: default
        lnn = CNN()
        print('Model generator parameters are:')
        print(lnn.model_gen_params)
        print('Model optimizer is:')
        print(lnn.optimizer)
        print('Model optimizer parameters are:')
        print(lnn.optimizer_params)
        print('\n\n#####')

        # Scenario 2: Some model gen inputs provided externally
        mgp = {'dense_layers': (400, 400)}
        lnn = CNN(model_gen_params=mgp)
        print('Model generator parameters are:')
        print(lnn.model_gen_params)
        print('Model optimizer is:')
        print(lnn.optimizer)
        print('Model optimizer parameters are:')
        print(lnn.optimizer_params)
        print('\n\n#####')

        # Scenario 3: Some model gen inputs provided externally and some optimizer parameters provide externally
        mgp = {'dense_layers': (400, 400)}
        opt = {'name': 'adamax', 'beta_1': 10}
        lnn = CNN(model_gen_params=mgp, optimizer_params=opt)
        print('Model generator parameters are:')
        print(lnn.model_gen_params)
        print('Model optimizer is:')
        print(lnn.optimizer)
        print('Model optimizer parameters are:')
        print(lnn.optimizer_params)

        lnn.build(input_shape=(96, 96, 1), num_of_responses=2)

    do_resnet = True
    if do_resnet:
        # Scenario 1: default
        resnet = ResNet()
        print('Model generator parameters are:')
        print(resnet.model_gen_params)
        print('Model optimizer is:')
        print(resnet.optimizer)
        print('Model optimizer parameters are:')
        print(resnet.optimizer_params)
        print('\n\n#####')

        # Scenario 2: Some model gen inputs provided externally
        mgp = {'dense_layers': (400, 400)}
        resnet = ResNet(model_gen_params=mgp)
        print('Model generator parameters are:')
        print(resnet.model_gen_params)
        print('Model optimizer is:')
        print(resnet.optimizer)
        print('Model optimizer parameters are:')
        print(resnet.optimizer_params)
        print('\n\n#####')

        # Scenario 3: Some model gen inputs provided externally and some optimizer parameters provide externally
        mgp = {'dense_layers': (400, 400)}
        opt = {'name': 'adamax', 'beta_1': 10}
        resnet = ResNet(model_gen_params=mgp, optimizer_params=opt)
        print('Model generator parameters are:')
        print(resnet.model_gen_params)
        print('Model optimizer is:')
        print(resnet.optimizer)
        print('Model optimizer parameters are:')
        print(resnet.optimizer_params)

        resnet.build(input_shape=(96, 96, 3), num_of_responses=2)
