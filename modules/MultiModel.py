import os
import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from modules.transforms import warp
from modules.data_augmentation import AugmentData


class MultiModel:
    """
    Simple wrapper class for the models to create multiple models
    """

    def __init__(self, model, model_g1=None, prefix='', eliminate=None, augment_params=None,
                 response_names=None, ):
        """
        There are two groups of data points:
            Group 0: Has ~30% of labeled data points
            Group 1: Has ~100% of labeled data points

        :param model: Model class applied to Group 1 in models.py. One of Means, CNN, ResNet, SimpleNN ...
        :param model_g1: Model class applied to Group 2. If not provide then Group 1 model class is used
        :param prefix: Prefix to give filenames for this particular run
        :param eliminate: Which features to eliminate from Group 0 or Group 1. Dictionary of the form:
                          {0: [<feature 1>, <feature 2>, ...], 0: [<feature 1>, <feature 2>, ...]}
                          For example  if you want to eliminate all "nose_tip" Gorup 0 data points from training and CV
                          then {0: ['nose_tip', ], 1: []}
        :param augment_params: Parameters sent to the Augmentation class, dictionary is unpacked to keywords.
                               Default(None) is to use the defaults of the class
        :param response_names: Which response names to model, if None then all responses
        """
        # What models to use for Group 1(few samples) and group2 (lot of samples)
        self._model = model
        self._model_g1 = model if model_g1 is None else model_g1
        self.prefix = prefix
        self.class_name = str(self._model).split('.')[-1].split('\'')[0]
        self.optimal_params = None
        self.group_optimal = None
        self.models = {}
        self.responses = []
        self.eliminate = {0: [], 1: []} if eliminate is None else eliminate

        # Make augment params an empty dictionary if the input is None
        self.augment_params = augment_params if augment_params is not None else {}

        # Which response names to
        all_response_names = ['mouth_center_bottom_lip', 'nose_tip', 'left_eye_center', 'right_eyebrow_outer_end',
                              'left_eyebrow_outer_end', 'right_eye_outer_corner', 'right_eye_inner_corner',
                              'left_eye_inner_corner', 'mouth_right_corner', 'mouth_left_corner',
                              'left_eyebrow_inner_end', 'mouth_center_top_lip',
                              'right_eyebrow_inner_end', 'left_eye_outer_corner', 'right_eye_center']
        self.response_names = response_names if response_names is not None else all_response_names

        self.group = (['left_eyebrow_outer_end', 'right_eye_outer_corner', 'right_eye_inner_corner',
                       'left_eye_inner_corner', 'mouth_right_corner', 'mouth_left_corner',
                       'right_eyebrow_outer_end', 'left_eyebrow_inner_end', 'mouth_center_top_lip',
                       'right_eyebrow_inner_end', 'left_eye_outer_corner'],
                      ['mouth_center_bottom_lip', 'nose_tip', 'right_eye_center', 'left_eye_center'])

        if prefix:
            prefix = prefix + '_'
        self.prefix = prefix

        try:
            basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        except AttributeError:
            basepath = '/content/drive/MyDrive/Colab Notebooks/cct/'

        self.outpath = os.path.join(basepath, 'data/analysis')
        self.subpath = os.path.join(basepath, 'data/')

    def _get_model_and_fit_params(self, response_name, model_params_dict, fit_params_dict, optimizer_params_dict,
                                  do_optimal, group_optimal):
        """

        :param response_name:
        :param model_params_dict:
        :param fit_params_dict:
        :param optimizer_params_dict:
        :param group_optimal:
        :return:
        """

        if self.optimal_params is not None and do_optimal:

            if not group_optimal:
                # This is the optimal parameter dictionary
                opt_params = copy.deepcopy(self.optimal_params[response_name])
                opt_params.update(model_params_dict)
                opt_params.update(fit_params_dict)

                # These are the model params and are marked by the keyword model: in front of it
                model_params = {k.split(':')[1]: v for k, v in opt_params.items() if 'model:' in k}

                # These are the fit params and are marked by the keyword fit: in front of it
                fit_params = {k.split(':')[1]: v for k, v in opt_params.items() if 'fit:' in k}

                # These are optimizer_params fit params and are marked by the keyword optimizer: in front of it
                opt_params = {k.split(':')[1]: v for k, v in opt_params.items() if 'optimizer:' in k}

            else:
                # This is the optimal parameter dictionary
                if response_name in self.group[0]:
                    opt_params = copy.deepcopy(self.group_optimal[0])
                else:
                    opt_params = copy.deepcopy(self.group_optimal[1])

                opt_params.update(model_params_dict)
                opt_params.update(fit_params_dict)

                # These are the model params and are marked by the keyword model: in front of it
                model_params = {k.split(':')[1]: v for k, v in opt_params.items() if 'model:' in k}

                # These are the fit params and are marked by the keyword fit: in front of it
                fit_params = {k.split(':')[1]: v for k, v in opt_params.items() if 'fit:' in k}

                # These are optimizer_params fit params and are marked by the keyword optimizer: in front of it
                opt_params = {k.split(':')[1]: v for k, v in opt_params.items() if 'optimizer:' in k}
        else:
            model_params = model_params_dict
            fit_params = fit_params_dict
            opt_params = optimizer_params_dict

        return model_params, fit_params, opt_params

    def load(self, response_names):
        """

        :param response_names: Dictionary of train DataFrames
        :return:
        """

        # Create a model for each of the responses
        for response_name in response_names:
            # depending on the group, the model is different
            if response_name in self.group[0]:
                _model = self._model
            else:
                _model = self._model_g1

            # Create a model for these optimization parameters
            this_model = _model(response_name=response_name, prefix=self.prefix)

            # Create and instance of the model
            self.models[response_name] = this_model

    def _get_group(self, df):
        # Where are all the NAN's in the data?
        nans = None
        for response in self.response_names:
            tmp = df[response].apply(lambda x1: np.any(np.isnan(x1))).values
            nans = np.logical_or(nans, tmp)

        # As group 0 is defined as one with nan's flip it around
        not_all_nans = np.logical_not(nans)

        return not_all_nans.astype(int)

    def _get_model_data(self, df, response_name, additional_columns=None):
        """

        :param df: DataFrame to extract data from
        :param response_name: Response name to extract
        :param additional_columns: List of additional columns
        :return:
        """

        if additional_columns is None:
            additional_columns = []

        # If these values are available then return it as part of the return df
        keep_columns = [x for x in ['unique_id', 'X', response_name + '_t', 'group'] if x in df.columns]
        return_df = df[keep_columns + additional_columns].copy()

        # Numpy arrays are not deep copied in functions so we need to explicitly do that
        return_df['X'] = return_df['X'].apply(lambda x: x.copy())

        #
        if response_name + '_t' in df.columns:
            if 'group' in return_df.columns:
                # Eliminate Group 0 or 1 as defined in eliminate
                # 1. Eliminate group 0
                eliminate_group = self.eliminate[0]
                # Eliminate
                if response_name in eliminate_group:
                    print('Group 0 eliminated from response %s' % response_name)
                    index = return_df['group'] != 0
                    return_df = return_df[index]

                # 2. Eliminate Group 1
                eliminate_group = self.eliminate[1]
                # Eliminate
                if response_name in eliminate_group:
                    print('Group 1 eliminated from response %s' % response_name)
                    index = return_df['group'] != 1
                    return_df = return_df[index]

            # Rename the response column to y
            return_df = return_df.rename(columns={response_name + '_t': 'y'})

            # Separate out NAN and not NAN data
            nans = return_df['y'].apply(lambda x: np.any(np.isnan(x))).values
            not_nans = np.logical_not(nans)
            not_nan_df = return_df[not_nans]
            nan_df = return_df[nans]

            return not_nan_df, nan_df
        else:
            return return_df, None

    def _model_fit(self, augmented, cv, unlabeled, response_name, model_params, opt_params, fit_params):
        """
        Fit and train the model

        :param augmented: Augmented data for training as a DataFrame
        :param cv: CV data for training also as a DataFrame
        :param unlabeled: Unlabeled data for training also as a DataFrame
        :param response_name: Response name that is being fit
        :param model_params: Parameters for the model
        :param opt_params: Parameters for the optimizer
        :return:
        """

        # depending on the group, the model is different
        if response_name in self.group[0]:
            _model = self._model
        else:
            _model = self._model_g1

        # Create a model for these optimization parameters
        this_model = _model(model_gen_params=model_params,
                            optimizer_params=opt_params,
                            response_name=response_name,
                            prefix=self.prefix)

        # Fit it with the fit parameters
        this_model.fit(x=augmented['X'].to_list(), y=augmented['y'].to_list(),
                       cv_data=[cv['X'].to_list(), cv['y'].to_list()],
                       x_unlabeled=unlabeled['X'].to_list(),
                       **fit_params)

        return this_model

    def fit(self, train_data, cv_data, model_params_dict=None, fit_params_dict=None, optimizer_params_dict=None):
        """

        :param train_data: Dictionary of train DataFrames
        :param cv_data: cv_data to check model performance
        :param model_params_dict: Parameters for model initialization
        :param fit_params_dict: parameters for fitting
        :param optimizer_params_dict: Model optimization parameters
        :return:
        """

        if model_params_dict is None:
            model_params_dict = {}

        if fit_params_dict is None:
            fit_params_dict = {}

        # Get the train data group
        train_data['group'] = self._get_group(train_data)
        cv_data['group'] = self._get_group(cv_data)

        # Create a model for each of the responses
        cnt = 1
        for response_name in self.response_names:
            # Get the dataframe for this response
            print('Fitting model %s which is %d/%d' % (response_name, cnt, len(self.response_names)))

            # Get the labeled and unlabeled train data
            labeled_df, unlabeled_df = self._get_model_data(train_data, response_name)

            # Get the labeled and unlabeled data
            cv_df, _ = self._get_model_data(cv_data, response_name)

            # Create an object of the Augment data class
            # 1. Get the augment parameters
            if response_name in self.augment_params:
                # If there is augment parameeter per response
                augment_params = self.augment_params[response_name]
            else:
                # If there is one augment parameters for all
                augment_params = self.augment_params

            # Create an augment data object with input parameters
            augment = AugmentData(**augment_params)

            #  Augment the data with the number of transforms
            augmented = augment.augment_one(labeled_df, response_name=response_name)

            #  Get the model and fit parameters for this model
            model_params, fit_params, opt_params = self._get_model_and_fit_params(
                response_name, model_params_dict, fit_params_dict, optimizer_params_dict,
                do_optimal=False, group_optimal=False)

            # Fit this model
            this_model = self._model_fit(augmented, cv_df, unlabeled_df, response_name, model_params, opt_params,
                                         fit_params)

            # Accumulate the models here
            self.models[response_name] = this_model
            print('Completed fitting')
            cnt += 1

    @staticmethod
    def _reformat(y_pred, debug=False):
        if isinstance(y_pred, np.ndarray):
            ret_val = pd.Series([x for x in y_pred])
            if debug:
                print('Return y_pred first 10', str(y_pred[:10]))
                print('Return shape:', ret_val.shape)
                print('Series 10:', ret_val.values[:10])
            return ret_val
        return y_pred

    def _process_predict_input(self, predict_data, response_name):
        """

        :param predict_data:
        :param response_name:
        :return:
        """

        # We want to keep these columns for later use depending on whether it is test data or submission
        keep_columns = ['ImageId', 'inverse', response_name, 'orig_Image']
        keep_columns = [x for x in keep_columns if x in predict_data.columns]
        labeled_df, _ = self._get_model_data(predict_data, response_name, additional_columns=keep_columns)

        # This is the true y value
        if response_name in predict_data.columns:
            labeled_df.rename(columns={response_name: 'y_true'}, inplace=True)

        return labeled_df

    def predict(self, predict_df, submission=False):
        """
        :param predict_df:  predict DataFrame
        :param submission: Is this a submission file?
        :return:
        """

        # Create a model for each of the responses
        metrics = {}
        responses = pd.DataFrame()

        for response_name in self.response_names:
            if response_name not in predict_df.columns:
                break
            # Add the column group to the data
            predict_df['group'] = self._get_group(predict_df)

        for response_name, model in self.models.items():

            # Get the data to predict
            labeled_df = self._process_predict_input(predict_df, response_name)
            print(labeled_df.columns)

            # Reset the index for assignment to be easy
            labeled_df = labeled_df.reset_index(drop=True)

            # Predict an dAssign it to the labeled DataFrame
            labeled_df['y_pred'] = model.predict(labeled_df['X'].to_list())
            labeled_df['y_invert'] = labeled_df[['inverse', 'y_pred']].apply(lambda x: warp(x[0], x[1]), axis=1)

            # Check if y_true is in column, if so then calculate rmse
            if 'y_true' in labeled_df:
                metrics[response_name] = \
                    mean_squared_error(np.vstack(labeled_df['y_true'].to_list()),
                                       np.vstack(labeled_df['y_invert'].to_numpy()))**0.5
                model.metric = metrics[response_name]

            # If this is a submission file then we need to add Location and FeatureName to the data
            # This doubles the DataFrame length as response gets split into its x and y components
            if submission:
                # Accumulate the x and y as separate DataFrames and concatenate it together
                accum_df = pd.DataFrame()
                for cnt, suffix in enumerate(['_x', '_y']):
                    tmp_df = labeled_df.copy()
                    tmp_df['Location'] = tmp_df['y_invert'].apply(lambda x: x[cnt])
                    tmp_df['FeatureName'] = response_name + suffix
                    accum_df = pd.concat((accum_df, tmp_df))
                labeled_df = accum_df

            # Update the model with the metric
            responses = pd.concat((responses, labeled_df))

        return responses, metrics

    def create_submission(self, submission_data):
        """
        Create a submission file with the created models
        :param submission_data: Output of SourceData's source_test_csv
        :return:
        """

        outfile = os.path.join(self.subpath, '%sMySubmission.csv' % self.class_name)

        infile = os.path.join(self.subpath, 'IdLookupTable.csv')
        submission_lookup = pd.read_csv(infile)

        # Make a prediction on it
        responses, _ = self.predict(submission_data, submission=True)

        # Convert the output into a DataFrame for submission
        responses = pd.DataFrame(responses)

        # Merge responses
        submission = submission_lookup.merge(responses, left_on=['ImageId', 'FeatureName'],
                                             right_on=['ImageId', 'FeatureName'], how='left')

        # Now create the output submission file as CSV
        submission[['RowId', 'Location_y']].rename(columns={'Location_y': 'Location'}).to_csv(outfile, index=False)

        return submission

    def plot_learning_curve(self, folder,  file_prefix, logbase=10):
        """
        Plot the learning curve using learning history
        :param folder: Folder with files
        :param file_prefix: Prefix to each file
        :param logbase:
        :return:
        """

        # Make it a square grid
        num_xs = int(len(self.response_names) ** 0.5 + 0.5)
        num_ys = num_xs

        fig, axs = plt.subplots(num_xs, num_ys, figsize=(15, 15 * (num_xs / num_ys) + 1))

        # Convert
        cnt = 0
        lowest_rmse = []
        response_names = self.response_names
        for x in range(num_xs):
            for y in range(num_ys):
                try:
                    file = os.path.join(self.subpath, 'models', self.class_name.lower(), folder, file_prefix + '_' +
                                        self.class_name.lower() + '_' + response_names[cnt] + '.params')
                    with open(file, 'rb') as infi:
                        params = pickle.load(infi)

                    # Get the models history in a DataFrame
                    df = pd.DataFrame(params['history'])
                    epoch = list(range(df.shape[0]))

                    # Turn teh grid on to make it easily readable
                    axs[x, y].grid(True)

                    # Create the RMSE plots here
                    # Use a log scale so that the slope in teh curve is visible
                    lowest_rmse.append(df['val_root_mean_squared_error'].min())
                    axs[x, y].loglog(epoch, df['root_mean_squared_error'], linestyle='-', color='k', base=logbase)
                    axs[x, y].loglog(epoch, df['val_root_mean_squared_error'], linestyle='-', color='b', base=logbase)
                    axs[x, y].legend(['Train rmse', 'Validation rmse'])
                    axs[x, y].set_title('%s rmse %.2f' %
                                        (response_names[cnt], lowest_rmse[-1]))
                    axs[x, y].set_xlim([1, 2**10])
                    axs[x, y].set_ylim([0.5, 2**8])
                    cnt += 1
                except (FileNotFoundError, IndexError):
                    continue

        # Create a bar plot to compare the lowest RMSE for each parameter
        x = np.arange(len(response_names))  # the label locations
        width = 0.9  # the width of the bars
        axs[num_xs-1, num_ys-1].bar(x, lowest_rmse, width)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[num_xs-1, num_ys-1].set_ylabel('RMSE')
        axs[num_xs-1, num_ys-1].set_title('Lowest RMSE by response')
        axs[num_xs-1, num_ys-1].set_xticks(x)
        axs[num_xs-1, num_ys-1].set_xticklabels(response_names, rotation=90)
        axs[num_xs-1, num_ys-1].set_ylim([0, 3])
        # Add annotation to bars
        for i in axs[num_xs-1, num_ys-1].patches:
            plt.text(i.get_x(), i.get_height() + i.get_height()/20, str(round((i.get_height()), 2)),
                     fontsize=6, fontweight='bold', color='grey')
        # axs[num_xs - 1, num_ys - 1].tight_layout()
        plt.tight_layout()

        # plt.suptitle('Training and validation RMSE with 3X data augmentation')
        plt.savefig(self.outpath + '/%s_learning_curve.png' % (folder+file_prefix))
        plt.show()
