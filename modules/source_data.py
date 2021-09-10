import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.data_processing import Center


class SourceData:

    def __init__(self, center_params=None, debug=False):
        """
        Center the face in the image. When multiple face are in the image then use the closest to the center

        :param center_params: Parameters for image centering
            :do_scale: Whether to scale the image or not
            :pick_center_image: If multiple faces pick one closest to center or
                                    pick largest area
            :do_debug: Verbose printout or not
        :param debug: set skip_center to True irrespective of its value
        :return:
        """
        center_params = {} if center_params is None else center_params

        # if its debug mode then set skip_center to True irrespective of its values
        if debug:
            center_params['skip_center'] = True

        # Create an object of the center class
        self.center = Center(**center_params)

    @staticmethod
    def cleanup(indf):
        """
        Convert the DataFrame columns into clean format for plotting and analyzing

        :param indf:
        :return:
        """
        for col in indf.columns:
            # If the columns is not the image column then convert it to number
            if col != 'Image':
                indf[col] = pd.to_numeric(indf[col])
            else:
                # Convert from string to float numpy array and the scale from 0 to 1
                indf[col] = indf[col].apply(lambda x: np.array(x.split(' '), dtype='float32')/255)
                # Convert it into a nXn image ndarray
                indf[col] = indf[col].apply(lambda x: x.reshape(int(x.size**0.5), int(x.size**0.5)))

        return indf

    def transform(self, data_df):
        """
        Center and scale the image to make it easier for CNN to learn

        :param data_df: DataFrame to be transformed
        :return:
        """

        # Create a column with a unique id
        data_df['unique_id'] = [i for i in range(data_df.shape[0])]

        # Rename the Image column because we will give the transformed image the column name image
        data_df = data_df.rename(columns={'Image': 'orig_Image'})

        # This returns a tuple of values with new_img, forward, inverse, (x, y, width, height)
        data_df['tmp'] = data_df['orig_Image'].apply(lambda x: self.center.center(x))

        # # This is the transformed image
        data_df['Image'] = data_df['tmp'].apply(lambda x: x[0])

        # This is the transformed image
        data_df['forward'] = data_df['tmp'].apply(lambda x: x[1])

        # This is the transformed image
        data_df['inverse'] = data_df['tmp'].apply(lambda x: x[2])

        drop_column = ['tmp', ]

        # Drop the unneccessary columns
        data_df = data_df.drop(columns=drop_column)

        return data_df

    def source_all_data_csv(self):
        """
        Gets the training
        :return:
        """

        infile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/training.csv')

        # Read all the data into a pandas DataFrame
        all_data = pd.read_csv(infile)

        # Reset the index and get it as a column name
        all_data = all_data.reset_index().rename(columns={'index': 'unique_id'})

        # Pandas reads everything as text, convert it into numbers
        all_data = self.cleanup(all_data)
        all_data = self.transform(all_data)

        return all_data

    def _source_data_csv(self, train_frac=0.7, cv_frac=0.2, seed=None, debug=False):
        """
        Gets the training
        :param train_frac: What fraction of data will be training data
        :param cv_frac: What fraction of data will be cross validation
        :param seed: Seed for random number generator, used for row shuffling
        :param debug: Debug mode or not
        :return:
        """

        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/transformed_data.p')
        try:
            # Check to see if train, cv, test data already exists
            with open(outfile, 'rb') as infi:
                train_data, cv_data, test_data = pickle.load(infi)
        except FileNotFoundError:

            infile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/training.csv')

            # Read all the data into a pandas DataFrame
            all_data = pd.read_csv(infile)

            # Reset the index and get it as a column name
            # all_data = all_data.reset_index().rename(columns={'index': 'unique_id'})

            # Without replacement sampling of the data. As 100% data is asked, it simply shuffles the lines
            if seed is None:
                seed = random.randint(1, 65535)
                
            print('Setting seed for random state as %d' % seed)
            all_data = all_data.sample(frac=1, random_state=seed)

            if debug:
                all_data = all_data[:300]

            # Pandas reads everything as text, convert it into numbers
            all_data = self.cleanup(all_data)

            print('Starting transform')
            all_data = self.transform(all_data)
            print('Transform complete')

            # Split the shuffled DataFrame into train, cv and test data
            stop_index_train = int(all_data.shape[0]*train_frac)
            stop_index_cv = int(all_data.shape[0]*(train_frac + cv_frac))

            # Split it as train, cv and test data
            train_data = all_data[:stop_index_train].copy()
            cv_data = all_data[stop_index_train:stop_index_cv].copy()
            # Test data keeps everything
            test_data = all_data[stop_index_cv:].copy()

            # Save the data as files
            with open(outfile, 'wb') as outfi:
                pickle.dump((train_data, cv_data, test_data), outfi)

        return train_data, cv_data, test_data

    @staticmethod
    def _split_on_response(df, response_columns):
        """

        :param df: The DataFrame to be split as x and Y responses
        :param response_columns: Names of the columns that are responses
        :return:
        """

        # Keep these columns
        all_df = pd.DataFrame()
        for response in response_columns:

            # Create a temporary DataFrame for the columns of interest
            keep_columns = ['unique_id', 'forward', 'inverse', 'orig_Image', 'Image']
            temp_df = df[[response + '_x', response + '_y'] + keep_columns].copy()

            # Combine the two response columns into one pair
            temp_df[response] = temp_df[[response + '_x', response + '_y']].apply(lambda x: np.array(x), axis=1)
            temp_df = temp_df.drop(columns=[response + '_x', response + '_y'])

            # Wrap the coordinates for the response
            temp_df[response + '_t'] = temp_df[['forward', response]].apply(lambda x: Center.warp(x[0], x[1]), axis=1)

            # Stack everything up in all_df
            if all_df.empty:
                all_df = temp_df
            else:
                # The rest of the columns already exist in all_df
                temp_df = temp_df[['unique_id', response, response + '_t']]

                # Join it back to all_df
                all_df = all_df.merge(temp_df, left_on='unique_id', right_on='unique_id')

        # rename the image column to 'X'
        all_df = all_df.rename(columns={'Image': 'X'})

        return all_df

    @staticmethod
    def plot_samples(indata):
        """

        :param indata:
        :return:
        """

        response_columns = ['left_eyebrow_outer_end', 'right_eye_outer_corner', 'right_eye_inner_corner',
                            'left_eye_inner_corner', 'mouth_right_corner', 'mouth_left_corner',
                            'right_eyebrow_outer_end', 'left_eyebrow_inner_end', 'mouth_center_top_lip',
                            'right_eyebrow_inner_end', 'left_eye_outer_corner',
                            'mouth_center_bottom_lip', 'nose_tip', 'right_eye_center', 'left_eye_center']

        # Where are all the NAN's in the data?
        nans = None
        for response in response_columns:
            tmp = indata[response].apply(lambda x1: np.any(np.isnan(x1))).values
            nans = np.logical_or(nans, tmp)

        # Not nans
        not_nans = np.logical_not(nans)
        filtered = indata[not_nans].copy()
        filtered.reset_index(inplace=True, drop=True)
        plt.figure(figsize=(4.8*2, 6.4*10))
        for i in range(10):
            #
            plt.subplot(10, 2, 2*i+1)
            index = random.randint(0, filtered.shape[0]-1)
            plt.imshow(filtered.iloc[index]['orig_Image'])
            plt.title('Original image')
            for response in response_columns:
                if response in filtered.columns:
                    x = [filtered.iloc[index][response][0], ]
                    y = [filtered.iloc[index][response][1], ]
                    plt.plot(x, y, marker='o', markersize=3, color='r')

            #
            plt.subplot(10, 2, 2*i+2)
            plt.imshow(filtered.iloc[index]['X'])
            plt.title('Pre-processed image')
            for response in response_columns:
                if response in filtered.columns:
                    x = [filtered.iloc[index][response + '_t'][0], ]
                    y = [filtered.iloc[index][response + '_t'][1], ]
                    plt.plot(x, y, marker='o', markersize=3, color='r')

        plt.show()

    def source_data(self, train_frac=0.7, cv_frac=0.15, seed=None, debug=False, combine_train_test=False):
        """
        Gets the training
        :param train_frac: What fraction of data will be training data
        :param cv_frac: What fraction of data will be cross validation
        :param seed: Seed for random number generator, used for row shuffling
        :param debug: Is it debug mode, if it is then it does not center the data
        :param combine_train_test: Combine the train and test data
        :return:
        """

        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/processed_data.p')
        try:
            # Check to see if train, cv, test data already exists
            with open(outfile, 'rb') as infi:
                return_dict = pickle.load(infi)

        except FileNotFoundError:

            # Source the data into training and test and CV
            train_data, cv_data, test_data = \
                self._source_data_csv(train_frac=train_frac, cv_frac=cv_frac, seed=seed, debug=debug)

            # All columns except the image columns are repsonse columns
            # We will pair the responses and hence remove _x or _y
            not_response = ['Image', 'orig_Image', 'forward', 'inverse', 'unique_id']
            response_columns = list(set([x[:-2] for x in train_data.columns if x not in not_response]))

            # Split the data by DataType
            train_data = self._split_on_response(train_data, response_columns)
            cv_data = self._split_on_response(cv_data, response_columns)
            test_data = self._split_on_response(test_data, response_columns)

            return_dict = {'train': train_data, 'cv':  cv_data, 'test': test_data}

            # Save the data as files
            with open(outfile, 'wb') as outfi:
                pickle.dump(return_dict, outfi)

        if combine_train_test:
            # Combining train and test data into one
            train_data = pd.concat((return_dict['train'], return_dict['test']))
            return_dict['train'] = train_data

        # Plot the samples
        self.plot_samples(return_dict['train'])

        return return_dict

    def source_test_csv(self):
        """
        Gets the Submission file
        :return:
        """

        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/submission_data.p')
        infile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/test.csv')
        try:
            # Check to see if train, cv, test data already exists
            with open(outfile, 'rb') as infi:
                submission_data = pickle.load(infi)
        except FileNotFoundError:
            print('Preprocessing submission data')
            # Read all the data into a pandas DataFrame
            submission_data = pd.read_csv(infile)

            # Pandas reads everything as text, convert it into numbers
            submission_data = self.cleanup(submission_data)
            submission_data = self.transform(submission_data)

            # Reset the index and get it as a column name
            submission_data = submission_data.rename(columns={'Image': 'X'})

            # Save the data as files
            with open(outfile, 'wb') as outfi:
                pickle.dump(submission_data, outfi)

        return submission_data
