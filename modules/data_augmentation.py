import os
import cv2
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.transforms import warp
from modules.transforms import affine_transform
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class AugmentData:
    def __init__(self, max_rotation=10, max_shift=10, max_shear=0.2, max_scale=0.2, num_transforms=2,
                 colab_compatibility=True, elastic_alpha=34, elastic_sigma=4, cartoon_prob=0):
        """
        :param max_rotation: Maximum angle to rotate the image by.
                            Actual Rotation will be random number in the interval (-max_rotation, max_rotation)
        :param max_shift: Maximum lateral shift of the image
                          Actual shift will be in the interval (-max_shift, max_shift)
        :param max_shear: Maximum shear angle for the image.
                          Actual shear will be in the interval (-max_shear, max_shear)
        :param max_scale: Maximum scale for the image. Must be between 0 and 1
                          Actual scale will be in the interval (1-max_scale, 1+max_scale)
        :param num_transforms:  Number of random transforms to perform
        :param colab_compatibility: Create a Google colab compatible scales value?
        :param elastic_sigma: sigma to use for the elastic transformation
        :param elastic_alpha: alpha to use for the elastic tarnsformation
        :param cartoon_prob: What fraction of images were cartoon
        """
        self.data = None
        self.max_rotation = max_rotation
        self.max_shift = max_shift
        self.max_shear = max_shear
        self.max_scale = max_scale
        self.num_transforms = num_transforms
        self.mode = "edge"
        self.do_plots = True
        self.colab_compatibility = colab_compatibility
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        self.cartoon_prob = cartoon_prob
        self.cartoon_line_size = (3, 7)
        self.cartoon_blur_value = (3, 7)

    def augment(self, data, force_create=False):
        """
        Transform all the images in the input DataFrame

        :param data: Dictionary of data for each response
        :param force_create: Force the creation of the augment file
        """

        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '../data/augment/augment_data_%d.p' % self.num_transforms)

        if not force_create:
            try:
                # Check to see if train, cv, test data already exists
                with open(outfile, 'rb') as infi:
                    augmented_data = pickle.load(infi)
                print('Already augmented data was found, delete the file if you want to recreate')
                return augmented_data
            except FileNotFoundError:
                pass

        # This object will take no memory if it just returns the file
        self.data = data

        # For each response create the transformed matrices
        all_transformed_data = {}
        for response_name, data_df in self.data.items():
            # Accumulated DataFrame
            all_transformed_data[response_name] = self.augment_one(data_df, response_name)

        # Check to see if train, cv, test data already exists
        with open(outfile, 'wb') as outfi:
            pickle.dump(all_transformed_data, outfi)

        return all_transformed_data

    def augment_one(self, data_df, response_name, do_one=False, force_create=False):
        """
        Transform all the images in the input DataFrame

        :param data_df: Dictionary of data for each response
        :param response_name:
        :param do_one: Do only one transformation of the image?
        :param force_create:
        """

        # These are the only two columns that augment is expecting
        data_df = data_df[['X', 'y']]

        # If 0 transformation then return the original DataFrame
        if self.num_transforms == 0:
            return data_df

        # This is the augmented data with the
        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '../data/augment/augdat_resp_%s_samples_%d.p' % (response_name, self.num_transforms))

        if not do_one and not force_create:
            try:
                # Check to see if train, cv, test data already exists
                with open(outfile, 'rb') as infi:
                    augmented_data = pickle.load(infi)
                print('Augmented data found for response %s with transforms %d and has %d samples input data as %d rows'
                      % (response_name, self.num_transforms, augmented_data.shape[0], data_df.shape[0]))
                return augmented_data
            except FileNotFoundError:
                pass

        print('Augmenting response %s' % response_name)
        plt_cnt = 0
        # If the image has a left or right characteristic then flipping it changes the landmark.
        if 'left' in response_name or 'right' in response_name:
            create_flip = False
        else:
            create_flip = True

        # All the transformations are by-image, do loop thorugh each image
        accum_df = pd.DataFrame()
        for row in range(data_df.shape[0]):
            image_x = data_df.iloc[row]['X']
            coord_y = data_df.iloc[row]['y']

            # affine transformation
            data_row_transform = self.transform_img(image_x, coord_y.copy(), create_flip)
            try:
                # Keep only the columns that are in the input DataFrame
                tmp_df = data_row_transform[[x for x in data_df.columns]]
            except KeyError:
                # This occurs when the transformations led to empty DataFrame
                # In this case skip this DataFrame
                continue

            accum_df = pd.concat((accum_df, tmp_df))
            data_row_transform.assign(parent=pd.Series([row for _ in range(data_row_transform.shape[0])]))
            if self.do_plots and plt_cnt < 1 and not data_row_transform.empty:
                # Produce five plots per response type
                self.plot_transformations(image_x, coord_y, data_row_transform, response_name)
                plt_cnt += 1

            if do_one:
                break

        # The return DataFrame, shuffle the rows to remove correlation in batches
        return_df = pd.concat((data_df, accum_df))
        return_df = return_df.sample(frac=1).reset_index(drop=True)

        if not do_one:
            # Check to see if train, cv, test data already exists
            with open(outfile, 'wb') as outfi:
                pickle.dump(return_df, outfi)

        return return_df

    @staticmethod
    def elastic_transform(image, alpha, sigma, random_gen=None):
        """
        Perform an elastic transformation of the image
        # Motivated by this code: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a

        :param image: Image to transform
        :param alpha: Alpha value for transformation (original value used 34)
        :param sigma: Sigma value for transformation (original value used 34)
        :param random_gen:
        :return:
        """

        # A random state creates a random number generator without existing with than existing one
        if random_gen is None:
            random_gen = np.random.RandomState(None)
        else:
            random_gen = np.random.RandomState(random_gen)

        # Create an x shift and y shift value for every pixel in the image
        shape = image.shape
        dx = gaussian_filter((random_gen.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_gen.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        # This is the original pixel location of the image
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        # We add a delta to it
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        # And map it to the new location
        return map_coordinates(image, indices, order=1).reshape(shape)

    @staticmethod
    def cartoon(image, line_size, blur_value):
        """
        Convert the image into a cartoon using edge detection
        :param image: Image to transform
        :param line_size: Thickness of the resulting edge line
        :param blur_value: Blur the lines using how many pixels?
        :return:
        """

        # gray_blur = cv2.medianBlur(train_data[2940, :].reshape(96, 96), blur_value)
        image = image * 255
        image = image.astype('uint8')
        edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
        edges = edges.astype('float32')/255

        return edges

    @staticmethod
    def _pixel_spread(coord, limits):
        """
        Spread the value of the pixel across multiple
        :param coord: X or y coordinate
        :param limits: Maximum limits each value can take
        :return:
        """

        # col = round(coord_y[1])
        # col = 0 if col < 0 else col
        # col = image_x.shape[1] - 1 if col > image_x.shape[1] - 1 else col
        axis = int(coord)

        # This is the value that got rounded off
        remainder = coord - axis

        # Remainder can only be greater than 0
        axis1 = axis + 1
        remainder = remainder / axis1

        print('Axis %d axis1 %d remainder %.2f' % (axis, axis1, remainder))
        if limits[0] < axis < limits[1] and limits[0] < axis1 < limits[1]:
            return (axis, 1), (axis1, remainder)
        else:
            return None

    def spread_pixel(self, image_x, coord_y):
        # Create an array of zeros the same shape as the image
        # Mark it with a one as to where we want it to transform
        # Note that X are columns and Y's are rows and hence this flip
        trans_im_y = np.zeros(image_x.shape)
        cols = self._pixel_spread(coord_y[1], (0, image_x.shape[1]))

        # As we are rounding the values, perhaps we can spread it over two pixels in teh ratio of the rounding
        # For example the value of 73.3 can be thought of as 73 + a74, rearranging a = (73.3-73)/74= 0.3/74
        rows = self._pixel_spread(coord_y[0], (0, image_x.shape[0]))

        if cols is None or rows is None:
            return None

        #
        for col, valc in cols:
            for row, valr in rows:
                trans_im_y[col, row] = valc * valr
                print('Col %d row %d value %.2f' % (col, row, trans_im_y[col, row]))

        return trans_im_y

    @staticmethod
    def one_pixel(image_x, coord_y):
        # Create an array of zeros the same shape as the image
        # Mark it with a one as to where we want it to transform
        # Note that X are columns and Y's are rows and hence this flip
        trans_im_y = np.zeros(image_x.shape)

        col = round(coord_y[1])
        if col > image_x.shape[1] - 1:
            return None

        row = round(coord_y[0])
        if row > image_x.shape[0] - 1:
            return None

        #
        trans_im_y[row, col] = 1

        return trans_im_y

    def do_cartoon(self, image_x, coord_y, all_transformations, do_debug_plots):
        """

        :param image_x:
        :param coord_y:
        :param all_transformations:
        :param do_debug_plots:
        :return:
        """

        for _ in range(self.num_transforms):
            line_size = random.randint(*self.cartoon_line_size)
            blur_value = random.randint(self.cartoon_blur_value[0], line_size)

            if line_size % 2 == 0:
                line_size += 1

            if blur_value > line_size:
                blur_value = line_size

            trans_x = self.cartoon(image_x, line_size=line_size, blur_value=blur_value)
            if do_debug_plots:
                plt.figure()
                plt.imshow(trans_x, cmap='bone')
                plt.plot(coord_y[0], coord_y[1], 'o')
                plt.title('Line size %d blur %d' % (line_size, blur_value))
                plt.show()

            all_transformations.append({'X': trans_x, 'y': coord_y, 'shear': 0, 'translation': 0, 'rotation': 0,
                                        'scale': 0})
        # Convert it into a DataFrame
        return pd.DataFrame(all_transformations)

    def transform_img(self, image_x, coord_y, create_flip=False):
        """
        :param image_x: The x image to be transformed
        :param coord_y: The y coordinates to be transformed
        :param create_flip: Whether to create flipped images or not
        """
        do_debug_plots = 0

        all_transformations = []
        # Just cartoon transform some of the images
        if random.random() < self.cartoon_prob:
            # Make all these transformation into cartoons
            return self.do_cartoon(image_x, coord_y, all_transformations, do_debug_plots)

        # All the scale transformation we will do to this image
        # If the random number is zero then we get a scaling of 1 - self.max_scale
        # If the random number is 1 then we get a scaling of self.max_scale - 2*self.max_scale = -self.max_scale
        scales = [1 + (self.max_scale - 2*random.random()*self.max_scale) for _ in range(self.num_transforms)]

        # The version in Google colab expects a tuple of sx and sy
        if self.colab_compatibility:
            scales = [(x, x) for x in scales]

        # All the rotations we will do to this image
        # If the random number is zero then we get a scaling of self.max_rotation
        # If the random number is 1 then rotation is self.max_rotation - 2*self.max_rotation = -self.max_rotation
        rotations = [self.max_rotation - 2 * random.random() * self.max_rotation for _ in range(self.num_transforms)]

        # All the translations we will do to this image
        # Translation is basically a shift in the x and y direction
        translations_x = [self.max_shift - 2 * random.random() * self.max_shift for _ in range(self.num_transforms)]
        translations_y = [self.max_shift - 2 * random.random() * self.max_shift for _ in range(self.num_transforms)]
        translations = [x for x in zip(translations_x, translations_y)]

        # All the translations we will do to this image
        shears = [self.max_shear - 2 * random.random() * self.max_shear for _ in range(self.num_transforms)]

        # Flip or not
        if create_flip:
            # Half the images have a possibility of being flipped
            flips = [random.random() > 0.5 for _ in range(self.num_transforms)]
        else:
            # Can't flip, it changes the landmark
            flips = [0 for _ in range(self.num_transforms)]

        # Elastic transform
        elastics = [random.random() > 0.5 for _ in range(self.num_transforms)]

        for scale, rotation, translation, shear, flip, elastic in \
                zip(scales, rotations, translations, shears, flips, elastics):
            # Create an object for the transformation
            # Perform the transformation
            trans_x, forward_matrix, _ = affine_transform(
                image_x.copy(), scale=scale, rotation=np.deg2rad(rotation), translation=translation,
                shear=np.deg2rad(shear))

            # Now convert the coordinates
            trans_y = warp(forward_matrix, coord_y)

            # Translate the coordinates into an image for
            # trans_im_y = self.spread_pixel(trans_x, trans_y)
            trans_im_y = self.one_pixel(trans_x, trans_y)
            if trans_im_y is None:
                continue

            # Elastic transformation of the image
            if elastic:
                alpha = self.elastic_alpha
                sigma = self.elastic_sigma
                seed = random.randint(1, 65535)

                # noinspection PyTypeChecker
                trans_x = self.elastic_transform(trans_x, alpha, sigma, random_gen=seed)
                # noinspection PyTypeChecker
                t_im_y = self.elastic_transform(trans_im_y, alpha, sigma, random_gen=seed)
            else:
                alpha = 0
                sigma = 0
                t_im_y = trans_im_y

            # Locations where the pixel is now spread
            w_ty = np.where(t_im_y)
            if w_ty[1].size == 0:
                # The coordinates are outside the image, we skip this one
                continue

            # Weighted average of the location that they went to
            do_new_method = False
            if do_new_method:
                # The pixel has spread across multiple locations
                # Take the weighted average of its value across rows
                weights_y = np.sum(t_im_y, axis=1)
                loc_y = np.where(weights_y)
                row = np.sum(loc_y[0]*weights_y[loc_y[0]]/np.sum(weights_y[loc_y[0]]))

                # Take the weighted average of its value across cols
                weights_x = np.sum(t_im_y, axis=0)
                loc_x = np.where(weights_x)
                col = np.sum(loc_x[0] * weights_x[loc_x[0]] / np.sum(weights_x[loc_x[0]]))
                trans_y = [row, col]
            else:
                trans_y = np.array((np.mean(w_ty[0]), np.mean(w_ty[1])))

            if do_debug_plots:
                plt.figure()
                plt.imshow(image_x, cmap='bone')
                w_y = np.where(trans_im_y)
                plt.plot(w_y[0], w_y[1], 'd', markersize=3, color='b')
                plt.plot(trans_y[0], trans_y[1], 'o', markersize=3, color='y')
                plt.plot(coord_y[0], coord_y[1], '*', markersize=3, color='g')
                plt.title('Plain')
                plt.show()

                plt.figure()
                plt.imshow(trans_x, cmap='bone')
                plt.plot(w_ty[0], w_ty[1], 'd', markersize=1, color='b')
                plt.plot(trans_y[0], trans_y[1], 'o', markersize=2, color='y')
                plt.plot(coord_y[0], coord_y[1], '*', markersize=2, color='g')
                # plt.plot(row, col, '+', markersize=3, color='r')

                plt.title('Alpha %.2f sigma %.2f' % (alpha, sigma))
                plt.show()

            if flip:
                trans_x = np.fliplr(trans_x)
                # Flip the x-coordinate around
                trans_y[0] = trans_x.shape[0] - trans_y[0] - 1

                if do_debug_plots:
                    plt.figure()
                    plt.imshow(trans_x, cmap='bone')
                    plt.plot(w_ty[0], w_ty[1], 'd', markersize=1, color='b')
                    plt.plot(trans_y[0], trans_y[1], 'o', markersize=2, color='y')
                    plt.plot(coord_y[0], coord_y[1], '*', markersize=2, color='g')
                    plt.ylim([0, 3])
                    plt.title('Flipped')
                    plt.show()

            # If the translated coordinated are within the image range then save the image
            # Otherwise throw it out
            all_transformations.append({'X': trans_x, 'y': trans_y, 'shear': shear, 'translation': str(translation),
                                        'rotation': rotation, 'scale': scale})

        return pd.DataFrame(all_transformations)

    @staticmethod
    def plot_transformations(original_image, original_landmark, transformation_df, response_name):
        """

        :param original_image: Original image before transformation
        :param original_landmark: Landmark on the original image
        :param transformation_df: DataFrame with the transformed images
        :param response_name: Nae of teh response that is being plotted
        :return:
        """

        # These are all the transformations
        all_records = transformation_df.to_dict('index')

        # 1. Plot the original image
        num_xs = len(all_records)+1
        fig, axs = plt.subplots(1, len(all_records)+1, figsize=(6.4*num_xs, 6.4))
        axs[0].imshow(original_image, cmap="bone")
        axs[0].plot([original_landmark[0], ], [original_landmark[1], ], 'o', color='r', markersize=2)
        axs[0].axis("off")
        axs[0].set_title(response_name)

        # 2. Plot each of the individual images
        for idx, record in all_records.items():
            axs[idx+1].imshow(record['X'], cmap="bone")
            axs[idx+1].plot([record['y'][0], ], [record['y'][1], ], 'o', color='r', markersize=2)
            axs[idx+1].axis("off")
            axs[idx+1].set_title('Transformation %d' % (idx+1))
        plt.show()
