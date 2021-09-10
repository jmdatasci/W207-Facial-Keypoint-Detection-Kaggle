import random
import numpy as np
from modules.transforms import warp
from modules.transforms import affine_transform


class Center:
    def __init__(self, skip_center=False, do_scale=True, pick_center_image=True, verbose=False):
        """

        :param skip_center: Skip centering of images
        :param do_scale: Scale the image after identifying it?
        :param pick_center_image: Amongst multiple images pick center image?
        :param verbose: Verbose printout?
        """

        if not skip_center:
            # noinspection PyUnresolvedReferences
            from mtcnn.mtcnn import MTCNN
            self.detector = MTCNN()
        else:
            self.detector = None

        self.skip_center = skip_center
        self.do_scale = do_scale
        self.pick_center_image = pick_center_image
        self.verbose = verbose

    @staticmethod
    def warp(warp_matrix, coord):
        return warp(warp_matrix, coord)

    @staticmethod
    def _expand_dim(axis, width, max_dim, fraction=0.1, do_debug=False):

        # Total expansion should  be that fraction
        fraction = fraction / 2

        #  Make the box wider to capture more features especially eye features at the edges
        x_offset = int(width * fraction)
        x_min = axis - x_offset
        x_min = 0 if x_min < 0 else x_min
        if do_debug:
            print('axis  old %d new %d ' % (axis, x_min))
        axis = x_min

        # Max width is limited by where x is and teh edge of the image
        # Say x is 7 and the image has width 96, the max width can only be 96 - 7 - 1= 88
        max_width = max_dim - x_min - 1
        new_width = width + 2 * x_offset
        if do_debug:
            print('Width old %d new %d' % (width, new_width))
        width = new_width
        width = max_width if width > max_width else width

        return axis, width

    def center_and_scale_image(self, image, do_scale=True, pick_center_image=True, do_debug=False):
        """
        Center the face in the image. When multiple face are in the image then use the closest to the center

        :param image: 2D image
        :param do_scale: Whether to scale the image or not
        :param pick_center_image: If multiple faces pick one closest to center or
                                pick largest area
        :param do_debug: Verbose printout or not
        :return:
        """

        do_debug = do_debug
        scaled_image = False

        # Make it a three channel image
        if np.max(image) <= 1:
            scaled_image = True
            # Convert it into a 0 to 255 image
            image = (image * 255).astype('uint8')
        image = np.stack((image, image, image), axis=-1)

        # Detect the face in the image
        faces = self.detector.detect_faces(image)
        print('%d|' % len(faces), end='')
        if random.random() < 1 / 200:
            print('')

        # This is the center of the image
        image_center = np.array([(image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2])
        if do_debug:
            print('Image center %s' % str(image_center))

        # Find the largest face and return that
        parameters = None
        if pick_center_image:
            previous_dist = 96 ** 2 + 96 ** 2
        else:
            previous_dist = 0
        for face in faces:
            # One face
            x, y, width, height = face['box']

            # Sometimes the height and width are off in the image
            # we need to truncate it
            width = image.shape[1] - x - 1 if width > image.shape[1] - x - 1 else width
            height = image.shape[0] - y - 1 if height > image.shape[0] - y - 1 else height

            # Location of this face
            face_center = np.array([x + width / 2, y + height / 2])

            if pick_center_image:
                # Euclidean distance of this face from the image center
                dist = np.sum((face_center - image_center) ** 2) ** 0.5
                condition = dist < previous_dist
            else:
                # Area of the face in pixels
                dist = width * height
                condition = dist > previous_dist

            if do_debug:
                print('Boxes: X %d Y %d width %d height %d center %s dist %.2f' %
                      (x, y, width, height, str(face_center), dist))

            if condition:
                # Location of the image
                parameters = (x, y, width, height)
                previous_dist = dist

        if parameters is not None:
            # These are the parameters for the image
            x, y, width, height = parameters
        else:
            x, y, width, height = (0, 0, image.shape[1] - 1, image.shape[0] - 1)

        # Expand the axis in the x dimension
        x, width = self._expand_dim(x, width, image.shape[1], fraction=0.15, do_debug=do_debug)
        y, height = self._expand_dim(y, height, image.shape[0], fraction=0.15, do_debug=do_debug)
        # This is the scale the image can be scaled by before exceeding image size
        scaling = (image.shape[0] - 1) / max(width, height)

        image = np.squeeze(image[:, :, 0])
        if scaled_image:
            image = image.astype('float32') / 255

        if not do_scale:
            #
            scaling = 1
            center_x = (image_center[0] - (x + width / 2)) * scaling
            center_y = (image_center[1] - (y + height / 2)) * scaling
        else:
            # Shift must be to get 0,0 to the left most
            center_x = -x * scaling
            center_y = -y * scaling

        # This is the translated image
        new_img, forward, inverse = affine_transform(image, (center_x, center_y), (scaling, scaling))

        if do_debug:
            print('X %d Y %d width %d height %d scaling %.2f' % (x, y, width, height, scaling))
            print('Center X %.2f Y %.2f' % (center_x, center_y))

            print('Forward %s' % str(forward))
            print('Inverse %s' % str(inverse))

        if not do_scale:
            # If the image is not scaled then make the area outside the face as 0's
            zero_padded = np.zeros(new_img.shape)
            x_min = int(image_center[0] - width / 2)
            x_max = int(image_center[0] + width / 2 + 1)
            y_min = int(image_center[1] - height / 2)
            y_max = int(image_center[1] + height / 2 + 1)
            if do_debug:
                print('x_min %d x_max %d y_min %d y_max %d' %
                      (x_min, x_max, y_min, y_max))
            zero_padded[x_min:x_max, y_min:y_max] = new_img[x_min:x_max, y_min:y_max]
        else:
            zero_padded = new_img

        return zero_padded, forward, inverse, (x, y, width, height)

    @staticmethod
    def center_and_scale_image_skip(image):
        """
        Center the face in the image
        :param image: 2D image
        :return:
        """

        #
        do_debug = False

        scaled_image = False
        # Make it a three channel image
        if np.max(image) <= 1:
            scaled_image = True
            # Convert it into a 0 to 255 image
            image = (image * 255).astype('uint8')

        # Find the largest face and return that
        parameters = (0, 0, image.shape[0] - 1, image.shape[1] - 1, 1)

        # These are the parameters for the image
        x, y, width, height, scaling = parameters

        # Sometimes the height and width are off in the image
        # we need to truncate it
        width = image.shape[1] if width > image.shape[1] - 1 else width
        height = image.shape[0] if height > image.shape[1] - 1 else height

        if scaled_image:
            image = image.astype('float32') / 255

        if do_debug:
            print('X %d Y %d width %d height %d scaling %.2f' % (x, y, width, height, scaling))

        new_img = image.copy()
        inverse = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        forward = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        return new_img, forward, inverse, (x, y, width, height)

    def center(self, image):
        """

        :param image: The image to process
        :return:
        """

        if self.skip_center:
            return self.center_and_scale_image_skip(image)
        else:
            return self.center_and_scale_image(image, do_scale=self.do_scale, pick_center_image=self.pick_center_image,
                                               do_debug=self.verbose)
