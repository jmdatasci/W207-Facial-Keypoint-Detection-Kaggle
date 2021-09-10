import numpy as np
from skimage import data
from skimage import transform
import matplotlib.pyplot as plt


def warp(warp_matrix, coord):

    # Find the inverse
    new_coord = np.dot(warp_matrix, np.concatenate((coord, np.array([1, ]))))

    return new_coord[:2]


def affine_transform(img, translation=(0, 0), scale=1.0, rotation=0, shear=0):
    """

    :param img: Image in a 2D format
    :param translation: x and y shift translations
    :param scale: Image scaling value
    :param rotation: Degree of rotation
    :param shear: Amount of shear in degress
    :return:
    """

    # Find the transformation matrix for the image
    tform = \
        transform.AffineTransform(scale=scale, rotation=np.deg2rad(rotation), translation=translation,
                                  shear=np.deg2rad(shear))

    # This is the inverse transformation
    new_img = transform.warp(img, tform.inverse)

    # Return the inverse function
    return new_img, tform.params.copy(), np.linalg.inv(tform.params)


def projective_transform(img, coord):
    """

    :param img:
    :param coord:
    :return:
    """
    matrix = np.array([[1, 0.99, 0], [0.1, 0.99, 0], [0.0015, 0.99, 0.5]])

    # Find the transformation matrix for the image
    tform = transform.ProjectiveTransform(matrix)

    # This is the inverse transformation
    new_img = transform.warp(img, tform.inverse)

    # Coordinates need a direct transformation
    if isinstance(coord, tuple):
        array_coord = np.array(coord + (1,))
    elif isinstance(coord, list):
        array_coord = np.array(coord + [1, ])
    elif isinstance(coord, np.ndarray):
        if len(coord.shape) == 2:
            coord = coord.flatten()
        array_coord = np.concatenate((coord, np.array([1, ])))
    else:
        raise TypeError('Input must be of type tuple, list or numpy ndarray')

    # Find the transformed coordinate
    new_coord = np.dot(tform.params, array_coord)

    return new_img, new_coord[:2]


def test_affine():
    img = getattr(data, 'camera')()
    new_img, forward_matrix, inverse_matrix = affine_transform(img, (10, 20), 1.2)
    # noinspection PyArgumentList
    coord = np.array((300, 150))
    new_coord = warp(forward_matrix, coord)
    inverted_coord = warp(inverse_matrix, new_coord)
    print(new_coord)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='bone')
    plt.plot([coord[0], ], [coord[1], ], 'o')
    plt.plot([inverted_coord[0], ], [inverted_coord[1], ], '*')

    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='bone')
    plt.plot([new_coord[0], ], [new_coord[1], ], 'x')
    plt.show()


def test_projective():
    img = getattr(data, 'camera')()
    coord = (300, 150)
    new_img, new_coord = projective_transform(img, coord)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='bone')
    plt.plot([coord[0], ], [coord[1], ], 'o')

    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='bone')
    plt.plot([new_coord[0], ], [new_coord[1], ], 'x')
    plt.show()


if __name__ == '__main__':
    test_affine()
    # test_projective()
