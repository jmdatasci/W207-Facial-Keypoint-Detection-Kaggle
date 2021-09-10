import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from modules.MultiModel import MultiModel
from modules.source_data import source_data
from modules.data_augmentation import AugmentData


def learning_curve_plots(model, folder, file_prefix):
    # Create a set of models using the means model
    mmo = MultiModel(model)
    mmo.plot_learning_curve(logbase=2, folder=folder, file_prefix=file_prefix)


def augmentation_sample_size(model, prefix):
    """
    Relationship between number of samples and cross validation error

    :param model:
    :param prefix:
    :return:
    """

    # Get the name of the class
    class_name = str(model).split('.')[-1].split('\'')[0]
    response_names = ['left_eyebrow_outer_end', 'right_eye_center', 'nose_tip']
    colors = list(mcolors.TABLEAU_COLORS.keys())

    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/analysis')
    legends = []
    for cnt, response_name in enumerate(response_names):
        try:
            filename = os.path.join(basepath, '%s%s_scaling_response_%s.p' % (prefix, class_name, response_name))
            with open(filename, 'rb') as infi:
                df = pickle.load(infi)
        except OSError:
            continue

        # Scaling of the
        scaling_df = df.groupby('scaling')[['val_root_mean_squared_error']].min()
        scaling_df = pd.DataFrame(scaling_df)
        scaling_df.reset_index(inplace=True)
        plt.plot(scaling_df['scaling'], scaling_df['val_root_mean_squared_error'], color=colors[cnt], marker='o')
        legends.append(response_name)

    # Reducing cross validation RMSE
    plt.title('Cross validation RMSE reduces with increasing augmented samples')
    plt.xlabel('Number of augmented images per original image')
    plt.ylabel('RMSE')
    plt.legend(legends)
    plt.show()


def augmentation_train_time(model, prefix=''):
    """
    Relationship between number of samples and training time

    :param model:
    :param prefix:
    :return:
    """

    # Get the name of the class
    class_name = str(model).split('.')[-1].split('\'')[0]
    response_names = ['left_eyebrow_outer_end', 'right_eye_center', 'nose_tip']
    colors = list(mcolors.TABLEAU_COLORS.keys())

    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/analysis')
    legends = []
    for cnt, response_name in enumerate(response_names):
        try:
            filename = os.path.join(basepath, '%s%s_scaling_response_%s.p' % (prefix, class_name, response_name))
            with open(filename, 'rb') as infi:
                df = pickle.load(infi)
        except OSError:
            continue

        # Scaling of the
        scaling_df = df.groupby('scaling')[['train_time']].mean()
        scaling_df = pd.DataFrame(scaling_df)
        scaling_df.reset_index(inplace=True)
        plt.plot(scaling_df['scaling'], scaling_df['train_time'], color=colors[cnt], marker='o')
        legends.append(response_name)

    # Reducing cross validation RMSE
    plt.title('Training time increases with samples')
    plt.xlabel('Number of augmented images per original image')
    plt.ylabel('Training time(hrs)')
    plt.legend(legends)
    plt.show()


def augmentation_images():
    """
    Relationship between number of samples and training time
    :return:
    """

    # Get the source data
    data, _ = source_data()
    for response_name in list(data['train_labeled'].keys()):
        augment = AugmentData(num_transforms=4, cartoon_prob=0)
        augment.augment_one(data['train_labeled'][response_name], response_name=response_name, do_one=True)


if __name__ == '__main__':
    #
    do_sample_graph = False
    if do_sample_graph:
        from modules.models import CNN
        augmentation_sample_size(CNN, prefix='')
        augmentation_train_time(CNN)

    #
    do_augmentation_graph = False
    if do_augmentation_graph:
        augmentation_images()

    do_learning_curves = True
    if do_learning_curves:
        from modules.models import CNN

        learning_curve_plots(CNN, folder='high_augmentation_high_skew', file_prefix='cnn_transformed')
        learning_curve_plots(CNN, folder='failed_centering', file_prefix='cnn_transformed')
        learning_curve_plots(CNN, folder='volume_augmentation', file_prefix='volume_augment')
