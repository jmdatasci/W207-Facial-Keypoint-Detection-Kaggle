from modules.models import CNN
from modules.models import Means
from modules.MultiModel import MultiModel
from modules.source_data import SourceData


def test_means_model():
    debug = False
    seed = 1427

    # Augmentation parameters
    augmentation_params = {'left_eye_inner_corner': {'num_transforms': 12},
                           'left_eye_outer_corner': {'num_transforms': 12},
                           'right_eye_inner_corner': {'num_transforms': 12},
                           'right_eye_outer_corner': {'num_transforms': 12},
                           'left_eyebrow_inner_end': {'num_transforms': 12},
                           'left_eyebrow_outer_end': {'num_transforms': 12},
                           'right_eyebrow_inner_end': {'num_transforms': 12},
                           'right_eyebrow_outer_end': {'num_transforms': 12},
                           'mouth_left_corner': {'num_transforms': 12},
                           'mouth_right_corner': {'num_transforms': 12},
                           'mouth_center_top_lip': {'num_transforms': 12},
                           'mouth_center_bottom_lip': {'num_transforms': 8},
                           'nose_tip': {'num_transforms': 8},
                           'left_eye_center': {'num_transforms': 8},
                           'right_eye_center': {'num_transforms': 8}}

    # Centering params
    centering_params = {'skip_center': False, 'do_scale': True, 'pick_center_image': True}

    # eliminate parameters
    eliminate_params = {0: [], 1: ['nose_tip', 'mouth_center_bottom_lip']}

    # Create a model of some model type
    mm = MultiModel(CNN, eliminate=eliminate_params, augment_params=augmentation_params)

    # Source the data here
    sd = SourceData(debug=debug, center_params=centering_params)
    data = sd.source_data(combine_train_test=True, seed=seed)

    # Now fit the model with optimal parameters
    mm.fit(train_data=data['train'], cv_data=data['cv'])

    # Now make a prediction on the test data and compare the ori
    test_data = data['test']
    pred, metrics = mm.predict(test_data)
    print(metrics)

    # Check for submission file creation
    submmision_data = sd.source_test_csv()
    mm.create_submission(submission_data=submmision_data)


if __name__ == '__main__':
    # Test the model flow using a means model
    test_means_model()

    # test_semisup_kfold()
