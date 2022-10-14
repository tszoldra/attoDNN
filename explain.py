import os

import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize as normalize_img
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam # solution: https://github.com/keisen/tf-keras-vis/issues/87 - change file ~/usr_local_bin/anaconda3/envs/tf-keras-vis/lib/python3.10/site-packages/tf_keras_vis/utils/model_modifiers.py


def turn_off_GPU():
    # turn off GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    warnings.warn('Disabling GPU support for explainability algorithms.')


def plot_explain(X, y, model, analyzer_name: str, score_function, subplot_kwargs=dict(), analyzer_kwargs=dict(),
                 cmap='jet', alpha=0.5):
    fig, axs = plt.subplots(X.shape[0], **subplot_kwargs)

    y_pred = model.predict(X)

    analyzer_dict = {
        'Saliency': Saliency,
        'Gradcam': Gradcam,
        'GradcamPlusPlus': GradcamPlusPlus,
        'Scorecam': Scorecam,
    }

    analyzer = analyzer_dict[analyzer_name](model, clone=True)
    analyzer_result = analyzer(score_function, X, **analyzer_kwargs)

    for ax, img, img_explain, label_true, label_pred in zip(axs, X, analyzer_result, y, y_pred):
        ax.imshow(img)
        ax.imshow(img_explain, cmap=cmap, alpha=alpha)
        ax.axis('off')
        ax.set_title(f'true: {label_true: .3f}, pred: {label_pred: .3f}', fontsize=16)

    return fig, axs



# TODO Evaluate explainability measures through AOPC as in innvestigate
