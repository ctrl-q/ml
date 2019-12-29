# %%
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.ticker import MaxNLocator
from sklearn import metrics
import torch
import itertools
import utils.prediction as prediction


def show_image(image, transform=None, model=None):
    """Shows image, its transformation(s) if any, and a model's predicted label if any
    Args:
        image (dict): original image containing keys "image" and "bboxes"
        transform (torchvision.transforms): torchvision transformation(s)
        model (torch.nn.Module): model to use if you like to visualize its prediction
    """
    plt.figure()
    for index, t in enumerate((image, transform)):
        if t is not None:
            ax = plt.subplot(1, 3, index + 1)
            plt.tight_layout()

            if index == 0:
                img = image
                ax.set_title("Original")
            else:
                img = t(image)
                ax.set_title(type(t).__name__)

            # show bounding box(es)
            for idx in range(len(img['bboxes']['label'])):
                i = img['bboxes']['left'][idx]
                j = img['bboxes']['top'][idx]
                h = img['bboxes']['height'][idx]
                w = img['bboxes']['width'][idx]
                box = patches.Rectangle(
                    (i, j), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(box)
                if model:
                    with torch.no_grad():
                        ax.axis("off")
                        outputs = model(image)
                        _, pred = torch.max(outputs, 1)
                        ax.set_title("Predicted length: {}".format(pred))

            ax.imshow(img["image"])

        plt.show()


def show_learning_curve(model_file, model_name=None, curve='accuracy', save_img=False, save_filename=None):
    """Shows loss curve or accuracy curve of a model

    Args:
        model_file (str): file_path to the saved model
        model_name: the title of the image
        save_img: if True, the image will be saved
        save_filename: the filename of the saving image
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_file, map_location=device)

    train_history = state.get("train_history", [])
    valid_history = state.get("valid_history", [])

    train_accuracy = [accuracy for accuracy, loss in train_history]
    valid_accuracy = [accuracy for accuracy, loss in valid_history]

    train_loss = [loss for accuracy, loss in train_history]
    valid_loss = [loss for accuracy, loss in valid_history]

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)

    if (len(train_history) != 0 and len(valid_history) != 0):
        if curve == 'accuracy':
            plt.plot(range(1, len(train_history) + 1),
                     train_accuracy, label='train accuracy')
            plt.plot(range(1, len(valid_history) + 1),
                     valid_accuracy, label='validation accuracy')
            plt.ylabel('Accuracy')

        if curve == 'loss':
            plt.plot(range(1, len(train_history) + 1),
                     train_loss, label='train loss')
            plt.plot(range(1, len(valid_history) + 1),
                     valid_loss, label='validation loss')
            plt.xlabel('Loss')

        if model_name is not None:
            plt.title(model_name)

        # ax.set_facecolor('white')
        # fig.patch.set_facecolor('white')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epochs')
        plt.legend()

        if save_img and save_filename is not None:
            plt.savefig(save_filename)
        else:
            plt.show()

    return


def plot_distribution(dataset, save_fig=False):
    """
    Args:
        dataset (SVHNDataset): SVHNDataset
        save_fig (bool): Optional argument to save plot. If true, save plot.
    """
    # Create array of lengths
    if dataset.transform is None:
        lengths = [len(sample["metadata"]["label"])
                   for sample in dataset.md]  # access metadata directly
    else:
        lengths = [sample["bboxes"]["length"]
                   for _, sample in enumerate(dataset)]

    # Calculate and plot histogram of label length
    bins = int(max(lengths))
    plt.hist(lengths, bins, facecolor='green', alpha=0.5,
             weights=[1 / len(lengths)] * len(lengths))
    plt.xlabel('Length')
    plt.ylabel('Probability')
    plt.title(r'Histogram of lengths')
    plt.show()
    if save_fig:
        plt.savefig('len_distribution.png')


def compare_models(names, paths, metric="Accuracy"):
    """Shows histogram of `metric` for all models

    Args:
        names (sequence): name of each model
        paths (sequence): path of each model in the same order as `names`
        metric (str): Metric to compare them on (either 'Accuracy' or 'Criterion' i.e loss)
    """
    metric = metric.title()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert len(names) == len(paths), "len(names) != len(paths)"
    # accept `loss` as metric just for clarity
    tuple_indexes = {"Accuracy": 0, "Criterion": 1, "Loss": 1}
    idx = tuple_indexes[metric]

    values = []
    for name, path in zip(names, paths):
        # get last metric reported by the model
        state = torch.load(path, map_location=device)
        valid_history = state["valid_history"]
        value = valid_history[-1][idx] if isinstance(
            valid_history[-1], tuple) else valid_history[-1]
        values.append(value)

    plt.bar(list(range(len(names))), values, tick_label=names)
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title("{} per model".format(metric))
    plt.show()


def show_confusion_matrix(labels, predictions, weights=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Shows confusion matrix

    Args:
        labels (sequence): true labels for each image
        predictions (sequence): model predictions for each image
        weights (sequence): weight to apply to each class (default: None)
    """
    unique_labels = set(labels)
    assert len(labels) == len(predictions), "len(labels) != len(predictions)"
    assert weights is None or len(weights) == len(
        unique_labels), "len(weights) != number of labels"

    matrix = metrics.confusion_matrix(
        labels, predictions, sample_weight=weights)
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = max(matrix) / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.tight_layout()


def show_metrics_report(labels, predictions, weights=None):
    """Shows Precision, Recall, F1-Score and Support per class

    Args:
        labels (sequence): true labels for each image
        predictions (sequence): model predictions for each image
        weights (sequence): weight to apply to each class (default: None)
    """
    unique_labels = set(labels)
    assert len(labels) == len(predictions), "len(labels) != len(predictions)"
    assert weights is None or len(weights) == len(
        set(unique_labels)), "len(weights) != number of labels"

    print(metrics.classification_report(
        labels, predictions, sample_weight=weights))


def get_confusion_matrix(dataset_dir, metadata_filename, model_filename):

    confusion_matrix = {}
    for i in range(7):
        confusion_matrix[str(i)] = {'true_negative': 0,
                                    'true_positive': 0,
                                    'false_negative': 0,
                                    'false_positive': 0}

    dataloader = prediction.get_validloader(dataset_dir, metadata_filename)
    predictions = prediction.eval_model(
        dataset_dir, metadata_filename, model_filename)

    idx = 0
    for data in dataloader:
        _, labels = data['image'], data['bboxes']['length']

        for label in labels:

            for i in range(7):
                # true positive
                if predictions[idx] == i and label == i:
                    confusion_matrix[str(i)]['true_positive'] += 1
                # false negative
                elif predictions[idx] != i and label == i:
                    confusion_matrix[str(i)]['false_negative'] += 1
                # true negative
                elif predictions[idx] != i and label != i:
                    confusion_matrix[str(i)]['true_negative'] += 1
                # false positive
                elif predictions[idx] == i and label != i:
                    confusion_matrix[str(i)]['false_positive'] += 1
            idx += 1

    print('true_positive, true_negative, false_positive, false_negative')
    for i in range(7):
        print('class: ', str(i))
        print(confusion_matrix[str(i)]['true_positive'],
              confusion_matrix[str(i)]['true_negative'],
              confusion_matrix[str(i)]['false_positive'],
              confusion_matrix[str(i)]['false_negative'])

    return confusion_matrix
