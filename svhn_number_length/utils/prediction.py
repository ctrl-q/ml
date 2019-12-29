import utils.transforms as tsfrm
import utils.dataset as dataset
import model
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

import sys
sys.path.append('../')


def get_validloader(dataset_dir, metadata_filename):

    transform = tsfrm.get(0.0, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    data_set = dataset.SVHNDataset(
        metadata_filename, dataset_dir, transform=transform)

    # by default we use 0.8 to split data
    train_size = int(len(data_set) * 0.8)
    valid_set = Subset(data_set, range(train_size, len(data_set)))

    validloader = torch.utils.data.DataLoader(valid_set, batch_size=16,
                                              shuffle=False, num_workers=1)

    return validloader


def eval_model(dataset_dir, metadata_filename, model_filename):
    '''
    Skeleton for your testing function. Modify/add
    all arguments you will need.

    '''
    model_param = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load your best model
    if model_filename:
        model_filename = Path(model_filename)
        print("\nLoading model from", model_filename.absolute())
        model_param = torch.load(model_filename, map_location=device)

    if model_param:

        testloader = get_validloader(dataset_dir, metadata_filename)

        net = model.Paper_Model()
        net.load_state_dict(model_param['model_state_dict'])
        net.to(device)

        y_pred = np.array([])

        with torch.no_grad():
            for data in testloader:
                images, labels = data['image'], data['bboxes']['length']
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # Pylint shows an error here, but it's working
                _, predicted = torch.max(outputs, 1)
                y_pred = np.append(y_pred, predicted.numpy())

    else:

        print("\nYou did not specify a model, generating dummy data instead!")
        n_classes = 5
        y_pred = np.random.randint(0, n_classes, (100))

    return y_pred
