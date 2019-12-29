import argparse
import itertools
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

import model
import utils.dataset as dataset
import utils.transforms as tsfrm
from utils.autoencoder import AutoEncoder

# global constant
HERE = os.path.dirname(os.path.abspath(__file__))  # this file's location
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(net, criterion, optimizer, epoch_num, trainloader, validloader, classes, outdir, early_stopping, print_acc, checkpoint, out_suffix, reduce_lr):
    """Model training

    Args:
        net (torch.nn.Module): neural network to train
        criterion (torch.nn.modules.loss): loss function to minimize
        optimizer (torch.optim): algorithm for minimizing `criterion`
        epoch_num (max. number of epochs to train for)
        trainloader (torch.utils.data.DataLoader): dataloader generating training samples
        validloader (torch.utils.data.DataLoader): dataloader generating validation samples
        classes (sequence): all possible class labels
        outdir (path-like, str): directory for saving training checkpoints
        early_stopping (bool): whether training should stop if overfitting is detected
        print_acc (bool): whether function should print out train & test accuracies after each epoch
        checkpoint (str, path-like, NoneType): path to the training checkpoint
        out_suffix (str): Text to append to the filenames of saved checkpoints
        reduce_lr (bool): Whether to reduce the learning rate when `criterion` stops improving

    Returns:
       correction_rates (tuple of lists): lists of train and validation accuracies by epoch
    """
    # variables used for early stopping
    is_autoencoder = type(net).__name__ == "AutoEncoder"
    # we won't check for early stopping w/ autoencoder
    early_stopping = None if is_autoencoder else early_stopping
    if reduce_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        if early_stopping == "Accuracy":
            scheduler.mode = "max"
        elif early_stopping == "Criterion":
            scheduler.mode = "min"

    train_history = []
    valid_history = []
    if early_stopping == "Accuracy":
        best_valid_performance = 0.0
    elif early_stopping == "Criterion":
        best_valid_performance = float("inf")

    overfitting_flag = False
    overfitting_counter = 0
    start_epoch = 0
    if checkpoint:
        state = torch.load(checkpoint, map_location=DEVICE)
        net.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state["epoch"] + 1
        # We used different names for best_valid_performance, train_history and valid_history over time
        # This ensures the loading works with the previous names
        try:
            best_valid_performance = state["validation accuracy"]  # old name
        except KeyError:
            try:
                # new name
                best_valid_performance = state["validation performance"]
            except KeyError:  # performance just wasn't saved
                pass  # keep value set at beginning of function

        train_history = state.get(
            "train_accu_list", state.get("train_history", []))
        valid_history = state.get(
            "valid_accu_list", state.get("valid_history", []))

        print("Loaded model", checkpoint, "at epoch",
              state["epoch"], "with loss of", state["loss"], "and validation performance of", best_valid_performance)

    # train the network
    try:
        for epoch in range(start_epoch, epoch_num):
            print("-" * 25)
            print("EPOCH", epoch)
            print("[Epoch, batch index]:  loss")

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):

                # get the inputs
                inputs, labels = data['image'], data['bboxes']['length']
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                labels = inputs if is_autoencoder else labels

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            print("-" * 25)
            print("Epoch", epoch,
                  "Finished training.", "" if is_autoencoder or early_stopping is None else "Calculating validation performance ...")

            if early_stopping:
                if early_stopping == "Accuracy":
                    if print_acc:
                        print("Training performance by class")
                        print("-" * 12)
                    train_accuracy, train_loss = check_accuracy(
                        net, criterion, trainloader, classes, print_acc=print_acc)

                    if print_acc:
                        print("Validation performance by class")
                        print("-" * 12)
                    valid_accuracy, valid_loss = check_accuracy(
                        net, criterion, validloader, classes, print_acc=print_acc)

                    train_performance, valid_performance = train_accuracy, valid_accuracy
                    stop_condition = best_valid_performance < valid_performance

                elif early_stopping == "Criterion":
                    if print_acc:
                        print("Training performance by class")
                        print("-" * 12)
                    train_accuracy, train_loss = check_loss(
                        net, criterion, trainloader, classes, print_loss=print_acc)

                    if print_acc:
                        print("Validation performance by class")
                        print("-" * 12)
                    valid_accuracy, valid_loss = check_loss(
                        net, criterion, validloader, classes, print_loss=print_acc)

                    train_performance, valid_performance = train_loss, valid_loss
                    stop_condition = best_valid_performance > valid_performance

                train_history.append((train_accuracy, train_loss))
                valid_history.append((valid_accuracy, valid_loss))

                print("Train performance at epoch",
                      epoch, ":", train_performance)
                print("Validation performance at epoch",
                      epoch, ":", valid_performance)

            # overfitting detected if no improvement to `early_stopping` for 5 straight epochs
                if reduce_lr:
                    scheduler.step(valid_performance)

                if stop_condition:
                    best_valid_performance = valid_performance
                    overfitting_flag = True
                    overfitting_counter = 0

                    model_path = os.path.join(
                        outdir, "{}-{}-{}.tar".format(type(net).__name__, type(optimizer).__name__, out_suffix))
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": running_loss,
                        "train_history": train_history,
                        "valid_history": valid_history,
                        "validation performance": valid_performance
                    }, model_path)
                    print("Model is saved at epoch", epoch)

                if overfitting_flag and not stop_condition:
                    overfitting_counter += 1
                    if overfitting_counter > 5:
                        print("Overfitting detected. Training terminated.")
                        break

        print('Training finished.')
    finally:
        # needed for saving to disk
        valid_performance = None if is_autoencoder else valid_performance

        model_path = os.path.join(  # Save model regardless of outcome
            outdir, "{}-{}-{}-FINAL.tar".format(type(net).__name__, type(optimizer).__name__, out_suffix))
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": running_loss,
            "train_history": train_history,
            "valid_history": valid_history,
            "validation performance": valid_performance
        }, model_path)
        return train_history, valid_history


def check_accuracy(net, criterion, dataloader, classes, print_acc=True):
    """Checks accuracy of a model

    Args:
        net (torch.nn.Module): neural network to train
        criterion (torch.nn.modules.loss): loss function to minimize
        dataloader (torch.utils.data.DataLoader): dataloader for samples to evaluate
        classes (sequence): set of all class labels
        print_acc (bool): Whether to print the accuracy per class

    Returns:
        performance (tuple): contains the overall accuracy and the overall loss calculated by criterion
    """
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    class_correct_rate = list(0. for i in range(len(classes)))

    overall_loss = 0.0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data['image'], data['bboxes']['length']
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss = criterion(outputs, labels)
            overall_loss += loss.item()
            # Pylint shows an error here, but it's working
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):

        if class_total[i] == 0:
            class_correct_rate[i] = 0
        else:
            class_correct_rate[i] = class_correct[i] / class_total[i]
        if print_acc:
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct_rate[i]))

    overall_correct_rate = sum(class_correct) / sum(class_total)
    overall_loss /= sum(class_total)

    return overall_correct_rate, overall_loss


def check_loss(net, criterion, dataloader, classes, print_loss=True):
    """Checks loss of a model

    Args:
        net (torch.nn.Module): neural network to train
        criterion (torch.nn.modules.loss): loss function to minimize
        dataloader (torch.utils.data.DataLoader): dataloader for samples to evaluate
        classes (sequence): set of all class labels
        print_loss (bool): Whether to print the loss per class

    Returns:
        performance (tuple): contains the overall accuracy and the overall loss calculated by criterion
    """

    class_losses = torch.Tensor([0. for i in range(len(classes))]).to(DEVICE)
    class_sizes = class_losses.clone()
    correct_pred = 0

    reduction = criterion.reduction  # save the reduction method to restore it later
    criterion.reduction = "none"
    with torch.no_grad():
        for data in dataloader:
            images, labels = data['image'], data['bboxes']['length']
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            losses = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct_pred += (predicted == labels).sum()
            for i, loss in enumerate(losses):
                label = int(labels[i])
                class_sizes[label] += 1
                class_losses[label] += loss

    mean_class_losses = class_losses / class_sizes
    overall_correct_rate = (correct_pred / class_sizes.sum()).item()

    if print_loss:
        for i in range(len(classes)):
            print('Loss of', classes[i], ':', mean_class_losses[i].item())

    overall_loss = (class_losses.sum() / class_sizes.sum()).item()
    criterion.reduction = reduction  # restore the reduction method
    return overall_correct_rate, overall_loss


def parse_args():
    models = ["Simple_Model", "Paper_Model",
              "ChenXi", "ChenXiPretrained", "AutoEncoder"]

    criteria = ["CrossEntropyLoss", "MSELoss"]

    optimizers = ["SGD", "Adam", "RMSProp"]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Trains all specified models with all specified optimizers using all specified criteria")
    parser.add_argument("models", choices=models,
                        nargs="+", help="Model name(s)")
    parser.add_argument("-m", "--metadata", dest="m", default=os.path.join(HERE, "data",
                                                                           "SVHN", "train_metadata.pkl"), help="Path to train metadata file")
    parser.add_argument("-d", "--traindir", dest="d", default=os.path.join(HERE, "data",
                                                                           "SVHN", "train"), help="Path to directory with training images")
    parser.add_argument("-O", "--outdir", dest="O", default=os.path.join(HERE,
                                                                         "models"), help="Directory for saving model(s)")
    parser.add_argument("-C", "--checkpoints", dest="C", nargs="+", default=[],
                        help="For each checkpoint, specify name of the model, name of the optimizer and path to checkpoint. Model and optimizer must be in the list of models and optimizers to run (e.g. Simple_Model SGD 'path/to/chkpt')")

    parser.add_argument("-c", "--criteria", default=criteria, dest="c", choices=criteria,
                        nargs="+", help="Loss function(s) for training")
    parser.add_argument("-o", "--optimizers", default=optimizers, dest="o", choices=optimizers,
                        nargs="+", help="Optimizing algorithm(s) for training")
    parser.add_argument("-e", "--epochs", dest="e", default=10000, type=int,
                        help="Number of training epochs")
    parser.add_argument("-s", "--stopmetric", choices=["Accuracy", "Criterion"], help="metric to use for detecting overfitting",
                        dest="s")
    parser.add_argument("-t", "--trainsize", dest="t", type=float, default=0.8,
                        help="Proportion of dataset to be used for training")
    parser.add_argument("-b", "--batchsize", dest="b", type=int,
                        default=4, help="Batch size")
    parser.add_argument("-w", "--workers", dest="w", type=int, default=2,
                        help="Number of workers for dataloaders")
    parser.add_argument("-v", action="count", dest="v", default=0,
                        help="-v == print per class accuracies for each epoch")
    parser.add_argument("-g", "--gray", dest="g", type=float, default=0.0,
                        help="Proportion of images to grayscale when training")
    parser.add_argument("-S", "--seed", dest="S", type=int,
                        default=123, help="Random seed for reproducibility")
    parser.add_argument("-f", "--filesuffix", dest="f", default="",
                        help="Text to append to the filenames of saved checkpoints")
    parser.add_argument("-r", "--reducelr", dest="r", action="store_true",
                        help="Whether to reduce the learning rate when `criterion` stops improving")
    parser.add_argument("-a", "--autoencoder", dest="a",
                        help="Path to autoencoder")

    args = parser.parse_args()
    # Validate some arguments
    assert 0 <= args.t <= 1, "Training size must be between 0 and 1 inclusively"
    assert 0 <= args.g <= 1, "Grayscale proportion must be between 0 and 1 inclusively"
    assert len(args.C) % 3 == 0, "You need to specify 3 arguments per checkpoint"

    return args


def main():
    args = parse_args()

    # Store each triplet of arguments in a dict
    checkpoints = {(args.C[i], args.C[i + 1]): args.C[i + 2]
                   for i in range(0, len(args.C), 3)}

    os.makedirs(args.O, exist_ok=True)  # Create outdir if not existing
    # Set random seed
    torch.manual_seed(args.S)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.S)

    # loading and normalizing images
    if args.a:
        autoencoder = AutoEncoder()
        autoencoder.load_state_dict(torch.load(
            args.a, map_location=DEVICE)["model_state_dict"])
    else:
        autoencoder = None
    transform = tsfrm.get(args.g, [0.5, 0.5, 0.5], [
                          0.5, 0.5, 0.5], autoencoder)

    data_set = dataset.SVHNDataset(args.m, args.d, transform=transform)
    train_size = int(len(data_set) * args.t)

    train_set = Subset(data_set, range(train_size))
    valid_set = Subset(data_set, range(train_size, len(data_set)))

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.b,
                                              shuffle=True, num_workers=args.w)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=args.b,
                                              shuffle=True, num_workers=args.w)

    classes = ('0', '1', '2', '3', '4', '5', '6')

    for m, c, o in itertools.product(args.models, args.c, args.o):
        if m == "Simple_Model":
            model_ = model.Simple_Model()
        elif m == "Paper_Model":
            model_ = model.Paper_Model()
        elif m == "ChenXi":
            model_ = model.ChenXi(pretrained=False)
        elif m == "ChenXiPretrained":
            model_ = model.ChenXi(pretrained=True)
        elif m == "AutoEncoder":
            model_ = AutoEncoder()

        model_.to(DEVICE)

        if c == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif c == "MSELoss":
            criterion = nn.MSELoss()

        if o == "SGD":
            optimizer = optim.SGD(
                model_.parameters(), lr=0.001, momentum=0.9)
        elif o == "Adam":
            optimizer = optim.Adam(model_.parameters())
        elif o == "RMSProp":
            optimizer = optim.RMSprop(
                model_.parameters(), lr=0.001, momentum=0.9)

        checkpoint = checkpoints.get((m, o))
        print(m, c, o)
        print("-" * 50)
        train_model(model_, criterion, optimizer, args.e,
                    trainloader, validloader, classes, args.O, args.s, bool(args.v), checkpoint, args.f, args.r)


if __name__ == "__main__":
    main()
