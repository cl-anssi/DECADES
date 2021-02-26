import numpy as np

import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader




def train_one_epoch(model, dataloader, optimizer, scaler, device):
    """
    Trains the model for one epoch.

    Arguments
    ---------
    model : DECADES
        Instance of the DECADES model to train.
    dataloader : torch.utils.data.DataLoader
        Dataloader for the training set.
    optimizer : torch.optim.Optimizer
        Optimizer to use for parameter updates.
    scaler : torch.cuda.amp.GradScaler
        Scaler used for automatic mixed precision training.
    device : torch.Device
        Device on which the model is loaded.

    Returns
    -------
    running_loss : float
        Total loss for this epoch.
    """
    model.train()
    running_loss = 0
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        with autocast():
            loss = model(inputs)
        running_loss += loss.data.item()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return running_loss

def validate_model(model, dataloader, device):
    """
    Computes validation scores for the model.

    Arguments
    ---------
    model : DECADES
        Instance of the DECADES model to validate.
    dataloader : torch.utils.data.DataLoader
        Dataloader for the validation set.
    device : torch.Device
        Device on which the model is loaded.

    Returns
    -------
    test_scores : list
        List of length num_event_types + 1.
        Each element of the list is a list of two floats (mean and
        standard deviation of the anomaly scores), and stands for all
        events of one type (first num_event_types elements) or for all
        malicious events (last element).
    """
    model.validate()
    n_itr = len(model.interactions)
    test_scores = [[0, 0] for i in range(n_itr+1)]
    with torch.no_grad():
        res = []
        for inputs, labels in dataloader:
            types = inputs[:,-1].long().numpy()
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()
            res += list(zip(outputs.detach().numpy(),
                labels.numpy(),
                types))
        res = np.array(res, dtype=float)
        for i in range(n_itr):
            scores = res[(res[:,2]==i)&(res[:,1]==0)][:,0]
            test_scores[i] = [scores.mean(), scores.std()]
        scores = res[res[:,1]==1][:,0]
        if len(scores) > 0:
            test_scores[-1] = [scores.mean(), scores.std()]
        else:
            test_scores[-1] = [0, 0]
    return test_scores

def estimate_moments(model, dataset, device):
    """
    Estimates the mean and standard deviation of the anomaly scores
    for each event type.

    Arguments
    ---------
    model : DECADES
        Instance of the DECADES model to use.
    dataset : LANLDataset
        Dataset used to estimate the moments.
    device : torch.Device
        Device on which the model is loaded.

    Returns
    -------
    means : list
        List of length num_event_types, each element of which is the
        mean anomaly score for one event type.
    stds : list
        List of length num_event_types, each element of which is the
        standard deviation of the anomaly scores for one event type.
    """
    model.validate()
    dataset.train()
    dataloader = DataLoader(dataset, batch_size=50000, shuffle=True)
    means = [[] for itr in model.interactions]
    stds = [[] for itr in model.interactions]
    with torch.no_grad():
        for inputs, _ in dataloader:
            types = set(inputs[:,-1].long().numpy().tolist())
            inputs = inputs.to(device)
            outputs = model(inputs)
            for t in types:
                means[t].append(outputs[inputs[:,-1]==t].mean().data.item())
                stds[t].append(outputs[inputs[:,-1]==t].std().data.item())
    means = [sum(x)/len(x) for x in means]
    stds = [sum(x)/len(x) for x in stds]
    return means, stds

def train_model(
        model, dataset, device, epochs, batch_size,
        lr=1e-3, weight_decay=1e-5):
    """
    Trains the model on the given dataset.

    Arguments
    ---------
    model : DECADES
        Instance of the DECADES model to train.
    dataset : LANLDataset
        Training set to use.
    device : torch.Device
        Device on which the model is loaded.
    epochs : int
        Number of training epochs to perform.
    batch_size : int
        Size of the training batches.
    lr : float (optional)
        Learning rate for the parameter updates (default: 1e-3).
    weight_decay : float (optional)
        Coefficient for the L2 regularization of the parameters
        (default: 1e-5).

    Returns
    -------
    running_losses : list
        List containing the total loss for each epoch.
    test_scores : list
        List of dictionaries.
        Each dictionary has two keys ('mean' and 'std').
        The 'mean' (resp. 'std') key maps to a list containing the mean
        validation score (resp. standard deviation of the validation
        scores) for each epoch.
        The first num_event_types dictionaries correspond to all events
        of one type, while the last one corresponds to all malicious
        events.
    logvars : list
        List of length equal to the number of epochs.
        Each element is a list of length num_event_types whose elements
        are the mean logarithmic MTL uncertainties for each event type.
    """
    scaler = GradScaler()
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    n_itr = len(model.interactions)
    running_losses = []
    test_scores = [{'mean': [], 'std': []} for i in range(n_itr+1)]
    logvars = [[0.]*n_itr]

    for epoch in range(epochs):
        dataset.validate()
        dataloader = DataLoader(dataset, batch_size=50000, shuffle=False)
        scores = validate_model(model, dataloader, device)
        for i in range(n_itr+1):
            test_scores[i]['mean'].append(scores[i][0])
            test_scores[i]['std'].append(scores[i][1])

        dataset.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        running_losses.append(
            train_one_epoch(model, dataloader, optimizer, scaler, device))
        logvars.append(
            [itr.logvars.mean().data.item() for itr in model.interactions])

    means, stds = estimate_moments(model, dataset, device)
    for i, itr in enumerate(model.interactions):
        itr.y_mean = means[i]
        itr.y_std = stds[i]

    return running_losses, test_scores, logvars

def test_one_period(model, dataset, device, T):
    """
    Tests the model on the T-th test period of the given dataset.

    Arguments
    ---------
    model : DECADES
        Instance of the DECADES model to test.
    dataset : LANLDataset
        Dataset used to test the model.
    device : torch.Device
        Device on which the model is loaded.
    T : int
        Index of the test period.

    Returns
    -------
    res : numpy.ndarray
        Array of shape (n_events, 3).
        The first column contains the anomaly scores, the second one
        contains the labels (0 or 1) and the third one contains the
        event types.
    """
    model.test()
    dataset.test(T)
    dataloader = DataLoader(dataset, batch_size=50000, shuffle=False)

    res = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            types = inputs[:,-1].long().numpy()
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()
            res += list(zip(outputs.detach().numpy(), labels.numpy(), types))
    res = np.array(res, dtype=float)
    return res

def retrain(
        model, dataset, device, T, batch_size, lambda_0, lambda_1,
        sizes, val_prop=.01, thresh=.01, lr=1e-4, max_epochs=25):
    """
    Retrains the model on the T-th test period of the given dataset.

    Arguments
    ---------
    model : DECADES
        Instance of the DECADES model to retrain.
    dataset : LANLDataset
        Dataset used to retrain the model.
    device : torch.Device
        Device on which the model is loaded.
    T : int
        Index of the test period.
    batch_size : int
        Size of the training batches.
    lambda_0 : float
        Regularization hyperparameter for new entity embeddings.
    lambda_1 : float
        Regularization hyperparameter for old entity embeddings.
    sizes : tuple
        Tuples of integers with one element per entity type,
        corresponding to the number of known entities of this type
        before this test period (used to distinguish new entities from
        old ones).
    val_prop : float (optional)
        Proportion of the retraining set used for validation
        (default: 1e-2).
    thresh : float (optional)
        Stopping criterion for the retraining: when the relative
        variation of the validation error between two successive epochs
        becomes smaller than this value, retraining ends.
        (default: 1e-2)
    lr : float (optional)
        Learning rate for the parameter updates (default: 1e-4).
    max_epochs : int (optional)
        Stopping criterion (maximum number of retraining epochs to
        perform).
        (default: 25)

    Returns
    -------
    None
    """
    dataset.make_validation_set(T, val_prop)
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    old_embeddings = [e.embeddings.weight.clone().detach()
        for e in model.entities]
    for emb in old_embeddings:
        emb.requires_grad = False
    n_itr = len(model.interactions)
    dataset.val_retrain()
    dataloader = DataLoader(dataset, batch_size=10000, shuffle=False)
    scores = validate_model(model, dataloader, device)
    old_score = sum(scores[i][0] for i in range(n_itr))/n_itr
    variation = 1
    epoch = 0
    while variation > thresh and epoch < max_epochs:
        model.retrain()
        dataset.test(T)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        running_loss = 0
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            with autocast():
                loss = model(inputs) + sum(
                    (e.embeddings.weight-old_embeddings[j]
                        ).pow(2).sum(1).dot(
                            torch.cat([
                                old_embeddings[j].new_full(
                                    (sizes[j],),
                                    lambda_1),
                                old_embeddings[j].new_full(
                                    (old_embeddings[j].shape[0]-sizes[j],),
                                    lambda_0
                                )
                            ])
                        )
                    for j, e in enumerate(model.entities))
                running_loss += loss.data.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        dataset.val_retrain()
        dataloader = DataLoader(dataset, batch_size=50000, shuffle=False)
        scores = validate_model(model, dataloader, device)
        new_score = sum(scores[i][0] for i in range(n_itr))/n_itr
        variation = abs(new_score-old_score)/old_score
        old_score = new_score
        print('Epoch {0} ; total loss: {1} ; validation score: {2}'.format(
            epoch+1, running_loss, old_score))
        epoch += 1
    dataset.delete_validation_set(T)

def test_model(
        model, dataset, device, batch_size,
        lambda_0, lambda_1):
    """
    Tests the model on the given dataset.
    This includes computation of the anomaly scores and retraining.

    Arguments
    ---------
    model : DECADES
        Instance of the DECADES model to evaluate.
    dataset : LANLDataset
        Dataset used to evaluate the model.
    device : torch.Device
        Device on which the model is loaded.
    batch_size : int
        Size of the training batches for the retraining phase.
    lambda_0 : float
        Regularization hyperparameter for new entity embeddings.
    lambda_1 : float
        Regularization hyperparameter for old entity embeddings.

    Returns
    -------
    res : np.ndarray
        Array of shape (n_events, 3).
        The first column contains the anomaly scores, the second one
        contains the labels (0 or 1) and the third one contains the
        event types.
    """
    res = []
    n_periods = len(dataset._test_idx)
    for i in range(n_periods):
        sizes = [e.n_entities for e in model.entities]
        res.append(test_one_period(model, dataset, device, i))

        if i == n_periods-1:
            break

        retrain(
            model, dataset, device, i, batch_size,
            lambda_0, lambda_1, sizes)
    return np.concatenate(res)
