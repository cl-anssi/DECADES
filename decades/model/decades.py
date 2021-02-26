import torch
import torch.nn as nn

from model.entity import Entity
from model.interaction import Interaction




class DECADES(nn.Module):
    """
    PyTorch module for the DECADES model.


    Attributes
    ----------
    entities : list
        List of Entity instances representing the entity types.
    interactions : list
        List of Interaction instances representing the interaction
        (event) types.
    n_noise_samples : int
        Number of negative samples generated for each training
        batch.
    return_pval : bool
        If False, the anomaly score of an event is the negative
        log-probability of that event (instead of the aggregated
        log-p-values).
    training : bool
        If True, the forward function returns the NCE loss.
        Otherwise, it returns the anomaly scores of the events.
    updating : bool
        If True, new embeddings are created for unknown entities
        found in the input to the forward function, and the noise
        distribution is updated at each call to the forward function.

    Methods
    -------
    forward(input):
        Computes the NCE/MTL loss (if self.training) or the anomaly
        scores (if not self.training).
    update_noise_distributions():
        Updates the noise distribution of each event type based on the
        current unigram counts.
    reinitialize_counts():
        Resets the unigram count for each event type.
    enable_gradients(state):
        If state is True, enables gradient computation for all
        parameters.
        Otherwise, disables gradient computation for all parameters
        except for the entity embeddings.
    train():
        Set the model in initial training mode.
    validate():
        Set the model in validation mode.
    test():
        Set the model in test/detection mode.
    retrain():
        Set the model in retraining mode (updating of the embeddings
        during detection phase).
    """

    def __init__(
            self, entities, interactions,
            embedding_dim=64, n_noise_samples=10, noise_dist='all', alpha=.75,
            return_pval=False):
        """
        Constructor for the DECADES model.

            Arguments
            ---------
                entities : list
                    A list of dictionaries, each of which represents
                    one entity type.
                    Keys of each dictionary are "name" and "n_entities"
                    (respectively name of the entity type and initial
                    number of entities of this type).
                interactions : list
                    A list of dictionaries, each of which represents
                    one interaction type (i.e. event type).
                    Keys of each dictionary are "entities" (list of
                    entity types involved in this interaction type),
                    "noise_dist" (list of IntTensor, length equal to
                    len(entities)-1, i-th tensor has shape
                    (entities[i+1].n_entities,) and contains the
                    occurrence counts of the entities), and "seen"
                    if the noise distribution is "unique" (set of
                    tuples seen in the training set).
                    The entity list refers to the indices of the list
                    of entity types passed in the entities parameter.
                embedding_dim : int
                    Dimension of the entity embeddings and event
                    type-specific latent factors.
                n_noise_samples : int
                    Number of negative samples generated for each
                    training batch.
                noise_dist : str
                    Type of noise distribution to use.
                    Possible values: "all" (unigram distribution),
                    "unique" (unigram distribution with only one
                    occurrence of each tuple), "log" (log-unigram
                    distribution) and "pow" (unigram distribution
                    to the power alpha, the value of alpha is passed
                    as parameter).
                alpha : float
                    Parameter of the power-unigram noise distribution
                    (only used if noise_dist == "pow").
                return_pval : bool
                    If False, the anomaly score of an event is the
                    negative log-probability of that event (instead of
                    the aggregated log-p-values).
        """
        super(DECADES, self).__init__()

        self.entities = []
        for x in entities:
            self.entities.append(Entity(
                x['n_entities'], embedding_dim, x['name']))

        self.interactions = []
        for i, x in enumerate(interactions):
            if noise_dist == 'unique':
                self.interactions.append(Interaction(
                    self.entities, x['entities'], embedding_dim,
                    noise_dist=noise_dist, counts=x['noise_dist'],
                    seen=x['seen']))
            else:
                self.interactions.append(Interaction(
                    self.entities, x['entities'], embedding_dim,
                    noise_dist=noise_dist, counts=x['noise_dist'],
                    alpha=alpha))
        self.interactions = nn.ModuleList(self.interactions)
        self.entities = nn.ModuleList(self.entities)

        for i in range(len(self.interactions)):
            print([self.entities[e].n_entities
                for e in self.interactions[i].entity_types])

        self.n_noise_samples = n_noise_samples
        self.return_pval = return_pval

        self.training = True
        self.updating = False

    def forward(self, input):
        """
        Forward pass of the model.
        The output depends on the current mode (train, test, validate
        or retrain): it is the NCE/MTL loss in train and retrain modes,
        and the anomaly score in validate and test modes.

        Arguments
        ---------
        input : torch.Tensor
            Events to be analyzed.
            The input tensor has shape (n_events, m), where m = largest
            possible number of entities + 2.
            Each line of the input has the following structure:
            [entity_1, ..., entity_Ne, [padding], timestamp, event_type]

        Returns
        -------
        loss : torch.Tensor
            If the model is in train/retrain mode, the output tensor
            has a single element (the NCE/MTL loss for the batch).
            If the model is in test/validate mode, the output tensor
            has shape (n_events,), with each element equal to the
            anomaly score for the corresponding event.
        """
        it = set(int(i) for i in input[:,-1])
        if self.training:
            loss = 0
            for k in it:
                idx = torch.where(input[:,-1]==k)[0]
                m = self.interactions[k].n_entities
                loss += self.interactions[k](
                    input[idx,:m].long(), training=self.training,
                    n_noise_samples=self.n_noise_samples)
        else:
            loss = input.new_zeros(input.shape[0]).float()
            for k in it:
                idx = torch.where(input[:,-1]==k)[0]
                m = self.interactions[k].n_entities
                loss[idx] = self.interactions[k](
                    input[idx,:m].long(), training=self.training,
                    updating=self.updating, return_pval=self.return_pval)
        return loss

    def update_noise_distributions(self):
        """
        Updates the noise distribution of each event type based on the
        current unigram counts.

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        for itr in self.interactions:
            itr.update_noise_dist()

    def reinitialize_counts(self):
        """
        Resets the occurrence counts of the entities for each event
        type.

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        for itr in self.interactions:
            itr.reinitialize_counts()

    def enable_gradients(self, state):
        """
        Enables or disables gradient computation for model
        parameters other than the entity embeddings.

        Arguments
        ---------
        state : bool
            If True, gradient computation is enabled.
            Otherwise, it is disabled.

        Returns
        -------
        None
        """
        for itr in self.interactions:
            for m in itr.linear:
                for p in m.parameters():
                    p.requires_grad = state
            for p in itr.weighted_dot.parameters():
                p.requires_grad = state
            itr.logvars.requires_grad = state

    def train(self):
        """
        Sets the model in initial training mode.

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        self.training = True
        self.updating = False
        self.enable_gradients(True)

    def validate(self):
        """
        Sets the model in validation mode.

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        self.training = False
        self.updating = False
        self.enable_gradients(False)

    def test(self):
        """
        Sets the model in test/detection mode.

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        self.training = False
        self.updating = True
        self.enable_gradients(False)

    def retrain(self):
        """
        Sets the model in retraining/updating mode.

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        self.training = True
        self.updating = False
        self.enable_gradients(False)
        self.update_noise_distributions()
