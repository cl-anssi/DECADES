from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.statistical_functions import mid_p_value
from utils.torch_modules import WeightedDot, SoftmaxWeightedAvg




class Interaction(nn.Module):
    """
    PyTorch module handling events of one given type.
    Each instance of the DECADES model has a list of instances of this
    class as an attribute.


    Attributes
    ----------
    entity_list : list
        List of Entity instances representing the entity types.
    entity_types : list
        List of integers corresponding to the ordered indices of the
        entity types involved in each event of the considered type.
    n_entities : int
        Number of entities involved in each event of the considered
        type.
    logvars : torch.Tensor
        Tensor containing the log-variances associated with each
        predicted entity (used for multi-task learning).
    counts : list
        List of torch.IntTensor containing the unigram counts for each
        entity in the event.
        Used to compute the noise distribution for NCE.
    exclude : dict
        Dictionary used to keep track of same-type entities involved in
        a given event.
        We assume that one entity cannot appear twice in the same
        event, thus if an event contains two entities of the same type,
        we must adjust the probability distribution when predicting the
        second one (since it cannot be the same as the first one).
        This dictionary maps indices between 1 and n_entities-1 to
        smaller indices referring to same-type entities.
    noise_dist : list
        List of torch.Tensor containing the normalized probabilities
        corresponding to the noise distribution for each involved
        entity.
    seen : set
        Set of entity tuples observed so far.
        Useful only if the 'unique' noise distribution is used.
    alpha : float
        Exponent parameter, useful only if the power-unigram noise
        distribution is used.
        Then the probability of entity v is proportional to
        count(v)**alpha.
    noise_type : str
        Name of the noise distribution used (among 'all', 'unique',
        'log' and 'pow').
    linear : torch.nn.ModuleList
        List of SoftmaxWeightedAvg modules computing the h_e,i vectors
        used to predict each entity in the event.
    weighted_dot : WeightedDot
        Torch module computing the weighted dot product of two batches
        of vectors, with weights beta_e specific to this event type.
    y_mean : float
        Mean anomaly score for this event type.
    y_std : float
        Standard deviation of the anomaly scores for this event type.
    loss : function
        Binary cross-entropy loss function used to compute the NCE
        loss.
    Methods
    -------
    forward(entities, training=True, updating=False,
            n_noise_samples=10, return_pval=False):
        Forward pass of the module.
        Returns the MTL-weighted NCE loss if training is True,
        and the anomaly scores otherwise.
    context_vector(self, idx, entities):
        Returns the h_e,i vector with i == idx.
        Used to compute the affinity function Kappa.
    conditional_affinity(self, idx, entities):
        Computes the affinity function Kappa for the entity at index
        idx.
    get_neg_samples(self, idx, n_noise_samples):
        Samples from the noise distribution for the entity at index
        idx.
    get_noise_prob_with_exclusions(self, idx, entities, to_eval):
        Computes the probability of the entities in to_eval under the
        noise distribution for the entity at index idx.
    partial_loss(self, idx, entities, n_noise_samples):
        Computes the MTL-weighted NCE loss for the entity at index idx.
    conditional_distribution(self, idx, entities):
        Computes the predicted probability of the entity at index idx
        given previous entities.
    conditional_p_value(self, idx, entities):
        Computes the discrete p-value associated with the entity at
        index idx given previous entities.
    update(self, entities):
        Creates embeddings for previously unseen entities, and updates
        the unigram counts for the entities passed as argument.
    update_noise_dist(self):
        Updates the noise distribution based on the current unigram
        counts.
    reinitialize_counts(self):
        Resets the unigram counts.
    """

    def __init__(
            self, entity_list, entity_types, embedding_dim,
            noise_dist='all', counts=None, seen=None, alpha=.75):
        """
        Constructor.

            Arguments
            ---------
                entity_list : list
                    List of Entity instances corresponding to the
                    existing entity types.
                    This list is shared by all Interaction instances
                    belonging to the same DECADES instance.
                entity_types : list
                    List of indices (referring to elements of
                    entity_list) corresponding to the ordered entity
                    types involved in each event.
                embedding_dim : int
                    Dimension of the embedding space.
                noise_dist : str (optional)
                    Type of noise distribution to use.
                    Must be one of 'all', 'unique', 'log' and 'pow'
                    (default: 'all').
                counts : list (optional)
                    List of objects (lists, one-dimensional tensors or
                    one-dimensional NumPy arrays) containing the
                    unigram counts for each entity involved in this
                    event type.
                    If None, all unigram counts are initialized to 1.
                    (default: None)
                seen : set (optional)
                    Set of observed entity tuples for this event type.
                    Useful only if the 'unique' noise distribution is
                    used.
                    (default: None)
                alpha : float (optional)
                    Exponent parameter for the power-unigram noise
                    distribution.
                    Useful only if the 'pow' noise distribution is
                    used.
                    (default: .75)
        """
        super(Interaction, self).__init__()
        self.entity_list = entity_list
        self.entity_types = entity_types

        self.n_entities = len(entity_types)
        self.logvars = nn.Parameter(torch.zeros(self.n_entities-1))

        if counts is None:
            dimensions = [entity_list[i].n_entities for i in entity_types[1:]]
            counts = [torch.ones(n) for n in dimensions]
        self.counts = [torch.IntTensor(x) for x in counts]

        self.exclude = defaultdict(list)
        for i in range(1, self.n_entities):
            for j in range(i):
                if entity_types[i] == entity_types[j]:
                    self.exclude[i].append(j)
                    if (j >= 1 and
                        self.counts[i-1].shape[0]
                        != self.counts[j-1].shape[0]):
                        m = max(
                            self.counts[i-1].shape[0],
                            self.counts[j-1].shape[0])
                        for k in (i-1, j-1):
                            self.counts[k] = torch.cat([
                                self.counts[k],
                                self.counts[k].new_zeros(
                                    m-self.counts[k].shape[0])
                                ])
        if noise_dist == 'all':
            self.noise_dist = [x.float() for x in self.counts]
        elif noise_dist == 'unique':
            if seen is None:
                raise ValueError(
                    'Set of seen interactions must be provided to use \
                    unique counts')
            self.noise_dist = [x.float() for x in self.counts]
            self.seen = seen
        elif noise_dist == 'log':
            self.noise_dist = [torch.log(1+x.float()) for x in self.counts]
        elif noise_dist == 'pow':
            self.noise_dist = [torch.pow(x.float(), alpha)
                for x in self.counts]
            self.alpha = alpha
        else:
            raise ValueError('Unknown noise distribution')
        self.noise_type = noise_dist
        for i, nd in enumerate(self.noise_dist):
            self.noise_dist[i] /= nd.sum()

        self.linear = nn.ModuleList(SoftmaxWeightedAvg(i)
            for i in range(1, len(entity_types)))
        self.weighted_dot = WeightedDot(embedding_dim)

        self.y_mean = 0
        self.y_std = 1

        self.loss = nn.BCEWithLogitsLoss()

    def forward(
            self, entities, training=True, updating=False,
            n_noise_samples=10, return_pval=False):
        """
        Forward pass of the module.
        The output depends on the training argument: if training is
        True, the aggregated MTL/NCE loss is returned.
        Otherwise, the anomaly scores of the events are returned.

        Arguments
        ---------
        entities : torch.LongTensor
            Batch of events to be analyzed.
            This tensor has shape (n_events, self.n_entities).
        training : bool
            If True, the aggregated MTL/NCE loss for the batch is
            returned.
            Otherwise, the anomaly scores are returned.
            (default: True)
        updating : bool
            If updating == True and training == False, the unigram
            counts are updated and embeddings are created for
            previously unseen entities.
            This corresponds to the test/detection phase.
            (default: False)
        n_noise_samples : int
            Number of negative samples to generate when computing the
            NCE loss.
            Useful only if training == True.
            (default: 10)
        return_pval : bool
            If True, the anomaly scores are defined as the aggregated
            negative discrete log-p-values for involved entities.
            Otherwise, they are defined as the aggregated negative
            logarithmic conditional affinities (Kappa function).
            (default: False)

        Returns
        -------
        y : torch.Tensor
            If training == True, y has a single element, equal to the
            aggregated MTL/NCE loss for the batch.
            Otherwise, y has shape (entities.shape[0],), with the i-th
            element equal to the anomaly score for the i-th event in
            the batch.
        """
        if training:
            return sum(self.partial_loss(i, entities, n_noise_samples)
                for i in range(1, self.n_entities))
        else:
            if updating:
                self.update(entities)
            if return_pval:
                pval = torch.stack(
                    [self.conditional_p_value(i, entities)
                    for i in range(1, self.n_entities)],
                    dim=1)
                y = -torch.log(torch.maximum(
                    pval,
                    torch.full_like(pval, 1e-40)
                    # clamp values to avoid infinite scores
                    )).sum(1)/(self.n_entities-1)
            else:
                y = -sum(self.conditional_affinity(i, entities)
                    for i in range(1, self.n_entities))
            return (y-self.y_mean)/self.y_std

    def context_vector(self, idx, entities):
        """
        Returns the h_e,i vectors for the given event batch, with
        i == idx.

        Arguments
        ---------
        idx : int
            Index of the entity for which the context vector is
            computed.
            The context vector is the weighted sum of the embeddings
            of entities up to index idx-1.
        entities : torch.LongTensor
            Input event batch (shape: (n_events, self.n_entities)).

        Returns
        -------
        h : torch.Tensor
            Context vectors for each event in the batch (shape:
            (n_events, embedding_dim)).
        """
        inputs = [self.entity_list[e](entities[:,i])
            for i, e in enumerate(self.entity_types[:idx])]
        return self.linear[idx-1](torch.stack(inputs, dim=1))

    def conditional_affinity(self, idx, entities):
        """
        Returns the logarithm of the affinity function Kappa_e,i for
        the input event batch, with i == idx.

        Arguments
        ---------
        idx : int
            Index of the entity for which the affinity function is
            computed.
        entities : torch.LongTensor
            Input event batch (shape: (n_events, self.n_entities)).

        Returns
        -------
        affinity : torch.Tensor
            Logarithm of the idx-th affinity function for each event
            in the batch (shape: (n_events,)).
        """
        context_vector = self.context_vector(idx, entities)
        target_vector = self.entity_list[self.entity_types[idx]](
            entities[:, idx])
        return self.weighted_dot(context_vector, target_vector)

    def get_neg_samples(self, idx, n_noise_samples):
        """
        Generates negative samples using the noise distribution for
        the idx-th entity involved in this event type.

        Arguments
        ---------
        idx : int
            Index of the entity for which negative samples are
            generated.
        n_noise_samples : int
            Number of negative samples to generate.

        Returns
        -------
        samples : torch.LongTensor
            Negative samples (shape: (n_noise_samples,)).
        """
        return torch.multinomial(
            self.noise_dist[idx-1],
            n_noise_samples, replacement=True)

    def get_noise_prob_with_exclusions(self, idx, entities, to_eval):
        """
        Returns the probability of the entities in to_eval under the
        idx-th noise distribution.
        These probabilities are adjusted to take into account that
        one entity cannot appear twice in the same event.

        Arguments
        ---------
        idx : int
            Index of the entity whose noise distribution should be
            used.
        entities : torch.LongTensor
            Input batch of events (shape: (n_events, self.n_entities)).
            Used to ensure that the probability of an entity appearing
            twice in the same event is zero.
        to_eval : torch.LongTensor
            Batch of entities whose noise probability will be computed.
            The shape is arbitrary; the output will have the same
            shape.

        Returns
        -------
        probabilities : torch.Tensor
            Noise probabilities of the given entities
            (shape == to_eval.shape).
        """
        nd = self.noise_dist[idx-1][to_eval].to(entities.device)
        if idx in self.exclude:
            corr = 1 - sum(self.noise_dist[idx-1][entities[:, i]]
                for i in self.exclude[idx])
            if len(nd.shape) > 1:
                corr.unsqueeze_(1)
            nd /= corr.to(entities.device)
        return nd

    def partial_loss(self, idx, entities, n_noise_samples):
        """
        Computes the MTL-weighted NCE loss for the idx-th entity of
        each event in the input batch.
        The mean entity-wise loss is returned.

        Arguments
        ---------
        idx : int
            Index of the entity for which the NCE loss will be
            computed.
        entities : torch.LongTensor
            Input batch of events (shape: (n_events, self.n_entities)).
        n_noise_samples : int
            Number of negative samples to generate when computing the
            NCE loss.

        Returns
        -------
        loss : torch.Tensor
            Mean MTL/NCE loss (shape: (1,)).
        """
        neg = self.get_neg_samples(idx, n_noise_samples).to(entities.device)
        context_vector = self.context_vector(idx, entities)
        target_vector = self.entity_list[self.entity_types[idx]](
            entities[:, idx])

        sel = torch.full(
            (entities.shape[0], n_noise_samples), True,
            device=entities.device)
        if idx in self.exclude:
            for j in self.exclude[idx]:
                sel *= (entities[:,j].unsqueeze(1) != neg.unsqueeze(0))

        neg_scores = self.weighted_dot(
            context_vector,
            self.entity_list[self.entity_types[idx]](neg), False) \
            - torch.log(sel.sum(1).float()).unsqueeze(1) \
            - torch.log(
                self.get_noise_prob_with_exclusions(
                idx, entities, neg.repeat((entities.shape[0], 1))))
        pos_scores = self.weighted_dot(context_vector, target_vector) \
            - torch.log(sel.sum(1).float()) \
            - torch.log(self.get_noise_prob_with_exclusions(
                idx, entities, entities[:, idx]))
        preds = torch.cat([neg_scores[sel], pos_scores])
        labels = torch.cat([
            preds.new_zeros(sel.sum()),
            preds.new_ones(pos_scores.numel())])
        return (torch.exp(-self.logvars[idx-1])*self.loss(preds, labels)
            + .5*self.logvars[idx-1])

    def conditional_distribution(self, idx, entities):
        """
        Returns the predicted probability of the idx-th entity of each
        event in the input batch.

        Arguments
        ---------
        idx : int
            Index of the entity whose predicted probability is
            computed.
        entities : torch.LongTensor
            Input batch of events (shape: (n_events, self.n_entities)).
        
        Returns
        -------
        probabilities : torch.Tensor
            Predicted probabilities of the entities
            (shape: (n_events,)).
        """
        ent = self.entity_types[idx]
        context_vector = self.context_vector(idx, entities)
        target_vector = self.entity_list[ent](
            torch.arange(
                self.entity_list[ent].n_entities
            ).to(entities.device))
        prod = self.weighted_dot(context_vector, target_vector, False)

        if idx in self.exclude:
            for j in self.exclude[idx]:
                prod[
                    torch.arange(entities.shape[0]).to(entities.device),
                    entities[:,j]] = -1e10
        return F.softmax(prod, dim=1)

    def conditional_p_value(self, idx, entities):
        """
        Returns the discrete p-value associated with the idx-th entity
        of each event in the input batch.

        Arguments
        ---------
        idx : int
            Index of the entity whose p-value is computed.
        entities : torch.LongTensor
            Input batch of events (shape: (n_events, self.n_entities)).
        
        Returns
        -------
        p_values : torch.Tensor
            Discrete p-values of the entities (shape: (n_events,)).
        """
        dist = self.conditional_distribution(idx, entities)
        return mid_p_value(entities[:, idx], dist.detach())

    def update(self, entities):
        """
        Creates new embeddings for previously unknown entities from
        the input batch, and updates the unigram counts.

        Arguments
        ---------
        entities : torch.LongTensor
            Input batch of events (shape: (n_events, self.n_entities)).
        
        Returns
        -------
        None
        """
        n = entities.shape[0]
        if self.noise_type == 'unique':
            ind = []
            for i in range(n):
                evt = tuple(entities[i,:].tolist())
                if evt not in self.seen:
                    ind.append(i)
                    self.seen.add(evt)
        else:
            ind = list(range(n))
        idx = torch.amax(entities, 0)
        for i in range(idx.shape[0]):
            self.entity_list[self.entity_types[i]].update(idx[i])
            if i >= 1:
                counts = torch.bincount(entities[ind,i]).to(
                    self.counts[i-1].device)
                if idx[i] >= self.counts[i-1].shape[0]:
                    self.counts[i-1] = torch.cat([
                        self.counts[i-1],
                        torch.zeros(
                            idx[i]-len(self.counts[i-1])+1
                        ).int()])
                else:
                    counts = torch.cat([
                        counts,
                        torch.zeros(
                            self.counts[i-1].shape[0]-counts.shape[0]
                        ).int().to(counts.device)])
                self.counts[i-1] += counts

    def update_noise_dist(self):
        """
        Updates the noise distribution based on the current unigram
        counts.

        Arguments
        ---------
        None
        
        Returns
        -------
        None
        """
        for i in self.exclude:
            for j in self.exclude[i]:
                if (j >= 1
                    and self.counts[i-1].shape[0]
                    != self.counts[j-1].shape[0]):
                    m = max(
                        self.counts[i-1].shape[0],
                        self.counts[j-1].shape[0])
                    for k in (i-1, j-1):
                        self.counts[k] = torch.cat([
                            self.counts[k],
                            self.counts[k].new_zeros(
                                m-self.counts[k].shape[0])
                            ])
        if self.noise_type in ('all', 'unique'):
            self.noise_dist = [x.float() for x in self.counts]
        elif self.noise_type == 'log':
            self.noise_dist = [torch.log(1+x.float()) for x in self.counts]
        elif self.noise_type == 'pow':
            self.noise_dist = [torch.pow(x.float(), self.alpha)
                for x in self.counts]
        for i, nd in enumerate(self.noise_dist):
            self.noise_dist[i] /= nd.sum()

    def reinitialize_counts(self):
        """
        Resets the unigram counts.

        Arguments
        ---------
        None
        
        Returns
        -------
        None
        """
        for i, c in enumerate(self.counts):
            self.counts[i] = torch.zeros_like(c)
