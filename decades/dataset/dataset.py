import gzip
import random
import pandas as pd

import torch
from torch.utils.data import Dataset



class IdxDict:
    """
    Helper class to store and index entities.
    Functions like a defaultdict, where the fallback value for
    unknown keys is the current length of the dictionary.


    Attributes
    ----------
    dict : dict
        Dictionary mapping keys to unique indices.

    Methods
    -------
    keys():
        Returns the keys of the underlying dictionary.
    """

    def __init__(self, initial_values=None):
        """
        Constructor.

            Arguments
            ---------
                initial_values : list (optional)
                    A list of values to add to the dictionary at its
                    creation.
        """
        self.dict = {}
        if initial_values is not None:
            for v in initial_values:
                self.dict[v] = len(self.dict)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, key):
        if key not in self.dict:
            self.dict[key] = len(self.dict)
        return self.dict[key]

    def keys(self):
        """
        Returns the keys of the underlying dictionary.

        Arguments
        ---------
        None

        Returns
        -------
        keys : dict_keys
            Keys of the underlying dictionary.
        """
        return self.dict.keys()

class LANLDataset(Dataset):
    """
    PyTorch dataset to feed the LANL data to the DECADES model.


    Attributes
    ----------
    _train : list
        List of lists representing the training events.
    _test : list
        List of lists representing the test events.
    _test_idx : list
        List of lists of lists.
        Each list at the first level represents one training period
        (typically one day), while each element of these lists contains
        the indices of the corresponding events in the _test list.
    _val_idx : list
        List of indices corresponding to events from the _test list
        used for validation.
    entity_dict : list
        List of IdxDict corresponding to the entity types.
        Each IdxDict maps the identifiers of entities of one type to
        integers between 0 and n_entities-1.
    seen : list
        List of sets, each of which corresponds to one event type and
        contains all entity tuples seen in the training set for this
        event type.
    arities : list
        List containing the number of distinct entities in the training
        set for each entity type.
    interaction_lengths : list
        List containing the number of entities involved in each event
        type.
    interaction_entities : list
        List of lists, each of which contains the ordered entity types
        involved in an event type.
    period : float
        Length of one testing period, i.e. time elapsed between two
        retraining phases.
    training : bool
        If True, the events returned by __getitem__ are from the
        training set.
    validating : bool
        If True, the events returned by __getitem__ are from the
        validation set.
    val_retraining : bool
        If True, the events returned by __getitem__ are from the
        retraining validation set.
    T : int
        Index of the current testing period.
    Methods
    -------
    get_unigram_counts(interaction_type, unique=False):
        Returns the unigram distribution of each entity for a given
        interaction type.
        If unique is True, unigram counts are computed with a unique
        occurrence of each observed tuple of entities.
    make_validation_set(T=None, proportion=.001):
        Samples a validation set from the test set for the T-th test
        period, containing the given proportion of events from this
        set.
        The sampled events are removed from the test set.
        If T is not given, the self.T attribute is used.
        This function is used in the retraining phase to evaluate
        convergence of the retraining algorithm.
    delete_validation_set(T=None):
        Deletes a previously created validation set and puts the events
        back in the T-th test set.
        If T is not given, the self.T attribute is used.
    train():
        Sets the dataset in initial training mode (i.e. returning
        events from the training set).
    validate():
        Sets the dataset in initial validation mode (i.e. returning
        events from the test/detection set involving only entities from
        the initial training set).
    test(T):
        Sets the dataset in test/detection mode for the T-th time
        period (i.e. returning events from the T-th test set).
    val_retrain():
        Sets the dataset in retraining validation mode (i.e. returning
        events from the current retraining validation set).
    load_test_data(file):
        Replaces the currently loaded test dataset.
        Useful when the complete test dataset is too large to fit in
        memory.
    """

    def __init__(self, train, test, conf, no_double=False):
        """
        Constructor for the LANL dataset.

            Arguments
            ---------
                train : file object
                    Iterator on the lines of the training dataset.
                    Lines should be comma-separated, with all values
                    being numbers.
                    Each line represents one event, structured as
                    follows: entity_1, ..., entity_Ne, [padding],
                    timestamp, event type, label (0 for benign and 1
                    for malicious).
                    Lines should be ordered chronologically.
                test : file object
                    Iterator on the lines of the test dataset.
                    The structure of the file is the same as that of
                    the training set.
                conf : dict
                    Dictionary containing information about the
                    dataset.
                    Structure of the dictionary:
                    {
                        'entities': [
                            {
                                'name': ...
                            },
                            ...
                        ],
                        'interactions': [
                            {
                                'name': ...,
                                'n_entities': (int) number of entities
                                    involved in each event of this
                                    type,
                                'entities': (list) ordered indices of
                                    the types of entities involved in
                                    each event of this type.
                                    The indices are from the entity
                                    list from the 'entities' key.
                            }
                        ]
                    }
                no_double : bool
                    If True, the training set is undersampled to
                    contain one unique occurrence of each observed
                    entity tuple for each event type.
                    (default: False)
        """
        super(LANLDataset).__init__()
        self._train = []
        self._test = []
        self._test_idx = []
        self._val_idx = []
        self.entity_dict = [IdxDict() for e in conf['entities']]
        self.seen = [set() for i in conf['interactions']]
        for s1, s2, s3 in zip(
            (train, test),
            (self._train, self._test),
            ('train', 'test')):
            for i, l in enumerate(s1):
                x = [int(float(y)) for y in l.strip().split(',')]
                if x[-2] >= len(conf['interactions']):
                    continue
                if i == 0:
                    start = x[-3]
                if s3 == 'test':
                    t = (x[-3]-start)//conf['time_between_updates']
                    for j in range(t-len(self._test_idx)+1):
                        self._test_idx.append([])
                add_to_val = True
                it = x[-2]
                V = [x[j]
                    for j in range(conf['interactions'][it]['n_entities'])]
                D = [self.entity_dict[j]
                    for j in conf['interactions'][it]['entities']]
                Y = [d[v] for v, d in zip(V, D)]
                for y, z in zip(Y, conf['interactions'][it]['entities']):
                    if s3 == 'test' and y >= max_idx[z]:
                        add_to_val = False
                evt = Y + x[len(Y):]
                if (not no_double or s3 == 'test'
                    or tuple(Y) not in self.seen[x[-2]]):
                    if s3 == 'train':
                        s2.append(evt)
                        self.seen[x[-2]].add(tuple(Y))
                    else:
                        s2.append(evt)
                        self._test_idx[t].append(len(s2)-1)
                        if add_to_val:
                            self._val_idx.append(len(s2)-1)
            if s3 == 'train':
                max_idx = [len(d) for d in self.entity_dict]

        self.arities = max_idx
        self.interaction_lengths = [itr['n_entities']
            for itr in conf['interactions']]
        self.interaction_entities = [itr['entities']
            for itr in conf['interactions']]

        self.period = conf['time_between_updates']

        self.training = True
        self.validating = False
        self.val_retraining = False
        self.T = 0

    def __len__(self):
        if self.training:
            return len(self._train)
        elif self.validating:
            return len(self._val_idx)
        elif self.val_retraining:
            return len(self._val_retrain_idx)
        else:
            return len(self._test_idx[self.T])

    def __getitem__(self, idx):
        if self.training:
            line = self._train[idx]
        elif self.validating:
            j = self._val_idx[idx]
            line = self._test[j]
        elif self.val_retraining:
            j = self._val_retrain_idx[idx]
            line = self._test[j]
        else:
            j = self._test_idx[self.T][idx]
            line = self._test[j]
        return torch.LongTensor(line[:-1]), torch.LongTensor([line[-1]])

    def get_unigram_counts(self, interaction_type, unique=False):
        """
        Returns the unigram counts in the training set for each entity,
        for the given event type.

        Arguments
        ---------
        interaction_type : int
            Index of the considered event type.
        unique : bool
            If True, the unigram counts are computed with only one
            occurrence of each observed entity tuple.

        Returns
        -------
        counts : list
            List of pandas.Series objects containing the unigram
            counts.
        """
        df = pd.DataFrame(self._train)
        df = df[df[df.shape[1]-2]==interaction_type]
        bound = self.interaction_lengths[interaction_type]
        if unique:
            df = pd.DataFrame(list(pd.unique([
                tuple(x[:bound]) for _, x in df.iterrows()])))
        return [df[i].value_counts()
            for i in range(1, bound)]

    def make_validation_set(self, T=None, proportion=.001):
        """
        Randomly samples a validation set from the T-th test set,
        containing a given proportion of the events from this set.
        The sampled events are removed from the test set and stored
        into the self._val_retrain_idx attribute.
        This function is used to evaluate convergence during the
        retraining phase of the model.

        Arguments
        ---------
        T : int
            Index of the test period (if None, the self.T attribute is
            used).
        proportion : float
            Ratio between the size of the sampled validation set and
            the size of the original test set.

        Returns
        -------
        None
        """
        if T is None:
            T = self.T
        n_events = int(proportion*len(self._test_idx[T]))
        events = sorted(
            random.sample(list(range(len(self._test_idx[T]))), n_events))
        self._val_retrain_idx = [self._test_idx[T].pop(i-j)
            for j, i in enumerate(events)]

    def delete_validation_set(self, T=None):
        """
        Deletes the validation set stored in the self._val_retrain_idx
        attribute and puts the events back into the T-th test set.

        Arguments
        ---------
        T : int
            Index of the test set from which the events were originally
            sampled (if None, the self.T attribute is used).
        
        Returns
        -------
        None
        """
        if T is None:
            T = self.T
        self._test_idx[T] = sorted(self._test_idx[T] + self._val_retrain_idx)
        self._val_retrain_idx = []

    def train(self):
        """
        Sets the dataset in initial training mode (i.e. returning
        events from the initial training set).

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        self.training = True
        self.validating = False
        self.val_retraining = False

    def validate(self):
        """
        Sets the dataset in initial validation mode (i.e. returning
        events from the test/detection set involving only entities from
        the initial training set).

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        self.training = False
        self.validating = True
        self.val_retraining = False

    def test(self, T):
        """
        Sets the dataset in test/detection mode for the T-th time
        period (i.e. returning events from the T-th test set).

        Arguments
        ---------
        T : int
            Index of the test period (if None, the self.T attribute
            is used).

        Returns
        -------
        None
        """
        if T >= len(self._test_idx):
            raise ValueError(
                'Period index {0} larger than number of testing periods \
                ({1})'.format(T, len(self._test_idx))
                )
        self.training = False
        self.validating = False
        self.val_retraining = False
        self.T = T

    def val_retrain(self):
        """
        Sets the dataset in retraining validation mode (i.e. returning
        events from the current retraining validation set, stored in
        the self._val_retrain_idx attribute).

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        self.training = False
        self.validating = False
        self.val_retraining = True

    def load_test_data(self, file):
        """
        Replaces the currently loaded test dataset.
        Useful when the complete test dataset is too large to fit in
        memory.

        Arguments
        ---------
        file : file object
            Iterator on the lines of the new test dataset.
            Lines should be comma-separated, with all values
            being numbers.
            Each line represents one event, structured as
            follows: entity_1, ..., entity_Ne, [padding],
            timestamp, event type, label (0 for benign and 1
            for malicious).
            Lines should be ordered chronologically.
        """
        self._test = []
        self._test_idx = []
        self._val_idx = []
        for i, l in enumerate(file):
            x = [int(float(y)) for y in l.strip().split(',')]
            if i == 0:
                start = x[-3]
            t = (x[-3]-start)//self.period
            for j in range(t-len(self._test_idx)+1):
                self._test_idx.append([])
            it = x[-2]
            V = [x[j]
                for j in range(self.interaction_lengths[it])]
            D = [self.entity_dict[j]
                for j in self.interaction_entities[it]]
            Y = [d[v] for v, d in zip(V, D)]
            evt = Y + x[len(Y):]
            self._test.append(evt)
            self._test_idx[t].append(len(self._test)-1)