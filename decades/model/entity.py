import torch
import torch.nn as nn
from torch.nn import functional as F




class Entity(nn.Module):
    """
    PyTorch module representing all entities of one given type.
    It is essentially an embedding layer, which also handles creation
    of new embeddings for previously unknown entities.


    Attributes
    ----------
    embeddings : torch.nn.Embedding
        PyTorch module mapping entities to their embeddings.
    name : str
        Name of this entity type.
    n_entities : int
        Number of entities of this type (equal to
        self.embeddings.weight.shape[0]).

    Methods
    -------
    forward(idx):
        Forward pass of the module.
        Takes a torch.LongTensor of entity indices and returns the
        corresponding embeddings.
    update(idx):
        Creates new embeddings and appends them to
        self.embeddings.weight until
        self.embeddings.weight.shape[0] >= idx+1.
        New embeddings are initialized as the mean of previously
        existing embeddings.
    """

    def __init__(self, n_entities, embedding_dim, name=''):
        """
        Constructor.
        Creates the initial embeddings (random initialization).

            Arguments
            ---------
                n_entities : int
                    Number of entity embeddings to initialize.
                embedding_dim : int
                    Dimension of the embeddings.
                name : str (optional)
                    Name of this entity type (default: '').
        """
        super(Entity, self).__init__()
        self.embeddings = nn.Embedding(n_entities, embedding_dim)
        self.name = name
        self.n_entities = n_entities

    def forward(self, idx):
        """
        Forward pass of the module.
        Takes a torch.LongTensor of entity indices and returns the
        corresponding embeddings.

        Arguments
        ---------
        idx : torch.LongTensor
            Indices of the entities whose embeddings are returned.
            Shape is arbitrary.

        Returns
        -------
        embeddings : torch.Tensor
            Embeddings of the queried entities (shape:
            (*idx.shape, embedding_dim)).
        """
        return self.embeddings(idx)

    def update(self, idx):
        """
        Creates new embeddings and appends them to
        self.embeddings.weight until
        self.embeddings.weight.shape[0] >= idx+1.
        New embeddings are initialized as the mean of previously
        existing embeddings.

        Arguments
        ---------
        idx : int
            Index of a previously unknown entity (if the index of an
            already known entity is passed, nothing happens).

        Returns
        -------
        None
        """
        if idx >= self.n_entities:
            self.embeddings.weight = nn.Parameter(
                torch.cat([
                    self.embeddings.weight,
                    self.embeddings.weight.detach().mean(0).repeat(
                        (idx-self.n_entities+1, 1))
                ]))
            self.n_entities = idx + 1
