import torch
import torch.nn as nn
from torch.nn import functional as F




class SoftmaxWeightedAvg(nn.Module):
    """
    PyTorch module computing the h_e,i vectors.
    It essentially learns the weights of a linear combination,
    using a softmax function to ensure these weights are all positive
    and sum to 1.


    Attributes
    ----------
    weights : torch.Tensor
        (Logarithmic, unnormalized) weights of the linear combination.

    Methods
    -------
    forward(input):
        Returns the linear combination of the input vectors.
    """

    def __init__(self, n_inputs):
        """
        Constructor.

            Arguments
            ---------
                n_inputs : int
                    Number of vectors in the linear combination.
        """
        super(SoftmaxWeightedAvg, self).__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs))

    def forward(self, input):
        """
        Forward pass of the module.
        Takes a batch of vector tuples and returns a batch of combined
        vectors.

        Arguments
        ---------
        input : torch.Tensor
            Batch of vector tuples to combine (shape:
            (batch_size, self.n_inputs, vector_dim)).

        Returns
        -------
        combination : torch.Tensor
            Linear combination for each tuple in the batch (shape:
            (batch_size, vector_dim)).
        """
        return torch.einsum(
            'xyz,y->xz',
            input,
            F.softmax(self.weights, dim=0))


class WeightedDot(nn.Module):
    """
    PyTorch module computing weighted dot product of two vectors.
    It is used by the Interaction module to compute the affinity
    function Kappa.


    Attributes
    ----------
    weights : torch.Tensor
        Weight vector used to compute the weighted dot product.
        In practice, it is the latent factor associated with one event
        type.

    Methods
    -------
    forward(input_left, input_right, reduce=True):
        Returns the weighted dot products of vector batches input_left
        and input_right.
        If reduce is True, input_left and input_right are assumed to
        contain the same number of vectors m, and m weighted dot
        products are computed (i-th left vector with i-th right
        vector).
        Otherwise, all pairwise weighted dot products between the left
        and right vectors are computed.
    """

    def __init__(self, dim):
        """
        Constructor.

            Arguments
            ---------
                dim : int
                    Dimension of the vectors to be given as input.
        """
        super(WeightedDot, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, dim))

    def forward(self, input_left, input_right, reduce=True):
        """
        Forward pass of the module.
        Returns the weighted dot products of vector batches input_left
        and input_right.
        If reduce is True, input_left and input_right are assumed to
        contain the same number of vectors m, and m weighted dot
        products are computed (i-th left vector with i-th right
        vector).
        Otherwise, all pairwise weighted dot products between the left
        and right vectors are computed.

        Arguments
        ---------
        input_left : torch.Tensor
            Batch of vectors (shape: (batch_size_left, self.dim)).
        input_right : torch.Tensor
            Batch of vectors (shape: (batch_size_right, self.dim)).
        reduce : bool (optional)
            If reduce is True, then implicitly
            batch_size_left == batch_size_right == batch_size, and
            the output is a tensor with shape (batch_size,) whose i-th
            coefficient is the weighted dot product between the i-th
            left vector and the i-th right vector.
            Otherwise, all pairwise weighted dot products are computed
            and returned as a (batch_size_left, batch_size_right)
            tensor.

        Returns
        -------
        products : torch.Tensor
            Weighted dot products of the inputs (shape: (batch_size,)
            if reduce is True, (batch_size_left, batch_size_right)
            otherwise).
        """
        if reduce:
            assert input_left.shape[0] == input_right.shape[0], \
                'Setting reduce to True implies that the input tensors \
                have the same shape'
            return torch.einsum(
                'xy,xy->x',
                input_left * self.weights,
                input_right)
        else:
            return torch.mm(
                input_left * self.weights,
                input_right.T)
