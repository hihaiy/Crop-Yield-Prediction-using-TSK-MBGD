import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import softmax
from torch.utils.data import DataLoader

from pytsk.gradient_descent.utils import NumpyDataLoader

def ur_loss(frs, tau=0.5):
    r"""
    The uniform regularization (UR) proposed by Cui et al.
    UR loss is computed as :math:`\ell_{UR} = \sum_{r=1}^R (\frac{1}{N}\sum_{n=1}^N f_{n,r} - \tau)^2`,
    where :math:`f_{n,r}` represents the firing level of the :math:`n`-th sample on the :math:`r`-th rule.

    :param torch.tensor frs: The firing levels (output of the antecedent) with the size of :math:`[N, R]`,
    where :math:`N` is the number of samples, :math:`R` is the number of rules.
    :param float tau: The expectation :math:`\tau` of the average firing level for each rule. For a
        :math:`C`-class classification problem, we recommend setting :math:`\tau` to :math:`1/C`,
        for a regression problem, :math:`tau` can be set as :math:`0.5`.
    :return: A scale value, representing the UR loss.
    """
    return ((torch.mean(frs, dim=0) - tau) ** 2).sum()

class Wrapper:
    """
    This class provides a training framework for beginners to train their fuzzy neural networks.

    :param torch.nn.Module model: The pre-defined TSK model.
    :param torch.Optimizer optimizer: PyTorch optimizer.
    :param torch.nn._Loss criterion: PyTorch loss. For example, :code:`torch.nn.CrossEntropyLoss()` for classification tasks,
        and :code:`torch.nn.MSELoss()` for regression tasks.
    :param int batch_size: Batch size during training & prediction.
    :param int epochs: Training epochs.
    :param [Callback] callbacks: List of callbacks.
    :param str label_type: Label type, "c" or "r", when :code:`label_type="c"`, label's dtype will be changed to
        "int64", when :code:`label_type="r"`, label's dtype will be changed to "float32".
    :param str device: Device to use for training and inference ("cpu" or "cuda").
    :param bool reset_param: Whether to reset model parameters before training.
    :param float ur: Weight for the uniform regularization loss.
    :param float ur_tau: Expectation for the average firing level for each rule.
    """
    def __init__(self, model, optimizer, criterion,
                 batch_size=512, epochs=1, callbacks=None, label_type="c",
                 device="cpu", reset_param=True, ur=0, ur_tau=0.5, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.model.to(self.device)
        self.label_type = label_type
        self.stop_training = False  # Add stop_training attribute

        self.ur = ur
        self.ur_tau = ur_tau

        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            raise ValueError("callbacks must be a list of Callback objects")
        
        self.reset_param = reset_param
        if self.reset_param:
            self.model.reset_parameters()
        self.cur_batch = 0
        self.cur_epoch = 0

        self.kwargs = kwargs

    def train_on_batch(self, input, target):
        """
        Define how to update a model with one batch of data.
        This method can be overwritten for custom training strategy.

        :param torch.tensor input: Feature matrix with the size of :math:`[N,D]`,
            :math:`N` is the number of samples, :math:`D` is the input dimension.
        :param torch.tensor target: Target matrix with the size of :math:`[N,C]`,
            :math:`C` is the output dimension.
        """
        # Update model once
        input, target = input.to(self.device), target.to(self.device)
        outputs, frs = self.model(input, get_frs=True)
        if self.ur > 0:
            ur_loss_value = ur_loss(frs, self.ur_tau)
            loss = self.criterion(outputs, target) + self.ur * ur_loss_value
        else:
            loss = self.criterion(outputs, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, X, y):
        """
        Train the model with numpy array.

        :param numpy.array X: Feature matrix with the size of :math:`[N, D]`.
        :param numpy.array y: Label matrix with the size of :math:`[N, C]`,
            for classification tasks, :math:`C=1`, for regression tasks, :math:`C` is the
            number of the output dimension of the model.
        """
        if isinstance(X, np.ndarray):
            X = X.astype("float32")
        if isinstance(y, np.ndarray):
            if self.label_type == "c":
                y = y.astype("int64")
            elif self.label_type == "r":
                y = y.astype("float32")
            else:
                raise ValueError('label_type can only be "c" or "r"!')

        train_loader = DataLoader(
            NumpyDataLoader(X, y),
            batch_size=self.batch_size,
            shuffle=self.kwargs.get("shuffle", True),
            num_workers=self.kwargs.get("num_workers", 0),
            drop_last=self.kwargs.get("drop_last", True if self.batch_size < X.shape[0] else False),
        )

        self.fit_loader(train_loader)
        return self


    def fit_loader(self, train_loader):
        """
        Train the model with a user-defined PyTorch DataLoader.

        :param torch.utils.data.DataLoader train_loader: Data loader. The output of the loader should be
            corresponding to the inputs of `train_on_batch`. For example, if the dataloader has two outputs,
            then `train_on_batch` should also have two inputs.
        """
        self.stop_training = False
        for e in range(self.epochs):
            self.cur_epoch = e
            self.__run_callbacks__("on_epoch_begin")
            for inputs in train_loader:
                self.__run_callbacks__("on_batch_begin")
                self.model.train()
                self.train_on_batch(inputs[0], inputs[1])
                self.__run_callbacks__("on_batch_end")
                self.cur_batch += 1
            self.__run_callbacks__("on_epoch_end")
            if self.stop_training:  # Check the stop_training flag
                break
        return self

    def predict(self, X, y=None):
        """
        Get the prediction of the model.

        :param numpy.array X: Feature matrix with the size of :math:`[N, D]`.
        :param y: Not used.
        :return: Prediction matrix with the size of :math:`[N, C]`,
            where :math:`C` is the output dimension of the model.
        """
        # X = X.astype("float32")
        test_loader = DataLoader(
            NumpyDataLoader(X),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.kwargs.get("num_workers", 0),
            drop_last=False
        )
        y_preds = []
        for inputs in test_loader:
            self.model.eval()
            y_pred = self.model(inputs.to(self.device)).detach().cpu().numpy()
            y_preds.append(y_pred)
        return np.concatenate(y_preds, axis=0)

    def predict_proba(self, X, y=None):
        """
        For classification problems only. Needs `label_type="c"`, returns the prediction after softmax.

        :param numpy.array X: Feature matrix with the size of :math:`[N, D]`.
        :param y: Not used.
        :return: Prediction matrix with the size of :math:`[N, C]`,
            where :math:`C` is the output dimension of the model.
        """
        if self.label_type == "r":
            raise ValueError('predict_proba can only be used when label_type="c"')
        y_preds = self.predict(X)
        return softmax(y_preds, axis=1)

    def __run_callbacks__(self, func_name):
        for cb in self.callbacks:
            getattr(cb, func_name)(self)

    def save(self, path):
        """
        Save the model.

        :param str path: Model save path.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Load the model.

        :param str path: Model load path.
        """
        self.model.load_state_dict(torch.load(path))
