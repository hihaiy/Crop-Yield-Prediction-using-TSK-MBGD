from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np

class Callback:
    """
    Similar to the callback class in Keras, our package provides a simplified version
    of callback, which allows users to monitor metrics during the training.
    We strongly recommend users to customize their callbacks. Here we provide two
    examples, :func:`EvaluateMSE <EvaluateMSE>` and :func:`EarlyStoppingMSE <EarlyStoppingMSE>`.
    """
    def on_batch_begin(self, wrapper):
        pass

    def on_batch_end(self, wrapper):
        pass

    def on_epoch_begin(self, wrapper):
        pass

    def on_epoch_end(self, wrapper):
        pass

class EvaluateMSE(Callback):
    """
    Evaluate the Mean Squared Error (MSE) during training.

    :param numpy.array X: Feature matrix with the size of :math:`[N, D]`.
    :param numpy.array y: Label matrix with the size of :math:`[N, 1]`.
    """
    def __init__(self, X, y, verbose=0):
        super(EvaluateMSE, self).__init__()
        self.X, self.y = X, y
        self.verbose = verbose
        self.logs = []

    def on_epoch_end(self, wrapper):
        cur_log = {}
        y_pred = wrapper.predict(self.X).squeeze()
        mse = mean_squared_error(y_true=self.y, y_pred=y_pred)
        cur_log["epoch"] = wrapper.cur_epoch
        cur_log["mse"] = mse
        self.logs.append(cur_log)
        if self.verbose > 0:
            print("[Epoch {:5d}] Test MSE: {:.4f}".format(cur_log["epoch"], cur_log["mse"]))

class EarlyStoppingMSE(Callback):
    """
    Early-stopping by Mean Squared Error (MSE).

    :param numpy.array X: Feature matrix with the size of :math:`[N, D]`.
    :param numpy.array y: Label matrix with the size of :math:`[N, 1]`.
    :param int patience: Number of epochs with no improvement after which training will be stopped.
    :param int verbose: verbosity mode.
    :param str save_path: If :code:`save_path=None`, do not save models, else save the model
        with the best MSE to the given path.
    """
    def __init__(self, X, y, patience=1, verbose=0, save_path=None):
        super(EarlyStoppingMSE, self).__init__()
        self.X, self.y = X, y
        self.verbose = verbose
        self.patience = patience
        self.best_mse = float('inf')
        self.cnt = 0
        self.logs = []
        self.save_path = save_path

    def on_epoch_end(self, wrapper):
        cur_log = {}
        y_pred = wrapper.predict(self.X).squeeze()
        mse = mean_squared_error(y_true=self.y, y_pred=y_pred)

        if mse < self.best_mse:
            self.best_mse = mse
            self.cnt = 0
            if self.save_path is not None:
                wrapper.save(self.save_path)
        else:
            self.cnt += 1
            if self.cnt > self.patience:
                wrapper.stop_training = True
        cur_log["epoch"] = wrapper.cur_epoch
        cur_log["mse"] = mse
        cur_log["best_mse"] = self.best_mse
        self.logs.append(cur_log)
        if self.verbose > 0:
            print("[Epoch {:5d}] EarlyStopping Callback MSE: {:.4f}, Best MSE: {:.4f}".format(cur_log["epoch"], cur_log["mse"], cur_log["best_mse"]))

class EarlyStoppingRMSE(Callback):
    """
    Early-stopping by Root-Mean Squared Error (RMSE).

    :param numpy.array X: Feature matrix with the size of :math:`[N, D]`.
    :param numpy.array y: Label matrix with the size of :math:`[N, 1]`.
    :param int patience: Number of epochs with no improvement after which training will be stopped.
    :param int verbose: verbosity mode.
    :param str save_path: If :code:`save_path=None`, do not save models, else save the model
        with the best MSE to the given path.
    """
    def __init__(self, X, y, patience=1, verbose=0, save_path=None):
        super(EarlyStoppingRMSE, self).__init__()
        self.X, self.y = X, y
        self.verbose = verbose
        self.patience = patience
        self.best_rmse = float('inf')
        self.cnt = 0
        self.logs = []
        self.save_path = save_path

    def on_epoch_end(self, wrapper):
        cur_log = {}
        y_pred = wrapper.predict(self.X)
        rmse = np.sqrt(mean_squared_error(y_true=self.y, y_pred=y_pred))
        mae = mean_absolute_error(y_true=self.y, y_pred=y_pred)

        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.cnt = 0
            if self.save_path is not None:
                wrapper.save(self.save_path)
        else:
            self.cnt += 1
            if self.cnt > self.patience:
                wrapper.stop_training = True
        cur_log["epoch"] = wrapper.cur_epoch
        cur_log["rmse"] = rmse
        cur_log["best_rmse"] = self.best_rmse
        self.logs.append(cur_log)
        if self.verbose > 0:
            print("[Epoch {:5d}] EarlyStopping Callback RMSE: {:.4f}, Best RMSE: {:.4f}".format(cur_log["epoch"], cur_log["rmse"], cur_log["best_rmse"]))
        # print("RMSE:", self.best_rmse)
        # print("MAE:", mae)
