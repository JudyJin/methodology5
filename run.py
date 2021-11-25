import torch
import numpy as np
import matplotlib.pyplot as plt

from torchts.nn.loss import quantile_loss
from torchts.nn.model import TimeSeriesModel
from torchts.nn.models.seq2seq import Encoder, Decoder, Seq2Seq

import argparse


class LSTM(TimeSeriesModel):
    def __init__(self, input_size, output_size, optimizer, hidden_size=8, batch_size=10, **kwargs):
        super(LSTM, self).__init__(optimizer, **kwargs)
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        return (torch.zeros(self.batch_size, self.hidden_size),
                torch.zeros(self.batch_size, self.hidden_size))

    def forward(self, x, y=None, batches_seen=None):
        hc = self.init_hidden()

        hidden, _ = self.lstm(x, hc)
        out = self.linear(hidden)
        return out


def run_model(x, y):

    # define hyper-parameters
    dropout_rate = 0.8
    num_layers = 1
    hidden_dim = 32
    input_dim = 1
    output_dim = 1
    horizon = 20
    batch_size = 64

    # uncertainty quantification
    quantiles = [0.025, 0.5, 0.975]

    # instantiate seq2seq model
    models = {}
    for quantile in quantiles:
        encoder = Encoder(input_dim, hidden_dim, num_layers, dropout_rate)
        decoder = Decoder(output_dim, hidden_dim, num_layers, dropout_rate)

        models[quantile] = Seq2Seq(
            encoder,
            decoder,
            output_dim,
            horizon,
            optimizer=torch.optim.Adam,
            criterion=quantile_loss,
            criterion_args={"quantile": quantile}
        )

    # train model
    for _, model in models.items():
        model.fit(
            torch.from_numpy(x),
            torch.from_numpy(y),
            max_epochs=300,
            batch_size=batch_size,
        )

    # inference
    y_preds = {}
    for q, model in models.items():
        y_preds[q] = model(torch.from_numpy(x)).detach().numpy()

    # plt.plot(x[-horizon:].flatten(), y_preds[0.025][-horizon].flatten(), label="p=0.025")
    # plt.plot(x[-horizon:].flatten(), y_preds[0.5][-horizon].flatten(), label="p=0.5")
    # plt.plot(x[-horizon:].flatten(), y_preds[0.975][-horizon].flatten(), label="p=0.975")
    # plt.plot(x.flatten(), y.flatten(), label="y_true")
    # plt.legend()
    #plt.show()

    inputDim = 1
    outputDim = 1
    optimizer_args = {"lr": 0.01}
    quantiles = [0.025, 0.5, 0.975]

    batch_size = 10
    models = {quantile: LSTM(
        inputDim,
        outputDim,
        torch.optim.Adam,
        criterion=quantile_loss,
        criterion_args={"quantile": quantile},
        optimizer_args=optimizer_args
    ) for quantile in quantiles}

    # Resize our x and y to remove unnecessary nested arrays
    x.resize([100, 1])
    y.resize([100, 1])

    for _, model in models.items():
        # train model
        model.fit(
            torch.from_numpy(x),
            torch.from_numpy(y),
            max_epochs=100,
            batch_size=batch_size,
        )

    # inference
    y_preds = {}
    for x_batch in torch.split(torch.from_numpy(x), batch_size):
        for q, model in models.items():
            if q not in y_preds:
                y_preds[q] = [model.predict(x_batch).detach().numpy()]
            else:
                y_preds[q].append(model.predict(x_batch).detach().numpy())
    y_preds = {q: np.concatenate(y_pred) for q, y_pred in y_preds.items()}

    # Plot our uncertainty quantification and save it locally
    plt.clf()
    plt.plot(x.flatten(), y_preds[0.025].flatten(), label="p=0.025")
    plt.plot(x.flatten(), y_preds[0.5].flatten(), label="p=0.5")
    plt.plot(x.flatten(), y_preds[0.975].flatten(), label="p=0.975")
    plt.plot(x.flatten(), y.flatten(), label="y_true")
    plt.legend()
    plt.savefig("uncertainty_fig")
    plt.show()


def test():
    """Generate test data and pass it into our pipeline for regression with
    quantile error and uncertainty quantification"""

    # generate linear time series data with some noise
    x = np.linspace(-10, 10, 100).reshape(-1, 1).astype(np.float32)
    y = 2*x+1 + \
        np.random.normal(0, 2, x.shape).reshape(-1, 1).astype(np.float32)
    #plt.plot(x.flatten(), y.flatten())
    #plt.show()
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]

    run_model(x, y)


# Get commandline arguments to inform what function we want to use
parser = argparse.ArgumentParser()
parser.add_argument("context", help="specify what pipeline to run")
args = parser.parse_args()

# If the argument "test" is specified, run our test function
if args.context == "test":
    test()
