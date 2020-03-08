from data_science_layer.machine_learning.not_sk_learn_ml_models. \
    pytorch_base import PytorchModel, nn, torch
import types
from itertools import tee
import numpy as np


class PytorchRnn(PytorchModel):
    first_stage_size = 400
    first_stage_layers = 1
    intermediate_stage_features = 4


    def _setup_model(self, x, y):
        models = []
        models.append(nn.RNN(
            input_size=x.shape[1],
            hidden_size=self.first_stage_size,
            num_layers=self.first_stage_layers,
            dropout=0.01
        ))
        models.append(nn.RNN(
            input_size=self.first_stage_size * 2,
            hidden_size=self.intermediate_stage_features,
            num_layers=1,
            dropout=0.01
        ))
        models.append(nn.RNN(
            input_size=self.intermediate_stage_features*2,
            hidden_size=y.shape[1],
            num_layers=1,
            dropout=0.01
        ))
        self._model = models

    def _prep_tensors(self, x, y):
        x, y = self.handle_pandas(x, y)
        y = torch.tensor(y)
        y = torch.reshape(y, (y.shape[0], 1, y.shape[1]))
        x = torch.tensor(x, requires_grad=True)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        x = x.float()
        y = self.set_y_data_type(y)
        if self._gpu_available:
            x.to(self._gpus)
            y.to(self._gpus)
        return x, y

    def predict(self, x):
        predictions_out = []
        stack_size = x.shape[0]
        for seq in range(stack_size):
            x_seq = x[seq, :, :]
            seq_length = x_seq.shape[0]
            x_seq = torch.tensor(x_seq, requires_grad=False)
            x_seq = torch.reshape(x_seq, (x_seq.shape[0], 1, x_seq.shape[1]))
            x_seq = x_seq.float()
            predictions, h0 = self._model[0](x_seq)
            h0_zeros = torch.zeros((seq_length, 1, 400))
            for z in range(seq_length):
                h0_zeros[z, 0, :] = h0
            p0 = torch.cat((predictions, h0_zeros), 2)
            # STAGE 2
            predictions, h1 = self._model[1](
                p0)
            h1_zeros = torch.zeros((seq_length, 1, 4))
            for z in range(seq_length):
                h1_zeros[z, 0, :] = h1
            p1 = torch.cat((predictions, h1_zeros), 2)
            # STAGE 3
            predictions, h2 = self._model[2](p1)
            predictions = torch.squeeze(predictions, dim=2)
            predictions_out.append(predictions.detach().numpy())
        return np.stack(predictions_out)

    def fit(self, x, y):
        last_loss = 100000000
        stack_size = x.shape[0]
        h0 = None
        first = True
        for t in range(self.train_time):
            counter = 0
            running_loss = 0.0

            for seq in range(stack_size):
                x_seq = x[seq, :, :]
                y_seq = y[seq, :, :]
                seq_length = x_seq.shape[0]
                if first:
                    outputs, size = self._prep_input(x_seq, y_seq)
                    optimizer = torch.optim.Adam(self._model[0].parameters(),
                                                 lr=0.05)
                    x_seq, y_seq = self._prep_tensors(x_seq, y_seq)
                    # STAGE 1
                    predictions, h0 = self._model[0](x_seq)
                    h0_zeros = torch.zeros((seq_length, 1, 400))
                    for z in range(seq_length):
                        h0_zeros[z, 0, :] = h0
                    p0 = torch.cat((predictions, h0_zeros), 2)
                    # STAGE 2
                    predictions, h1 = self._model[1](
                        p0)
                    h1_zeros = torch.zeros((seq_length, 1, 4))
                    for z in range(seq_length):
                        h1_zeros[z, 0, :] = h1
                    p1 = torch.cat((predictions, h1_zeros), 2)
                    # STAGE 3
                    predictions, h2 = self._model[2](p1)
                else:
                    x_seq, y_seq = self._prep_tensors(x_seq, y_seq)
                    predictions, h0 = self._model[0](x_seq, h0)
                    h0_zeros = torch.zeros((seq_length, 1, 400))
                    for z in range(seq_length):
                        h0_zeros[z, 0, :] = h0
                    p0 = torch.cat((predictions, h0_zeros), 2)
                    # STAGE 2
                    predictions, h1 = self._model[1](
                        p0, h1)
                    h1_zeros = torch.zeros((seq_length, 1, 4))
                    for z in range(seq_length):
                        h1_zeros[z, 0, :] = h1
                    p1 = torch.cat((predictions, h1_zeros), 2)
                    # STAGE 3
                    predictions, h2 = self._model[2](p1, h2)

                loss = self.loss_func(predictions,
                                      y_seq.view(size, outputs))
                first = False
                self._model[0].zero_grad()
                self._model[1].zero_grad()
                self._model[2].zero_grad()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                running_loss += loss.item() / size
                optimizer.step()
                counter += 1
                print('field {}'.format(counter), running_loss / counter)
            print(t, running_loss / counter)
            if abs(last_loss - running_loss) < 0.000001:
                break
            last_loss = running_loss
            print(last_loss)
            print(((running_loss) ** (0.5)) / counter)
