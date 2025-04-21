import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Base(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, training_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.training_size = training_size

    def train_model(
        self,
        x_train,
        y_train,
        task_idx,
        num_epochs=1000,
        batch_size=100,
        lr=0.001,
        display_epoch=5,
    ):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        N = x_train.size(0)
        batch_size = N if N < batch_size else batch_size

        self.train()
        for epoch in range(num_epochs):
            avg_cost = 0.0
            num_batches = len(loader)
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                loss = self.cost(batch_x, batch_y, task_idx)
                loss.backward()
                self.optimizer.step()

                avg_cost += loss.item() / num_batches

            if (epoch + 1) % display_epoch == 0:
                print(f"Epoch: {epoch+1}, cost= {avg_cost:.9f}")

        print("Optimization Finished!")

    def prediction(self, x_test, task_idx):
        with torch.no_grad():
            outputs = self(x_test, task_idx)
            return outputs.cpu().numpy()


class Vanilla_NN(Base):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        training_size,
        prev_weights=None,
    ):
        super().__init__(input_size, hidden_size, output_size, training_size)
        self.hidden_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()  # multihead
        self.task_idx = 0

        # hidden layers
        prev_size = input_size
        for i, h_size in enumerate(hidden_size):
            layer = nn.Linear(prev_size, h_size)
            with torch.no_grad():
                if prev_weights is not None:
                    layer.weight.copy_(prev_weights[0][i].clone())
                    layer.bias.copy(prev_weights[1][i].clone())
                else:
                    layer.weight.copy_(torch.randn(h_size, prev_size) * 0.1)
                    layer.bias.copy_(torch.randn(h_size) * 0.1)
            self.hidden_layers.append(layer)
            prev_size = h_size

        # output layers
        # if has prev_weights and prev_weights has weights, biases from previous layer but ALSO weights for last layer (head weights)
        if prev_weights is not None:
            # for w in previous head weights
            num_prev_tasks = len(prev_weights[2])
            for i in range(num_prev_tasks):
                layer = nn.Linear(prev_size, output_size)
                with torch.no_grad():
                    layer.weight.copy_(prev_weights[2][i].clone())
                    layer.bias.copy_(prev_weights[3][i].clone())
                # recreates output layers (previous heads) and adds new head layer to the list of heads (output_layers).
                self.output_layers.append(layer)

        output_layer = nn.Linear(prev_size, output_size)
        with torch.no_grad():
            output_layer.weight.copy_(torch.randn(output_size, prev_size) * 0.1)
            output_layer.bias.copy_(torch.randn(output_size) * 0.1)
        self.output_layers.append(output_layer)

        self.criterion = nn.MSELoss()

    def forward(self, x, task_idx):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layers[task_idx](x)
        return x

    def _logpred(self, x, y, task_idx):
        pred = self.forward(x, task_idx)
        return self.criterion(pred, y)  # check shape?

    def prediction_prob(self, x_test, task_idx):
        with torch.no_grad():
            return self.forward(x_test, task_idx).cpu().numpy()

    def cost(self, x, y, task_idx):
        c = self._logpred(x, y, task_idx)
        return c

    def get_weights(self):  # gets the distributions parameterized by weights of layers.
        w_h = [layer.weight.detach().clone().t() for layer in self.hidden_layers]
        b_h = [layer.bias.detach().clone() for layer in self.hidden_layers]

        w_o = [layer.weight.detach().clone().t() for layer in self.output_layers]
        b_o = [layer.bias.detach().clone() for layer in self.output_layers]

        return [w_h, b_h, w_o, b_o]


class MFVI_NN(Base):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        training_size,
        no_train_samples=10,
        no_pred_samples=100,
        prev_means=None,
        prev_log_vars=None,
        prior_mean=0.0,
        prior_var=0.0,
    ):
        super(MFVI_NN, self).__init__(
            input_size, hidden_size, output_size, training_size
        )
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.size = [input_size] + hidden_size + [output_size]
        self.params_hW_m, self.params_hW_v = nn.ParameterList(), nn.ParameterList()
        self.params_oW_m, self.params_oW_v = nn.ParameterList(), nn.ParameterList()

        self.params_hb_m, self.params_hb_v = nn.ParameterList(), nn.ParameterList()
        self.params_ob_m, self.params_ob_v = nn.ParameterList(), nn.ParameterList()

        self.init_distr(input_size, hidden_size, output_size, prev_means, prev_log_vars)
        self.init_prior(
            input_size,
            hidden_size,
            output_size,
            prev_means,
            prev_log_vars,
            prior_mean,
            prior_var,
        )
        self.kl_div = torch.distributions.kl.kl_divergence
        self.to(device)

    def init_distr(
        self, input_size, hidden_size, output_size, prev_means, prev_log_vars
    ):
        # Initialize distributions
        # If this is the network initialised just after vanilla
        if prev_means is not None and prev_log_vars is None:
            prev_size = input_size
            # W_v, b_v stores log variances, NOT just variances
            for i, h_size in enumerate(hidden_size):
                W_m = nn.Parameter(prev_means[0][i])
                b_m = nn.Parameter(prev_means[1][i])
                W_v = nn.Parameter(torch.full((prev_size, h_size), -6.0, device=device))
                b_v = nn.Parameter(0.5 * torch.full((h_size,), -6.0, device=device))

                self.params_hW_m.append(W_m)
                self.params_hb_m.append(b_m)
                self.params_hW_v.append(W_v)
                self.params_hb_v.append(b_v)

                prev_size = h_size

            W_last_m = nn.Parameter(prev_means[2][0])
            b_last_m = nn.Parameter(prev_means[3][0])

        # if this follows another bayesian (prev_l)
        # prev_means should be [hidden_W_d, output_W_d]
        # prev_log_vars should be [hidden_b_d, output_b_d]
        else:
            hW_params = [self.parameterize(d) for d in prev_means[0]]
            self.params_hW_m, self.params_hW_v = map(nn.ParameterList, zip(*hW_params))
            oW_params = [self.parameterize(d) for d in prev_means[1]]
            self.params_oW_m, self.params_oW_v = map(nn.ParameterList, zip(*oW_params))
            hb_params = [self.parameterize(d) for d in prev_log_vars[0]]
            self.params_hb_m, self.params_hb_v = map(nn.ParameterList, zip(*hb_params))
            ob_params = [self.parameterize(d) for d in prev_log_vars[1]]
            self.params_ob_m, self.params_ob_v = map(nn.ParameterList, zip(*ob_params))

            W_last_m = nn.Parameter(
                torch.randn(hidden_size[-1], output_size, device=device) * 0.1
            )
            b_last_m = nn.Parameter(torch.randn(output_size, device=device) * 0.1)

        W_last_v = nn.Parameter(
            torch.full((hidden_size[-1], output_size), -6.0, device=device)
        )
        b_last_v = nn.Parameter(0.5 * torch.full((output_size,), -6.0, device=device))

        self.params_oW_m.append(W_last_m)
        self.params_oW_v.append(W_last_v)
        self.params_ob_m.append(b_last_m)
        self.params_ob_v.append(b_last_v)

    def init_prior(
        self,
        input_size,
        hidden_size,
        output_size,
        prev_means,
        prev_log_vars,
        prior_mean,
        prior_var,
    ):
        # Prior parameters
        self.ph_W_d, self.po_W_d = [], []
        self.ph_b_d, self.po_b_d = [], []
        if prev_means is not None and prev_log_vars is None:
            prev_size = input_size
            for i, h_size in enumerate(hidden_size):
                W_m = torch.full((prev_size, h_size), prior_mean, device=device)
                b_m = torch.full((h_size,), prior_mean, device=device)
                W_v = torch.full((prev_size, h_size), prior_var, device=device)
                b_v = torch.full((h_size,), prior_var, device=device)

                W_d = Normal(loc=W_m, scale=torch.exp(0.5 * W_v))
                b_d = Normal(loc=b_m, scale=torch.exp(0.5 * b_v))
                self.ph_W_d.append(W_d)
                self.ph_b_d.append(b_d)

                prev_size = h_size

        else:
            self.ph_W_d = prev_means[0]  # need .copy()?
            self.po_W_d = prev_means[1]
            self.ph_b_d = prev_log_vars[0]
            self.po_b_d = prev_log_vars[1]

        W_last_m = torch.full((hidden_size[-1], output_size), prior_mean, device=device)
        b_last_m = torch.full((output_size,), prior_mean, device=device)
        W_last_v = torch.full((hidden_size[-1], output_size), prior_var, device=device)
        b_last_v = torch.full((output_size,), prior_var, device=device)

        W_last_d = Normal(loc=W_last_m, scale=torch.exp(0.5 * W_last_v))
        b_last_d = Normal(loc=b_last_m, scale=torch.exp(0.5 * b_last_v))
        self.po_W_d.append(W_last_d)
        self.po_b_d.append(b_last_d)

    def cost(self, x, y, task_idx):
        kl = self._KL()
        lp = self._logpred(x, y, task_idx)
        return kl / self.training_size - lp

    def _logpred(self, x, y, task_idx):
        pred = self.forward(x, task_idx, self.no_train_samples)
        pred_mean = pred.mean(dim=0)
        loss = F.mse_loss(pred_mean, y)
        # print(loss)
        return -loss

    def prediction_prob(self, x_test, task_idx):
        with torch.no_grad():
            pred = self(
                torch.from_numpy(x_test).to(device), task_idx, self.no_pred_samples
            )
        return pred.cpu().detach().numpy()

    def get_weights(self):
        hidden_W_d, hidden_b_d, output_W_d, output_b_d = self.get_distr(
            self.params_hW_m,
            self.params_hW_v,
            self.params_hb_m,
            self.params_hb_v,
            self.params_oW_m,
            self.params_oW_v,
            self.params_ob_m,
            self.params_ob_v,
            deparam=True,
        )
        return ([hidden_W_d, output_W_d], [hidden_b_d, output_b_d])

    def _normal(self, m, logv):
        return Normal(loc=m, scale=torch.exp(0.5 * logv))

    def forward(self, inputs, task_idx, num_samples):
        K = num_samples
        # (batch_size, input_dim) -> (K, batch_size, input_dim)
        x = inputs.unsqueeze(0).repeat(K, 1, 1)

        for i, (W_m, W_v, b_m, b_v) in enumerate(
            zip(self.params_hW_m, self.params_hW_v, self.params_hb_m, self.params_hb_v)
        ):
            W_d = self._normal(W_m, W_v)
            b_d = self._normal(b_m, b_v)
            W = W_d.rsample((K,))  # shape [K, din, dout]
            b = b_d.rsample((K, 1))  # shape [K, 1, dout]
            x = torch.einsum("mni,mio->mno", x, W) + b  # [K, N, dout]
            x = F.relu(x)

        W_task_m, W_task_v = self.params_oW_m[task_idx], self.params_oW_v[task_idx]
        b_task_m, b_task_v = self.params_ob_m[task_idx], self.params_ob_v[task_idx]
        W_task_d = self._normal(W_task_m, W_task_v)
        b_task_d = self._normal(b_task_m, b_task_v)
        W_task = W_task_d.rsample((K,))
        b_task = b_task_d.rsample((K, 1))
        x = torch.einsum("mni,mio->mno", x, W_task) + b_task  # [K, N, 10] for mnist

        return x

    def get_distr(
        self,
        hW_m_list,
        hW_v_list,
        hb_m_list,
        hb_v_list,
        oW_m_list,
        oW_v_list,
        ob_m_list,
        ob_v_list,
        deparam=False,
    ):
        hidden_W_d = [self._normal(W_m, W_v) for W_m, W_v in zip(hW_m_list, hW_v_list)]
        hidden_b_d = [self._normal(b_m, b_v) for b_m, b_v in zip(hb_m_list, hb_v_list)]
        output_W_d = [self._normal(W_m, W_v) for W_m, W_v in zip(oW_m_list, oW_v_list)]
        output_b_d = [self._normal(b_m, b_v) for b_m, b_v in zip(ob_m_list, ob_v_list)]
        if deparam:
            hidden_W_d = [self.deparameterize(d) for d in hidden_W_d]
            hidden_b_d = [self.deparameterize(d) for d in hidden_b_d]
            output_W_d = [self.deparameterize(d) for d in output_W_d]
            output_b_d = [self.deparameterize(d) for d in output_b_d]

        return hidden_W_d, hidden_b_d, output_W_d, output_b_d

    def _KL(self):
        kl_tot = 0
        hidden_W_d, hidden_b_d, output_W_d, output_b_d = self.get_distr(
            self.params_hW_m,
            self.params_hW_v,
            self.params_hb_m,
            self.params_hb_v,
            self.params_oW_m,
            self.params_oW_v,
            self.params_ob_m,
            self.params_ob_v,
        )

        posteriors = hidden_W_d + hidden_b_d + output_W_d + output_b_d
        priors = self.ph_W_d + self.ph_b_d + self.po_W_d + self.po_b_d
        for post, prior in zip(posteriors, priors):
            kl_tot += torch.sum(self.kl_div(post, prior))

        return kl_tot

    def deparameterize(self, dist: torch.distributions.Distribution):
        params = {
            name: getattr(dist, name).detach().clone() for name in dist.arg_constraints
        }
        return type(dist)(**params)

    def parameterize(self, dist: torch.distributions.Normal):

        loc = dist.loc.detach().clone()
        scale = dist.scale.detach().clone()
        log_var = torch.log(scale.pow(2))
        loc_param = nn.Parameter(loc)
        log_var_param = nn.Parameter(log_var)
        return (loc_param, log_var_param)
