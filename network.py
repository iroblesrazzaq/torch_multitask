import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import tools


def torch_popvec(y):
    """Population vector read-out in PyTorch."""
    num_units = y.size(-1)
    pref = torch.arange(0, 2*np.pi, 2*np.pi/num_units, dtype=y.dtype, device=y.device)
    cos_pref = torch.cos(pref)
    sin_pref = torch.sin(pref)
    temp_sum = torch.sum(y, dim=-1)
    temp_cos = torch.sum(y * cos_pref, dim=-1) / temp_sum
    temp_sin = torch.sum(y * sin_pref, dim=-1) / temp_sum
    loc = torch.atan2(temp_sin, temp_cos)
    return torch.fmod(loc, 2*np.pi)


class LeakyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, sigma_rec=0, activation='softplus', w_rec_init='diag', rng=None):
        super(LeakyRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.sigma = np.sqrt(2 / alpha) * sigma_rec
        self.activation = activation
        self.w_rec_init = w_rec_init

        if activation == 'softplus':
            self.act_fn = torch.nn.functional.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'tanh':
            self.act_fn = torch.tanh
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self.act_fn = torch.nn.functional.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'power':
            self.act_fn = lambda x: torch.pow(torch.nn.functional.relu(x), 2)
            self._w_in_start = 1.0
            self._w_rec_start = 0.01
        elif activation == 'retanh':
            self.act_fn = lambda x: torch.tanh(torch.nn.functional.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        w_in0 = self.rng.randn(input_size, hidden_size) / np.sqrt(input_size) * self._w_in_start

        if w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(hidden_size)
        elif w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(hidden_size, rng=self.rng)
        elif w_rec_init == 'randgauss':
            w_rec0 = self._w_rec_start * self.rng.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)

        matrix0 = np.concatenate((w_in0, w_rec0), axis=0)

        self.weight = nn.Parameter(torch.Tensor(matrix0))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    @property
    def state_size(self):
        return self.hidden_size

    @property
    def output_size(self):
        return self.hidden_size

    def forward(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        if state is None:
            state = torch.zeros(inputs.shape[0], self.hidden_size, device=inputs.device)
            #print(f"Initialized state with state.shape: {state.shape}")
        else:
            #print(f"Stack: RNN forward \nstate.shape: {state.shape}")

            state = state.squeeze(0)

        #print(f"pre batch inputs.shape: {inputs.shape}\nstate.shape: {state.shape}")

        inputs = inputs.squeeze(0)

        if inputs.ndim == 2:
            # Transpose the weight matrix during the computation
            #print("Stack: RNN forward, Using batch processing")
            #print(f"Stack: RNN forward, pre batch inputs.shape: {inputs.shape}\nstate.shape: {state.shape}")
            mult_out = torch.matmul(torch.cat((inputs, state), dim=1), self.weight)
            #print("Stack: RNN forward, completed mult")
        else:    
            #print("Stack: RNN forward, Using single processing")
            #print(f"Stack: RNN forward, inputs.shape: {inputs.shape}\nstate.shape: {state.shape}")
            # Transpose the weight matrix during the computation
            mult_out = torch.matmul(torch.cat((inputs, state), dim=0), self.weight.T)
            #print("Stack: RNN forward, completed mult")
        gate_inputs = mult_out + self.bias

        noise = torch.randn_like(state) * self.sigma
        gate_inputs = gate_inputs + noise

        output = self.act_fn(gate_inputs)

        output = (1 - self.alpha) * state + self.alpha * output
        return output, output


class Model(nn.Module):
    def __init__(self, model_dir, hp=None):
        super(Model, self).__init__()
        
        if hp is None:
            hp = tools.load_hp(model_dir)
            if hp is None:
                raise ValueError('No hp found for model_dir {:s}'.format(model_dir))
        
        self.model_dir = model_dir
        self.hp = hp
        
        self._build()
        
    def _build(self):
        hp = self.hp
        
        # Input
        if hp['use_separate_input']:
            self.sensory_rnn_input = nn.Linear(hp['rule_start'], hp['n_rnn'])
            if 'mix_rule' in hp and hp['mix_rule']:
                self.mix_rule = nn.Linear(hp['n_rule'], hp['n_rule'], bias=False)
                self.mix_rule.weight.requires_grad = False
                nn.init.orthogonal_(self.mix_rule.weight)
            self.rule_rnn_input = nn.Linear(hp['n_rule'], hp['n_rnn'], bias=False)
        
        # Recurrent
        if hp['rnn_type'] == 'LeakyRNN':
            self.rnn = LeakyRNNCell(hp['n_input'], hp['n_rnn'], hp['alpha'], 
                                    sigma_rec=hp['sigma_rec'], 
                                    activation=hp['activation'],
                                    w_rec_init=hp['w_rec_init'])
        # no support for leakyGRU yet
        #elif hp['rnn_type'] == 'LeakyGRU':
            #self.rnn = LeakyGRUCell(hp['n_input'], hp['n_rnn'], hp['alpha'],
                                    #sigma_rec=hp['sigma_rec'])
        else:
            raise ValueError('Unknown RNN type')
        
        # Output
        self.output = nn.Linear(hp['n_rnn'], hp['n_output'])
        
    def forward(self, x, y=None, c_mask=None):
        hp = self.hp
        n_batch, n_time, _ = x.shape
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']
        
        if hp['use_separate_input']:
            sensory_inputs, rule_inputs = torch.split(x, [hp['rule_start'], hp['n_rule']], dim=-1)
            sensory_rnn_inputs = self.sensory_rnn_input(sensory_inputs)
            if 'mix_rule' in hp and hp['mix_rule']:
                rule_inputs = self.mix_rule(rule_inputs)
            rule_rnn_inputs = self.rule_rnn_input(rule_inputs)
            rnn_inputs = sensory_rnn_inputs + rule_rnn_inputs
        else:
            rnn_inputs = x
        
        h = torch.zeros(n_time + 1, n_batch, n_rnn).to(x.device)
        
        for t in range(n_time):
            h[t+1], _ = self.rnn(rnn_inputs[:, t, :], h[t])
        
        h_shaped = h[1:].reshape(-1, n_rnn)
        y_hat_ = self.output(h_shaped)
        
        if hp['loss_type'] == 'lsq':
            # Least-square loss
            y_hat = torch.sigmoid(y_hat_)
            if y is not None and c_mask is not None:
                y_shaped = y.reshape(-1, n_output)
                self.cost_lsq = torch.mean(
                    torch.pow((y_shaped - y_hat) * c_mask, 2))
        else:
            y_hat = torch.softmax(y_hat_, dim=-1)
            if y is not None and c_mask is not None:
                y_shaped = y.reshape(-1, n_output)
                self.cost_lsq = torch.mean(
                    -c_mask * torch.sum(y_shaped * torch.log(y_hat), dim=-1))
        
        self.y_hat = y_hat.view(n_time, n_batch, n_output)
        
        y_hat_fix, y_hat_ring = torch.split(self.y_hat, [1, n_output - 1], dim=-1)
        self.y_hat_loc = torch_popvec(y_hat_ring)
        
        return self.y_hat, self.cost_lsq
    
    def set_optimizer(self, extra_cost=None):
        hp = self.hp
        
        self.cost_reg = torch.tensor(0.).to(next(self.parameters()).device)
        if hp['l1_h'] > 0:
            self.cost_reg += hp['l1_h'] * torch.mean(torch.abs(self.h))
        if hp['l2_h'] > 0:
            self.cost_reg += hp['l2_h'] * torch.sum(torch.pow(self.h, 2))
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                if hp['l1_weight'] > 0:
                    self.cost_reg += hp['l1_weight'] * torch.mean(torch.abs(param))
                if hp['l2_weight'] > 0:
                    self.cost_reg += hp['l2_weight'] * torch.sum(torch.pow(param, 2))
        
        cost = self.cost_lsq + self.cost_reg
        if extra_cost is not None:
            cost += extra_cost
        
        if hp['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=hp['learning_rate'])
        else:
            raise ValueError('Unknown optimizer')
        
    def save(self):
        """Save the model."""
        save_path = os.path.join(self.model_dir, 'model.pt')
        torch.save(self.state_dict(), save_path)
        print("Model saved in file: %s" % save_path)
    
        
    def restore(self):
        self.load_state_dict(torch.load(os.path.join(self.model_dir, 'model.pt')))