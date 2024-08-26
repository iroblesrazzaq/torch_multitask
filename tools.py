import os
import json
import pickle
import numpy as np
import torch

def mkdir_p(path):
    """Create a directory if it doesn't exist"""
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def save_hp(hp, model_dir):
    """Save hyperparameters"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f, indent=2)

def load_hp(model_dir):
    """Load hyperparameters"""
    with open(os.path.join(model_dir, 'hp.json'), 'r') as f:
        hp = json.load(f)
    
    # Reconstruct non-serializable objects
    if 'rng' in hp:
        rng = np.random.RandomState()
        rng.set_state(tuple(hp['rng']))
        hp['rng'] = rng
    
    return hp

def save_log(log):
    """Save training log"""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, 'log.json')
    with open(fname, 'w') as f:
        json.dump(log, f)

def load_log(model_dir):
    """Load training log"""
    fname = os.path.join(model_dir, 'log.json')
    if not os.path.isfile(fname):
        return None
    with open(fname, 'r') as f:
        log = json.load(f)
    return log

def get_perf(y_hat, y_loc):
    """Get performance"""
    y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
    
    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = np.argmax(y_hat[..., 1:], axis=-1)

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return np.mean(perf)

def get_default_hp(ruleset):
    '''Get a default hp.
    Useful for debugging.
    Returns:
        hp : a dictionary containing training hyperparameters
    '''
    num_ring = 2  # This might need to be adjusted based on your specific implementation
    n_rule = 20  # Assuming all 20 tasks

    n_eachring = 32
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    hp = {
         # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 512,
            # input type: normal, multi
            'in_type': 'normal',
            # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
            'rnn_type': 'LeakyRNN',
            # whether rule and stimulus inputs are represented separately
            'use_separate_input': False,
            # Type of loss functions
            'loss_type': 'lsq',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation runctions, relu, softplus, tanh, elu
            'activation': 'relu',
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 20,
            # discretization time step/time constant
            'alpha': 0.2,
            # recurrent noise
            'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
            'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 0,
            # l2 regularization on activity
            'l2_h': 0,
            # l2 regularization on weight
            'l1_weight': 0,
            # l2 regularization on weight
            'l2_weight': 0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0,
            # proportion of weights to train, None or float between (0, 1)
            'p_weight_train': None,
            # Stopping performance
            'target_perf': 1.,
            # number of units each ring
            'n_eachring': n_eachring,
            # number of rings
            'num_ring': num_ring,
            # number of rules
            'n_rule': n_rule,
            # first input index for rule units
            'rule_start': 1+num_ring*n_eachring,
            # number of input units
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            # number of recurrent units
            'n_rnn': 256,
            # number of input units
            'ruleset': ruleset,
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.001,
            # intelligent synapses parameters, tuple (c, ksi)
            'c_intsyn': 0,
            'ksi_intsyn': 0,
            
    }

    return hp

def gen_feed_dict(model, trial, hp):
    """Generate feed_dict for session run"""
    feed_dict = {
        'x': torch.tensor(trial.x).float(),
        'y': torch.tensor(trial.y).float(),
        'c_mask': torch.tensor(trial.c_mask).float()
    }
    return feed_dict

def gen_ortho_matrix(dim, rng=None):
    """Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    """
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim-n+1,))
        else:
            x = rng.normal(size=(dim-n+1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = -D*(np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    return H