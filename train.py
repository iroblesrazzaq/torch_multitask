import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim

from network import Model
from task import generate_trials, rules_dict
import tools

def train(model_dir,
          hp=None,
          max_steps=1e7,
          display_step=500,
          ruleset='all',
          rule_trains=None,
          rule_prob_map=None,
          seed=0):

    tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = tools.get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # Rules to train and test
    if rule_trains is None:
        hp['rule_trains'] = rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign rule probabilities
    if rule_prob_map is None:
        rule_prob_map = dict()
    hp['rule_probs'] = None
    if isinstance(hp['rule_trains'], (list, tuple)):
        rule_prob = np.array([rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))
    
    tools.save_hp(hp, model_dir)

    # Build the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model_dir, hp=hp).to(device)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])

    step = 0
    while step * hp['batch_size_train'] <= max_steps:
        try:
            # Validation
            if step % display_step == 0:
                log['trials'].append(step * hp['batch_size_train'])
                log['times'].append(time.time() - t_start)
                log = do_eval(model, log, hp['rule_trains'], hp, device)
                if log['perf_avg'][-1] > hp['target_perf']:
                    print('Perf reached the target: {:0.2f}'.format(hp['target_perf']))
                    break

            # Training
            rule_train_now = hp['rng'].choice(hp['rule_trains'], p=hp['rule_probs'])
            trial = generate_trials(rule_train_now, hp, 'random', batch_size=hp['batch_size_train'])

            feed_dict = tools.gen_feed_dict(model, trial, hp)
            x = feed_dict['x'].to(device)
            y = feed_dict['y'].to(device)
            c_mask = feed_dict['c_mask'].to(device)
            
            optimizer.zero_grad()
            y_hat, loss = model(x, y, c_mask)
            loss.backward()
            optimizer.step()

            step += 1

        except KeyboardInterrupt:
            print("Optimization interrupted by user")
            break

    print("Optimization finished!")

def do_eval(model, log, rule_trains, hp, device):
    if not isinstance(rule_trains, (list, tuple)):
        rule_name_print = rule_trains
    else:
        rule_name_print = ' & '.join(rule_trains)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training ' + rule_name_print)

    for rule_test in hp['rules']:
        n_rep = 16
        batch_size_test_rep = int(hp['batch_size_test'] / n_rep)
        clsq_tmp = list()
        perf_tmp = list()
        
        for i_rep in range(n_rep):
            trial = generate_trials(rule_test, hp, 'random', batch_size=batch_size_test_rep)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            x = feed_dict['x'].to(device)
            y = feed_dict['y'].to(device)
            c_mask = feed_dict['c_mask'].to(device)
            
            with torch.no_grad():
                y_hat, loss = model(x, y, c_mask)
            
            clsq_tmp.append(loss.item())
            perf_tmp.append(tools.get_perf(y_hat.cpu().numpy(), trial.y_loc))

        log['cost_'+rule_test].append(np.mean(clsq_tmp, dtype=np.float64))
        log['perf_'+rule_test].append(np.mean(perf_tmp, dtype=np.float64))
        print('{:15s}'.format(rule_test) +
              '| cost {:0.6f}'.format(np.mean(clsq_tmp)) +
              '  | perf {:0.2f}'.format(np.mean(perf_tmp)))

    if isinstance(rule_trains, (list, tuple)):
        rule_tmp = rule_trains
    else:
        rule_tmp = [rule_trains]
    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)

    # Saving the model
    model.save(model.model_dir)
    tools.save_log(log)

    return log

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modeldir', type=str, default='data/debug')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {'activation': 'softplus',
          'n_rnn': 256,
          'learning_rate': 0.001,
          'l2_h': 0.,
          'use_separate_input': False}
    train(args.modeldir,
          seed=0,
          hp=hp,
          ruleset='all',
          display_step=500)