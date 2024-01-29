# System imports.
import os

# Local imports.
from .environment import Config

class config():
    # env config
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = True
    high             = 1000000.

    # output config
    output_path  = os.path.join("results", "dqn_linear_results")
    model_output = os.path.join(output_path, "model.weights")
    log_path     = os.path.join(output_path, "log.txt")
    plot_output  = os.path.join(output_path, "scores.png")
    record_path  = os.path.join(output_path, "monitor")

    # model and training config
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 1000
    log_freq          = 50
    eval_freq         = 1000
    record_freq       = 1000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 50000 #5000000
    batch_size         = 32
    buffer_size        = 10000 #1000000
    target_update_freq = 10   #10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr_begin           = 0.25 # 0.00025
    lr_end             = 0.05 # 0.00005
    lr_nsteps          = nsteps_train/2/10
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 10000 #1000000
    learning_start     = 5*Config.MAX_STEPS   #50000
