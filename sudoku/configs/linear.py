# System imports.
import os



class config():
    # env config
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = True
    high             = 9 # 255.

    # output config
    output_path = os.path.join("results", "729-actions")
    model_output = os.path.join(output_path, "model.weights")
    log_path = os.path.join(output_path, "log.txt")
    plot_output = os.path.join(output_path, "scores.png")
    record_path = os.path.join(output_path, "records")

    # model and training config
    num_episodes_test = 20
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 500000
    log_freq          = 50
    eval_freq         = 500000
    record_freq       = 500000
    soft_epsilon      = 0.05

    # hyper params
    nsteps_train       = 5000000  # 500000
    batch_size         = 128
    buffer_size        = 100000  # 100000
    target_update_freq = 500
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    lr_begin           = 0.1  # 0.00025
    lr_end             = 0.005  # 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.01
    eps_nsteps         = 1000000  # 100000
    learning_start     = 50000  # 50000
