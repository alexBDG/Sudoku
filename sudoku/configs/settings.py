# System imports.
import os



# data path
sudoku_path = os.path.join(os.path.dirname(__file__), "data", "sudoku-3m.csv")
min_difficulty = 0.

# env config
render_train     = False
render_test      = False
overwrite_render = True
record           = True
high             = 1.

# output config
output_path = "results"
model_output = "model.weights"
log_path = "log.txt"
summarize_output = "summarize.png"
record_path = "records"
buffer_path = "data"

# model and training config
num_episodes_test = 50
grad_clip         = True
clip_val          = 10
saving_freq       = 500000
log_freq          = 50
eval_freq         = 500000
record_freq       = 500000
soft_epsilon      = 0.05

# hyper params
nsteps_train       = 5000000
batch_size         = 128
buffer_size        = 500000
target_update_freq = 10000
gamma              = 0.99
learning_freq      = 4
state_history      = 4
skip_frame         = 4
lr_begin           = 0.00025
lr_end             = 0.00005
lr_nsteps          = nsteps_train/2
eps_begin          = 1
eps_end            = 0.1
eps_nsteps         = 2000000
learning_start     = 50000


# Environment settings
MAX_STEPS = 10000


try:
    from .local_settings import *
except:
    # No local settings - use previous ones
    pass