# System imports.



class Config:
    # "alpha" or "f(s)"
    CASE_STUDY = "alpha"

    # "r2_coef" or "mse"
    REWARD_TYPE = "mse" 

    IS_TENSORBOARD = True
    TRAIN = True

    TRAIN_TIME_STEPS = 2000 # Inutile
    TEST_TIME_STEPS  = 100 # Inutile
    MAX_STEPS        = 1000
    MAX_INITIALIZE   = 1000000

    # Critères d'arrêt
    MAX_R2  = 0.95
    MIN_MSE = 0.5

    N_ACTIONS = 9 * 3
    OBS_SHAPE = (9, 9)

    DIM_DENSE_1 = 4*N_ACTIONS
    DIM_DENSE_2 = 4*N_ACTIONS
