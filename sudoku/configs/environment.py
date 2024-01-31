# System imports.



class Config:
    MAX_STEPS = 100

    N_ACTIONS = 9 * 9 * 9
    OBS_SHAPE = (9, 9, 1)

    DIM_DENSE_1 = N_ACTIONS // 9
    DIM_DENSE_2 = N_ACTIONS // 9
