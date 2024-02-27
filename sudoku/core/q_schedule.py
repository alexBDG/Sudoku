# System imports.
import numpy as np



class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon        = eps_begin
        self.eps_begin      = eps_begin
        self.eps_end        = eps_end
        self.nsteps         = nsteps


    def update(self, t):
        """
        Updates epsilon

        Args:
            t: int
                frame number
        """
        
        if t < self.nsteps:
            coef = (self.eps_end - self.eps_begin)/self.nsteps
            self.epsilon = self.eps_begin + t*coef
        else:
            self.epsilon = self.eps_end


class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: float
                initial exploration rate
            eps_end: float
                final exploration rate
            nsteps: int
                number of steps taken to linearly decay eps_begin to eps_end
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action

        Args:
            best_action: int 
                best action according some policy
        Returns:
            an action
        """
        
        pred = np.random.choice(2, 1, p=[self.epsilon, 1-self.epsilon])
        if (pred[0] == 0):
            return self.env.action_space.sample()
        else:
            return best_action
