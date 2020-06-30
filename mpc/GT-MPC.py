
from mpc.MPC import MPC

class GT_MPC(MPC):
    """
    MPC which uses ground-truth / simulated dynamics.
    Assumes open-loop control sequendes for efficiency.
    """


    def __init__(self, params):

        super().__init__(params)

        #TODO: spin-up environments

    def reset(self):
        """Resets this controller.
        """
        raise NotImplementedError("Must be implemented in subclass.")


