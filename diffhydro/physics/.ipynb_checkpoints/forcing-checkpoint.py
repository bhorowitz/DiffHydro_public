
class NoForcing:
    def timestep(x):
        return 1E10
    def force(i,sol,params):
        return sol #no froce