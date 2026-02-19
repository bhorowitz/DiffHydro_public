
class NoForcing:
    def timestep(self, x):
        return 1E10
    def force(self,i,sol,params,dt):
        return sol, params #no froce
