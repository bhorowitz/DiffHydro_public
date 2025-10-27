import numpy as np
import jax.numpy as jnp
from functools import partial
import jax

#from Ben's DiffAPM implementation, need to clean up and probably put somewhere else...

@jax.jit
def shift(arr, dx, dy, dz):
        return jnp.roll(jnp.roll(jnp.roll(arr, dx, axis=0), dy, axis=1), dz, axis=2)

@jax.jit
def adjacency_sum(U):
    adj = jnp.roll(U,1,axis=0) + jnp.roll(U,-1,axis=0) + jnp.roll(U,1,axis=1) + jnp.roll(U,-1,axis=1) + jnp.roll(U,1,axis=2) + jnp.roll(U,-1,axis=2)
    return adj

@jax.jit
def apply_poisson(U, h=None):
    """Apply the 3D poisson operator to U."""
    alpha = len(U.shape)
    x = jnp.empty_like(U)

    if h is None:
        h = 1 / U.shape[0]

    if alpha == 3:
        x = (-6* U + 
        adjacency_sum(U)) / (h*h)
        
    else:
        raise ValueError('residual: invalid dimension')

    return x


#### RESTRICTION FUNCTIONS

def restriction(A):
    """
        applies simple restriction to A
        @param A n x n matrix
        @return (n//2, n//2) matrix
    """
    # indicator for Dimension
    alpha = len(A.shape)
    # initialize result with respect to the wanted shape
    # Index of the second to the last element to mention in ret (depends on
    # the shape of A)

    # Case: Dimension 1
    if alpha == 3:
        ret = restriction_3D(A)
    # Case: Error
    else:
        raise ValueError('restriction: invalid dimension')

    return ret

#@jit(nopython=True, fastmath=True)
@jax.jit
def restriction_3D(A):
    # get every second element in A
    ret = A[::2, ::2, ::2]
    return ret


def weighted_restriction(A):
    # indicator for Dimension
    alpha = len(A.shape)
    # initialize result with respect to the wanted shape
    ret = restriction(A)

    # min length is 3
    assert(A.shape[0] >= 3)

    if alpha == 3:
        ret = weighted_restriction_3D(A, ret)
    else:
        raise ValueError('weighted restriction: invalid dimension')
    return ret

#@(nopython=True, fastmath=True)

@partial(jax.jit)
def weighted_restriction_3D(A, ret):
    # Base weight: center point
    weighted = 8 * A

    # Face neighbors (6)
    for ax in range(3):
        weighted += 4 * jnp.roll(A, shift=1, axis=ax)
        weighted += 4 * jnp.roll(A, shift=-1, axis=ax)

    # Edge neighbors (12)
    for ax1 in range(3):
        for ax2 in range(ax1 + 1, 3):
            for shift1 in [-1, 1]:
                for shift2 in [-1, 1]:
                    weighted += 2 * jnp.roll(
                        jnp.roll(A, shift1, axis=ax1), shift2, axis=ax2
                    )

    # Corner neighbors (8)
    for shift in [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                  (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]:
        weighted += jnp.roll(jnp.roll(jnp.roll(A, shift[0], 0), shift[1], 1), shift[2], 2)

    # Average and sample every 2nd point
    restricted = weighted[::2, ::2, ::2] / 64.0

    # Store into interior of ret
    ret = restricted
    return ret

### prolongation


@partial(jax.jit, static_argnames=['fine_shape'])
def prolongation(e, fine_shape):
    """
    This interpolates/ prolongates to a grid of fine_shape
    @param e
    @param fine_shape targeted shape
    @return grid with fine_shape
    """
    # indicator for Dimension
    alpha = len(e.shape)
    # initialize result with respect to the wanted shape
    w = jnp.zeros(fine_shape)

    if alpha == 3:
        w = prolongation_3D(w, e)

    # Case: Error
    else:
        raise ValueError("prolongation: invalid dimension")
    return w

@jax.jit
def prolongation_3D(w, e):
    # Assign coarse grid points directly
    w = w.at[::2, ::2, ::2].set(e)

    # Neighbors in y
    w = w.at[::2, 1::2, ::2].set((e + jnp.roll(e, -1, axis=1)) / 2)
    # Neighbors in z
    w = w.at[::2, ::2, 1::2].set((e + jnp.roll(e, -1, axis=2)) / 2)
    # Neighbors in x
    w = w.at[1::2, ::2, ::2].set((e + jnp.roll(e, -1, axis=0)) / 2)

    # Face centers: xy, xz, yz
    w = w.at[::2, 1::2, 1::2].set((e +
                                  jnp.roll(e, -1, axis=1) +
                                  jnp.roll(e, -1, axis=2) +
                                  jnp.roll(jnp.roll(e, -1, axis=1), -1, axis=2)) / 4)

    w = w.at[1::2, ::2, 1::2].set((e +
                                  jnp.roll(e, -1, axis=0) +
                                  jnp.roll(e, -1, axis=2) +
                                  jnp.roll(jnp.roll(e, -1, axis=0), -1, axis=2)) / 4)

    w = w.at[1::2, 1::2, ::2].set((e +
                                  jnp.roll(e, -1, axis=0) +
                                  jnp.roll(e, -1, axis=1) +
                                  jnp.roll(jnp.roll(e, -1, axis=0), -1, axis=1)) / 4)

    # Center of the coarse cell (xyz)
    w = w.at[1::2, 1::2, 1::2].set((
        e +
        jnp.roll(e, -1, axis=0) +
        jnp.roll(e, -1, axis=1) +
        jnp.roll(e, -1, axis=2) +
        jnp.roll(jnp.roll(e, -1, axis=0), -1, axis=1) +
        jnp.roll(jnp.roll(e, -1, axis=0), -1, axis=2) +
        jnp.roll(jnp.roll(e, -1, axis=1), -1, axis=2) +
        jnp.roll(jnp.roll(jnp.roll(e, -1, axis=0), -1, axis=1), -1, axis=2)
    ) / 8)

    return w

#### multigrid cycles

from abc import abstractmethod

class AbstractCycle:
    def __init__(self, F, v1, v2, mu, l, eps=1e-8, h=None,laplace= apply_poisson):
        self.v1 = v1
        self.v2 = v2
        self.mu = mu
        self.F = F
        self.l = l
        self.eps = eps
        if h is None:
            self.h = 1 / F.shape[0]
        else:
            self.h = h
        if (self.l == 0):
            self.l = int(np.log2(self.F.shape[0])) - 1
        # ceck if l is plausible
        if np.log2(self.F.shape[0]) < self.l:
            raise ValueError('false value of levels')
        self.poisson = laplace
        
    def __call__(self, U):
        return self.do_cycle(self.F, U, self.l, self.h)

    @abstractmethod
    def _presmooth(self, F, U, h):
        pass

    @abstractmethod
    def _postsmooth(self, F, U, h):
        pass

    @abstractmethod
    def _compute_residual(self, F, U, h):
        pass

    @abstractmethod
    def _solve(self, F, U, h):
        pass

    @abstractmethod
    def norm(self, U):
        pass

    @abstractmethod
    def restriction(self, r):
        pass

    def _residual(self, U):
        return self._compute_residual(self.F, U, self.h)

    def _compute_correction(self, r, l, h):
        e = jnp.zeros_like(r)
        for _ in range(self.mu):
            e = self.do_cycle(r, e, l, h)
        return e

    def do_cycle(self, F, U, l, h):
       # print(l)
        if l <= 1 or U.shape[0] <= 1:
            return self._solve(F, U, h)

        U = self._presmooth(F=F, U=U, h=h)

        r = self._compute_residual(F=F, U=U, h=h)

        r = self.restriction(r)
        
        e = self._compute_correction(r, l - 1, 2 * h)

        e = prolongation(e, U.shape)

        # correction
        U += e

        return self._postsmooth(F=F, U=U, h=h)


class PoissonCycle(AbstractCycle):
    def __init__(self, F, v1, v2, mu, l, eps=1e-8, h=None,laplace= apply_poisson):
        super().__init__(F, v1, v2, mu, l, eps, h,laplace)

    def _presmooth(self, F, U, h=None):
        return GS_RB(
            F,
            U=U,
            h=h,
            max_iter=self.v1,
            eps=self.eps,
            laplace=self.poisson)

    def _postsmooth(self, F, U, h=None):
        return GS_RB(
            F,
            U=U,
            h=h,
            max_iter=self.v2,
            eps=self.eps,
        laplace=self.poisson)

    def _compute_residual(self, F, U, h):
        return F - self.poisson(U, h)

    def _solve(self, F, U, h):
        return GS_RB(
            F=F,
            U=U,
            h=h,
            max_iter=1000,
            eps=self.eps,
            norm_iter=5,
        laplace=self.poisson)

    def norm(self, U):
        residual = self._residual(U)
        return jnp.linalg.norm(residual)

    def restriction(self, r):
        return weighted_restriction(r)

    

def poisson_multigrid(F, U, l, v1, v2, mu, iter_cycle, eps=1e-6, h=None, laplace = apply_poisson):
    """Implementation of MultiGrid iterations
       should solve AU = F
       A is poisson equation
       @param U n x n Matrix
       @param F n x n Matrix
       @param v1 Gauss Seidel iterations in pre smoothing
       @param v2 Gauss Seidel iterations in post smoothing
       @param mu iterations for recursive call
       @return x n vector
    """

    cycle = PoissonCycle(F, v1, v2, mu, l, eps, h,laplace)
    return multigrid(cycle, U, eps, iter_cycle)

#jax.lax.while_loop(cond_fun, body_fun, init_val)
#def body_func(U,cycle):



@partial(jax.jit, static_argnames=['cycle','iter_cycle','eps'])
def multigrid_optimizer(cycle, U, eps, iter_cycle):
    
    def cond(arg):
        step, U = arg
        norm = cycle.norm(U)
        return (step < iter_cycle) & (norm > eps)
    
    def body(arg):
        step, U = arg
        U = cycle(U)
        return (step + 1, U)

    return jax.lax.while_loop(
        cond,
        body,
        (0, U)
    )

def multigrid(cycle, U, eps, iter_cycle):

    # scale the epsilon with the number of gridpoints
    eps *= U.shape[0] * U.shape[0] * U.shape[0]
    _,U = multigrid_optimizer(cycle, U, eps, iter_cycle)
    #for i in range(1, iter_cycle + 1):
    #    U = cycle(U)
    #    norm = cycle.norm(U)
      #  print(f"Residual has a L2-Norm of {norm:.4} after {i} MGcycle")
       # if norm <= eps:
         #   print(
         #       f"converged after {i} cycles with {norm:.4} error")
       #     break
    return U


@partial(jax.jit, static_argnames=['h','iter_cycle','eps','laplace'])
def GS_RB_optimizer(F, U, h, eps, iter_cycle,laplace):
    
    def cond(arg):
        step, U = arg
        norm = jnp.linalg.norm(F - laplace(U, h))
        return (step < iter_cycle) & (norm > eps)
    
    def body(arg):
        step, U = arg
        #red
        U= sweep_3D(1, F, U, h*h)
        # black
        U = sweep_3D(0, F, U, h*h)
        return (step + 1, U)

    return jax.lax.while_loop(
        cond,
        body,
        (0, U)
    )



import jax
def GS_RB(
    F,
    U=None,
    h=None,
    max_iter=1000,
    eps=1e-8,
    norm_iter=1000,
    laplace = apply_poisson
):
    """
    red-black
    Solve AU = F, the poisson equation.

    @param F n vector
    @param h is distance between grid points | default is 1/N
    @return U n vector
    """
    if U is None:
        U = jnp.zeros_like(F)
    if h is None:
        h = 1 / (U.shape[0])

    h2 = h * h

    if len(F.shape) == 3:
        sweep = sweep_3D
    else:
        raise ValueError("Wrong Shape!!!")

    norm = 0.0  # declarate norm so we can output later
    it = 0
    #Gauss-Seidel-Iterationen
    _,U = GS_RB_optimizer(F, U, h, eps, max_iter, laplace)

    #print(f"converged after {it} iterations with {norm:.4} error")

    return U


@partial(jax.jit, static_argnames=['color','h2'])
def sweep_3D(color, F, U, h2):
    """
    Perform one red-black Gauss-Seidel sweep with periodic BCs.

    @param color: 0 (black) or 1 (red)
    @param F: Right-hand side
    @param U: Current solution
    @param h2: Grid spacing squared
    """
    # Grid shape
    m, n, o = F.shape

    # Create mask for red or black points on a checkerboard
    i = jnp.arange(m)[:, None, None]
    j = jnp.arange(n)[None, :, None]
    k = jnp.arange(o)[None, None, :]

    checkerboard = (i + j + k) % 2
    mask = (checkerboard == color)

    # Compute neighbor sums using jnp.roll for periodic access
    neighbor_sum = (
        jnp.roll(U, 1, axis=0) + jnp.roll(U, -1, axis=0) +
        jnp.roll(U, 1, axis=1) + jnp.roll(U, -1, axis=1) +
        jnp.roll(U, 1, axis=2) + jnp.roll(U, -1, axis=2)
    )

    # Update only the selected color cells
    U_new = jnp.where(
        mask,
        (neighbor_sum - F * h2) / 6.0,
        U
    )

    return U_new
    


def gauss_seidel(A, F, U=None, eps=1e-10, max_iter=1000):
    """Implementation of Gauss Seidl iterations
       should solve AU = F
       @param A n x m Matrix
       @param F n vector
       @return n vector
    """
    raise
    n, *_ = A.shape
    if U is None:
        U = jnp.zeros_like(F)

    for _ in range(max_iter):
        U_next = jnp.zeros_like(U)
        for i in range(n):
            left = jnp.dot(A[i, :i], U_next[:i])
            right = jnp.dot(A[i, i + 1:], U[i + 1:])
            U_next[i] = (F[i] - left - right) / (A[i, i])

        U = U_next
        if np.linalg.norm(F - A @ U) < eps:
            break

    return U
