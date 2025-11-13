from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

def pressure_from_cons(U, gamma=None, eps=1e-30):
    if gamma is None:
        gamma = eq.gamma  # or pass eq in explicitly

    rho = U[0]
    mom = U[1:-1]                 # <-- only the momentum components
    E   = U[-1]

    rho_safe = jnp.maximum(rho, eps)
    v2  = (mom**2).sum(axis=0) / (rho_safe**2)
    KE  = 0.5 * rho_safe * v2
    Emag = 0.0                     # add magnetic energy here if/when needed

    Et = E - KE - Emag
    Et = jnp.maximum(Et, eps)      # numerical floor; avoids negative p from roundoff

    return (gamma - 1.0) * Et



class HeatCoolForce:
    """
    Improved radiative cooling with robust overcooling protection.
    Key changes:
    - Fixed interpolation formula
    - Consistent temperature calculation
    - Per-substep floor enforcement preventing negative pressures
    - Better coordinated fractional limiters
    """

    def __init__(
        self,
        equation_manager,
        pressure_fn,
        logT_table,
        logLambda_m20_table,
        *,
        ALPHA = 813.142554365,
        BETA  = 406571277.182611837,
        GAMMA = 406.571277183,
        MHKB  = 115.98518596699539,
        include_heating = True,
        ctime = 1.0,
        dtmax = 1.0,
        temp_floor = 1.0e4,
        subcycles = 10,
        early_exit = True,
        eps = 1e-30
    ):
        self.eq = equation_manager
        self.pressure_fn = pressure_fn
        self.logT = jnp.asarray(logT_table)
        self.logL = jnp.asarray(logLambda_m20_table)
        self.ALPHA = ALPHA
        self.BETA  = BETA
        self.GAMMA = GAMMA
        self.MHKB = MHKB
        self.include_heating = include_heating
        self.ctime = ctime
        self.dtmax = dtmax
        self.temp_floor = temp_floor
        self.subcycles = int(subcycles)
        self.early_exit = early_exit
        self.eps = eps

        self.i_rho = self.eq.mass_ids
        self.i_E   = self.eq.energy_ids

        self._Tmin = 1.0e4
        self._Tmax = 10.0**8.5
        self._Tcoef = 14.0 / 11.0
        
        # Compute Et_floor once at initialization
        self.gamma = getattr(self.eq, "gamma", 5.0/3.0)
        self.Rgas  = getattr(self.eq, "R", 1.0)

    def _temp_from_rhoP(self, rho, P):
        """Single consistent temperature calculation method."""
        return self._Tcoef * self.MHKB * P / jnp.maximum(rho, self.eps)

    def _Et_floor_from_rho(self, rho):
        """Compute minimum thermal energy density for given density."""
        if self.temp_floor is None:
            return 0.0
        Tf = self.temp_floor
        # Et = rho * R * T / (gamma - 1) for ideal gas
        return (rho * self.Rgas * Tf) / (self.gamma - 1.0)
        
    def _interp_logLambda_m20(self, logT):
        """Fixed interpolation formula."""
        # Grid spacing assumed to be 0.05
        idx_float = (logT - 4.0) / 0.05
        p = jnp.floor(idx_float).astype(int)
        p = jnp.clip(p, 0, self.logL.size - 2)
        
        # Correct linear interpolation
        frac = idx_float - p
        y = self.logL[p] * (1.0 - frac) + self.logL[p+1] * frac
        
        # Extrapolation above 10^8.5 K
        over = logT > 8.5
        slope = 20.0 * (self.logL[-1] - self.logL[-2])
        y_over = slope * (logT - 8.5) + self.logL[-1]
        
        return jnp.where(over, y_over, y)

    def _cooling_heating_rate(self, rho, T):
        """Compute energy change rate density (positive = heating, negative = cooling)."""
        logT = jnp.log10(jnp.maximum(T, self.eps))

        # Low-T branch: Koyama & Inutsuka 2002
        low = T < self._Tmin
        low_cool = self.ALPHA * rho * rho * (
            1.0e7 * jnp.exp(-118400.0 / (T + 1000.0)) + 
            0.014 * jnp.sqrt(jnp.maximum(T, 0.0)) * jnp.exp(-92.0 / jnp.maximum(T, self.eps))
        )
        low_heat = jnp.where(self.include_heating, self.GAMMA * rho * rho, 0.0)
        dotE_low = -low_cool + low_heat

        # High-T branch: Sutherland & Dopita 1993
        y = self._interp_logLambda_m20(logT)
        Lambda = 10.0**y
        dotE_high = -self.BETA * rho * rho * Lambda

        return jnp.where(low, dotE_low, dotE_high)

    def timestep(self, U):
        """Compute cooling timestep limit."""
        W = self.eq.get_primitives_from_conservatives(U)
        rho = jnp.maximum(W[self.i_rho], self.eps)
        P   = jnp.maximum(W[self.i_E],   self.eps)
        Et = P / (self.gamma - 1.0)
    
        T  = self._temp_from_rhoP(rho, P)
        T_eff = jnp.maximum(T, self.temp_floor if self.temp_floor is not None else 0.0)
    
        dotE = self._cooling_heating_rate(rho, T_eff)
        tcool = jnp.abs(Et / jnp.maximum(jnp.abs(dotE), self.eps))
        dt_cool = self.ctime * jnp.min(tcool)
        
        return jnp.minimum(dt_cool, self.dtmax)

    def force(self, i_step, U, params, dt):
        """
        Cooling/heating with per-substep floor enforcement.
        Key improvement: enforce Et >= Et_floor at every substep.
        """
        # Extract thermodynamic state
        W     = self.eq.get_primitives_from_conservatives(U)
        rho   = jnp.maximum(W[self.i_rho], self.eps)
        P     = jnp.maximum(W[self.i_E],   self.eps)
        Et0   = P / (self.gamma - 1.0)
        T0    = self._temp_from_rhoP(rho, P)
        
        # Compute floor per cell (density-dependent)
        Et_floor = self._Et_floor_from_rho(rho)
        Tf = self.temp_floor if self.temp_floor is not None else 0.0
        
        # --- Early-exit criterion ---
        T_eff0 = jnp.maximum(T0, Tf)
        dotE0  = self._cooling_heating_rate(rho, T_eff0)
        tcool0 = jnp.abs(Et0 / jnp.maximum(jnp.abs(dotE0), self.eps))
        
        # More conservative checks for early exit
        # Use 1.1× floor threshold (wider margin)
        near_floor0 = (T_eff0 <= 1.1 * Tf)
        
        # Check for sign flip
        dotE_raw0 = self._cooling_heating_rate(rho, T0)
        sign_flip = (dotE0 * dotE_raw0) < 0.0
        
        # Require small kick relative to Et AND to the margin above floor
        f_ee = 0.2  # conservative for early exit
        margin = Et0 - Et_floor
        small_kick = (jnp.abs(dotE0 * dt) <= f_ee * jnp.maximum(Et0, self.eps)) & \
                     (jnp.abs(dotE0 * dt) <= 0.5 * jnp.maximum(margin, self.eps))
        
        cand = tcool0 > (5.0 * dt / jnp.maximum(self.ctime, self.eps))
        
        slow = (self.early_exit & cand & (~near_floor0) & (~sign_flip) & small_kick)
        
        # --- Subcycling path with per-substep floor enforcement ---
        dt_loc = dt / float(self.subcycles)
        
        def body(_, Et_cur):
            # Current state
            P_cur = (self.gamma - 1.0) * Et_cur
            T_cur = self._temp_from_rhoP(rho, P_cur)
            
            # Compute rate (use floor temperature if below)
            T_eff = jnp.maximum(T_cur, Tf)
            # Disable cooling/heating if already at floor
            near_floor = (T_cur <= 1.05 * Tf)
            dotE = jnp.where(near_floor, 0.0, self._cooling_heating_rate(rho, T_eff))
            
            # Proposed update
            dE = dotE * dt_loc
            Et_new = Et_cur + dE
            
            # KEY FIX: Enforce floor at every substep
            # This prevents cascading violations across substeps
            Et_new = jnp.maximum(Et_new, Et_floor)
            
            # Additional fractional limiter for stability
            # (this is now secondary protection, floor is primary)
            abs_dE = jnp.abs(Et_new - Et_cur)
            max_frac_change = 0.5  # allow 50% change per substep max
            Et_new = jnp.where(
                abs_dE > max_frac_change * Et_cur,
                Et_cur + jnp.sign(Et_new - Et_cur) * max_frac_change * Et_cur,
                Et_new
            )
            
            return Et_new
        
        Et_sub = lax.fori_loop(0, int(self.subcycles), body, Et0)
        
        # --- Early-exit path (trapezoidal) ---
        # Predictor
        Et_pred = jnp.maximum(Et0 + dotE0 * dt, Et_floor)
        P_pred  = (self.gamma - 1.0) * Et_pred
        T_pred  = self._temp_from_rhoP(rho, P_pred)
        
        # Corrector
        dotE1 = self._cooling_heating_rate(rho, jnp.maximum(T_pred, Tf))
        
        # Trapezoidal update with conservative limiter
        dE_trap = 0.5 * (dotE0 + dotE1) * dt
        margin = Et0 - Et_floor
        dE_trap = jnp.clip(dE_trap, 
                          -f_ee * jnp.maximum(margin, self.eps),  # don't exceed margin to floor
                          f_ee * jnp.maximum(Et0, self.eps))      # don't heat too fast
        
        Et_slow = jnp.maximum(Et0 + dE_trap, Et_floor)
        
        # --- Select path ---
        Et_final = jnp.where(slow, Et_slow, Et_sub)
        
        # --- Write back ---
        dE = Et_final - Et0
        return U.at[self.i_E].add(dE)
    

class HeatCoolForce_old:
    """
    will be depreciated when I get around to it. Other heat_cool_force seem to behave better numerically, but 
    haven't gotten around to testing.
    
    Radiative cooling and heating as a source/forcing term:
      - T < 1e4 K: Koyama & Inutsuka (2002) cooling (+ optional heating)
      - 1e4 K <= T <= 10^8.5 K: Sutherland & Dopita (1993) via tabulated log10 Λ(T)-20
      - T  > 10^8.5 K: linear extrapolation using last two table points
    Subcycling: 10 micro-steps per hydro step with early exit if t_cool large.
    Timestep limiter: min(ctime * |E/Edot|, dtmax), computed on current state.
    Units/coeffs match those in Athena.
    """

    def __init__(
        self,
        equation_manager,
        pressure_fn,
        logT_table,                # shape [N], log10 T grid (monotonic, e.g. 4.0 .. 8.5)
        logLambda_m20_table,       # shape [N], values of (log10 Λ - 20) at logT_table
        *,
        # --- coefficients taken from Athena multiblast config (simulation units):
        ALPHA = 813.142554365,         # low-T cooling prefactor
        BETA  = 406571277.182611837,   # high-T cooling prefactor
        GAMMA = 406.571277183,         # low-T heating prefactor (toggle via include_heating)
        MHKB  = 115.98518596699539,    # m_H / k_B in code units
        include_heating = True,
        # timestep control used there:
        ctime = 1.0,       # coefficient 
        dtmax = 1.0,       # ceiling 
        temp_floor = 1.0e4,    # optional temperature floor 
        subcycles = 10,    # 10 micro-steps
        early_exit = True, # early exit if t_cool > dt/ctime (as in their code)
        eps = 1e-30
    ):
        self.eq = equation_manager
        self.pressure_fn = self.pressure_from_manager
        self.logT = jnp.asarray(logT_table)
        self.logL = jnp.asarray(logLambda_m20_table)
        self.ALPHA = ALPHA
        self.BETA  = BETA
        self.GAMMA = GAMMA
        self.MHKB = MHKB
        self.include_heating = include_heating
        self.ctime = ctime
        self.dtmax = dtmax
        self.temp_floor = temp_floor
        self.subcycles = int(subcycles)
        self.early_exit = early_exit
        self.eps = eps

        # indices from equation manager (mass, momentum, energy)
        self.i_rho = self.eq.mass_ids
        # momentum indices not needed; we only touch thermal energy (via total energy slot)
        self.i_E   = self.eq.energy_ids

        # constants and thresholds
        self._Tmin = 1.0e4
        self._Tmax = 10.0**8.5
        # convenience: factor 1.4/1.1 = 14/11 ≈ 1.272727... used in that file for T
        self._Tcoef = 14.0 / 11.0

    # -----------------------
    # Internal helpers
    # -----------------------
    def pressure_from_manager(self,U):
        #threshholded helper, can overwrite
        W = self.eq.get_primitives_from_conservatives(U)
        return jnp.maximum(W[self.eq.energy_ids], self.eq.eps)
        
    def _temp_from_rhoP(self, rho, P):
        # T = (14/11) * (m_H / k_B) * P / rho  
        return self._Tcoef * self.MHKB * P / jnp.maximum(rho, self.eps)

    def _Et_from_P(self, P, gamma):
        # Thermal energy density: Et = P / (gamma - 1); for gamma=5/3, Et=1.5 P
        return P / (gamma - 1.0)
        
    def _interp_logLambda_m20(self, logT):
        # assumes table grid is logT = 4.0 + 0.05*i  (i=0..N)
        p = jnp.floor((logT - 4.0) / 0.05).astype(int)
        p = jnp.clip(p, 0, self.logL.size - 2)
        q = (logT - 4.0) / 0.05
        y = self.logL[p]   * (q - 1.0 * p) + \
            self.logL[p+1] * (1.0 * p - q + 1.0)
        # above 10^8.5: extrapolate with the last two points 
        over = logT > 8.5
        slope = 20.0 * (self.logL[-1] - self.logL[-2])
        y_over = slope * (logT - 8.5) + self.logL[-1]
        return jnp.where(over, y_over, y)

    def _cooling_heating_rate(self, rho, T):
        """
        Compute dotE (energy change rate density):
          dotE = rho^2 * [ -Λ(T)  +  (heating if enabled & T<1e4) ]
        Returns dotE in simulation units matching ALPHA/BETA/GAMMA usage.
        """
        logT = jnp.log10(jnp.maximum(T, self.eps))

        # Low-T branch (T < 1e4 K): Koyama & Inutsuka 2002 + optional heating
        low = T < self._Tmin
        # KI02 cooling piece:
        low_cool = self.ALPHA * rho * rho * (
            1.0e7 * jnp.exp(-118400.0 / (T + 1000.0)) + 0.014 * jnp.sqrt(jnp.maximum(T, 0.0)) * jnp.exp(-92.0 / jnp.maximum(T, self.eps))
        )
        # Kim & Ostriker 2015 heating:
        low_heat = jnp.where(self.include_heating, self.GAMMA * rho * rho, 0.0)
        dotE_low = -low_cool + low_heat

        # High-T table branch (1e4 <= T): Sutherland & Dopita 1993 (tabulated)
        y = self._interp_logLambda_m20(logT)  # y = log10 Λ - 20
        Lambda = 10.0**y                       # this is dimensionless “Λ * 1e20” factor
        dotE_high = -self.BETA * rho * rho * Lambda

        return jnp.where(low, dotE_low, dotE_high)

    # -----------------------
    # Public API: dt limiter
    # -----------------------
    def timestep(self, U):
        W = self.eq.get_primitives_from_conservatives(U)
        rho = jnp.maximum(W[self.i_rho], self.eps)
        P   = jnp.maximum(W[self.i_E],   self.eps)
        gamma = getattr(self.eq, "gamma", 5.0/3.0)
        Et = P / (gamma - 1.0)
    
        T  = (14.0/11.0) * self.MHKB * P / rho
        T_eff = jnp.maximum(T, self.temp_floor if self.temp_floor is not None else 0.0)
    
        dotE = self._cooling_heating_rate(rho, T_eff)
        tcool = jnp.abs(Et / jnp.maximum(jnp.abs(dotE), 1e-30))
        dt_cool = self.ctime * jnp.min(tcool)
        idx = 444656#jnp.argmin(tcool)
     #   jax.debug.print("cool min: Et={e:.3e} T={t:.3e} rho={d:.3e} dotE={ed:.3e} idx={id:.10e}",
      #                  e=Et.reshape(-1)[idx],
     #                   t=T.reshape(-1)[idx], d=rho.reshape(-1)[idx],ed=dotE.reshape(-1)[idx],id=idx)
        return jnp.minimum(dt_cool, self.dtmax)
    def _thermo_from_U(self, U):
        # Use EquationManager to avoid unit/definition mismatches
        W = self.eq.get_primitives_from_conservatives(U)
        rho = jnp.maximum(W[self.i_rho], self.eps)
        P   = jnp.maximum(W[self.i_E],   self.eps)      # 'energy_ids' indexes pressure in primitives
        T   = self.eq.get_temperature(P, rho)
        gamma = getattr(self.eq, "gamma", 5.0/3.0)
        Et  = P / (gamma - 1.0)                         # thermal energy density
        return rho, P, T, Et, gamma

    # -----------------------
    # Public API: apply forcing
    # -----------------------
    def force(self, i_step, U, params, dt):
        """
        Cooling/heating source update with safe early-exit:
          - Subcycling path: explicit with symmetric fractional cap (no extra fac throttle).
          - Early-exit path: only when safe; trapezoidal update + fractional cap.
        """
        # --- state & thermodynamics ---
        W     = self.eq.get_primitives_from_conservatives(U)
        rho   = jnp.maximum(W[self.i_rho], self.eps)
        P     = jnp.maximum(W[self.i_E],   self.eps)         # primitives' "energy" slot is pressure
        gamma = getattr(self.eq, "gamma", 5.0/3.0)
        Et0   = P / (gamma - 1.0)
    
        # Kelvin-like "Athena" temperature: T = (14/11)*(mH/kB)*P/rho
        def T_phys(P_, rho_):
            return (14.0/11.0) * self.MHKB * P_ / jnp.maximum(rho_, self.eps)
    
        T0 = T_phys(P, rho)
        Tf = self.temp_floor if self.temp_floor is not None else 0.0
    
        # --- early-exit predicate (compute on current state) ---
        T_eff0 = jnp.maximum(T0, Tf)
        dotE0  = self._cooling_heating_rate(rho, T_eff0)
        tcool0 = jnp.abs(Et0 / jnp.maximum(jnp.abs(dotE0), 1e-30))
    
        # cells already at/near the floor: do NOT early-exit; keep subcycling (safer)
        near_floor0 = (T_eff0 <= 1.01 * Tf)
    
        # if the physical (raw-T) rate would flip sign relative to floored-T rate, avoid early-exit
        dotE_raw0 = self._cooling_heating_rate(rho, T0)
        sign_flip = (dotE0 * dotE_raw0) < 0.0
    
        # also require that a full-step kick would be small compared to Et (otherwise subcycle)
        f_ee = 0.3  # max fractional change allowed in slow path
        small_kick = jnp.abs(dotE0 * dt) <= (f_ee * jnp.maximum(Et0, self.eps))
    
        # more conservative threshold than dt/ctime (avoid marginal cases)
        cand = tcool0 > (5.0 * dt / jnp.maximum(self.ctime, self.eps))
    
        slow = (self.early_exit &
                cand &
                (~near_floor0) &
                (~sign_flip) &
                small_kick)
    
        # --- subcycling path (explicit; strong but stable) ---
        dt_loc    = dt / float(self.subcycles)
        f_max_hi  = 0.9   # aggressive cap at high T (smooth regime)
        f_max_lo  = 0.3   # conservative cap near the floor
    
        def body(_, carry):
            Tcur, Etcur = carry
    
            # rate at this substep (heating/cooling off right at the floor)
            T_eff     = jnp.maximum(Tcur, Tf)
            near_floor = (T_eff <= 1.01 * Tf)
            dotE      = jnp.where(near_floor, 0.0, self._cooling_heating_rate(rho, T_eff))
    
            # single symmetric fractional limiter (no extra stability factor)
            dE    = dotE * dt_loc
            f_max = jnp.where(T_eff >= 1.0e4, f_max_hi, f_max_lo)
            dE    = jnp.clip(dE, -f_max * Etcur, f_max * Etcur)
    
            Etnew = jnp.maximum(Etcur + dE, 1e-20)  # numerical floor only
            Pnew  = (gamma - 1.0) * Etnew
            Tnew  = T_phys(Pnew, rho)
            return (Tnew, Etnew)
    
        T_sub, Et_sub = lax.fori_loop(0, int(self.subcycles), body, (T0, Et0))
    
        # --- early-exit (slow) path: trapezoidal update + fractional cap ---
        # predictor using start-of-step rate
        Et_pred = Et0 + dotE0 * dt
        P_pred  = (gamma - 1.0) * jnp.maximum(Et_pred, self.eps)
        T_pred  = T_phys(P_pred, rho)
    
        # corrector rate at predicted state (respect floor for the rate)
        dotE1   = self._cooling_heating_rate(rho, jnp.maximum(T_pred, Tf))
    
        # trapezoidal increment, then cap to the same f_ee fraction
        dE_trap = 0.5 * (dotE0 + dotE1) * dt
        dE_trap = jnp.clip(dE_trap, -f_ee * jnp.maximum(Et0, self.eps), f_ee * jnp.maximum(Et0, self.eps))
        Et_slow = jnp.maximum(Et0 + dE_trap, 1e-20)
    
        # --- choose path per-cell ---
        Et_final = jnp.where(slow, Et_slow, Et_sub)
    
        # --- write back thermal-energy change into total energy slot ---
        # Compute the physical Et floor corresponding to the temp floor
        gamma = getattr(self.eq, "gamma", 5.0/3.0)
        Rgas  = getattr(self.eq, "R", 1.0)  # EOS gas constant in code units
        Tf    = self.temp_floor if self.temp_floor is not None else 0.0
        Et_floor = (rho * Rgas * Tf) / (gamma - 1.0)
        
        # Enforce only at write-back (no per-substep injection)
        Et_final = jnp.maximum(Et_final, Et_floor)
        
        # Write back into total energy slot
        dE = Et_final - Et0
        return U.at[self.i_E].add(dE)
    
    def force_sec(self, i_step, U, params, dt):
        ## SEEMS TO UNDER/OVERCOOL EASILY! Need to debug more, might be more efficient/faster than the subcycling thing
        """
        Implicit (Backward Euler) cooling/heating update:
            
          - Solves for T_{n+1} in: T_{n+1} = T_n + dt * C_T * dotE(rho, max(T_{n+1}, T_floor))
          - Secant iterations with damping, fully vectorized.
          - No energy injection during solve; temperature floor enforced at write-back.
        """
        # --- State & EOS helpers ---
        W     = self.eq.get_primitives_from_conservatives(U)
        rho   = jnp.maximum(W[self.i_rho], self.eps)
        P     = jnp.maximum(W[self.i_E],   self.eps)        # primitives' pressure slot
        gamma = getattr(self.eq, "gamma", 5.0/3.0)
    
        # Athena-style temperature: T = (14/11)*(mH/kB)*P/rho
        def T_from_PR(P_, rho_):
            return (14.0/11.0) * self.MHKB * P_ / jnp.maximum(rho_, self.eps)
    
        def P_from_rhoT(rho_, T_):
            return (11.0/14.0) * (1.0 / self.MHKB) * rho_ * T_
    
        T0 = T_from_PR(P, rho)
        Tf = self.temp_floor if self.temp_floor is not None else 0.0
    
        # Cells already at/near the floor: skip cooling (keeps behavior sharp near floor)
        near_floor0 = (jnp.maximum(T0, Tf) <= 1.01 * Tf)
    
        # C_T factor mapping dotE -> dT/dt
        CT = (14.0/11.0) * self.MHKB * (gamma - 1.0) / jnp.maximum(rho, self.eps)
    
        # --- Define F(T) = T - T0 - dt * CT * dotE(rho, max(T,Tf)) ---
        def F(T):
            Teff = jnp.maximum(T, Tf)
            return T - T0 - dt * CT * self._cooling_heating_rate(rho, Teff)
    
        # --- Secant solver (vectorized), with damping & fallbacks ---
        # Initial guesses: T0 and an explicit predictor
        Teff0   = jnp.maximum(T0, Tf)
        dotE0   = self._cooling_heating_rate(rho, Teff0)
        T1_pred = T0 + dt * CT * dotE0                     # forward-Euler predictor
        T_prev  = jnp.clip(T0,        1.0, self._Tmax)
        T_curr  = jnp.clip(T1_pred,   1.0, self._Tmax)
    
        def secant_step(carry, _):
            Tm1, Tm = carry
            Fm1 = F(Tm1)
            Fm  = F(Tm)
            denom = Fm - Fm1
            # If denom ~ 0, fall back to damped fixed-point
            use_secant = jnp.abs(denom) > 1e-20
            T_next_sec = Tm - Fm * (Tm - Tm1) / denom
            T_next_fp  = T0 + dt * CT * self._cooling_heating_rate(rho, jnp.maximum(Tm, Tf))
            T_next     = jnp.where(use_secant, T_next_sec, 0.5 * (Tm + T_next_fp))
    
            # Damping & bounds
            T_next = 0.5 * T_next + 0.5 * Tm
            T_next = jnp.clip(T_next, 1.0, self._Tmax)
            return (Tm, T_next), None
    
        # Iterate a small, fixed number of times (robust & JIT-friendly)
        (T_prev_fin, T_curr_fin), _ = lax.scan(secant_step, (T_prev, T_curr), xs=None, length=8)
    
        # If a cell was near-floor initially, keep its T (no source update there)
        T_new = jnp.where(near_floor0, T0, T_curr_fin)
    
        # --- Write-back temperature floor ONLY at the end (no energy injection during solve) ---
        T_new = jnp.maximum(T_new, Tf)
    
        # Map back to pressure & thermal energy, then update the energy slot conservatively
        P_new  = P_from_rhoT(rho, T_new)
        Et0    = P / (gamma - 1.0)
        Et_new = P_new / (gamma - 1.0)
    
        # Optional: prevent tiny numeric underflows
        Et_new = jnp.maximum(Et_new, 1e-20)
    
        dE = Et_new - Et0
        return U.at[self.i_E].add(dE)
    