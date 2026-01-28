import numpy as np
import math
from scipy import signal

    
class DrydenGustModel():
    """
    Robust implementation of the Dryden Turbulence Model (MIL-F-8785C).

    This class simulates atmospheric turbulence by passing white Gaussian noise 
    through coloring filters (transfer functions) shaped according to the 
    Dryden spectral density. It specifically implements the Low-Altitude 
    model (below 1000 ft) where turbulence is non-isotropic and dependent on 
    height above ground level.

    Attributes:
        dt (float): Simulation time step in seconds.
        turbulence_level (float): Intensity of turbulence in knots (Severity).
        mean_wind (np.ndarray): Steady-state wind vector [u, v, w] in m/s.
        _zi_u, _zi_v, _zi_w (np.ndarray): Internal delay buffers (initial states) 
                                      for the digital filters.
    """
    def __init__(self, dt, turbulence_intensity_knots=15, mean_wind=[0, 0, 0]):
        """
        Initializes the Dryden gust generator with simulation and environmental parameters.

        Args:
            dt (float): Discrete time step for the simulation (s).
            turbulence_intensity_knots (float): Wind speed at 20ft altitude, 
                                        defining the severity (Light/Moderate/Severe).
            mean_wind (list/np.ndarray): The constant background wind vector in m/s.
        """
        self.dt = float(dt)
        self.turbulence_level = float(turbulence_intensity_knots)
        self.mean_wind = mean_wind 
        
        # Internal states (Memory) for filters (u, v, w)
        self._zi_u = None
        self._zi_v = None
        self._zi_w = None
        
        # Digital filter coefficients (b, a) stored as 1D vectors
        self._sys_d_u = None
        self._sys_d_v = None
        self._sys_d_w = None
        
        # Cache to avoid constant recalculation
        self._last_h = -1.0
        self._last_V = -1.0

    def _update_filters(self, h_meters, V_ms):
        """
        Recalculates digital filter coefficients based on current flight conditions.

        As altitude and airspeed change, the scale lengths ($L_u, L_v, L_w$) and 
        intensities ($\sigma_u, \sigma_v, \sigma_w$) of the turbulence change. This 
        method updates the Laplace transfer functions and discretizes them using 
        the Tustin (bilinear) transformation.

        Args:
            h_meters (float): Current altitude above ground level (AGL) in meters.
            V_ms (float): Current true airspeed (TAS) of the vehicle in m/s.

        Note:
            Internal calculations are performed in Imperial units (ft, ft/s) 
            per the MIL-F-8785C standard before being stored.
        """
        # 1. Conversions and Safety (Force float)
        h = float(h_meters) * 3.28084  # m -> ft
        V = float(V_ms) * 3.28084      # m/s -> ft/s
        
        # Clamp to avoid division by zero (V=0) or negative altitudes
        if V < 0.1: V = 0.1
        if h < 10.0: h = 10.0 

        # 2. Dryden Parameters (Low Altitude Model < 1000ft)
        # Scale lengths (L)
        L_w = h
        L_u = h / ((0.177 + 0.000823 * h) ** 1.2)
        L_v = L_u 

        # Intensities (Sigma)
        sigma_w = 0.1 * self.turbulence_level
        sigma_u = sigma_w / ((0.177 + 0.000823 * h) ** 0.4)
        sigma_v = sigma_u

        # 3. Continuous Transfer Functions (Laplace)
        # Form: $H(s) = \frac{Num(s)}{Den(s)}$
        
        # --- U-AXIS (Longitudinal) ---
        K_u = sigma_u * math.sqrt((2 * L_u) / (math.pi * V))
        T_u = L_u / V
        num_u = [K_u]
        den_u = [T_u, 1.0] 

        # --- V-AXIS (Lateral) ---
        # Watch parentheses for the square root calculation
        K_v = sigma_v * math.sqrt(L_v / (math.pi * V)) 
        T_v = L_v / V
        num_v = [K_v * math.sqrt(3.0) * T_v, K_v]
        den_v = [T_v**2, 2 * T_v, 1.0]

        # --- W-AXIS (Vertical) ---
        K_w = sigma_w * math.sqrt(L_w / (math.pi * V))
        T_w = L_w / V
        num_w = [K_w * math.sqrt(3.0) * T_w, K_w]
        den_w = [T_w**2, 2 * T_w, 1.0]

        # 4. Discretization (Continuous -> Discrete)
        # cont2discrete returns 2D matrices [[b...]], [[a...]]
        # We must extract the first row [0] to obtain 1D vectors
        
        sys_u = signal.cont2discrete((num_u, den_u), self.dt, method='bilinear')
        self._sys_d_u = (sys_u[0][0], sys_u[1][0]) # (b, a) as 1D

        sys_v = signal.cont2discrete((num_v, den_v), self.dt, method='bilinear')
        self._sys_d_v = (sys_v[0][0], sys_v[1][0])

        sys_w = signal.cont2discrete((num_w, den_w), self.dt, method='bilinear')
        self._sys_d_w = (sys_w[0][0], sys_w[1][0])

        # 5. State Reinitialization (zi)
        # lfilter_zi requires 1D vectors, which is now guaranteed.
        if self._zi_u is None:
            self._zi_u = signal.lfilter_zi(self._sys_d_u[0], self._sys_d_u[1]) * 0.0
            self._zi_v = signal.lfilter_zi(self._sys_d_v[0], self._sys_d_v[1]) * 0.0
            self._zi_w = signal.lfilter_zi(self._sys_d_w[0], self._sys_d_w[1]) * 0.0

        self._last_h = float(h_meters)
        self._last_V = float(V_ms)

    def step(self, h_meters, V_ms):
        """
        Computes the atmospheric wind gust vector for the current time step.

        Generates zero-mean white Gaussian noise and passes it through the 
        pre-computed digital filters to produce the stochastic gust components. 
        The filters are automatically updated if altitude or velocity changes 
        exceed defined thresholds to ensure physical consistency.

        Args:
            h_meters (float): Current altitude (AGL) in meters.
            V_ms (float): Current true airspeed in m/s.

        Returns:    
            np.ndarray: Total wind vector (Mean Wind + Gusts) in m/s, 
               expressed in the World frame $[u_{wind}, v_{wind}, w_{wind}]$.
        """
        h = float(h_meters)
        V = float(V_ms)

        # Update filters if flight conditions change significantly
        if (abs(h - self._last_h) > 1.0 or abs(V - self._last_V) > 0.5):
            self._update_filters(h, V)
        elif self._sys_d_u is None:
            self._update_filters(h, V)

        # White noise (filter input)
        noise = np.random.normal(0, 1, 3)

        # Filtering
        # Unpacking b and a which are now guaranteed to be 1D vectors
        
        # U Axis
        b_u, a_u = self._sys_d_u
        val_u, self._zi_u = signal.lfilter(b_u, a_u, [noise[0]], zi=self._zi_u)
        
        # V Axis
        b_v, a_v = self._sys_d_v
        val_v, self._zi_v = signal.lfilter(b_v, a_v, [noise[1]], zi=self._zi_v)
        
        # W Axis
        b_w, a_w = self._sys_d_w
        val_w, self._zi_w = signal.lfilter(b_w, a_w, [noise[2]], zi=self._zi_w)

        # Conversion ft/s -> m/s and adding mean wind
        gusts_ms = np.array([val_u[0], val_v[0], val_w[0]]) * 0.3048
        
        return self.mean_wind + gusts_ms