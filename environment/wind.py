import numpy as np
import math
from scipy import signal

    
class DrydenGustModel():
    """
    Implémentation du modèle de turbulence Dryden (MIL-F-8785C) robuste.
    Corrigé pour éviter les erreurs de dimension (2D vs 1D) avec scipy.signal.
    """
    def __init__(self,dt,turbulence_intensity_knots=15,mean_wind=[0,0,0]):
        self.dt = float(dt)
        self.turbulence_level = float(turbulence_intensity_knots)
        self.mean_wind = mean_wind 
        # États internes (Memory) pour les filtres (u, v, w)
        self._zi_u = None
        self._zi_v = None
        self._zi_w = None
        
        # Coefficients des filtres numériques (b, a) stockés en 1D
        self._sys_d_u = None
        self._sys_d_v = None
        self._sys_d_w = None
        
        # Mémoire pour éviter le recalcul constant
        self._last_h = -1.0
        self._last_V = -1.0

    def _update_filters(self, h_meters, V_ms):
        """
        Recalcule les coefficients des filtres numériques.
        """
        # 1. Conversions et Sécurités (Force float)
        h = float(h_meters) * 3.28084  # m -> ft
        V = float(V_ms) * 3.28084      # m/s -> ft/s
        
        # Clamp pour éviter divisions par zéro (V=0) ou altitudes négatives
        if V < 0.1: V = 0.1
        if h < 10.0: h = 10.0 

        # 2. Paramètres Dryden (Low Altitude Model < 1000ft)
        # Longueurs d'échelle (L)
        L_w = h
        L_u = h / ((0.177 + 0.000823 * h) ** 1.2)
        L_v = L_u 

        # Intensités (Sigma)
        sigma_w = 0.1 * self.turbulence_level
        sigma_u = sigma_w / ((0.177 + 0.000823 * h) ** 0.4)
        sigma_v = sigma_u

        # 3. Fonctions de Transfert Continues (Laplace)
        # Forme: H(s) = Num(s) / Den(s)
        
        # --- AXE U (Longitudinal) ---
        K_u = sigma_u * math.sqrt((2 * L_u) / (math.pi * V))
        T_u = L_u / V
        num_u = [K_u]
        den_u = [T_u, 1.0] 

        # --- AXE V (Latéral) ---
        # Attention aux parenthèses pour la racine carrée
        K_v = sigma_v * math.sqrt(L_v / (math.pi * V)) 
        T_v = L_v / V
        num_v = [K_v * math.sqrt(3.0) * T_v, K_v]
        den_v = [T_v**2, 2 * T_v, 1.0]

        # --- AXE W (Vertical) ---
        K_w = sigma_w * math.sqrt(L_w / (math.pi * V))
        T_w = L_w / V
        num_w = [K_w * math.sqrt(3.0) * T_w, K_w]
        den_w = [T_w**2, 2 * T_w, 1.0]

        # 4. Discrétisation (Continuous -> Discrete)
        # cont2discrete renvoie des matrices 2D [[b...]], [[a...]]
        # Il faut extraire la première ligne [0] pour avoir des vecteurs 1D
        
        sys_u = signal.cont2discrete((num_u, den_u), self.dt, method='bilinear')
        self._sys_d_u = (sys_u[0][0], sys_u[1][0]) # (b, a) en 1D

        sys_v = signal.cont2discrete((num_v, den_v), self.dt, method='bilinear')
        self._sys_d_v = (sys_v[0][0], sys_v[1][0])

        sys_w = signal.cont2discrete((num_w, den_w), self.dt, method='bilinear')
        self._sys_d_w = (sys_w[0][0], sys_w[1][0])

        # 5. Réinitialisation des états (zi)
        # lfilter_zi a besoin de vecteurs 1D, maintenant c'est garanti.
        if self._zi_u is None:
            self._zi_u = signal.lfilter_zi(self._sys_d_u[0], self._sys_d_u[1]) * 0.0
            self._zi_v = signal.lfilter_zi(self._sys_d_v[0], self._sys_d_v[1]) * 0.0
            self._zi_w = signal.lfilter_zi(self._sys_d_w[0], self._sys_d_w[1]) * 0.0

        self._last_h = float(h_meters)
        self._last_V = float(V_ms)

    def step(self, h_meters, V_ms):
        """
        Calcule la prochaine rafale de vent.
        """
        h = float(h_meters)
        V = float(V_ms)

        # Mise à jour des filtres si conditions changent significativement
        if (abs(h - self._last_h) > 1.0 or abs(V - self._last_V) > 0.5):
            self._update_filters(h, V)
        elif self._sys_d_u is None:
            self._update_filters(h, V)

        # Bruit blanc (entrée du filtre)
        noise = np.random.normal(0, 1, 3)

        # Filtrage
        # On unpack b et a qui sont maintenant garantis d'être des vecteurs 1D
        
        # Axe U
        b_u, a_u = self._sys_d_u
        val_u, self._zi_u = signal.lfilter(b_u, a_u, [noise[0]], zi=self._zi_u)
        
        # Axe V
        b_v, a_v = self._sys_d_v
        val_v, self._zi_v = signal.lfilter(b_v, a_v, [noise[1]], zi=self._zi_v)
        
        # Axe W
        b_w, a_w = self._sys_d_w
        val_w, self._zi_w = signal.lfilter(b_w, a_w, [noise[2]], zi=self._zi_w)

        # Conversion ft/s -> m/s et ajout du vent moyen
        gusts_ms = np.array([val_u[0], val_v[0], val_w[0]]) * 0.3048
        
        return self.mean_wind + gusts_ms