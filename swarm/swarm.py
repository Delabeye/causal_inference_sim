# entities/swarm.py
import numpy as np
import pybullet as p

from entities.uav import UAV


class Swarm:
    """
    Classe représentant un essaim de drones (leader + followers).

    - Le leader suit ses propres waypoints / consignes (gérés dans UAV).
    - Les followers se placent en FORMATION TRIANGULAIRE derrière le leader,
      avec des offsets exprimés dans le REPÈRE DU LEADER (axe x vers l'avant).
    - À chaque pas de temps, on met à jour la cible de chaque follower :
        target = position_leader + Rz(yaw_leader) * offset_body
    - On ajoute une correction de "répulsion" pour maintenir une distance
      minimale entre les drones (éviter de se rentrer dedans).

    Les paramètres suivants viennent du SimulationManager / config :
      - min_sep : distance minimale désirée entre drones
      - avoid_gain : gain de la correction de répulsion
    """

    def __init__(
        self,
        agents: list[UAV],
        leader_name: str | None = None,
        min_sep: float = 0.6,
        avoid_gain: float = 0.5,
        formation_body_offsets: dict[str, np.ndarray] | None = None,
    ):
        if len(agents) == 0:
            raise ValueError("Swarm nécessite au moins un agent UAV.")

        self.agents = agents

        # Choix du leader
        if leader_name is not None:
            leader_list = [a for a in agents if getattr(a, "name", "") == leader_name]
            if len(leader_list) == 0:
                raise ValueError(f"Aucun UAV avec name='{leader_name}' trouvé pour le leader.")
            self.leader = leader_list[0]
        else:
            # par défaut, le premier UAV de la liste est le leader
            self.leader = agents[0]

        # Followers = tous les autres
        self.followers: list[UAV] = [a for a in agents if a is not self.leader]

        self.physics_client_id = self.leader.physics_client_id

        # ----------------- PARAMÈTRES D'ÉVITAGE -----------------
        # Distance minimale désirée entre les drones (en m)
        self.min_sep = float(min_sep)
        # Gain de correction (0 -> pas de correction, 1 -> correction maxi)
        self.avoid_gain = float(avoid_gain)

        # ----------------- OFFSETS DE FORMATION -----------------
        self.formation_body_offsets: dict[str, np.ndarray] = {}

        if formation_body_offsets is not None:
            # On utilise ce qu'on nous donne (dict name -> np.array)
            for f in self.followers:
                if f.name not in formation_body_offsets:
                    raise ValueError(
                        f"formation_body_offsets ne contient pas d'offset pour follower '{f.name}'."
                    )
                off = np.array(formation_body_offsets[f.name], dtype=float)
                if off.shape != (3,):
                    raise ValueError("Chaque offset doit être un vecteur 3D [x, y, z].")
                self.formation_body_offsets[f.name] = off
        else:
            # Formation triangulaire par défaut derrière le leader
            self._assign_default_triangular_offsets()

        print(
            f"[Swarm] Essaim créé avec leader='{self.leader.name}', "
            f"{len(self.followers)} follower(s). "
            f"(min_sep={self.min_sep:.2f}, avoid_gain={self.avoid_gain:.2f})"
        )

    # ------------------------------------------------------------------
    def _assign_default_triangular_offsets(self):
        """
        Assigne automatiquement une formation triangulaire derrière le leader
        pour tous les followers.

        Convention :
          - axe x du leader = avant
          - followers rangés en "rangs" derrière lui (x négatif),
            avec un triangle élargi en y.

        Exemple pour 6 followers (vue de dessus) :

             L
           F1 F2
         F3 F4 F5
        """
        spacing_x = 0.7  # distance entre rangées en x (m)
        spacing_y = 0.7  # distance latérale en y (m)

        followers = self.followers
        n = len(followers)
        if n == 0:
            return

        idx = 0
        row = 1
        while idx < n:
            num_in_row = row
            center = 0.5 * (num_in_row - 1)
            for j in range(num_in_row):
                if idx >= n:
                    break
                f = followers[idx]

                x_offset = -row * spacing_x
                y_offset = (j - center) * spacing_y
                z_offset = 0.0  # même altitude que le leader

                self.formation_body_offsets[f.name] = np.array(
                    [x_offset, y_offset, z_offset], dtype=float
                )
                idx += 1
            row += 1

    # ------------------------------------------------------------------
    def update(self):
        """
        Met à jour la cible de chaque follower en fonction de la pose du leader
        et applique une correction de répulsion pour maintenir une distance
        de sécurité entre les drones.

        À appeler à chaque pas de temps AVANT d'appeler uav.think_and_act().
        """
        if len(self.followers) == 0:
            return

        # 1) Lecture de la pose du leader
        try:
            state_leader = self.leader.get_ground_truth_state()
        except p.error:
            return

        pos_leader = state_leader["pos"]      # [x, y, z] monde
        orn_leader = state_leader["orn_q"]    # quaternion
        roll, pitch, yaw = p.getEulerFromQuaternion(orn_leader)

        # Matrice de rotation yaw (autour de Z) pour passer du repère leader au repère monde
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        R_yaw = np.array(
            [
                [cy, -sy, 0.0],
                [sy,  cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # 2) Récupérer la position courante de tous les drones pour l'évitement
        all_agents = [self.leader] + self.followers
        positions: dict[str, np.ndarray] = {}

        for a in all_agents:
            try:
                st = a.get_ground_truth_state()
                positions[a.name] = st["pos"]
            except p.error:
                continue

        # 3) Calcul de la cible de chaque follower
        for follower in self.followers:
            off_body = self.formation_body_offsets.get(follower.name, None)
            if off_body is None:
                # Pas d'offset défini -> on le laisse suivre son propre setpoint
                continue

            # --- Cible nominale en repère monde (formation idéale) ---
            off_world = R_yaw @ off_body
            base_target = pos_leader + off_world  # [x, y, z]

            # --- Correction de répulsion pour éviter les collisions ---
            correction = np.zeros(3, dtype=float)

            pos_f = positions.get(follower.name, None)
            if pos_f is None:
                pos_f = base_target  # fallback

            for other in all_agents:
                if other is follower:
                    continue

                pos_o = positions.get(other.name, None)
                if pos_o is None:
                    continue

                # Évitement en XY uniquement
                diff = base_target - pos_o
                diff[2] = 0.0

                dist = float(np.linalg.norm(diff))
                if dist < 1e-6:
                    continue

                if dist < self.min_sep:
                    repulse_dir = diff / dist
                    amplitude = (self.min_sep - dist)
                    correction += amplitude * repulse_dir

            # Applique la correction (amortie par avoid_gain)
            final_target = base_target + self.avoid_gain * correction
            # On garde la même altitude que la cible de base
            final_target[2] = base_target[2]

            # 4) Mettre à jour la cible du follower
            if hasattr(follower, "set_target_pos"):
                follower.set_target_pos(final_target)
            else:
                follower.target_pos = final_target
