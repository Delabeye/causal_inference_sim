
Sim_duration = 1000  # in second
Time_step = 0.1  # in second

Num_GNSS_sensors = 0  # number of GNSS sensors in the simulation
GNSS_position_noise_std = []  # list of float in meter
GNSS_velocity_noise_std = []  # list of float in meter/second

Num_drones = 0 # number of drones in the simulation
Drone_coords = []  # list of tuples (x, y, z) in meter
Drone_angles = []  # list of tuples (roll, pitch, yaw) in radian
Drone_speeds = []  # list of float in meter/second
Drone_mass = []  # list of float in kg
Drone_drag_coefficient = []  # list of float dimensionless
Max_drone_speed = []  # list of float in meter/second
Max_drone_acceleration = []  # list of float in meter/second^2
Sensors=[]  # list of tuples of sensor objects assigned to each drone

World_size = (100.0, 100.0, 20.0)  # in meter

Num_obstacles = 10







