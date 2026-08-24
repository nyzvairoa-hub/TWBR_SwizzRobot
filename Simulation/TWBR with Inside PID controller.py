import numpy as np
import control as ct
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------
# 1. Plant

params = {
    'r': 0.0325,
    'm_w': 0.053,
    'l': 0.045,
    'm_b': 0.8,
    'd': 0.165,
    'J_J': 0.005518,
    'J_K': 0.00135,
    'n': 43.8,
    'J_r': 0.00000274,
    'R': 4.019,
    'L': 0.0000222,
    'K_T': 0.005443,
    'K_E': 0.005443,
    'b': 0.00000727,
    'f_v': 0.000054,
    'gra': 9.81
}

def pendulum_update(t, x, u, params):
    # Unpack states. Assuming x is a combination of q, q_dot, and currents (I)
    # Adjust the unpacking based on the exact state vector dimension in the paper.
    # x = [theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot, I_L, I_R]
    
    # Example state unpacking (modify to match your exact state vector size)
    q = x[0:3]       # Generalized coordinates
    q_dot = x[3:6]   # Velocities
    I = x[6:8]       # Motor currents
    
    r = params['r']
    m_w = params['m_w']
    l = params['l']
    m_b = params['m_b']
    d = params['d']
    J_J = params['J_J']
    J_K = params['J_K']
    n = params['n']
    J_r = params['J_r']
    R = params['R']
    L = params['L']
    K_T = params['K_T']
    K_E = params['K_E']
    b = params['b']
    f_v = params['f_v']
    gra = params['gra']
    
    # A. Calculate Mechanical terms (plug in equations here)
    
    m11 = 3/2*m_w*r*r + 1/4*m_b*r*r + n*n*J_r + (((l*l)*(r*r))/(d*d))*m_b*np.sin(q[2])**2 + J_J*((r*r)/(d*d))
    m22 = m11
    m33 = m_b*l*l + J_K + 2*n*n*J_r
    m12 = 1/4*m_b*r*r - (l*l*r*r)/(d*d)*m_b*np.sin(q[2])**2 - J_J*((r*r)/(d*d))
    m21 = m12
    m13 = 1/2*m_b*l*r*np.cos(q[2]) - n*n*J_r
    m23 = m13
    m31 = m13
    m32 = m13
    
    v11 = (2*l*l*r*r)/(d*d)*m_b*np.sin(q[2])*np.cos(q[2])*q_dot[2]*(q_dot[0]-q_dot[1]) - 1/2*m_b*l*r*np.sin(q[2])*q_dot[2]
    v21 = (2*l*l*r*r)/(d*d)*m_b*np.sin(q[2])*np.cos(q[2])*q_dot[2]*(q_dot[1]-q_dot[0]) - 1/2*m_b*l*r*np.sin(q[2])*q_dot[2]
    v31 = -(l*l*r*r)/(d*d)*m_b*np.sin(q[2])*np.cos(q[2])*((q_dot[0]-q_dot[1])**2) - m_b*l*gra*np.sin(q[2])
    
    M_q = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
    v_q_qdot = np.array([v11, v21, v31])
    tau_a = K_T*n*np.dot(np.array([[1, 0], [0, 1], [-1, -1]]), I)
    tau_f = np.dot(np.array([[b+f_v, 0, -b], [0, b+f_v, -b], [-b, -b, b+b]]), q_dot) 

    rhs = tau_a - tau_f - v_q_qdot
    q_ddot = (np.linalg.solve(M_q, rhs)).flatten()

    # B. Calculate Electrical terms (plug in equations here)

    coupling_matrix = np.array([[1, 0, -1], [0, 1, -1]])
    I_dot = (1 / params['L']) * (u - params['K_E'] * params['n'] * (coupling_matrix @ q_dot) - params['R'] * I)

    # Combine derivatives into a single state derivative vector

    x_dot = np.concatenate([q_dot, q_ddot, I_dot])
     
    return x_dot

# --- Define the Output Function (y) ---
def pendulum_output(t, x, u, params):
    # As per the paper, y(t) = x(t)
    return x

twsbr_sys = ct.NonlinearIOSystem(
    updfcn=pendulum_update,
    outfcn=pendulum_output,
    name = 'plant',
    inputs=['u_L', 'u_R'],    # e.g., Left and Right motor voltages [u_L, u_R]
    outputs=['theta1', 'theta2', 'theta3', 'theta1_dot', 'theta2_dot', 'theta3_dot', 'I_L', 'I_R'],   # e.g., Full state vector output
    states=8,    # e.g., Full state vector size
)

# ---------------------------

# ---------------------------
# 2. Controller
def pid_up(t, x, u, params):
    ref_angle = u[0]  # Desired tilt angle (upright)
    current_angle = u[1]  # Assuming theta3 is the tilt angle

    error = ref_angle - current_angle

    return [error]

def pid_out(t, x, u, params):
    ref_angle = u[0]  # Desired tilt angle (upright)
    current_angle = u[1]  # Assuming theta3 is the tilt angle
    current_angle_dot = u[2]  # Assuming theta3_dot is the tilt angular velocity
    e_int = x[0]  # Integral of error (assuming it's stored in the state)

    error = ref_angle - current_angle

    Kp = 0
    Ki = 0
    Kd = 0

    Voltage = -(Kp * error + Ki * e_int - Kd * current_angle_dot)

    Voltage = np.clip(Voltage, -12, 12)  # Limit voltage to [-12V, 12V]

    return [Voltage, Voltage]

pid_sys = ct.NonlinearIOSystem(
    updfcn=pid_up,
    outfcn=pid_out,
    name = 'controller',
    inputs=['ref_angle', 'theta3', 'theta3_dot'],  # e.g., Reference angle and current tilt angle
    outputs=['u_L', 'u_R'],  # e.g., Output voltage to motors
    states = ['e_int']
)

# ---------------------------

# ---------------------------
# 3. Connection
closed_loop_sys = ct.InterconnectedSystem(
    [twsbr_sys, pid_sys],
    connections = [
        # Connect Controller outputs to Plant inputs
        ['plant.u_L', 'controller.u_L'],
        ['plant.u_R', 'controller.u_R'],

        #connect plant output to contrller inputs
        ['controller.theta3', 'plant.theta3'],
        ['controller.theta3_dot', 'plant.theta3_dot']
    ],
    inplist = ['controller.ref_angle'],
    outlist = ['plant.theta3', 'plant.theta1', 'plant.theta2', 'controller.u_L']
)

# 4. Simulation
# Define time vector (e.g., 5 seconds, 500 points)
time = np.linspace(0, 10, 100)

# Define input vector (e.g., 2 inputs over 500 time steps)
# Example: Apply a small step voltage to both motors at t=0
ref_in = np.zeros(len(time))

# Define initial conditions (e.g., robot starting slightly tilted)
X0 = np.zeros(9)
X0[2] = 0.3 # Example: small initial tilt angle

# Simulate
t, y = ct.input_output_response(
    closed_loop_sys, 
    time, 
    ref_in, 
    X0, 
    params=params
    )

theta3_sim = y[0, :]
theta1_sim = y[1, :]
theta2_sim = y[2, :]
voltage_sim = y[3, :]

# 5. Plot the Results
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(t, theta3_sim, label='Tilt Angle')
plt.axhline(0, color='r', linestyle='--', label='Target')
plt.title('Robot Balancing over Time')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(t, voltage_sim)
plt.title('Controller Effort (Motor Voltage)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid()
plt.tight_layout()
plt.show()

r = params['r']
l = params['l']
x_base = r * (theta1_sim + theta2_sim) / 2.0
y_base = r

x_tip = x_base + l * np.sin(theta3_sim)
y_tip = y_base + l * np.cos(theta3_sim)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_aspect('equal')
ax.grid(True)
ax.set_ylim(-0.1, r + l + 0.1)

ground = ax.axhline(0, color='black', lw=2)
wheel = plt.Circle((x_base[0], y_base), r, color='blue', fill=True, alpha=0.5)
ax.add_patch(wheel)
(body_line, ) = ax.plot([], [], 'o-', lw=4, color='orange', markersize=8)

def update(frame):
    ax.set_xlim(x_base[frame] - 0.3, x_base[frame] + 0.3)

    wheel.set_center((x_base[frame], y_base))
    body_line.set_data([x_base[frame], x_tip[frame]], [y_base, y_tip[frame]])
    return wheel, body_line

dt = (time[-1] - time[0]) / len(time)
ani = animation.FuncAnimation(fig, update, frames=len(time), interval=dt, blit=True)
plt.title('Two-Wheeled Inverted Pendulum Animation')
plt.show()

