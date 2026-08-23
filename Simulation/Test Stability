import numpy as np
import control as ct
import matplotlib.pyplot as plt

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
    v31 = -(l*l*r*r)/(d*d)*m_b*np.sin(q[2])*np.cos(q[2])*((q_dot[0]-q_dot[1])**2) + m_b*l*gra*np.sin(q[2])
    
    M_q = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
    v_q_qdot = np.array([v11, v21, v31])
    tau_a = K_T*n*np.dot(np.array([[1, 0], [0, 1], [-1, -1]]), I)
    tau_f = np.dot(np.array([[b+f_v, 0, -b], [0, b+f_v, -b], [-b, -b, b+b]]), q_dot) 

    rhs = tau_a - tau_f - v_q_qdot
    q_ddot = np.linalg.solve(M_q, rhs)

    # B. Calculate Electrical terms (plug in equations here)

    coupling_matrix = np.array([[1, 0, -1], [0, 1, -1]])
    I_dot = (1 / params['L']) * (u - params['K_E'] * params['n'] * (coupling_matrix @ q_dot) - params['R'] * I)

    # Combine derivatives into a single state derivative vector

    x_dot = np.concatenate([q_dot, q_ddot, I_dot])
     
    return x_dot

# --- 3. Define the Output Function (y) ---
def pendulum_output(t, x, u, params):
    # As per the paper, y(t) = x(t)
    return x

twsbr_sys = ct.NonlinearIOSystem(
    updfcn=pendulum_update,
    outfcn=pendulum_output,
    inputs=2,    # e.g., Left and Right motor voltages [u_L, u_R]
    outputs=8,   # e.g., Full state vector output
    states=8,    # e.g., Full state vector size
    name='twsbr'
)

# --- 5. Run the Simulation ---
# Define time vector (e.g., 5 seconds, 500 points)
time = np.linspace(0, 10, 1000)

# Define input vector (e.g., 2 inputs over 500 time steps)
# Example: Apply a small step voltage to both motors at t=0
U = np.zeros((2, len(time)))
U[0, :] = 5.0  # 5V to Left motor
U[1, :] = 5.0  # 5V to Right motor

# Define initial conditions (e.g., robot starting slightly tilted)
X0 = np.zeros(8)
X0[2] = 0.3 # Example: small initial tilt angle

# Simulate
t, y = ct.input_output_response(
    twsbr_sys, 
    time, 
    U, 
    X0, 
    params=params
    )

# --- 6. Plot the Results ---
plt.figure(figsize=(10, 6))
plt.plot(t, y[2, :], label='Tilt Angle (theta3)')
plt.title('Two-Wheeled Inverted Pendulum - Nonlinear Simulation')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)
plt.show()
