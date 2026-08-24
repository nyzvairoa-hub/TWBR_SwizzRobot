<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Two-Wheeled Robot PyScript Simulator</title>
    <link rel="stylesheet" href="https://pyscript.net/releases/2024.1.1/core.css" />
    <script type="module" src="https://pyscript.net/releases/2024.1.1/core.js"></script>

    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f4f6f8; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .controls { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 300px; }
        .slider-group { margin-bottom: 15px; }
        .slider-group label { display: block; font-weight: bold; margin-bottom: 5px; }
        .slider-group input { width: 100%; }
        canvas { background: #ffffff; border: 1px solid #ccc; border-radius: 8px; }
    </style>
</head>
<body>

    <h2>Interactive Balancing Robot Simulator (PyScript)</h2>

    <div class="container">
        <!-- Controls UI -->
        <div class="controls">
            <h3>Outer Loop (Position)</h3>
            <div class="slider-group">
                <label>Kx (Position): <span id="val_kx">0.1</span></label>
                <input type="range" id="kx" min="0" max="1" value="0.1" step="0.05">
            </div>
            <div class="slider-group">
                <label>Kv (Velocity): <span id="val_kv">0.2</span></label>
                <input type="range" id="kv" min="0" max="1" value="0.2" step="0.05">
            </div>
            
            <hr>
            
            <h3>Inner Loop (Balance)</h3>
            <div class="slider-group">
                <label>Kp: <span id="val_kp">60</span></label>
                <input type="range" id="kp" min="0" max="150" value="60" step="1">
            </div>
            <div class="slider-group">
                <label>Ki: <span id="val_ki">0</span></label>
                <input type="range" id="ki" min="0" max="20" value="0" step="0.5">
            </div>
            <div class="slider-group">
                <label>Kd: <span id="val_kd">4</span></label>
                <input type="range" id="kd" min="0" max="30" value="4" step="0.5">
            </div>
            <div class="slider-group">
                <label>Initial Tilt (rad): <span id="val_theta">0.25</span></label>
                <input type="range" id="theta0" min="-0.5" max="0.5" value="0.25" step="0.01">
            </div>
            <button id="run_btn" style="width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Simulate & Animate</button>
        </div>

        <!-- Animation Canvas -->
        <div>
            <canvas id="robotCanvas" width="600" height="400"></canvas>
        </div>
    </div>

    <!-- Python Engine -->
    <script type="py" config='{"packages": ["numpy"]}'>
import numpy as np
import js
from pyodide.ffi import create_proxy

# Robot Parameters
params = {
    'r': 0.0325, 'm_w': 0.053, 'l': 0.045, 'm_b': 0.8, 'd': 0.165,
    'J_J': 0.005518, 'J_K': 0.00135, 'n': 43.8, 'J_r': 0.00000274,
    'R': 4.019, 'L': 0.0000222, 'K_T': 0.005443, 'K_E': 0.005443,
    'b': 0.00000727, 'f_v': 0.000054, 'gra': 9.81
}

def plant_dynamics(x, u, params):
    q = x[0:3]
    q_dot = x[3:6]

    r, m_w, l, m_b, d = params['r'], params['m_w'], params['l'], params['m_b'], params['d']
    J_J, J_K, n, J_r = params['J_J'], params['J_K'], params['n'], params['J_r']
    gra, b, f_v = params['gra'], params['b'], params['f_v']

    m11 = 1.5*m_w*r**2 + 0.25*m_b*r**2 + n**2*J_r + (((l*r)/d)**2)*m_b*np.sin(q[2])**2 + J_J*(r/d)**2
    m12 = 0.25*m_b*r**2 - ((l*r)/d)**2*m_b*np.sin(q[2])**2 - J_J*(r/d)**2
    m13 = 0.5*m_b*l*r*np.cos(q[2]) - n**2*J_r

    M_q = np.array([[m11, m12, m13], [m12, m11, m13], [m13, m13, m_b*l**2 + J_K + 2*n**2*J_r]])

    v11 = (2*l**2*r**2/d**2)*m_b*np.sin(q[2])*np.cos(q[2])*q_dot[2]*(q_dot[0]-q_dot[1]) - 0.5*m_b*l*r*np.sin(q[2])*q_dot[2]
    v21 = (2*l**2*r**2/d**2)*m_b*np.sin(q[2])*np.cos(q[2])*q_dot[2]*(q_dot[1]-q_dot[0]) - 0.5*m_b*l*r*np.sin(q[2])*q_dot[2]
    v31 = -(l**2*r**2/d**2)*m_b*np.sin(q[2])*np.cos(q[2])*((q_dot[0]-q_dot[1])**2) - m_b*l*gra*np.sin(q[2])
    v_q = np.array([v11, v21, v31])

    # FIX: Removed the @Pro typo that broke the execution
    coupling = np.array([[1, 0, -1], [0, 1, -1]])
    I = (u - params['K_E'] * n * (coupling @ q_dot)) / params['R']

    tau_a = params['K_T'] * n * np.dot(np.array([[1, 0], [0, 1], [-1, -1]]), I)
    tau_f = np.dot(np.array([[b+f_v, 0, -b], [0, b+f_v, -b], [-b, -b, 2*b]]), q_dot)

    q_ddot = np.linalg.solve(M_q, tau_a - tau_f - v_q)
    return np.concatenate([q_dot, q_ddot])

def run_simulation(event=None):
    kx = float(js.document.getElementById('kx').value)
    kv = float(js.document.getElementById('kv').value)
    kp = float(js.document.getElementById('kp').value)
    ki = float(js.document.getElementById('ki').value)
    kd = float(js.document.getElementById('kd').value)
    theta0 = float(js.document.getElementById('theta0').value)

    dt = 0.01
    sim_time = 10.0
    steps = int(sim_time / dt)

    r, l = params['r'], params['l']
    if r < l:
        max_tilt = np.arccos(-r / l) 
    else:
        max_tilt = np.pi / 2 

    x = np.zeros(7)
    x[2] = theta0  

    history_x_base = []
    history_theta3 = []

    for _ in range(steps):
        if abs(x[2]) >= max_tilt:
            x[2] = np.sign(x[2]) * max_tilt 
            x[3:6] = 0.0                    
            u = np.array([0.0, 0.0])        
        else:
            # Calculate position and velocity
            x_base = r * (x[0] + x[1]) / 2.0
            v_wheel = r * (x[3] + x[4]) / 2.0
            
            # CASCADED CONTROL LOGIC
            # 1. Outer Loop: Find target tilt to return to center
            # If x_base is positive (drifted right), target_tilt becomes negative (leans left)
            target_tilt = -(kx * x_base + kv * v_wheel)
            target_tilt = np.clip(target_tilt, -0.2, 0.2) # Prevent commanding a crash-level lean
            
            # 2. Inner Loop: Balance at target tilt
            error = target_tilt - x[2]
            voltage = -(kp * error + ki * x[6] - kd * x[5])
            voltage = np.clip(voltage, -12.0, 12.0)
            u = np.array([voltage, voltage])

            # RK4 Integration
            plant_state = x[:6]
            k1 = plant_dynamics(plant_state, u, params)
            k2 = plant_dynamics(plant_state + 0.5*dt*k1, u, params)
            k3 = plant_dynamics(plant_state + 0.5*dt*k2, u, params)
            k4 = plant_dynamics(plant_state + dt*k3, u, params)

            x[:6] += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x[6] += error * dt 

        history_x_base.append(r * (x[0] + x[1]) / 2.0)
        history_theta3.append(x[2])

    animate_robot(history_x_base, history_theta3)

current_anim_id = [None]
draw_proxy = [None]

def animate_robot(x_base, theta3):
    canvas = js.document.getElementById("robotCanvas")
    ctx = canvas.getContext("2d")

    if current_anim_id[0] is not None:
        js.cancelAnimationFrame(current_anim_id[0])
    if draw_proxy[0] is not None:
        draw_proxy[0].destroy()

    frame = [0]
    total_frames = len(x_base)

    def draw(timestamp=None):
        i = frame[0]
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        scale = 800
        origin_x = canvas.width / 2 
        ground_y = 300

        ctx.beginPath()
        ctx.moveTo(0, ground_y)
        ctx.lineTo(canvas.width, ground_y)
        ctx.strokeStyle = "#333"
        ctx.lineWidth = 2
        ctx.stroke()

        wheel_x = origin_x + x_base[i] * scale
        wheel_y = ground_y - (params['r'] * scale)

        tip_x = wheel_x + (params['l'] * scale) * np.sin(theta3[i])
        tip_y = wheel_y - (params['l'] * scale) * np.cos(theta3[i])

        ctx.beginPath()
        ctx.moveTo(wheel_x, wheel_y)
        ctx.lineTo(tip_x, tip_y)
        ctx.strokeStyle = "#007bff"
        ctx.lineWidth = 8
        ctx.lineCap = "round"
        ctx.stroke()

        ctx.beginPath()
        ctx.arc(wheel_x, wheel_y, params['r'] * scale, 0, 2 * np.pi)
        ctx.fillStyle = "#6c757d"
        ctx.fill()
        ctx.stroke()

        frame[0] = (frame[0] + 1) % total_frames
        current_anim_id[0] = js.requestAnimationFrame(draw_proxy[0])

    draw_proxy[0] = create_proxy(draw)
    draw()

btn = js.document.getElementById("run_btn")
btn.addEventListener("click", create_proxy(run_simulation))

def update_labels(e):
    js.document.getElementById('val_kx').innerText = js.document.getElementById('kx').value
    js.document.getElementById('val_kv').innerText = js.document.getElementById('kv').value
    js.document.getElementById('val_kp').innerText = js.document.getElementById('kp').value
    js.document.getElementById('val_ki').innerText = js.document.getElementById('ki').value
    js.document.getElementById('val_kd').innerText = js.document.getElementById('kd').value
    js.document.getElementById('val_theta').innerText = js.document.getElementById('theta0').value

for el in ['kx', 'kv', 'kp', 'ki', 'kd', 'theta0']:
    js.document.getElementById(el).addEventListener('input', create_proxy(update_labels))

run_simulation()
    </script>
</body>
</html>
