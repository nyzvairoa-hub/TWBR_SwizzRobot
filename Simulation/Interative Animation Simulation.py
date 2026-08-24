<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Real-Time Balancing Robot</title>
    <link rel="stylesheet" href="https://pyscript.net/releases/2024.1.1/core.css" />
    <script type="module" src="https://pyscript.net/releases/2024.1.1/core.js"></script>

    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f4f6f8; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .controls { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 300px; }
        .slider-group { margin-bottom: 12px; }
        .slider-group label { display: block; font-weight: bold; margin-bottom: 4px; font-size: 14px; }
        .slider-group input { width: 100%; }
        canvas { background: #ffffff; border: 1px solid #ccc; border-radius: 8px; }
        .instructions { background: #e7f3ff; border-left: 4px solid #007bff; padding: 10px; margin-bottom: 15px; font-size: 13px; }
    </style>
</head>
<body>

    <h2>Real-Time Remote Control Balancing Robot</h2>

    <div class="container">
        <!-- Controls UI -->
        <div class="controls">
            <div class="instructions">
                <strong>Controls:</strong> Use <b>Left / Right Arrow Keys</b> or <b>A / D Keys</b> to drive the robot!
            </div>

            <h3>Outer Loop (Position)</h3>
            <div class="slider-group">
                <label>Kx (Position): <span id="val_kx">0.15</span></label>
                <input type="range" id="kx" min="0" max="1" value="0.15" step="0.01">
            </div>
            <div class="slider-group">
                <label>Kv (Velocity): <span id="val_kv">0.2</span></label>
                <input type="range" id="kv" min="0" max="1" value="0.2" step="0.01">
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
            <button id="reset_btn" style="width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Reset Robot Position</button>
        </div>

        <!-- Canvas -->
        <div>
            <canvas id="robotCanvas" width="600" height="400"></canvas>
        </div>
    </div>

    <!-- Python Engine -->
    <script type="py" config='{"packages": ["numpy"]}'>
import numpy as np
import js
from pyodide.ffi import create_proxy

params = {
    'r': 0.0325, 'm_w': 0.053, 'l': 0.045, 'm_b': 0.8, 'd': 0.165,
    'J_J': 0.005518, 'J_K': 0.00135, 'n': 43.8, 'J_r': 0.00000274,
    'R': 4.019, 'L': 0.0000222, 'K_T': 0.005443, 'K_E': 0.005443,
    'b': 0.00000727, 'f_v': 0.000054, 'gra': 9.81
}

# State Variables [q0, q1, theta3, dq0, dq1, dtheta3, integral_error]
state = np.zeros(7)
x_target = 0.0
keys = {'left': False, 'right': False}

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

    coupling = np.array([[1, 0, -1], [0, 1, -1]])
    I = (u - params['K_E'] * n * (coupling @ q_dot)) / params['R']

    tau_a = params['K_T'] * n * np.dot(np.array([[1, 0], [0, 1], [-1, -1]]), I)
    tau_f = np.dot(np.array([[b+f_v, 0, -b], [0, b+f_v, -b], [-b, -b, 2*b]]), q_dot)

    q_ddot = np.linalg.solve(M_q, tau_a - tau_f - v_q)
    return np.concatenate([q_dot, q_ddot])

def reset_robot(event=None):
    global state, x_target
    state = np.zeros(7)
    state[2] = 0.1 # Small initial tilt push
    x_target = 0.0

# Key event handling
def on_keydown(e):
    if e.key == "ArrowLeft" or e.key.lower() == "a":
        keys['left'] = True
    elif e.key == "ArrowRight" or e.key.lower() == "d":
        keys['right'] = True

def on_keyup(e):
    if e.key == "ArrowLeft" or e.key.lower() == "a":
        keys['left'] = False
    elif e.key == "ArrowRight" or e.key.lower() == "d":
        keys['right'] = False

js.window.addEventListener("keydown", create_proxy(on_keydown))
js.window.addEventListener("keyup", create_proxy(on_keyup))
js.document.getElementById("reset_btn").addEventListener("click", create_proxy(reset_robot))

# Real-time simulation loop
def game_loop(timestamp=None):
    global state, x_target

    kx = float(js.document.getElementById('kx').value)
    kv = float(js.document.getElementById('kv').value)
    kp = float(js.document.getElementById('kp').value)
    ki = float(js.document.getElementById('ki').value)
    kd = float(js.document.getElementById('kd').value)

    # 1. Update Target Position Setpoint based on Keyboard Input (Controller Stick)
    drive_speed = 0.25 # m/s target driving speed
    cmd_velocity = 0.0

    if keys['left']:
        cmd_velocity = -drive_speed
    elif keys['right']:
        cmd_velocity = drive_speed

    # Frame step sub-integration (5 sub-steps of dt=0.003s for physical accuracy)
    sub_dt = 0.003
    r, l = params['r'], params['l']
    max_tilt = np.arccos(-r / l) if r < l else np.pi / 2

    for _ in range(5):
        # Update target position
        x_target += cmd_velocity * sub_dt

        if abs(state[2]) >= max_tilt:
            state[2] = np.sign(state[2]) * max_tilt
            state[3:6] = 0.0
            u = np.array([0.0, 0.0])
        else:
            x_base = r * (state[0] + state[1]) / 2.0
            v_wheel = r * (state[3] + state[4]) / 2.0

            # CASCADED CONTROL (Outer loop -> Target tilt angle)
            target_tilt = -(kx * (x_base - x_target) + kv * (v_wheel - cmd_velocity))
            target_tilt = np.clip(target_tilt, -0.15, 0.15) # Max allowable tilt angle (~8.5 deg)

            # INNER BALANCE LOOP
            error = target_tilt - state[2]
            voltage = -(kp * error + ki * state[6] - kd * state[5])
            voltage = np.clip(voltage, -12.0, 12.0)
            u = np.array([voltage, voltage])

            # RK4 Integration
            plant_state = state[:6]
            k1 = plant_dynamics(plant_state, u, params)
            k2 = plant_dynamics(plant_state + 0.5*sub_dt*k1, u, params)
            k3 = plant_dynamics(plant_state + 0.5*sub_dt*k2, u, params)
            k4 = plant_dynamics(plant_state + sub_dt*k3, u, params)

            state[:6] += (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            state[6] += error * sub_dt

    # 2. Render canvas frame
    render_robot()

    # Request next frame
    js.requestAnimationFrame(create_proxy(game_loop))

def render_robot():
    canvas = js.document.getElementById("robotCanvas")
    ctx = canvas.getContext("2d")
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    scale = 600
    origin_x = canvas.width / 2 
    ground_y = 300

    # Draw Ground
    ctx.beginPath()
    ctx.moveTo(0, ground_y)
    ctx.lineTo(canvas.width, ground_y)
    ctx.strokeStyle = "#333"
    ctx.lineWidth = 2
    ctx.stroke()

    x_base = params['r'] * (state[0] + state[1]) / 2.0
    wheel_x = origin_x + x_base * scale
    wheel_y = ground_y - (params['r'] * scale)

    # Draw target setpoint marker on ground
    target_px = origin_x + x_target * scale
    ctx.beginPath()
    ctx.arc(target_px, ground_y, 4, 0, 2*np.pi)
    ctx.fillStyle = "red"
    ctx.fill()

    # Draw body line
    tip_x = wheel_x + (params['l'] * scale) * np.sin(state[2])
    tip_y = wheel_y - (params['l'] * scale) * np.cos(state[2])

    ctx.beginPath()
    ctx.moveTo(wheel_x, wheel_y)
    ctx.lineTo(tip_x, tip_y)
    ctx.strokeStyle = "#007bff"
    ctx.lineWidth = 8
    ctx.lineCap = "round"
    ctx.stroke()

    # Draw wheel
    ctx.beginPath()
    ctx.arc(wheel_x, wheel_y, params['r'] * scale, 0, 2 * np.pi)
    ctx.fillStyle = "#6c757d"
    ctx.fill()
    ctx.stroke()

def update_labels(e):
    js.document.getElementById('val_kx').innerText = js.document.getElementById('kx').value
    js.document.getElementById('val_kv').innerText = js.document.getElementById('kv').value
    js.document.getElementById('val_kp').innerText = js.document.getElementById('kp').value
    js.document.getElementById('val_ki').innerText = js.document.getElementById('ki').value
    js.document.getElementById('val_kd').innerText = js.document.getElementById('kd').value

for el in ['kx', 'kv', 'kp', 'ki', 'kd']:
    js.document.getElementById(el).addEventListener('input', create_proxy(update_labels))

reset_robot()
game_loop()
    </script>
</body>
</html>
