import autograd.numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from tqdm import tqdm

@dataclass
class Body:
    name: str = None
    mass: float = None
    radius: float = None
    initial_position: np.ndarray = None
    initial_velocity: np.ndarray = None

    def __post_init__(self):
        self.initial_position = np.asarray(self.initial_position, dtype=float)
        self.initial_velocity = np.asarray(self.initial_velocity, dtype=float)

        if self.initial_position.shape != (3,):
            raise ValueError("initial_position must be a 3-element array")

        if self.initial_velocity.shape != (3,):
            raise ValueError("initial_velocity must be a 3-element array")

        self.x = [self.initial_position.copy()]
        self.v = [self.initial_velocity.copy()]
        
        self.t = []

        self.K = []
        self.U = []



def make_binary_system(m_1, m_2, a, e, i, phi):

    if m_1 == 0 and m_2 == 0:
        raise ValueError("At least one body must have non-zero mass")

    # --------------------------------------------------
    # Rotation matrix (define ONCE)
    # --------------------------------------------------
    cphi, sphi = np.cos(phi), np.sin(phi)
    ci, si = np.cos(i), np.sin(i)

    R = np.array([
        [ cphi,       -sphi,        0.0 ],
        [ ci*sphi,    ci*cphi,     -si  ],
        [ si*sphi,    si*cphi,      ci  ]
    ])

    # --------------------------------------------------
    # Pericenter state (relative orbit)
    # --------------------------------------------------
    r_peri = a * (1 - e)
    r_rel = np.array([r_peri, 0.0, 0.0])

    # --------------------------------------------------
    # TEST PARTICLE CASE
    # --------------------------------------------------
    if m_1 == 0 or m_2 == 0:

        M = max(m_1, m_2)

        # vis-viva at periapsis
        v_rel_mag = np.sqrt(M * (1 + e) / (a * (1 - e)))
        v_rel = np.array([0.0, v_rel_mag, 0.0])

        r_rel = R @ r_rel
        v_rel = R @ v_rel

        if m_1 > 0:
            body_massive = Body(
                mass=m_1,
                initial_position=np.zeros(3),
                initial_velocity=np.zeros(3)
            )
            body_test = Body(
                mass=0.0,
                initial_position=r_rel,
                initial_velocity=v_rel
            )
        else:
            body_massive = Body(
                mass=m_2,
                initial_position=np.zeros(3),
                initial_velocity=np.zeros(3)
            )
            body_test = Body(
                mass=0.0,
                initial_position=r_rel,
                initial_velocity=v_rel
            )

        return body_massive, body_test

    # --------------------------------------------------
    # TRUE BINARY CASE
    # --------------------------------------------------
    M = m_1 + m_2

    # relative speed at periapsis
    v_rel_mag = np.sqrt(M * (1 + e) / (a * (1 - e)))
    v_rel = np.array([0.0, v_rel_mag, 0.0])

    # Split into CM frame
    r_1 =  (m_2 / M) * r_rel
    r_2 = -(m_1 / M) * r_rel

    v_1 =  (m_2 / M) * v_rel
    v_2 = -(m_1 / M) * v_rel

    # Rotate
    r_1 = R @ r_1
    r_2 = R @ r_2
    v_1 = R @ v_1
    v_2 = R @ v_2

    body_1 = Body(mass=m_1, initial_position=r_1, initial_velocity=v_1)
    body_2 = Body(mass=m_2, initial_position=r_2, initial_velocity=v_2)

    return body_1, body_2
    
    
    
    
    
    
    
    

def compute_acc_and_jerk(b, bodies):
    """Compute acceleration and jerk on body b from bodies."""
    a = np.zeros(3)
    j = np.zeros(3)

    for bb in bodies:
        if bb is b:
            continue
        if bb.mass == 0:
            continue

        r = bb.x[-1] - b.x[-1]
        v = bb.v[-1] - b.v[-1]

        r2 = np.dot(r, r)
        r1 = np.sqrt(r2)
        r3 = r2 * r1

        a += bb.mass * r / r3
        j += bb.mass * (v / r3 - 3.0 * r * np.dot(r, v) / (r2 * r3))

    return a, j


def evolve_system(bodies, time, dt=None, adapt_dt=True, eta=0.03):
    """
    4th-order Hermite integrator with global adaptive timestep.
    If a body has mass zero, it's gravitational influence is null. It just moves in the potential
    """

    # Initial energies
    for b in bodies:
        if not b.K:
            b.K.append(0.5 * b.mass * np.linalg.norm(b.v[-1])**2)

        if not b.U:
            U = 0
            for bb in bodies:
                if bb is not b:
                    U += -bb.mass * b.mass / np.linalg.norm(b.x[-1] - bb.x[-1])
            b.U.append(U)
            
        if not b.t:
            b.t.append(0)

    if not adapt_dt and dt is None:
        raise ValueError("Provide dt when adapt_dt=False")

    t = 0.0
    eps = 1e-12

    # --- tqdm progress bar (time-based) ---
    pbar = tqdm(total=time, unit="time", desc="Evolving system",leave=True)

    while t < time:

        # --------------------------------------------------
        # STEP 0: choose global timestep
        # --------------------------------------------------
        if adapt_dt:
            dt = np.inf
            for b in bodies:
                a, j = compute_acc_and_jerk(b, bodies)
                a_norm = np.linalg.norm(a)
                j_norm = np.linalg.norm(j)

                if j_norm > 0:
                    dt_i = eta * np.sqrt(a_norm / (j_norm + eps))
                    dt = min(dt, dt_i)

        # Prevent overshoot of final time
        if t + dt > time:
            dt = time - t

        
        # --------------------------------------------------
        # STEP 1: predictor
        # --------------------------------------------------
        for b in bodies:
            b.a, b.j = compute_acc_and_jerk(b, bodies)

            b.x_pred = b.x[-1] + b.v[-1] * dt + 0.5 * b.a * dt**2 + (1/6) * b.j * dt**3
            b.v_pred = b.v[-1] + b.a * dt + 0.5 * b.j * dt**2

        # --------------------------------------------------
        # STEP 2: evaluate forces at predicted positions
        # --------------------------------------------------
        for b in bodies:
            a_new = np.zeros(3)
            j_new = np.zeros(3)

            for bb in bodies:
                if bb is b:
                    continue
                if bb.mass == 0:
                    continue

                r = bb.x_pred - b.x_pred
                v = bb.v_pred - b.v_pred

                r2 = np.dot(r, r)
                r1 = np.sqrt(r2)
                r3 = r2 * r1

                a_new += bb.mass * r / r3
                j_new += bb.mass * (v / r3 - 3 * r * np.dot(r, v) / (r2 * r3))

            b.a_new = a_new
            b.j_new = j_new

        # --------------------------------------------------
        # STEP 3: snap and crackle
        # --------------------------------------------------
        for b in bodies:
            b.s = (2 / dt**2) * (b.a_new - b.a - b.j * dt)
            b.c = (6 / dt**3) * (b.j_new - b.j - b.s * dt)

        # --------------------------------------------------
        # STEP 4: corrector
        # --------------------------------------------------
        for b in bodies:
            x_new = b.x_pred + (1 / 24) * b.s * dt**4
            v_new = b.v_pred + (1 / 6) * b.s * dt**3 + (1 / 24) * b.c * dt**4
        
            b.x.append(x_new)
            b.v.append(v_new)
            b.t.append(b.t[-1] + dt)


        
        # Energies
        for b in bodies:
            b.K.append(0.5 * b.mass * np.linalg.norm(b.v[-1])**2)

            U = 0
            for bb in bodies:
                if bb is not b:
                    U += -bb.mass * b.mass / np.linalg.norm(b.x[-1] - bb.x[-1])
            b.U.append(U)

        # Advance time + progress bar
        t += dt
        pbar.update(dt)

    pbar.close()




def evolve_in_fixed_potential(bodies, potential, field, hessian, time, dt=None, adapt_dt=True, eta=0.03):
    """
    4th-order Hermite integrator with global adaptive timestep.
    PARTICLES FEEL A FIXED POTENTIAL. COLLISIONLESS CODE
    """
    
    # Initial energies
    for b in bodies:
        if not b.K:
            b.K.append(0.5 * b.mass * np.linalg.norm(b.v[-1])**2)

        if not b.U:
            b.U.append(b.mass * potential(b.x[-1]))
        
        if not b.t:
            b.t.append(0)

    if not adapt_dt and dt is None:
        raise ValueError("Provide dt when adapt_dt=False")

    t = 0.0
    eps = 1e-12

    # --- tqdm progress bar (time-based) ---
    pbar = tqdm(total=time, unit="time", desc="Evolving system",leave=True)

    
    while t < time:

        # --------------------------------------------------
        # STEP 0: choose global timestep
        # --------------------------------------------------
        if adapt_dt:
            dt = np.inf
            for b in bodies:
                a_norm = np.linalg.norm(field(b.x[-1])) 
                j_norm = np.linalg.norm(-hessian(b.x[-1]) @ b.v[-1]) 

                if j_norm > 0:
                    dt_i = eta * np.sqrt(a_norm / (j_norm + eps))
                    dt = min(dt, dt_i)

        # Prevent overshoot of final time
        if t + dt > time:
            dt = time - t
        
        # --------------------------------------------------
        # STEP 1: predictor
        # --------------------------------------------------
        for b in bodies:
            b.a = field(b.x[-1]) 
            b.j = - hessian(b.x[-1]) @ b.v[-1] 

            b.x_pred = b.x[-1] + b.v[-1] * dt + 0.5 * b.a * dt**2 + (1/6) * b.j * dt**3
            b.v_pred = b.v[-1] + b.a * dt + 0.5 * b.j * dt**2

        # --------------------------------------------------
        # STEP 2: evaluate forces at predicted positions
        # --------------------------------------------------
        for b in bodies:
            b.a_new = field(b.x_pred) 
            b.j_new = - hessian(b.x_pred) @ b.v_pred 

        # --------------------------------------------------
        # STEP 3: snap and crackle
        # --------------------------------------------------
        for b in bodies:
            b.s = (2 / dt**2) * (b.a_new - b.a - b.j * dt)
            b.c = (6 / dt**3) * (b.j_new - b.j - b.s * dt)

        # --------------------------------------------------
        # STEP 4: corrector
        # --------------------------------------------------
        for b in bodies:
            x_new = b.x_pred + (1 / 24) * b.s * dt**4
            v_new = b.v_pred + (1 / 6) * b.s * dt**3 + (1 / 24) * b.c * dt**4
            
            b.x.append(x_new)
            b.v.append(v_new)
            b.t.append(b.t[-1] + dt)

        # Energies
        for b in bodies:
            b.K.append(0.5 * b.mass * np.linalg.norm(b.v[-1])**2)
            b.U.append(b.mass * potential(b.x[-1]))
        
        # Advance time + progress bar
        t += dt
        pbar.update(dt)

    pbar.close()

    
    
    
