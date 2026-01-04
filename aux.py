import autograd.numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from tqdm import tqdm



def plot_3d_orbits(bodies, 
                   t_max, 
                   year_IU,
                   dt_step_marker=None,
                   max_range=None,
                   figsize=(10,8), 
                   colors=None, 
                   marker_size=None):
    """
    Plot 3D orbits of multiple bodies up to a given time t_max.
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # --- Color handling (FIXED) ---

    if colors is None:
        # Option 1: qualitative colormap (good for up to 10 bodies)
        cmap = plt.cm.get_cmap('tab10', len(bodies))
        colors = [cmap(i) for i in range(len(bodies))]
        
    all_positions = []

    for i, b in enumerate(bodies):
        times = np.asarray(b.t)
        positions = np.asarray(b.x)

        idx_max = np.searchsorted(times, t_max, side='right')
        positions = positions[:idx_max]

        if len(positions) == 0:
            continue

        all_positions.append(positions)

        # Orbit line
        ax.plot(
            positions[:,0],
            positions[:,1],
            positions[:,2],
            lw=1,
            color=colors[i]
        )

        # --- Markers along orbit every dt_step_marker ---
        if dt_step_marker is not None:
            n_markers = int(t_max / dt_step_marker)
            marker_indices = [np.digitize(k*dt_step_marker,times)-1 for k in range(n_markers)]
            
            ax.scatter(
                positions[marker_indices,0],
                positions[marker_indices,1],
                positions[marker_indices,2],
                marker='x',
                s=5,
                color=colors[i],
                alpha=0.6
            )

        
        # Marker size
        if marker_size is None:
            size = max(10, 100 * np.sqrt(b.mass))
        else:
            size = marker_size

        # Final position marker
        ax.scatter(
            positions[-1,0],
            positions[-1,1],
            positions[-1,2],
            s=size,
            color=colors[i],
            marker='o',
            label=b.name or f"Body {i}"
        )

    # Labels and legend
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    ax.legend()

    # Equal aspect ratio
    all_positions = np.concatenate(all_positions)
    mid = [0,0,0]
    if max_range is None:
        max_range = (all_positions.max(axis=0) - all_positions.min(axis=0)).max() / 2
        mid = (all_positions.max(axis=0) + all_positions.min(axis=0)) / 2
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    if t_max > year_IU:
        ax.set_title(f"Time: {t_max/year_IU:.2f} years")
    else:
        ax.set_title(f"Time: {t_max/day_IU:.2f} days")

    plt.show()
    
    
    
def plot_projected_orbits(bodies, 
                   t_max, 
                   year_IU,
                   time=None,
                   planets=[],
                   observer='Earth',
                   unwrap=False,
                   dt_step_marker=None,
                   max_range=None,
                   figsize=(10,8), 
                   colors=None, 
                   marker_size=None):

    if time: unwrap = False

    fig,ax = plt.subplots(figsize=(10,10))


    j = 0
    selected_bodies = []
    for i,b in enumerate(bodies): 
        if b.name == observer:
            j = i
        if b.name in planets:
            selected_bodies.append(b)
    if len(selected_bodies) == 0: 
        selected_bodies = bodies.copy()  # make a copy
        selected_bodies.pop(j)           # remove element at index j

    idx_min = 0
    if time: t_max = time
    idx_max = np.searchsorted(bodies[0].t, t_max, side='right')
    if time: idx_min = idx_max - 1
    
    obs_x = bodies[j].x
    for b in selected_bodies:
        theta_array = []
        phi_array = []
        for i in range(idx_min,idx_max):
            vector = b.x[i] - obs_x[i]
            r = np.linalg.norm(vector)
            theta = np.arcsin(vector[2]/r) * 180/np.pi
            phi = np.arctan2(vector[1], vector[0]) + np.pi
            theta_array.append(theta)
            phi_array.append(phi)

        phi_array = np.array(phi_array)
        if unwrap:
            phi_array = np.unwrap(phi_array)                  # unwrap across discontinuities
        phi_array *= 180/np.pi  # to degrees

        s = 20 if time else 0.5
        ax.scatter(phi_array,theta_array,s=s)
        if not unwrap:
            ax.set_xlim(0,360)
        ax.set_ylim(-90,90)
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('DEC [deg]')
        if t_max > year_IU:
            ax.set_title(f"Time: {t_max/year_IU:.2f} years")
        else:
            ax.set_title(f"Time: {t_max/day_IU:.2f} days")
    plt.show()

    
