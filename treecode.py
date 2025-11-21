# Author: Roberto Infurna
# Description: This script simulates the evolution of a galaxy system over time using an N-body simulation with 
#              the Barnes-Hut algorithm for efficient gravitational force computation.
#              The script includes classes and functions to represent stars, an octree data structure (Tree),
#              and an evolution function (evolve) that calculates the motion of stars using the Leap-Frog method.


import numpy as np

class Star:
    def __init__(self, m, x, v): # x to be passed as array
        self.m = m
        self.x = x                                                           # position vector in cartesian coordinates
        self.v = v                                                           # velocity vector in cartesian coordinates
        self.r = np.linalg.norm(x)                                           # distance from the origin
        self.v_magn = np.linalg.norm(v)                                      # velocity magnitude
        self.kinetic_energy = 0.5 * self.m * self.v_magn**2                  # kinetic energy in internal units
        self.angular_momentum = self.m * np.cross(x, v)                      # angular momentum in internal units
        self.angular_momentum_magn = np.linalg.norm(self.angular_momentum)   # angular momentum magnitude in internal units


class Tree:
    def __init__(self, particles, n_max=1, idx=()):
        """
        Initialize the Tree structure.
        
        Parameters:
        - particles: List of particles (each with mass `m` and position `x` as a numpy array).
        - n_max: Maximum number of particles in a node before it splits into subnodes.
        - idx: Tuple representing the hierarchical index of the node.
        """

        self.n = len(particles)  # Number of particles in the node
        self.idx = idx              # Current index
        self.M = np.sum([particle.m for particle in particles])  # Total mass
        self.cm = np.sum([particle.m * particle.x for particle in particles], axis=0) / self.M  # Center of mass
        self.L = 2 * max([np.linalg.norm(particle.x - self.cm) for particle in particles]) if self.n > 1 else 0  # Size of the region
        
        if self.n <= n_max: # it's a leaf
            return

        else:   

            # Build subnodes
            particles_subnodes = [[] for _ in range(8)]  # Octree structure
            for particle in particles:
                # Create a binary index (0 to 7) based on the particle's position relative to the CM
                sub_idx = (
                    (1 if particle.x[0] >= self.cm[0] else 0) +
                    (2 if particle.x[1] >= self.cm[1] else 0) +
                    (4 if particle.x[2] >= self.cm[2] else 0)
                )
                particles_subnodes[sub_idx].append(particle)
    
            # Recursively initialize subnodes
            self.subnodes = [
                Tree(particles_subnodes[i], n_max, idx + (i,)) if particles_subnodes[i] else None
                for i in range(8)
            ]

    def __getitem__(self, idx): 
        """
        Access a specific node in the tree using its index path (tuple).
        """
        
        current_node = self
    
        # Traverse the tree according to the index path
        for i in idx:
            if not hasattr(current_node, 'subnodes') or current_node.subnodes is None:
                return current_node  # Return the current node if it is a leaf
            if not (0 <= i < 8) or current_node.subnodes[i] is None:
                return None  # Invalid index or empty subnode
            current_node = current_node.subnodes[i]  # Move to the next subnode
        return current_node

    def print_tree_details(self,node = ()): #None
        """
        Recursively print details of all nodes in the tree.        
        Parameters:
        - node: The current node being processed. Defaults to the root node.
        """
        
        # Print index, size (number of particles), mass, and the size of the region for the current node
        print(f"Index: {node.idx}, Size: {node.n}, Mass: {node.M}, Region size: {node.L}")
        
        # Check if the node has subnodes and recursively print their details
        if hasattr(node, 'subnodes'): 
            for subnode in node.subnodes:
                if subnode is not None:
                    subnode.print_tree_details(subnode)  # Recursively print subnodes
                    
                  
### ======================================================================================================================================================== ###

def grav_force_star_node(star,F,tree,current_index,eps,theta):
    """
    Computes the gravitational force on a star from one node of the tree and adds it to the force vector F.

    This function uses the Barnes-Hut algorithm to efficiently compute gravitational forces by approximating distant
    particles as a single node, rather than calculating forces between all pairs of particles.

    Parameters:
    - star: The star (particle) for which the gravitational force is being calculated.
    - F: The total gravitational force vector on the star. This will be updated during the computation.
    - tree: The tree structure representing the hierarchical spatial decomposition of particles.
    - current_index: A tuple representing the path to the current node in the tree, used to access the node.
    - eps: A small value used to avoid division by zero and stabilize force calculations (softening parameter).
    - theta: The threshold used to determine when a node can be treated as a point mass for efficiency.

    The method recursively traverses the tree and computes the gravitational force on the given star based on the tree's nodes.
    The force is computed in two stages:
    1. If the node is sufficiently far from the star (as determined by the ratio of the node's size to the distance between 
       the node's center of mass and the star), the node is treated as a single point mass, and the gravitational force 
       from that node is calculated.
    2. If the node is not sufficiently distant, the function recursively computes the gravitational forces from each of 
       the node's subnodes.

    This method optimizes the computation by reducing the number of pairwise force calculations, making it more efficient 
    for large systems of particles.
    """

    node = tree[current_index]  # Access the current node of the tree
    
    if node is None:  # If the node does not exist, return
        return
            
    d = np.linalg.norm(star.x - node.cm)
    L = node.L
    M = node.M

    if d == 0:     # don't compute gravitational force of the star with itself (also it makes the loop infinite)
        return
       
    if L/d < theta:
            
        F += M * star.m / (d**2 + eps**2) * (node.cm - star.x)/d
    
    else: 
                       
        for i in range(8):
                    
                new_index = current_index + (i,)
                
                grav_force_star_node(star,F,tree,new_index,eps,theta)

from copy import deepcopy
from tqdm.notebook import tqdm
def evolve(galaxy, tstop, dt, dtout, eps, theta=1, orbit_indexes=None):
    """
    Evolves the galaxy system over time using an N-body simulation with a Barnes-Hut algorithm
    for efficient gravitational force computation.

    This function simulates the motion of stars in the galaxy under gravitational interactions,
    using the Leap-Frog integration method. The gravitational forces are computed using an octree
    data structure to optimize force calculation. The system is evolved until the specified time stop (tstop).

    Parameters:
    - galaxy (Galaxy): The galaxy object containing the system of stars. It must include a list of "Star" objects "galaxy.system", and a list of times "galaxy.t"
    - tstop (float): The stopping time for the simulation.
    - dt (float): The time step for the integration.
    - dtout (float): The interval at which the galaxy's state should be saved to the history.
    - eps (float): Softening parameter used in the gravitational force calculation to avoid singularities.
    - theta (float): The Barnes-Hut parameter that controls the threshold for approximating distant nodes.
    - orbit_indexes (list, optional): List of star indexes to track orbits for. If None, no orbits are tracked.

    Returns:
    None (modifies the galaxy object in place).

    This function uses a `Tree` data structure to represent the spatial division of the galaxy for fast gravitational 
    force calculations. The Leap-Frog method is used to update the positions and velocities of the stars.

    The simulation progresses through discrete time steps, and the state of the galaxy is saved periodically 
    based on the `dtout` parameter.
    """


    if orbit_indexes:
        galaxy.orbit_indexes = orbit_indexes
        galaxy.orbits = [[galaxy.system[0][i].x] for i in orbit_indexes]
        galaxy.t_orbits = []
    
    frame = deepcopy(galaxy.system[-1]) 
    t = galaxy.t[-1]

    with tqdm(total=tstop, desc="Evolving system") as pbar:
        while t < tstop:
        
            tree = Tree(frame)  # build the tree at each interaction

            for i,star in enumerate(frame):    # Cycle over all N stars
                
                F = np.zeros(3)
                
                current_index = ()
                
                grav_force_star_node(star,F,tree,current_index,eps,theta)

                a = F / star.m
              
                # Leap frog algorithm
                v_i_12 = star.v + dt / 2 * a
                r_i_1 = star.x + dt * v_i_12
                v_i_1 = v_i_12 + dt / 2 * a

                frame[i] = Star(star.m, r_i_1, v_i_1)

                if orbit_indexes:
                    for k in range(len(orbit_indexes)):
                        if i == orbit_indexes[k]: 
                            galaxy.orbits[k].append(r_i_1)

            if orbit_indexes:
                galaxy.t_orbits.append(t)
                
            if t - galaxy.t[-1] >= dtout:
                galaxy.system.append(deepcopy(frame))
                galaxy.t.append(t)

            t += dt 
    
            pbar.update(dt)
            
            
            
            


