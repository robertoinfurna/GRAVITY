# Function to calculate the second time derivative of the quadrupole moment (Q̈)
def double_dot_qmm(M1, M2, r1, v1, a1, r2, v2, a2):
    """
    Calculates the second time derivative of the mass quadrupole moment tensor.

    Parameters:
    - M1, M2: Masses of the two bodies
    - r1, r2: Position vectors of the two bodies
    - v1, v2: Velocity vectors of the two bodies
    - a1, a2: Acceleration vectors of the two bodies
    
    Returns:
    - double_dot_q: The second time derivative of the quadrupole moment tensor
    """
    double_dot_q = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # Diagonal elements
            if i == j:
                double_dot_q[i][j] = (M1*(2*v1[i]**2 + 2*r1[i]*a1[j]) 
                    + M2*(2*v2[i]**2 + 2*r2[i]*a2[j]))
            # Off-diagonal elements
            else:
                double_dot_q[i][j] = (M1*(a1[i]*r1[j] + 2*v1[i]*v1[j] + a1[j]*r1[i])  
                    + M2*(a2[i]*r2[j] + 2*v2[i]*v2[j] + a2[j]*r2[i])) 
    return double_dot_q


# Function to calculate the third time derivative of the quadrupole moment (Q⃛)
def triple_dot_qmm(M1, M2, r1, v1, a1, J1, r2, v2, a2, J2):
    """
    Calculates the third time derivative of the mass quadrupole moment tensor.

    Parameters:
    - M1, M2: Masses of the two bodies
    - r1, r2: Position vectors of the two bodies
    - v1, v2: Velocity vectors of the two bodies
    - a1, a2: Acceleration vectors of the two bodies
    - J1, J2: Third time derivatives of the position vectors (jerk vectors)
    
    Returns:
    - triple_dot_q: The third time derivative of the quadrupole moment tensor
    """
    triple_dot_q = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # Diagonal elements
            if i == j:
                triple_dot_q[i][j] = (M1*(6*v1[i]*a1[j] + 2*r1[i]*J1[j]) 
                    + M2*(6*v2[i]*a2[j] + 2*r2[i]*J2[j])) 
            # Off-diagonal elements
            else:
                triple_dot_q[i][j] = (M1*(J1[i]*r1[j] + 3*a1[i]*v1[j] + 3*a1[j]*v1[i] + J1[j]*r1[i])
                    + M2*(J2[i]*r2[j] + 3*a2[i]*v2[j] + 3*a2[j]*v2[i] + J2[j]*r2[i]))
    return triple_dot_q


# Function to find a new orthonormal basis in the plane perpendicular to the observation direction (k)
def construct_transverse_basis(k):
    """
    Constructs a new orthonormal basis (e1_prime, e2_prime) in the plane 
    perpendicular to the observation direction (k).

    Parameters:
    - k: Observation direction vector
    
    Returns:
    - e1_prime, e2_prime: New transverse basis vectors
    """
    # Find a vector not collinear with k (arbitrary choice)
    if np.allclose(k, [1, 0, 0]):
        v = np.array([0, 1, 0])
    else:
        v = np.array([1, 0, 0])
    
    # First orthonormal vector e1_prime (perpendicular to k)
    e1_prime = np.cross(k, v)
    e1_prime /= np.linalg.norm(e1_prime)
    
    # Second orthonormal vector e2_prime (also in the transverse plane)
    e2_prime = np.cross(k, e1_prime)
    e2_prime /= np.linalg.norm(e2_prime)
    
    return e1_prime, e2_prime


# Function to rotate the tensor into the new transverse basis
def rotate_to_new_frame(A, e1_prime, e2_prime):
    """
    Rotates the tensor A into the new frame defined by the transverse basis vectors.

    Parameters:
    - A: The tensor to be rotated
    - e1_prime, e2_prime: Transverse basis vectors
    
    Returns:
    - A_rotated: Rotated tensor
    """
    # Construct the rotation matrix from the new basis
    R = np.array([e1_prime, e2_prime, np.cross(e1_prime, e2_prime)]).T
    
    # Rotate the tensor using the rotation matrix
    A_rotated = R.T @ A @ R
    
    return A_rotated


# Function to project the quadrupole tensor into the transverse-traceless (TT) gauge
def TT_projector(v, q):  
    """
    Projects the quadrupole tensor into the Transverse-Traceless (TT) gauge.

    Parameters:
    - v: Observation direction vector
    - q: Quadrupole moment tensor
    
    Returns:
    - q_tt: Projected tensor in TT gauge
    """
    # Construct the TT projector
    P = [
        [1 - v[0]**2, -v[0]*v[1], -v[0]*v[2]],
        [-v[1]*v[0], 1 - v[1]**2, -v[1]*v[2]],
        [-v[2]*v[0], -v[2]*v[1], 1 - v[2]**2]
    ]
    q_tt = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    q_tt[i][j] += (P[i][k]*P[j][l]-1/2*P[i][j]*P[k][l])*q[k][l]
    return q_tt


# Function to calculate the gravitational wave (GW) strains
def polarized_strains(M1, M2, r1, v1, a1, r2, v2, a2, n_obs, d):
    """
    Calculates the gravitational wave strains (h_plus, h_cross) for a binary system.

    Parameters:
    - M1, M2: Masses of the two bodies
    - r1, r2: Position vectors of the two bodies
    - v1, v2: Velocity vectors of the two bodies
    - a1, a2: Acceleration vectors of the two bodies
    - n_obs: Observation direction vector
    - d: Distance to the source
    
    Returns:
    - h_plus, h_cross: Gravitational wave strains
    """
    # Calculate the second time derivative of the quadrupole moment
    ddot_q = double_dot_qmm(M1, M2, r1, v1, a1, r2, v2, a2)
    
    # Project into the Transverse-Traceless gauge
    ddot_q_tt = TT_projector(n_obs, ddot_q)             
        
    # Construct a transverse basis (e1_prime, e2_prime) orthogonal to k
    e1_prime, e2_prime = construct_transverse_basis(n_obs)

    # Rotate the matrix into the new frame
    ddot_q_pol_basis = rotate_to_new_frame(ddot_q_tt, e1_prime, e2_prime)

    # Extract the 2x2 part from the rotated matrix (upper-left 2x2 block)
    ddot_q_pol_basis = ddot_q_pol_basis[:2, :2]
    h_plus = 2/d * ddot_q_pol_basis[0][0]  
    h_cross = 2/d * ddot_q_pol_basis[0][1]  

    return h_plus, h_cross
    
    
def step(M1, M2, r1, r2, v1, v2, n_obs, d):     
    """
    Advances the positions and velocities of a binary system by one time step,
    while calculating the gravitational wave (GW) strains and luminosity.

    Parameters:
    - M1, M2: Masses of the two bodies
    - r1, r2: Position vectors of the two bodies
    - v1, v2: Velocity vectors of the two bodies
    - n_obs: Observation direction vector
    - d: Distance to the source
    
    Returns:
    - Updated positions and velocities (r1, r2, v1, v2)
    - Gravitational wave strains (h_plus, h_cross)
    - Time step used (dt)
    """
    
    # Calculate the separation between the two bodies
    r = np.linalg.norm(r1 - r2)
   
    # Calculate the accelerations due to gravitational attraction
    a1 = M2/r**3 * (r2 - r1)
    a2 = M1/r**3 * (r1 - r2)

    # Calculate the third time derivatives of position (jerks)
    # This includes relativistic corrections to account for varying accelerations
    J1 = M2/r**3 * (v2 - v1 - 3 * (r2 - r1) * np.dot(v2 - v1, r2 - r1) / r**2)
    J2 = M1/r**3 * (v1 - v2 - 3 * (r1 - r2) * np.dot(v1 - v2, r1 - r2) / r**2)

    # Calculate the gravitational wave strains (h_plus, h_cross)
    h_plus, h_cross = polarized_strains(M1, M2, r1, v1, a1, r2, v2, a2, n_obs, d)    
   
    ## Calculate Gravitational Wave Luminosity ##        
    # Use the third time derivative of the quadrupole moment tensor
    tdot_q = triple_dot_qmm(M1, M2, r1, v1, a1, J2, r2, v2, a2, J2)                  
    L_gw = 1/5 * np.sum(tdot_q**2)  # GW luminosity as per the quadrupole formula

    ## Determine the Time Step (dt) ##
    # The time step is chosen to ensure numerical stability
    v = np.linalg.norm(v1 - v2)
    a = np.linalg.norm(a1 - a2)
    dt = 0.01 * min(r/v, np.sqrt(r/a))  
     
    ## Update Positions and Velocities ##
    # Using a third-order Taylor expansion to account for the jerks (J1, J2)
    # Also includes a correction term for the energy loss due to GW emission

    # Update positions with third-order corrections and GW energy loss
    r1 += v1*dt + 1/2 * a1 * dt**2 + 1/6 * J1 * dt**3  -  (r1 - r2)/r * 2*r**2/(M1*M2) * L_gw * dt  
    r2 += v2*dt + 1/2 * a2 * dt**2 + 1/6 * J2 * dt**3  -  (r2 - r1)/r * 2*r**2/(M1*M2) * L_gw * dt

    # Update velocities with second-order corrections
    v1 += a1 * dt + 1/2 * J1 * dt**2 
    v2 += a2 * dt + 1/2 * J2 * dt**2   

    return r1, r2, v1, v2, h_plus, h_cross, dt



