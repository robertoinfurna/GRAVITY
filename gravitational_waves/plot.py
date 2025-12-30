def plot(binary,tstop,tstart=0,pbar=False):

    tstart = time_IU(tstart)
    
    R1, R2, separation, h_plus, h_cross, t = binary.evolve(tstop,pbar,save_ram=False)

    start_idx = np.searchsorted(t, tstart,side='left')
    t = np.array(t[start_idx:])
    R1, R2 = np.array(R1[start_idx:]), np.array(R2[start_idx:])
    separation = np.array(separation[start_idx:])
    h_plus, h_cross = h_plus[start_idx:], h_cross[start_idx:]

    f_kepler = frequency_kepler(binary.M,separation)                        #iu
    t_gw, f_gw = extract_frequencies(np.array(t), np.array(h_plus))
    f_of_t = interp1d(t_gw, f_gw, kind='linear', fill_value="extrapolate")  #f_of_t = instantaneous_freq(t,h_plus)
    f_gw_fun = f_of_t(t)

    #create figure and set axes
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 3) 
    
    # 3D Plot on the left
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.set_box_aspect([1, 1, 1])  
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
   
    ## 3d plot ##

    R1, R2, separation = R1/binary.d_factor, R2/binary.d_factor, separation/binary.d_factor   #rescale

    ax1.scatter(R1[:,0],R1[:,1],R1[:,2],s=0.1, color='blue') 
    ax1.scatter(R1[-1,0],R1[-1,1],R1[-1,2],s=5,label='%.1e $M_\odot$'%binary.M1, color='blue') 
    ax1.scatter(R2[:,0],R2[:,1],R2[:,2],s=0.1,color='red') 
    ax1.scatter(R2[-1,0],R2[-1,1],R2[-1,2],s=5,label='%.1e $M_\odot$'%binary.M2, color='red') 

    R1_max_dist = np.max(np.sqrt(R1[:, 0]**2 + R1[:, 1]**2 + R1[:, 2]**2))
    R2_max_dist = np.max(np.sqrt(R2[:, 0]**2 + R2[:, 1]**2 + R2[:, 2]**2))
    d_lim = 1.1 * max(R1_max_dist, R2_max_dist)
        
    ax1.set_xlim(-d_lim,d_lim) ; ax1.set_ylim(-d_lim,d_lim) ; ax1.set_zlim(-d_lim,d_lim)
    ax1.set_xlabel("x %s" %binary.d_label) ; ax1.ticklabel_format(style='sci', axis='both', scilimits=(0,0)) 

    ax1.quiver(0, 0, 0, (d_lim/2)*binary.n_obs[0], (d_lim/2)*binary.n_obs[1], (d_lim/2)*binary.n_obs[2], color='black', arrow_length_ratio=0.3)
    ax1.legend() 

    
    ## h_plus h_cross plot ##

    t = t/binary.t_factor
    f_kepler = f_kepler*s_IU
    f_gw_fun = f_gw_fun*s_IU
    
    ax2.plot(t,h_plus,label="$h_+$",c='purple')
    ax2.plot(t,h_cross,label="$h_\\times$",c='blue')
    
    ax2.set_xlim(t[0],t[-1])
    ax2.set_xlabel("time %s" %binary.t_label)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), borderaxespad=0.)
    
    ## separation and frequency plot ##

    # Create a second y-axis that shares the same x-axis
    ax3_ = ax3.twinx()  
    ax3.plot(t,separation, c='black',label='separation')
    ax3_.plot(t, f_gw_fun, color='tab:green', label='GW frequency (principal harmonic)', linestyle='-.', alpha = 0.8)
    ax3.axhline(y=binary.R_isco/binary.d_factor,linestyle=':',c='black') 
    ax3_.axhline(y=binary.f_ringdown*s_IU,linestyle='--',c='tab:green')   
    ax3_.text(t[-1], binary.f_ringdown * s_IU, "$f_{ringdown}$", color='tab:green', verticalalignment='bottom')
    ax3.text(t[-1], binary.R_isco / binary.d_factor, "$R_{ISCO}$", color='black', verticalalignment='bottom')
    
    ax3.set_xlim(t[0],t[-1])
    ax3.set_ylim(0,1.5*max(separation))
    ax3.set_xlabel("time %s" %binary.t_label)
    ax3.set_ylabel("separation %s" %binary.d_label) 
    
    ax3_.set_ylabel('Frequency [Hz]')
    ax3_.set_ylim(min(f_kepler),1.1*binary.f_ringdown*s_IU) 
    ax3_.set_yscale("log")
    
    lines, labels = ax3.get_legend_handles_labels()  
    lines2, labels2 = ax3_.get_legend_handles_labels()  
    ax3_.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1) 

    plt.tight_layout()
    plt.show()




def animation(binary, tstop, deltaT, tstart=0,pbar=False):
    
    R1, R2, separation, h_plus, h_cross, t = binary.evolve(tstop,pbar,save_ram=False)

    tstart = time_IU(tstart)
    tstop = time_IU(tstop)
    deltaT = time_IU(deltaT)
    
    start_idx = np.searchsorted(t, tstart,side='left')
    t = np.array(t[start_idx:])
    R1, R2 = np.array(R1[start_idx:]), np.array(R2[start_idx:])
    separation = np.array(separation[start_idx:])
    h_plus, h_cross = h_plus[start_idx:], h_cross[start_idx:]

    R1, R2 = np.array(R1) / binary.d_factor, np.array(R2) / binary.d_factor
    separation = np.array(separation) / binary.d_factor
    t = np.array(t) / binary.t_factor
    dt = np.diff(t)

    e2_prime, e1_prime = construct_transverse_basis(binary.n_obs)

    R1_p, R2_p = [], []
    for i in range(len(R1)):
        x1_prime, y1_prime, x2_prime, y2_prime = np.dot(R1[i], e1_prime), np.dot(R1[i], e2_prime), np.dot(R2[i], e1_prime), np.dot(R2[i], e2_prime)
        R1_p.append([x1_prime, y1_prime])
        R2_p.append([x2_prime, y2_prime])
    R1_p = np.array(R1_p); R2_p = np.array(R2_p)

    # create figure and set axes
    fig = plt.figure(figsize=(14, 4))
    plt.suptitle("$d$ = %.2e parsec, $i$ = %.2f" % (binary.d / parsec_IU, binary.i))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ## projection plot 1
    R1_max_dist = np.max(np.sqrt(R1_p[:, 0]**2 + R1_p[:, 1]**2))
    R2_max_dist = np.max(np.sqrt(R2_p[:, 0]**2 + R2_p[:, 1]**2))
    d_lim = 1.1 * max(R1_max_dist, R2_max_dist)
    ax1.set(xlim=(-d_lim, d_lim),ylim=(-d_lim, d_lim),xlabel="x %s" % binary.d_label)
    ax1.set_xlabel("x %s" % binary.d_label); ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.set_aspect('equal', adjustable='box') 
    ax1.grid()
    # Initialize scatter plots
    scatter_R1_p = ax1.scatter([], [], s=6, label='%.1e $M_\odot$' % binary.M1, color='blue')
    scatter_R2_p = ax1.scatter([], [], s=6, label='$%.1e M_\odot$' % binary.M2, color='red')
    line_R1_p, = ax1.plot([], [], color='blue', alpha=0.5)  # Trajectory of R1
    line_R2_p, = ax1.plot([], [], color='red', alpha=0.5)   # Trajectory of R2        

    ## initialize GW waveform plot 2
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(min(min(h_plus),min(h_cross)),max(max(h_plus),max(h_cross)))
    #ax2.set_ylim(min(np.nanmin(h_plus), np.nanmin(h_cross)), max(np.nanmax(h_plus), np.nanmax(h_cross)))
    ax2.set_xlabel("time %s" % binary.t_label)
    ax2.set_ylabel("strain h")
    h_plus_plot, = ax2.plot([], [], color='purple', label="$h_+$")
    h_cross_plot, = ax2.plot([], [], color='blue', label="$h_\\times$")
    ax2.grid()
    
    ## ring of test masses plot 3
    # Initialize scatter plots
    angles_ring = np.linspace(0, 2 * np.pi - np.pi / 8, 30)
    x_ring = np.cos(angles_ring)
    y_ring = np.sin(angles_ring)

    scatter_ring = ax3.scatter(x_ring, y_ring, s=6)  
    stretch_scale = 1.5*max(max(np.abs(h_plus)),max(np.abs(h_cross)))
    ax3.set(xlim=(-2, 2),ylim=(-2, 2), xlabel="$e_1$", ylabel="$e_2$")
    ticks = np.arange(-2, 2, 0.5)  # Define tick spacing (from -2 to 2 with step of 1)
    ax3.set_xticks(ticks)
    ax3.set_yticks(ticks)
    ax3.set_aspect('equal', adjustable='box') 
    ax3.set_title("$\\times$ %.2e" %stretch_scale)
    ax3.grid()

    def update(frame):
        scatter_R1_p.set_offsets(R1_p[frame, [0, 1]])
        scatter_R2_p.set_offsets(R2_p[frame, [0, 1]])
        line_R1_p.set_data(R1_p[:frame + 1, 0], R1_p[:frame + 1, 1])
        line_R2_p.set_data(R2_p[:frame + 1, 0], R2_p[:frame + 1, 1])
        ax1.set_title("t = %.2e %s" % (t[frame], binary.t_units))

        h_plus_plot.set_data(t[:frame], h_plus[:frame])
        h_cross_plot.set_data(t[:frame], h_cross[:frame])

        if frame == 0:
            scatter_ring.set_offsets(np.column_stack((x_ring, y_ring)))
            scatter_ring.set_offsets(np.column_stack((x_ring, y_ring)))

        delta_x = x_ring + 1/2 * (h_plus[frame] * x_ring + h_cross[frame] * y_ring) / stretch_scale
        delta_y = y_ring + 1/2 * (-h_plus[frame] * y_ring + h_cross[frame] * x_ring) / stretch_scale

        scatter_ring.set_offsets(np.column_stack((delta_x, delta_y)))

        return scatter_R1_p, scatter_R2_p, line_R1_p, line_R2_p, h_plus_plot, h_cross_plot, scatter_ring

    # frames spaced by deltaT
    frame_index = [0]
    k = frame_index[-1]
    while t[k] < t[-1]:
        i = k
        while i < len(t) - 1 and t[i] - t[k] < deltaT / binary.t_factor:
            i += 1
        k = i
        frame_index.append(i)

    ax1.legend(loc='upper left') #, bbox_to_anchor=(0.3, 0.3), borderaxespad=0.)
    ax2.legend(loc='upper left') #, bbox_to_anchor=(0.3, 0.3), borderaxespad=0.)
    # Create the animation
    ani = FuncAnimation(fig, update, frames=frame_index, blit=True, repeat=False)  # int(len(R1)/speed)
    display(HTML(ani.to_jshtml()))

    plt.tight_layout()
    fig.subplots_adjust()
    plt.close()
