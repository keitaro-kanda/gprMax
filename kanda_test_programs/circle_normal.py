import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


#* Defin function to make circle
def circle(center, radius, num_points=100):
    """
    Parameters:
    - center: tuple (cx, cy), center of the circle
    - radius: float, radius of the circle
    - num_points: int, number of points to make the circle

    Returns:
    - x: np.array, x-coordinates of the circle
    - y: np.array, y-coordinates of the circle
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x, y


#* Define function to make input wave
def initial_wave(position, angle, num):
    """
    Parameters:
    - position: tuple (x, y), starting point of the wave
    - angle: float, angle of the wave
    - length: float, length of the wave

    Returns:
    - x: np.array, x-coordinates of the wave
    - y: np.array, y-coordinates of the wave
    """

    theta = np.linspace(3/2*np.pi - angle, 3/2*np.pi, num, endpoint=True)
    x =  np.cos(theta)
    y =  np.sin(theta)

    vector_list = np.zeros((num, 2))
    vector_list[:, 0] = x
    vector_list[:, 1] = y

    return vector_list



#* Define function to calculate the refraction at the surface
def refraction_at_surface(source_position, incident, surface_position, n1, n2):
    """
    Parameters:
    - source_position: tuple (x, y), starting point of the wave
    - incident: np.array([dx, dy]), normalized direction vector
    - surface_position: float, y-coordinate of the surface

    Returns:
    - intersection: np.array([x, y]), point of intersection
    - None: if no intersection
    """
    # レイの方程式: p = p0 + t*d
    p0 = np.array(source_position)
    d = np.array(incident)

    # 平面の方程式: y = surface_position
    # 代入して解く: p0[1] + t*d[1] = surface_position
    t = (surface_position - p0[1]) / d[1]
    if t > 0:
        intersection = p0 + t * d
    else:
        intersection = None

    if incident[1] > 0:
        normal = np.array([0, -1])
    else:
        normal = np.array([0, 1])
    R, T = calc_vec(incident, normal, n1, n2)

    return intersection, t, R, T



#* Define function to calculate the intersection of a ray and a circle
def compute_circle_intersection(p0, d, center, radius, tol=1e-8):
    """
    Parameters:
    - p0: np.array([x, y]), point on the circle, starting point of the ray
    - d: np.array([dx, dy]), normalized direction vector
    - center: tuple (cx, cy), center of the circle
    - radius: float, radius of the circle
    - tol: float, tolerance for numerical errors

    Returns:
    - intersection: np.array([x, y]), point of intersection
    - None: if no intersection
    """
    # 円の方程式: ||p - center||^2 = radius^2
    # レイの方程式: p = p0 + t*d
    # 代入して解く: ||p0 + t*d - center||^2 = radius^2
    p0 = np.array(p0)
    d = np.array(d)
    center = np.array(center)
    
    # 定数項
    a = np.dot(d, d)
    b = 2 * np.dot(d, p0 - center)
    c = np.dot(p0 - center, p0 - center) - radius**2
    
    # 判別式
    discriminant = b**2 - 4*a*c
    
    if discriminant < -tol:
        # 交点なし
        return None
    elif np.abs(discriminant) < tol:
        # 接線の場合、交点が1つ
        t = -b / (2*a)
        if t > tol:
            return p0 + t*d
        else:
            return None
    else:
        # 交点が2つ
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2*a)
        t2 = (-b - sqrt_disc) / (2*a)
        # t=0 は開始点なので、t > tol の解を選ぶ
        t_values = [t for t in [t1, t2] if t > tol]
        if not t_values:
            return None
        # 最小の正の t を選ぶ
        t_min = min(t_values)
        return p0 + t_min * d, t_min



#* Defin function to calculate the reflection and refraction of light
def calc_vec(incident, normal, epsilon1, epsilon2):
    """
    Parameters:
    - incident: vector of incident light
    - normal: vector of normal to the surface
    - n1: refractive index of incident medium
    - n2: refractive index of transmission medium

    Returns:
    - R: vector of reflected light
    - T: vector of refracted light
    """


    #* Calculate the reflection vector
    R = incident - 2 * np.dot(incident, normal) * normal
    R = R / np.linalg.norm(R)

    #* Calculate the refraction vector
    n1 = np.sqrt(epsilon1)
    n2 = np.sqrt(epsilon2)
    eta = n2 / n1
    cos_theta_i = -np.dot(normal, incident)
    sin_theta_i_sq = 1 - cos_theta_i**2
    sin_theta_t_sq = (1 / eta**2) * sin_theta_i_sq
    if sin_theta_t_sq > 1:
        # Total reflection
        T = None
    else:
        T = (1 / eta) * incident - 1 / eta * (np.dot(incident, normal) + np.sqrt(eta**2 - 1 + np.dot(incident, normal)**2)) * normal
        T = T / np.linalg.norm(T)

    return R, T




#* Physical constants
c = 299792458 # speed of light in vacuum [m/s]

#* Make circle
center = (1.5, 3)
radius = 0.15
x_circle, y_circle = circle(center, radius)


#* Calculate the surface refraction
source_position = (1.5, 6)
incident_wave_vector = initial_wave(source_position, np.pi/180*3.988, 15)

surface = 5.0




#* Plot and calculate
fig, ax = plt.subplots(1, 3, figsize=(18,8), tight_layout=True)

#* Plot the circle
ax[0].plot(x_circle, y_circle, c='k')
ax[1].plot(x_circle, y_circle, c='k')
#* Plot the surface
ax[0].axhline(y=surface, color='k', linestyle='-')
ax[1].axhline(y=surface, color='k', linestyle='-')
#* Plot the source position
ax[0].scatter(source_position[0], source_position[1], c='r')
ax[1].scatter(source_position[0], source_position[1], c='r')



#* Calculate the ray paths
total_times = []


for i in range(incident_wave_vector.shape[0]):
    #* Source -> Surface
    surface_intersection, distance1, R1, T1 = refraction_at_surface(
        source_position, incident_wave_vector[i], surface, 1, 3)


    #* Surface -> Upper boundary of the circle
    if surface_intersection is None:
        continue
    else:
        intersection2, distance2 = compute_circle_intersection(
            p0=surface_intersection,
            d=T1,
            center=center,
            radius=radius
        )

    if not intersection2 is None:
        normal2 = intersection2 - center # to the outside
        normal2 = normal2 / np.linalg.norm(normal2)
        R2, T2 = calc_vec(T1, normal2, 3, 9)

    #* Upper boundary of the circle -> Lower boundary of the circle
    intersection3, distance3 = compute_circle_intersection(
        p0=intersection2,
        d=T2,
        center=center,
        radius=radius
    )

    if not intersection3 is None:
        normal3 = - (intersection3 - center) # minus: to the inside
        normal3 = normal3 / np.linalg.norm(normal3)
        R3, T3 = calc_vec(T2, normal3, 9, 3)



    #* Lower boundary of the circle -> Upper boundary of the circle
    intersection4, distance4 = compute_circle_intersection(
        p0=intersection3,
        d=R3,
        center=center,
        radius=radius
    )

    if not intersection4 is None:
        normal4 = - (intersection4 - center)
        normal4 = normal4 / np.linalg.norm(normal4)
        R4, T4 = calc_vec(R3, normal4, 9, 3)


    #* Upper boundary of the circle -> Surface
    intersection5, distance5, R5, T5 = refraction_at_surface(
        intersection4, T4, surface, 3, 1)
    

    #* At height of the source
    intersection6, distance6, R6, T6 = refraction_at_surface(
        intersection4, T5, 6, 1, 1)



    total_time = (distance1 + distance6) / c + (distance2 + distance5) / (c / np.sqrt(3)) + (distance3 + distance4) / (c /np.sqrt(9))
    total_times.append([intersection3[0], intersection3[1], total_time / 1e-9]) # ns


    #* 単位ベクトルの長さ調整
    incident_wave_vector[i] = incident_wave_vector[i]/10
    R1 = R1/10
    T1 = T1/10
    R2 = R2/10
    T2 = T2/10
    R3 = R3/10
    T3 = T3/10
    R4 = R4/10
    T4 = T4/10
    R5 = R5/10
    T5 = T5/10
    R6 = R6/10
    T6 = T6/10

    arrow_list = [R1, T1, R2, T2, R3, T3, R4, T4, R5, T5, R6, T6]
    position_list = [surface_intersection, intersection2, intersection3, intersection4, intersection5, intersection6]


    #* Source position
    ax[0].scatter(source_position[0], source_position[1], c='r')
    ax[0].arrow(source_position[0], source_position[1], incident_wave_vector[i][0], incident_wave_vector[i][1],
                head_width=0.01, head_length=0.01,  fc='r', ec='r')

    #* Plot the ray arrows
    for j in range(len(position_list)):
        if j >= 3:
            I = arrow_list[(j-1)*2]
        else:
            I = arrow_list[j*2-1]
        R = arrow_list[j*2]
        T = arrow_list[j*2+1]
        position = position_list[j]

        if i == 0 and j == 0:
            #* Incident wave
            ax[0].arrow(position[0] - I[0], position[1] - I[1], I[0], I[1],
                        head_width=0.01, head_length=0.01,  fc='r', ec='r', label='Incident wave', alpha=0.5)
            #* Reflected wave
            ax[0].arrow(position[0], position[1], R[0], R[1],
                        head_width=0.01, head_length=0.01,  fc='g', ec='g', alpha=0.5, label='Reflected wave')
            #* Refracted wave
            ax[0].arrow(position[0], position[1], T[0], T[1],
                        head_width=0.01, head_length=0.01,  fc='b', ec='b', label='Refracted wave')
        elif j ==5:
            #* Refracted wave
            ax[0].arrow(position[0], position[1], T[0], T[1],
                        head_width=0.01, head_length=0.01,  fc='b', ec='b')
        else:
            #* Incident wave
            ax[0].arrow(position[0] - I[0], position[1] - I[1], I[0], I[1],
                        head_width=0.01, head_length=0.01,  fc='r', ec='r', alpha=0.5)
            #* Reflected wave
            ax[0].arrow(position[0], position[1], R[0], R[1],
                        head_width=0.01, head_length=0.01,  fc='g', ec='g')
            #* Refracted wave
            ax[0].arrow(position[0], position[1], T[0], T[1],
                        head_width=0.01, head_length=0.01,  fc='b', ec='b')


    #* Calculate the scatter points
    scatters = []
    scatter1 = np.linspace(source_position, surface_intersection, 100)
    scatters.append(scatter1)
    scatter2 = np.linspace(surface_intersection, intersection2, 100)
    scatters.append(scatter2)
    scatter3 = np.linspace(intersection2, intersection3, 100)
    scatters.append(scatter3)
    scatter4 = np.linspace(intersection3, intersection4, 100)
    scatters.append(scatter4)
    scatter5 = np.linspace(intersection4, intersection5, 100)
    scatters.append(scatter5)
    scatter6 = np.linspace(intersection5, intersection6, 100)
    scatters.append(scatter6)

    scatters = np.array(scatters)

    #* Calculate the time at each scatter point
    scatters_flattened = scatters.reshape(-1, 2)  # Reshape for scatter plotting

    total_scatter_points = scatters_flattened.shape[0]  # Number of scatter points
    time_list = np.linspace(0, total_time, total_scatter_points)  # Time for each point
    ax[1].scatter(scatters_flattened[:, 0], scatters_flattened[:, 1], c=time_list, cmap='jet', s=1)



    """
    #* Plt scatter from source to surface and show the time in color
    ax[1].scatter(surface_intersection[0], surface_intersection[1], c=total_time, cmap='jet')
    #* plt scatter from surface to the circle and show the time in color
    ax[1].scatter(intersection2[0], intersection2[1], c=total_time, cmap='jet')
    #* plt scatter from the circle to the circle and show the time in color
    ax[1].scatter(intersection3[0], intersection3[1], c=total_time, cmap='jet')
    """

total_times = np.array(total_times)


ax[0].set_xlim(0, 3)
ax[0].set_ylim(2.5, 6.5)
ax[0].set_aspect('equal')
ax[0].grid(True)
ax[0].legend()
ax[0].set_title('Ray paths', fontsize=20)
ax[0].set_xlabel('x', fontsize=20)
ax[0].set_ylabel('y', fontsize=20)
ax[0].tick_params(labelsize=16)

ax[1].set_xlim(0, 3)
ax[1].set_ylim(2.5, 6.5)
ax[1].set_aspect('equal')
ax[1].grid(True)
ax[1].set_title('Propagation time', fontsize=20)
ax[1].set_xlabel('x', fontsize=20)
ax[1].set_ylabel('y', fontsize=20)
ax[1].tick_params(labelsize=16)


#* Add colorbar
norm = plt.cm.colors.Normalize(vmin=0, vmax=max(total_times[:, 2]))
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=1)
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation='vertical')
cbar.set_label('Time [ns]', fontsize=18)
cbar.ax.tick_params(labelsize=16)


#* Calculate and plot the phase lag
time_list = total_times[:, 2]
time_list = time_list[1:] # Remove the first element
time_list = time_list - time_list.min()
pulse_width = 4
phase_lag = time_list / pulse_width * 2 * np.pi

ax[2].plot(time_list, phase_lag)
ax[2].set_title('Phase lag', fontsize=20)
ax[2].set_xlabel('Lag of time [ns]', fontsize=20)
ax[2].set_ylabel('Phase [rad]', fontsize=20)
ax[2].tick_params(labelsize=16)
ax[2].grid(True)


plt.savefig('kanda_test_programs/circle_normal.png')
plt.show()

print(total_times)