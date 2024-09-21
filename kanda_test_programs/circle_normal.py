import matplotlib.pyplot as plt
import numpy as np


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
def initial_wave(position, angle, length=1):
    """
    Parameters:
    - position: tuple (x, y), starting point of the wave
    - angle: float, angle of the wave
    - length: float, length of the wave

    Returns:
    - x: np.array, x-coordinates of the wave
    - y: np.array, y-coordinates of the wave
    """

    theta = np.linspace(3/2*np.pi - angle, 3/2*np.pi, 10, endpoint=True)
    x = length * np.cos(theta)
    y = length * np.sin(theta)

    vector_list = np.zeros((10, 2))
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

    normal = np.array([0, 1])
    R, T = calc_vec(incident, normal, n1, n2)

    return np.array(intersection), t, R, T



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






#* Make circle
center = (1.5, 4)
radius = 0.15
x_circle, y_circle = circle(center, radius)


#* Calculate the surface refraction
source_position = (1.5, 6)
incident_wave_vector = initial_wave(source_position, np.pi/180*5, length=1)
print(incident_wave_vector)

surface = 5.0




#* Plot and calculate
fig, ax = plt.subplots(figsize=(8,8))

#* Plot the circle
ax.plot(x_circle, y_circle, c='k')
#* Plot the surface
ax.axhline(y=surface, color='k', linestyle='-')
#* Plot the source position
ax.scatter(source_position[0], source_position[1], c='r')



#* Calculate the ray paths
for i in range(incident_wave_vector.shape[0]):
    #* Source -> Surface
    surface_intersection, distance1, R1, T1 = refraction_at_surface(
        source_position, incident_wave_vector[i], surface, 1, 3)
    if i==0:
        print('surface_intersection: ', surface_intersection)
        print('T1: ', T1)
        print(' ')

    #* Surface -> Upper boundary of the circle
    intersection2, distance2 = compute_circle_intersection(
        p0=surface_intersection,
        d=T1,
        center=center,
        radius=radius
    )
    if i==0:
        print('intersection2: ', intersection2)

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
    if i==0:
        print('intercestion3: ', intersection3)

    if not intersection3 is None:
        normal3 = - (intersection3 - center) # minus: to the inside
        normal3 = normal3 / np.linalg.norm(normal3)
        R3, T3 = calc_vec(T2, normal3, 9, 3)

    total_distance = distance1 + distance2 + distance3


    #* 単位ベクトルの長さ調整
    incident_wave_vector[i] = incident_wave_vector[i]/10
    R1 = R1/10
    T1 = T1/10
    R2 = R2/10
    T2 = T2/10
    R3 = R3/10
    T3 = T3/10

    #* Plot the vectors
    if i == 0:
        #* Source
        ax.arrow(source_position[0], source_position[1], incident_wave_vector[i, 0], incident_wave_vector[i, 1],
                    head_width=0.01, head_length=0.01,  fc='r', ec='r', label='Incident wave')

        #* Surface
        ax.arrow(surface_intersection[0] - incident_wave_vector[i, 0], surface_intersection[1] - incident_wave_vector[i, 1],
                    incident_wave_vector[i, 0], incident_wave_vector[i, 1],
                    head_width=0.01, head_length=0.01,  fc='r', ec='r')
        ax.arrow(surface_intersection[0], surface_intersection[1], T1[0], T1[1],
                    head_width=0.01, head_length=0.01,  fc='b', ec='b', label='Refracted wave')
        ax.arrow(surface_intersection[0], surface_intersection[1], R1[0], R1[1],
                    head_width=0.01, head_length=0.01,  fc='g', ec='g', label='Reflected wave', alpha=0.5)

        #* Upper boundary of the circle
        ax.arrow(surface_intersection[0] - T1[0], surface_intersection[1] - T1[1], T1[0], T1[1],
                    head_width=0.01, head_length=0.01,  fc='r', ec='r')
        ax.arrow(intersection2[0], intersection2[1], T2[0], T2[1],
                    head_width=0.01, head_length=0.01,  fc='b', ec='b')
        ax.arrow(intersection2[0], intersection2[1], R2[0], R2[1],
                    head_width=0.01, head_length=0.01,  fc='g', ec='g', alpha=0.5)

        #* Lower boundary of the circle
        ax.arrow(intersection3[0] - T2[0], intersection3[1] - T2[1], T2[0], T2[1],
                    head_width=0.01, head_length=0.01,  fc='r', ec='r')
        ax.arrow(intersection3[0], intersection3[1], T3[0], T3[1],
                    head_width=0.01, head_length=0.01,  fc='b', ec='b')
        ax.arrow(intersection3[0], intersection3[1], R3[0], R3[1],
                    head_width=0.01, head_length=0.01,  fc='g', ec='g')
    else:
        #* Source
        ax.arrow(source_position[0], source_position[1], incident_wave_vector[i, 0], incident_wave_vector[i, 1],
                    head_width=0.01, head_length=0.01,  fc='r', ec='r')

        #* Surface
        ax.arrow(surface_intersection[0] - incident_wave_vector[i, 0], surface_intersection[1] - incident_wave_vector[i, 1],
                    incident_wave_vector[i, 0], incident_wave_vector[i, 1],
                    head_width=0.01, head_length=0.01,  fc='r', ec='r')
        ax.arrow(surface_intersection[0], surface_intersection[1], T1[0], T1[1],
                    head_width=0.01, head_length=0.01,  fc='b', ec='b')
        ax.arrow(surface_intersection[0], surface_intersection[1], R1[0], R1[1],
                    head_width=0.01, head_length=0.01,  fc='g', ec='g', alpha=0.5)

        #* Upper boundary of the circle
        ax.arrow(intersection2[0] - T1[0], intersection2[1] - T1[1], T1[0], T1[1],
                    head_width=0.01, head_length=0.01,  fc='r', ec='r')
        ax.arrow(intersection2[0], intersection2[1], T2[0], T2[1],
                    head_width=0.01, head_length=0.01,  fc='b', ec='b')
        ax.arrow(intersection2[0], intersection2[1], R2[0], R2[1],
                    head_width=0.01, head_length=0.01,  fc='g', ec='g', alpha=0.5)

        #* Lower boundary of the circle
        ax.arrow(intersection3[0] - T2[0], intersection3[1] - T2[1], T2[0], T2[1],
                    head_width=0.01, head_length=0.01,  fc='r', ec='r')
        ax.arrow(intersection3[0], intersection3[1], T3[0], T3[1],
                    head_width=0.01, head_length=0.01,  fc='b', ec='b')
        ax.arrow(intersection3[0], intersection3[1], R3[0], R3[1],
                    head_width=0.01, head_length=0.01,  fc='g', ec='g')


ax.set_xlim(0, 3)
ax.set_ylim(3.5, 6.5)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True)
ax.legend()

plt.savefig('kanda_test_programs/circle_normal.png')
plt.show()