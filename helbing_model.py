# -*- coding: utf-8 -*-
"""
Model based on the Helbing model of "Simulation dynamical features of 
escape panic" to simulate fish going through a hole in a wall.
"""


import numpy as np
import sys
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class Fish():
    """
    Fish that wants to swim to an objective out of the box.

    Attributes
    ----------
    size : float
        Radius of the sphere representing the size of the fish
    mass : float
        Mass of the fish
    position : numpy.array
        Position of the fish as an array [x, y]
    speed : numpy.array
        Speed of the fish as an array [vx, vy]
    desired_speed : float
        Instant speed that the fish wants to have
    objective : numpy.array
        Actual position of the objective that the fish want to go to
    first_objective : numpy.array
        First objective of the fish, once attained, change to the second
    second_objective : numpy.array
        Second objective of the fish, once the first is attained
    force : numpy.array
        Force and social force on the fish as an array [fx, fy]
    char_time : float
        Characteristic time for a fish to get to his desired speed
    color : str
        Color of the fish, changes when the objective changes.
    """
    def __init__(self, s, m, pos, tau, des_v, fobj, sobj):
        self.size = s
        self.mass = m
        self.position = pos
        self.speed = np.array([0., 0.])
        self.desired_speed = des_v
        self.objective = fobj
        self.first_objective = fobj
        self.second_objective = sobj
        self.force = np.array([0., 0.])
        self.char_time = tau
        self.color = 'b'
    
    
    def getCoords(self):
        " Returns the position of the fish"
        return self.position
    
    def getColor(self):
        " Returns the color of the fish"
        return self.color
    
    def getSize(self):
        " Returns the radius of the fish"
        return self.size
    
    def objective_speed(self):
        """
        Returns the objective speed of the fish, which is what speed the fish
        wants to go, depending on his own position, the position of the
        objective and his desired speed. That way, the closer the objective
        is the smaller the objective speed is, even though his desired speed
        is a constant.

        Returns
        -------
        numpy.array
            Objective speed of the fish

        """
        return (self.desired_speed*
                (self.objective - self.position)/
                    np.linalg.norm((self.objective - self.position)))
    
    
    def get_neighbors(self, fish_list, N = 8):
        """
        Finds the N neighbors of a fish

        Parameters
        ----------
        fish_list : list
            List of every fish.
        N : int, optional
            Number of neighbors for a fish. The default is 5.

        Returns
        -------
        n_list : list
            List of the N fish considered as the neighbors.
        
        Notes
        ----------
        See also get_positions()

        """
        neigh = NearestNeighbors(n_neighbors = N)
        pos_list, exc_list = get_positions(self, fish_list)
        neigh.fit(pos_list) #fit the NN to the values
        closest = neigh.kneighbors([self.position], 
                                   return_distance = False) #just take indices
        n_list = []
        for i in closest[0] :
            n_list.append(exc_list[i])
        return n_list
    
    
    def force_friction(self, other_fish, A, B, k, kappa):
        """
        Returns the force of friction of a fish onto another.

        Parameters
        ----------
        other_fish : Fish
            Fish with who we measure the force of friction.
        A : float
            Repulsion constant.
        B : float
            Repulsion constant.
        k : float
            Body force constant.
        kappa : float
            Sliding friction force constant.

        Returns
        -------
        numpy.array
            Returns the force of friction of other_fish on self, as [fx, fy].
        
        Notes
        ----------
        See also gfunc()

        """
        d = np.linalg.norm((self.position - other_fish.position)) #distance
        n = np.array((self.position - other_fish.position) / d) #normalized
        # vector going from the other fish to self
        sum_r = self.size + other_fish.size #sum of the radius of both fish
        t = np.array([-n[1], n[0]]) #normalized tangential vector
        tvd = np.dot((self.speed - other_fish.speed), t) #tangential velocity
        # difference
        rep_f = A * np.exp((sum_r - d)/B) * n #repulsion force
        bod_f = k * gfunc(sum_r - d, d, sum_r) * n #body force
        sli_f = kappa * gfunc(sum_r - d, d, sum_r) * tvd * t #sliding
        #friction force
        return rep_f + bod_f + sli_f
    
    
    def force_wall(self, wall, A, B, k, kappa):
        """
        Returns the force between the fish and the walls.

        Parameters
        ----------
        wall : list
            Position of the upper and down coordinates of the hole in the 
            wall, as [[xu, yu], [xd, yd]]
        A : float
            Repulsion constant.
        B : float
            Repulsion constant.
        k : float
            Body force constant.
        kappa : float
            Sliding friction force constant.

        Returns
        -------
        numpy.array
            Returns the force of friction of the wall on self, as [fx, fy].
        
        Notes
        ----------
        See also gfunc()

        """
        #find the closest position of the wall to the fish
        if self.position[1] > wall[0][1] or self.position[1] < wall[1][1] :
            if self.position[0] - wall[0][0] < 0:
                wall_pos = [wall[0][0], self.position[1]]
            #Make a thickness to the walls, as thick as the hole_size
            elif self.position[0] - wall[0][0] < wall[0][1] + wall[1][1]:
                wall_pos = [self.position[0] + 0.1, self.position[1]]
            else :
                wall_pos = [wall[0][0], self.position[1]]
        elif self.position[1] >= 0 :
            wall_pos = wall[0]
        elif self.position[1] < 0 :
            wall_pos = wall[1]
        d = np.linalg.norm((self.position - wall_pos)) #see force_friction
        n = np.array((self.position - wall_pos) / d)
        sum_r = self.size
        t = np.array([-n[1], n[0]])
        tvd = np.dot(self.speed, t)
        rep_f = A * np.exp((sum_r - d)/B) * n
        bod_f = k * gfunc(sum_r - d, d, sum_r) * n
        sli_f = kappa * gfunc(sum_r - d, d, sum_r) * tvd * t
        return rep_f + bod_f - sli_f
    
    
    def total_force(self, fish_list, wall, A, B, k, kappa):
        """
        Returns the total force exerted on a fish from other fish, the wall
        and the urge to go to the objective.

        Parameters
        ----------
        other_fish : Fish
            Fish with who we measure the force of friction.
        wall : list
            Position of the upper and down coordinates of the hole in the 
            wall, as [[xu, yu], [xd, yd]]
        A : float
            Repulsion constant.
        B : float
            Repulsion constant.
        k : float
            Body force constant.
        kappa : float
            Sliding friction force constant.

        Returns
        -------
        numpy.array
            Total force on the fish as an array [fx, fy].
        
        Notes
        ----------
        See also get_neighbors(), force_friction(), and force_wall()

        """
        ff = np.array([0.,0.])
        neighbors = self.get_neighbors(fish_list) #get every neighbors
        for j in range(len(neighbors)): #add friction_force for each one
            ff += self.force_friction(neighbors[j], A, B, k, kappa)
        fw = self.force_wall(wall, A, B, k, kappa)
        fs = self.mass * (self.objective_speed() - self.speed)/self.char_time
        return fs + ff + fw
            
    
    def update_force(self, fish_list, wall, A, B, k, kappa):
        " Update the force on the fish, see total_force()"
        self.force = self.total_force(fish_list, wall, A, B, k, kappa)
        
        
    def update_status(self, dt):
        " Update the position and speed of the fish, see verlet_alg()"
        self.position, self.speed = verlet_alg(self.position, self.speed,
                                               self.force, self.mass, dt)
    
    # def update_objective(self, d = 0.2):
    #     " Update the position of the actual objective of the fish"
    #     if self.objective == self.first_objective:
    #         if (self.position[0] > self.objective[0]) or (
    #                 angle_between(-(self.objective - self.position),
    #                             [-1, 0]) < (np.pi/12)) or (
    #                             np.linalg.norm((self.position -
    #                             self.first_objective[0])) < d ) :
    #             self.objective = self.second_objective
    #             self.color = 'orange' #change the color with the objective
    #     elif self.objective == self.second_objective:
    #         if (angle_between(-(self.objective - self.position), [-1, 0]) >
    #             (np.pi/12)) and (np.linalg.norm((
    #                 self.position - self.first_objective)) > d) and (
    #                     self.position[0] < self.first_objective[0]):
    #             self.objective = self.first_objective
    #             self.color = 'b' #change the color with the objective
    
    def update_objective(self, d = 0.35):
        " Update the position of the actual objective of the fish"
        if self.objective == self.first_objective:
            if ((np.linalg.norm((self.position - self.first_objective)) < d) 
                or (self.position[0] > self.first_objective[0])) :
                self.objective = self.second_objective
                self.color = 'orange' #change the color with the objective
                
    
    def evolveTimeStep(self, fish_list, wall, A, B, k, kappa, dt):
        " Make one timestep with an update of the characteristics of the fish"
        self.update_force(fish_list, wall, A, B, k, kappa)
        self.update_status(dt)
        self.update_objective()


def verlet_first(pos, vel, force, m, delta_t):
    "First part of the Verlet algorithm, see verlet_alg"
    r = np.array(pos)
    v = np.array(vel)
    a = np.array(force/m)
    return r + delta_t * v + 0.5 * a * delta_t**2, v + 0.5 * delta_t * a


def verlet_second(pos, vel, force, m, delta_t):
    "Second part of the Verlet algorithm, see verlet_alg"
    r = np.array(pos)
    v = np.array(vel)
    a = np.array(force/m)
    return r, v + 0.5 * delta_t * a


def verlet_alg(pos, vel, force, m, delta_t):
    """
    Verlet algorithm for the evolution of the position of a particle. Need to
    be on two steps because we need "half-time" values. Useful because energy
    is conserved (with small oscillations around the value).

    Parameters
    ----------
    pos : numpy.array
        Initial position as an array [x, y].
    vel : numpy.array
        Initial speed as an array [vx, vy].
    force : numpy.array
        Initial force as an array [fx, fy].
    m : float
        Mass of the particle.
    delta_t : float
        Time passed between the initial and final time.

    Returns
    -------
    r : numpy.array
        Updated position as an array [x, y].
    v : numpy.array
        Updated speed as an array [vx, vy].

    """
    r, v = verlet_first(pos, vel, force, m, delta_t)
    r, v = verlet_second(r, v, force, m, delta_t)
    return r, v

def gfunc(x, d, r):
    "g function from Helbing, useful for contact forces."
    if d > r :
        return 0
    else :
        return x


def get_positions(target, fish_list):
    """
    We want to get the positions (and the list of corresponding fish) of 
    every fish except the target fish.

    Parameters
    ----------
    target : Fish
        Target fish.
    fish_list : list
        List of every fish.

    Returns
    -------
    pos_list : list
        List of positions of every fish except the target fish.
    exc_list : list
        List of every fish except the target fish.

    """
    pos_list = []
    exc_list = []
    for fish in fish_list:
        if fish == target :
            pass
        else :
            pos_list.append([fish.position[0], fish.position[1]])
            exc_list.append(fish)
    return pos_list, exc_list


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def crystal_init_pos(xmin, xmax, ymin, ymax, n):
    """
    Initialized ordered positions of n**2 object in a box between xmin and 
    xmax and ymin and ymax such that they are not on top of each other. We use
    n such that we know that there is an integer as a square root of the 
    number of object to get the same number of full rows and columns.

    Parameters
    ----------
    xmin : float
        Minimal x value.
    xmax : float
        Maximal x value.
    ymin : float
        Minimal y value.
    ymax : float
        Maximal y value.
    n : int
        Square root of the number of fish.

    Returns
    -------
    xp : list
        x coordinates of every ordered object.
    yp : list
        y coordinates of every ordered object.

    """
    xp = []
    yp = []
    if n == 1: #avoid division by 0, put the only object in the middle
        xp.append((xmax - xmin)/2 + xmin)
        yp.append((ymax - ymin)/2 + ymin)
    else :
        for i in range(n): #variation of y then x, so columns after columns
            for j in range(n): 
                xp.append(xmin + (i/(n-1))*(xmax-xmin))
                yp.append(ymin + (j/(n-1))*(ymax-ymin))
    return xp, yp


def no_overlap(x, y, s, others):
    """
    Verifies that there is no overlap between the target and other objects,
    they're all circles but with different sizes.

    Parameters
    ----------
    x : float
        x-position of the target.
    y : float
        y-position of the target.
    s : float
        Radius of the target.
    others : list
        Other object listed as [[x1, y1, s1], [x2, y2, s2], ...].

    Returns
    -------
    bool
        True if there is no overlap, False otherwise.

    """
    for o in others :
        if np.linalg.norm(np.array([x,y]) - np.array(o[:2])) < s + o[2]:
            return False
        else :
            pass
    return True


def random_init_pos(C, R, size, others):
    """
    Returns a random initial position within a circle of center C and radius
    R of an circle with a radius size, with no overlap with other objects.

    Parameters
    ----------
    C : list
        Position of the center of the circle, as [xc, yc].
    R : float
        Radius of the circle.
    size : float
        Radius of the randomly initialized object.
    others : list
        Other object listed as [[x1, y1, s1], [x2, y2, s2], ...].

    Returns
    -------
    list
        Position and raidus of the object as [x, y, size].
    
    Notes
    ----------
    See also no_overlap().

    """
    pos = []
    while len(pos) < 1:
        r = R * np.sqrt(np.random.random()) #random polar coordinates
        theta = np.random.random() * 2 * np.pi
        x = C[0] + r * np.cos(theta) #swith to cartesian coordinates
        y = C[1] + r * np.sin(theta)
        if no_overlap(x, y, size, others) == True:
            pos.append([x, y, size]) #add if there is no overlap with others
        else :
            pass
    return pos[0]


def no_overlap_random_pos(C, R, size_list, N):
    """
    Returns random initial position of circle object of various radii within
    a circle of center C and radius R.

    Parameters
    ----------
    C : list
        Position of the center of the circle, as [xc, yc].
    R : float
        Radius of the circle.
    size_list : list
        List of the radii of the objects we want to randomly initialize.
    N : int
        Number of objects.
    
    Raises
    ------
    ValueError
        If there is not as much radii as the number of objects wanted.

    Returns
    -------
    numpy.array
        Array of the position of every object, as [[x1, y1], ..., [xN, yN]].
        
    Notes
    ----------
    See also random_init_pos().

    """
    pos_list = []
    if len(size_list) != N :
        raise ValueError('Not as much radii as object')
    for i in range(N):
        pos_list.append(random_init_pos(C, R, size_list[i], pos_list))
    return np.array(pos_list)[:,:2]


def make_fish(L, N, rmin, rmax, C, R, m, tau,
              desired_speed, first_obj, second_obj, init = 'random'):
    """
    Makes a list of fish with various parameters

    Parameters
    ----------
    L : float
        Size of the aquarium.
    N : int
        Number of fish.
    rmin : float
        Minimal radius.
    rmax : float
        Maximal radius.
    C : list
        Position of the center of the circle, as [xc, yc].
    R : float
        Radius of the circle.
    m : float
        Mass of the fish.
    tau : float
        Characteristic time of acceleration.
    desired_speed : float
        Desired maximal speed of the fish.
    first_obj : list
        Position of the first objective, as [x, y].
    second_obj : list
        Position of the second objective, as [x, y].
    init : str, optional
        How the initialization of the positions is made. Can either be
        'crystal', where they are ordered as a square with a fixed distance
        between each fish, or with 'random' as random positions of the fish 
        within a circle. The default is 'random'.

    Raises
    ------
    ValueError
        If the init option is not used correctly, with either 'random' or
        'crystal'.

    Returns
    -------
    fish_list : list
        List of fish.
    
    Notes
    ----------
    See also Fish(), no_overlap_random_pos(), and crystal_init_pos().

    """
    if init == 'random':
        fish_list = []
        size_list = []
        for i in range(N):
            size_list.append(np.random.uniform(rmin, rmax))
        pos_list = no_overlap_random_pos(C, R, size_list, N)
        for i in range(N):
            fish_list.append(Fish(size_list[i], m, pos_list[i],
                                  tau, desired_speed, first_obj, second_obj))
    elif init == 'crystal':
        xx, yy = crystal_init_pos(-2*L + L/3, L/2 - L/3,
                                  -L + L/3, L - L/3, n )
        for i in range(N):
            radius = np.random.random(rmin, rmax)
            fish_list.append(Fish(radius, m, [xx[i], yy[i]],
                                  tau, ds, fobj, sobj))
    else :
        raise ValueError('Wrong init option')
    return fish_list



def helbing_constant():
    "Return constant from the original article Helbing et. al, 2000"
    m = 80. #mass
    A = 2. * 10**3 #amplitude of long-range repulsion 
    B = 0.08 #characteristic distance for long-range repulsion
    ds = 0.8 #desired speed
    k = 1.2 * 10**5 #body force constant
    kappa = 2.4 * 10**5 #friction force constant
    rmin = 0.25 #minimum radius
    rmax = 0.35 #maximum radius
    tau = 0.5 #characteristic time for acceleration
    return m, A, B, ds, k, kappa, rmin, rmax, tau


def adaptative_timestep(fish_list, default_dt = 0.01, 
                        v_changelimit = 0.01, tms_mul = 0.95):
    """
    Returns the timestep adapted to the biggest velocity change, such that
    nothing due to a too big of a timestep occurs, such as described for
    Helbing et al., 2000.

    Parameters
    ----------
    fish_list : list
        List of every fish.
    default_dt : float, optional
        Initial timestep. The default is 0.01.
    v_changelimit : float, optional
        Maximum velocity change in a timestep. The default is 0.01.
    tms_mul : float, optional
        Multiplier used to reduce the timestep. The default is 0.95.

    Returns
    -------
    t : float
        Adaptative timestep.

    """
    max_acc = 0
    for fish in fish_list : #find the highest acceleration
        if max_acc < np.linalg.norm(fish.force)/fish.mass :
            max_acc = np.linalg.norm(fish.force)/fish.mass
    t = default_dt
    while t * max_acc > v_changelimit: #while velocity change superior to limit
        t *= tms_mul #reduce the timestep to reduce the velocity change
    return t
    
    
L = 10

m, A, B, ds, k, kappa, rmin, rmax, tau = helbing_constant()
ds = 1.2
kappa = 4.8 * 10**5

uw = [L/2, 2 * rmin]
dw = [L/2, -(2 * rmin)]

#Place first objective on the hole
fobj = [(1/2) * L + (uw[1] - dw[1])/2 , 0] 
#Place second objective further away so there is no jam near the exit
sobj = [(11/4) * L, 0]


#Number of individual is N, with an integer square root n
n = 7
N = n**2

#Position and radius of the circle of random initial position with no overlap
C = [-L/3, 0]
R = L/2

for i in range(10):     
    fish_list = make_fish(L, N, rmin, rmax, C, R, m, tau, ds, fobj, sobj)
    
    
    #Put this way, we get timestep, fish id, x, y, radius
    tmax = 150
    hist = []
    f_hist = []
    ts = 0
    timer = 0.01
    while ts < tmax:
        f_count = 0
        
        for fish in fish_list :
            fish.update_force(fish_list, [uw, dw], A, B, k, kappa)
        
        dt = adaptative_timestep(fish_list, default_dt = 0.01,
                                  v_changelimit = 0.01)
        ts += dt
        
        for fish in fish_list :
            fish.update_status(dt)
            fish.update_objective()
            hist.append([ts, f_count, fish.getCoords()[0], fish.getCoords()[1],
                          fish.getSize(), fish.getColor()])
            if ts >= timer :
                f_hist.append([ts, f_count, fish.getCoords()[0],
                                fish.getCoords()[1], fish.getSize(),
                                fish.getColor()])
            f_count += 1
        
        if ts >= timer :
            timer += 0.01
            
        sys.stdout.write("\r{0}%".format(round((ts-dt)/tmax*100,2)))
        sys.stdout.flush()
    
    #We create the corresponding pandas DataFrame
    df = pd.DataFrame(hist, columns =['Time', 'FishID', 'X', 'Y', 'R', 'C'])
    
    #We save it into a csv file
    title = 'history_49_ds12_ka48_{}.csv'.format(i)
    df.to_csv(title, index=False)
    
    #We create the corresponding pandas DataFrame
    df2 = pd.DataFrame(f_hist, columns =['Time', 'FishID', 'X', 'Y', 'R', 'C'])
    
    #We save it into a csv file
    title = 'fixed_history_49_ds12_ka48_{}.csv'.format(i)
    df2.to_csv(title, index=False)