"""Monte Carlo Simulation of magnetic spectrometer for electron beams

Requirements
"""

################################################################################
#                               Imports                                        #
################################################################################

import xlrd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
from scipy.interpolate import interp2d
import pickle

mpl.rc('font', family='serif', size=18)

################################################################################
#                               Constants                                      #
################################################################################

# import speed of light, electron mass, and elementary charge
from scipy.constants import c, m_e, e 
mc2 = m_e * c**2 # mc^2
CONVERT_TO_UNITS = {
                    'Joules':1.60217657*10**(-19), # eV -> Joules
                    'eV':1., # eV -> eV
                    'm':1., # m -> m
                    'mm':10**3, # m -> mm
                    }


################################################################################
#                               Parameters                                     #
################################################################################

PARALLEL = True # Run program on all available processors 
PARTICLES = 10**4 # Number of particles in simulation
BLOCK_NUMBER = 10
PINHOLE_DIAMETER = 2*10**(-3) # m
CROSS_POINT = 10 *10**(-3) # m   - distance from pinhole to the wire cross point
MAGNETIC_MAPPING_FILE = './magnetic_mapping.xlsx' # location of magnet excel file  
MAGNET_WIDTH = 24 * 10**(-3) # m
MAGNET_LENGTH = 24 * 10**(-3) # m
dt = 1.*10**-13 # Time step (s) 
EMIN = 30 # Minimum energy in simulation (keV)
EMAX = 9000 # maximum energy in simulation (keV)



################################################################################



class Electron(object):

    _m_0 = m_e # 9.10938291e-31 kg  rest mass of electron
    _q = e # 1.602176565e-19 Coulombs

    def __init__(self, energy=10**5, position=(0,0,0), direction=(0,1,0)):

        self.set_position(np.array(position, dtype='float')) # (x,y,z) m
        self.set_direction(np.array(direction, dtype='float')) # (vx_hat, vy_hat, vz_hat) m/s
        self._energy = float(energy) # eV

    def gamma(self):
        """Lorentz factor
        (1-v^2/c^2)^(-1/2)
        """ 
        return 1./np.sqrt(1.-self.speed**2/c**2)

    def energy(self, units='eV'):
        """Return the energy of the particle with appropriate units
        """
        return self._energy*CONVERT_TO_UNITS[units]

    @property
    def q(self):
        """Charge of the particle 
        """
        return self._q

    @property
    def m_0(self):
        """Rest mass of the particle
        """
        return self._m_0

    @property
    def m(self):
        """Relativistic mass of the particle
        gamma*m_0  w/ m_0 = rest mass

        Units: 
            kg
        """
        return self.gamma()*self.m_0

    @property
    def speed(self):
        """The relativistic speed of the particle
        Solution from E = mc^2(gamma-1)

        Units: 
            m/s
        """
        return c * np.sqrt(1.-(mc2/(self.energy('Joules')+mc2))**2)

    @property
    def direction(self):
        """The unit vector of the current direction of travel

        Units:
            m/s

        Return:
            numpy array of the from [vx,vy,vz]

        """
        if self._direction[2] != 0.0:
            raise ValueError("non-zero z-component of direction")
        return np.array(self._direction)

    @property
    def angle(self):
        """angle from the x axis in radians

        Ignore the warning for arctan(inf)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phi = np.arctan(self.direction[1]/self.direction[0])
        return phi 

    def set_direction(self, new_direction):
        """Set the direction attribute of the particle

        Normalize the direction in case the vector passed is not a unit vector 
        """
        if new_direction[2] != 0.0:
            raise ValueError("non-zero z-component of direction")

        self._direction = np.array(new_direction)/np.linalg.norm(new_direction)

    @property
    def position(self):
        """The current position of the particle 

        assert that the position never has a z component 

        Units:
            m

        Return:
            numpy array of the form [x,y,z]
        """
        if self._position[2] != 0.0:
            raise ValueError("non-zero z-component of position")
        return np.array(self._position)

    def set_position(self, new_position):
        """Set the position vector of the particle

        Raise exception if new_position has non-zero z-component
        """
        if len(new_position) < 3:
            raise ValueError("Position needs to be in 3 dimensions")
        self._position = np.array(new_position)


 
class Magnet(object):
    """
                            y, j 
               -x/2 o---------->
                    | xxxxxxxx
                    | xxxxxxxx
                    | xxxxxxxx
        (pinhole) = + xxxxxxxx
                    | xxxxxxxx
                    | xxxxxxxx
                    | xxxxxxxx
                    |
             +x/2,i v


    """

    def __init__(self, field, xx, yy, physical_dimensions=(MAGNET_WIDTH, MAGNET_LENGTH,0.)):
        """
        Args:
            field:
            physical_dimensions 
                Magnetic_WIdth: 
        """
        self._field = np.array(field)
        self._physical_dimensions = np.array(physical_dimensions)
        x_max = physical_dimensions[0]
        y_max = physical_dimensions[1]
        x = np.linspace(-x_max/2., x_max/2., self.logic_count[0])
        y = np.linspace(0, y_max, self.logic_count[1])
        self.xx, self.yy = np.meshgrind(np.linspace(x, y))

    @property
    def logic_count(self):
        """length of the logic space

        Number of elements in the field array
        """
        return np.array(self._field.shape)


    @property
    def logic_dimensions(self):
        """Dimensions of the logic space. 
        
        The max indices of the field array (i, j)
        """
        return self.logic_count - 1

    def physical_dimensions(self, units='m'):
        """Dimensions of the physical space.
        (x, y, z)

        Args:
            units (str: m or mm): select in which units value is returned as

        Default units:
            m
        """
        return self._physical_dimensions*CONVERT_TO_UNITS[units]

    def map_physical_space_to_logic_space(self, position):
        """Convert an x,y position to i,j space, z is ignored

        -x_max/2 @ i=0, x_max/2 @ i=i_max
        -y=0 @ j=0, y_max @ j=j_max

        Args:
            position (float, tuple): (x,y,z) w/ x, y, z = type:float

        Returns:
            (int, tuple): (i,j) w/ i, j = int
        """
        x, y, z = position
        i_max, j_max = map(float, self.logic_dimensions)
        i_count, j_count = map(float, self.logic_count)
        x_max, y_max, z_max = self.physical_dimensions()
        i = int(np.rint(x*i_count/x_max + i_max/2.))
        j = int(np.rint(y*j_count/y_max - .5))
        return (i,j)

    def field_strength_at_location(self, position):
        """Strength of the magnetic field at a specific physical location (x,y,z)
        Units: Tesla

        Args:
            position (float, tuple): (x,y) w/ x, y, z = type:float

        Returns:
            B-field (float, np-array): [0,0,field-strength]
        """

        i,j = self.map_physical_space_to_logic_space(position)
        return np.array([0.,0.,self._field[i,j]])

    def is_in_bounds(self, position):
        """Check to see if a point is inside the boundary of the magnet

        Map the physical position to logic space, and check if it is valid

        Args:
            position (float, tuple): (x,y,z,) w/ x, y, z = type:float

        Return:
            position_is_valid (Boolean)
        """
        i,j = self.map_physical_space_to_logic_space(position)
        try:
            self._field[i,j]
        except IndexError:
            return False
        else:
            return True


class Detector(object):
    """Faraday Cup

    All faraday cups are located on a 10 cm radius with the origin located at 
    the far side of the magnet (the opposite side from which electrons enter). 
    The coordinate transformation from electron position to the Faraday
    coor-system is then:
        e coor-sys -> fc coor-sys :(x,y) -> (x,y-y_max) = (x',y')

    Attributes:
        aperture (float): The diameter opening of the faraday cup
        placement (float): The location of the faraday cup in degrees
        electrons_captured (Electron): array of all electrons that intersect the
            faraday cup
    """

    _aperture = 5*10**(-3) # m  a 5mm diameter aperture
    _distance_from_origin = 10**(-1) # m 10 cm away from the origin

    def __init__(self, placement, magnet):
        self._placement = placement
        self._electrons_captured = []
        self._r = self.distance*np.array([np.cos(self.theta),np.sin(self.theta),0.])

    @property
    def apature(self):
        """The opening of the faraday cup 
        """
        return self._aperture

    @property
    def placement(self):
        """The location of the detector in degrees
        """
        return self._placement

    @property
    def theta(self):
        """The location of the detector in radians
        """
        return self.placement*np.pi/180.

    @property
    def vector_location(self):
        """Return the vector location of the detector
        r = |r|<cos theta, sin theta>
        wherein
            r = distance from origin
            theta = placement
        """
        return self._r

    @property
    def distance(self):
        """The distance away from the origin
        """
        return self._distance_from_origin

    def tally_electron(self, electron):
        """Append electron to the tally
        """
        self._electrons_captured.append(electron)

    @property
    def electrons_captured(self):
        """Return an array of all electron objects that have entered the FC
        """
        return self._electrons_captured

    def collision_detected(self, electron, return_percent=False):
        """Does an electron intersect the detector
        """
        p = electron.position
        r_p_vec = self._r - p
        r_p_mag = np.linalg.norm(r_p_vec)
        phi = self.apature/(2*r_p_mag)
        dhp = np.dot(electron.direction, r_p_vec)/r_p_mag

        if return_percent:
            return dhp
        else:
            if dhp >= np.cos(phi):
                self.tally_electron(electron)
                return True
            else:
                return False

    def analyze(self):
        """Return an analysis of the electrons captured by the cup
        """
        analysis = {}
        energies = np.fromiter(
                map(lambda x:x.energy()/1000., self.electrons_captured),
                dtype=float
                )
        if len(energies)>0:
            analysis['energies'] = energies
            analysis['count'] = len(energies)
            analysis['max'] = energies.max()
            analysis['min'] = energies.min()
            analysis['mean'] = energies.mean()
            analysis['std'] = np.std(energies)
        else:
            analysis = None

        return analysis
    
    def report(self):
        """Read out the analysis
        """
        analysis = self.analyze()

        print("Detector location: {} deg".format(self.placement)) 
        if analysis:
            template = '''
            {count} number of tallies
            maximum captured energy: {max:.3f} keV
            minimum captured energy: {min:.3f} keV
            Average captured energy: {mean:.3f} keV
            Standard Deviation: {std:.3f} keV
            '''
            print(template.format(**analysis))
        else:
            print("No electrons captured")


def import_magnet_data():
    """Read in the magnet data from the excel file

    Returns 3 dimensional array with size 25, 13, 7 corresponding to the logic
    space dimensions i, j, k
    """
    book = xlrd.open_workbook(MAGNETIC_MAPPING_FILE)
    sheet = book.sheet_by_index(0)
    n = 0
    layer = []
    layers = []
    layer_name = None


    while True:
        row = sheet.row_values(n) 
        # Look for the start of a new layer, reading a line with nothing 
        # will throw an IndexError, reading a line with a number will throw
        # an AttributionError.
        try:
            if row[0].split()[0] == 'Layer':
                # If we are already reading in data from a layer, save it before
                # starting a new layer
                if layer_name:
                    layers.append(np.array(layer))
                layer_name = row[0]
                layer = []
        except (IndexError, AttributeError):
             # If we are in a layer, and the line is not empty, read in the data
            if layer_name and row[0]:
                data = np.array(row[2:], dtype='U9')
                data[data == ''] = np.nan # convert empty strings to nan
                layer.append(data.astype(float))

        # Break loop if at EOF, otherwise increment the spreadsheet row.
        if row[0] == 'END':
            break
        else:
            n+=1

    # append the last layer recorded
    layers.append(np.array(layer))

    # Concatenate the layers along the third dimension. get shape (23, 13, 7) 
    # if only use np.array, gives shape (7, 23, 13)
    return np.dstack(layers) 

def average_layers(magnet):
    """For each i-j point average together all k values. 

    Nan's are ignored in the averaging. If all values are nan then the average 
    is saved as 0

    Parameters
    ----------
    magnet : ndarray
        magnetic field values in kgauss

    Returns
    -------
    ndarray
        value in Tesla
    """
    i_length, j_length, k_length = magnet.shape

    averaged_field = np.zeros((i_length, j_length))

    for i in range(i_length):
        for j in range(j_length):
            # take that average for a given i,j location over all z values, 
            # ignore nan's
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                averaged_value = np.nanmean(magnet[i,j,:])
            if not np.isnan(averaged_value):
                averaged_field[i,j] = averaged_value

    return averaged_field*.1 # Convert to Tesla

def smooth(values, x_max=MAGNET_WIDTH, y_max=MAGNET_LENGTH):
    plt.close('all')
    x = np.linspace(0,x_max,values.shape[1])
    y = np.linspace(0,y_max,values.shape[0])
    plt.pcolor(x*1000, y*1000, values)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylim(0,x_max*1000)
    plt.xlim(0,y_max*1000)

    f = interp2d(x, y, values, kind='cubic')
    x = np.linspace(0,x_max,300)
    y = np.linspace(0,y_max,300)

    plt.figure()
    values = f(x,y)
    values[values<0] = 0
    plt.pcolor(x*1000-16, y*1000-24, 10000*values)
    plt.gca().set_aspect('equal', adjustable='box')
    cb = plt.colorbar()
    plt.xlim(-16,8)
    plt.ylim(-24,0)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    xx, yy = np.meshgrid(x, y)
    cb.set_label("Gauss")

    return xx, yy, values


def runge_kutta_step(electron, magnet):
    """
    """

    def acceleration(direction, position):
        alpha = electron.q/(electron.m*c) 
        b_field = magnet.field_strength_at_location(position)
        dxB = np.cross(direction, b_field)
        return alpha*dxB

    def direction(position):
        k1=dt*acceleration(electron.direction, position)
        k2=dt*acceleration(electron.direction+k1/2., position)
        k3=dt*acceleration(electron.direction+k2/2., position)
        k4=dt*acceleration(electron.direction+k3, position)
        new_direction = electron.direction + 1/6.*(k1+2*k2+2*k3+k4)
        return new_direction/np.linalg.norm(new_direction) 

    def position():
        try:
            k1=dt*electron.speed*direction(electron.position)
            k2=dt*electron.speed*direction(electron.position+k1/2.)
            k3=dt*electron.speed*direction(electron.position+k2/2.)
            k4=dt*electron.speed*direction(electron.position+k3)
        except IndexError:
            dx = electron.speed * electron.direction * dt
            return dx + electron.position 
        else:
            return electron.position+1/6.*(k1+2*k2+2*k3+k4)

    new_direction = direction(electron.position)
    new_position = position()
    electron.set_direction(new_direction)
    electron.set_position(new_position)


def eularian_step(electron, magnet):

    def update_direction(electron, magnet):
        """Solution to d/dt(gamma*m*v)=q/c*vxB for constant speed

        v=s*d_hat

        Args:
            electron (Electron object): electron at current location and direction 
            magnet (Magnet object): The current magnetic field being used in the 
                simulation
        Return:
            electron (Electron object): with updated direction unit-vector

        """
        alpha = electron.q/(electron.m) 
        b_field = magnet.field_strength_at_location(electron.position)
        dxB = np.cross(electron.direction, b_field)
        new_direction = alpha*dxB*dt+electron.direction
        electron.set_direction(new_direction/np.linalg.norm(new_direction))

    def update_position(electron):
        """increment the position of the electron given its current speed, 
        direction, and time-step used 

        x_f = speed*direction*dt + x_i

        Return:
            electron (Electron object): with updated position vector
        """
        dx = electron.speed * electron.direction * dt
        new_position = dx + electron.position 
        electron.set_position(new_position)

    update_direction(electron, magnet)
    update_position(electron)

    return electron


def electron_from_random_source():
    """Generate an electron from a random source
    """

    def random_energy_generator(e_min, e_max):
        """Generate a random energy value between e_min and e_max

        Parameters
        ----------
        min : int
            minimum val in keV
        max : int 
            maximum val in keV
        """
        return np.random.uniform(e_min, e_max) * 10**3

    def random_angle_generator():
        """Generate a random launch angle for the electron

        Using pinhole diameter and the distance from the crossing point, 
        generate a possible launch angle for the electron

        Return
        ------
        phi : float
            units of radians
        """
        min_phi = np.pi/2 - np.arctan(PINHOLE_DIAMETER/(2*CROSS_POINT))
        max_phi = np.pi - min_phi
        return np.random.uniform(min_phi,max_phi)

    def random_position_generator():
        """Generate random value for the starting position of the electron 
        between the limits of the pinhole size
        """
        x_val = np.random.uniform(-PINHOLE_DIAMETER/2, PINHOLE_DIAMETER/2)
        return (x_val,0.,0.) 

    np.random.seed()
    energy = random_energy_generator(EMIN,EMAX)
    phi = random_angle_generator()
    direction = (np.cos(phi), np.sin(phi), 0.)
    position = random_position_generator()
    electron = Electron(energy=energy, 
                        position=position,
                        direction=direction)
    return electron

def set_up_detector_array(magnet):
    """Set up the detector array with corresponding angles
    """
    detectors = []
    detectors.append(Detector(20.0, magnet))
    detectors.append(Detector(40.0, magnet))
    detectors.append(Detector(60.0, magnet))
    detectors.append(Detector(80.0, magnet))
    detectors.append(Detector(100.0, magnet))
    return detectors

def final_report(data):
    """Print out the detector responses
    """
    d = data['data']
    plt.ion()
    plt.close('all')

    plt.figure()
    plt.plot(np.array(d['angle'])*180./np.pi,np.array(d['energy'])/1000., 'ro')
    plt.xlabel("Degrees")
    plt.ylabel("Energy (keV)")
    #plt.yscale('log')
    plt.draw()
    plt.figure()
    detectors = data['detectors']
    box_plot_data = []
    for detector in detectors:
        analysis = detector.analyze()
        if analysis:
            box_plot_data.append(analysis['energies'])
        else:
            box_plot_data.append([])
    plt.boxplot(box_plot_data)
    plt.yscale('log')
    tick_labels = [detector.placement for detector in detectors]
    ticks = np.arange(1,len(tick_labels))
    plt.xticks(ticks, tick_labels)
    plt.xlabel("Detector Position")
    plt.ylabel("Energy (keV)")
    ax = plt.gca()
    plt.xlim(.5,4.5)
    plt.grid(which='both')
    ax.set_xticklabels([r"20${}^\circ$", r"40${}^\circ$", r"60${}^\circ$", r"80${}^\circ$"])
    for line in plt.gca().lines:
        line.set_linewidth(2)
    plt.tight_layout()


def summary_report(histories, detector_array):
    """Print update of simulation to console
    """
    completed_percent =  float(histories)/PARTICLES*100
    print("\n\n" + "-"*50)
    print("{}% completed".format(int(completed_percent)))

    for detector in detector_array:
        detector.report()

def simulate_trajectory(electron, magnet, mode='eulerian'):
    """Move the electron forward using the appropriate scheme

    NOTE(RK doesn't do shit for speeding up the code)

    Check to make sure the angle of the electron is such that it doesn't get 
    caught in the magnetic field. 
    """
    while magnet.is_in_bounds(electron.position):
        if electron.angle < np.pi and electron.angle > 0.:
            if mode == 'eulerian':
                eularian_step(electron, magnet)
            elif mode == 'RK':
                runge_kutta_step(electron, magnet)
        else:
            #print("electron with energy ~ {:.3f} keV caught in loop".format(
            #                        electron.energy()/1000))
            return None

    return electron


def dump(obj, count):
    """Save the detector array in a dump file

    https://docs.python.org/2/library/pickle.html
    """
    filename = './dump.pk1'.format(count) 
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def run_traces(magnetic_field):
    """Run traces. If electron is caught in field None is returned, assign another
    energy to the electron and run again
    """

    electron = None

    while electron is None:
        electron = electron_from_random_source()
        electron = simulate_trajectory(electron, magnet)

    return electron

if __name__ == '__main__':
    """Run the Simulation

    Generate an electron from a random source, and move it through the applied
    magnetic field. 

    When electron has exited the field, check to see if it interacts with any
    of the detectors in the array based on the trajectory of the electron.

    Print out the final report
    """
    

    #Import magnetic field from excel file and run simulation
    magnetic_field = import_magnet_data()
    averaged_field = average_layers(magnetic_field)
    magnet = Magnet(averaged_field)

    pool = Pool()

    detector_array = set_up_detector_array(magnet)
    data = {'energy':[], 'angle':[]}
    block_count = int(PARTICLES/BLOCK_NUMBER)
    histories = 0

    while histories < PARTICLES:

        if PARALLEL:
            # start simulation in parallel on all available processors 
            processes = [pool.apply_async(run_traces, [magnet]) 
                                for e in range(block_count)]
            electrons = [p.get() for p in processes]
        else:
            electrons = [run_traces(magnet) for e in range(block_count)]

        # record information for each electron in the run
        for electron in electrons:
            data['energy'].append(electron.energy())
            data['angle'].append(electron.angle)
            for detector in detector_array:
                detector.collision_detected(electron)

        histories += len(electrons)
        summary_report(histories, detector_array) 
        #dump(detector_array)

    data =  {'detectors':detector_array, 'data':data}
    final_report(data)


