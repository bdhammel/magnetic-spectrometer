"""Monte Carlo Simulation of magnetic spectrometer for electron beams

Requirements
"""

################################################################################
#                               Imports                                        #
################################################################################

import xlrd
import numpy as np

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

PARTICLES = 10**6
PINHOLE_DIAMETER = 1 # mm
CROSS_POINT = 10 # mm   - distance from pinhole to the wire cross point
MAGNETIC_MAPPING_FILE = './magnetic_mapping.xlsx' # location of magnet excel file  
MAGNET_WIDTH = 25 # mm
MAGNET_LENGTH = 14 # mm
dt = 1.*10**-9 # Time step (s)



################################################################################



class Electron(object):

    _m_0 = m_e # 9.10938291e-31 kg  rest mass of electron
    _q = e # 1.602176565e-19 Coulombs

    def __init__(self, energy):
        self._position = [0.,0.,0.] # (x,y,z) m
        self._direction = [0.,1.,0.] # (vx_hat, vy_hat, vz_hat) m/s
        self._energy = energy*10**3 # eV

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
        return e

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
            mm 

        Return:
            numpy array of the form [x,y,z]
        """
        if self._position[2] != 0.0:
            raise ValueError("non-zero z-component of position")
        return np.array(self._positon)

    def set_position(self, new_position):
        """Set the position vector of the particle

        Raise exception if new_position has non-zero z-component
        """
        if new_position[2] != 0.0:
            raise ValueError("non-zero z-component of position")
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

    def __init__(self, field, physical_dimensions=(MAGNET_WIDTH, MAGNET_LENGTH,0.)):
        self._field = np.array(field)
        self._physical_dimensions = np.array(physical_dimensions)

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
        j = int(np.rint(y*j_max/y_max))
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
        apature (float): The opening of the faraday cup
        placment (float): The location of the faraday cup in degrees
        electrons_captured (Electron): array of all electrons that intercect the
            faraday cup
    """

    _apature = 10**(-3) # mm aperture

    def __init__(self, placment):
        self._placment = placment
        self._electrons_captured = []

    def tally_electron(self, electron):
        """Append electron to the tally
        """
        self._electrons_captured.append(electron)

    def collesion_detected(self, electron):
        """Does an electron intersect the detector
        """

        if True:
            self.tally_electron(electron)

    def analyze(self):
        pass
    
    def report(self):
        """read out the analysis
        """
        pass


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
                data = np.array(row[2:], dtype='S9')
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
    """
    i_length, j_length, k_length = magnet.shape()

    averaged_field = np.zeros((i_length, j_length))

    for i in range(i_length):
        for j in range(j_length):
            # take that average for a given i,j location over all z values, 
            # ignore nan's
            averaged_value = np.nanmean(magnet[i,j,:])
            if not np.isnan(averaged_value):
                averaged_field[i,j] = averaged_value

    return averaged_field

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
    alpha = electron.q/(electron.m*c) 
    b_field = magnet.field_strength_at_location(electron.position)
    dxB = np.cross(electron.direction, b_field)
    new_direction = alpha*dxB*dt+electron.direction
    electron.set_direction(new_direction/np.linalg.norm(new_direction))
    return electron

def update_position(electron):
    """increment the position of the electron given its current speed, direction,
    and time-step used 

    x_f = speed*direction*dt + x_i

    Return:
        electron (Electron object): with updated position vector
    """
    dx = electron.speed*electron.direction.dt
    new_position = dx + electron.position 
    electron.set_position(new_position)
    return electron

def electron_from_random_source():
    """Generate an electron from a random source
    """
    energy = ''
    direction = ''
    position = ''
    electron = Electron()
    return electron

def step(electron, magnet):
    """Increment the simulation by one time step
    """
    electron = update_direction(electron, magnet)
    electron = update_position(electron)
    return electron

def run_traces():

    for particle in PARTICLES:
        electron = Electron()
        while magnet.is_in_bounds(electron.position):
            electron = step(electron, magnet)


if __name__ == '__main__':
    magnet = import_magnet_data()
    averaged_field = average_layers(magnet)



