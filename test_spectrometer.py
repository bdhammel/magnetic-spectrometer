import unittest
from numpy import testing as nptest
import numpy as np
from magnetic_spectrometer import *

SIG_FIG_ACCURACY = 4

class ElectronTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.electron = Electron(energy=10**6) # 1 MeV
        pass

    def test_relativistic_speed(self):
        """Check that the calculated speed is correct with units of m/s

        http://hyperphysics.phy-astr.gsu.edu/hbase/relativ/releng.html
        """
        speed_for_MeV_e = 28212.5780447867 # cm/us 
        nptest.assert_approx_equal(self.electron.speed, speed_for_MeV_e, SIG_FIG_ACCURACY)

    def test_gamma(self):
        """Check that the lorentz factor is correct
        """
        gamma = 2.9567341484994287
        nptest.assert_approx_equal(self.electron.gamma(), gamma, SIG_FIG_ACCURACY)

    def test_relativistic_mass(self):
        """Check that the correct value for relativistic mass is returned

        http://hyperphysics.phy-astr.gsu.edu/hbase/relativ/releng.html
        """
        gamma = 2.9567341484994287
        mass_of_MeV_e = gamma*m_e
        nptest.assert_approx_equal(self.electron.m, mass_of_MeV_e, SIG_FIG_ACCURACY)

    def test_energy_in_eng_units(self):
        """Check that the correct value for energy is returned. 
        Check the conversion of eV to g cm2 us-2
        """
        expected_energy = 1.602e-18 # g cm2 us-2
        calculated_energy = self.electron.energy("eng")
        nptest.assert_approx_equal(calculated_energy, expected_energy, SIG_FIG_ACCURACY)

    def test_electron_cant_have_non_0_z_position(self):
        """Make sure the code breaks if an electron has a non-0 z position
        """
        self.electron._position = [0,0,1]
        with self.assertRaises(ValueError):
            self.electron.position

    def test_electron_cant_have_non_0_z_velocity(self):
        """Make sure the code breaks if an electron has a non-0 z velocity
        """
        self.electron._direction = [0,0,1]
        with self.assertRaises(ValueError):
            self.electron.direction

    def test_electron_direction_only_returns_unit_vector(self):
        """Make sure the direction is a unit vector 
        """
        direction = [4,8,0]
        self.electron.set_direction(direction)
        unit_direction = direction/np.linalg.norm(direction)
        nptest.assert_approx_equal(np.linalg.norm(self.electron.direction), 1., SIG_FIG_ACCURACY)
        nptest.assert_array_equal(self.electron.direction, unit_direction)


class MagnetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        field = np.ones((11,6))
        cls.x_max = 2.0
        cls.y_max = 2.4

        x = np.linspace(-cls.x_max/2., cls.x_max/2., field.shape[1])
        y = np.linspace(-cls.y_max, 0., field.shape[0])
        xx, yy = np.meshgrid(x, y)
        cls.magnet = Magnet(field, xx, yy)

    def test_map_physical_orgin_to_logic(self):
        """Check that the correct logic location is returned for the origin 
        """
        expected_logic_coor = [10, 2]
        logic_coor = self.magnet.map_physical_space_to_logic_space((0,0,0))
        nptest.assert_array_equal(logic_coor, expected_logic_coor)

    def test_map_negative_physical_to_logic(self):
        """Check that the correct logic location is returned for the minimum x
        distance
        """
        expected_logic_coor = [10, 0]
        pos = (-self.x_max/2.,0,0)
        logic_coor = self.magnet.map_physical_space_to_logic_space(pos)
        nptest.assert_array_equal(logic_coor, expected_logic_coor)

    @unittest.skip("Skipping because of even odd rounding")
    def test_position_on_boundary(self):
        """Check that a particle on the boundary is treated as inside
        """
        self.assertTrue(self.magnet.is_in_bounds((self.x_max/2., 0., 0.)))

    def test_x_position_out_of_bounds(self):
        """Check that the magnet successfully tests for the boundaries

        Move particle outside the boundary by the significant figure accuracy in 
        mm
        """
        outside = self.x_max/2 + 10**(-SIG_FIG_ACCURACY) 
        self.assertFalse(self.magnet.is_in_bounds((outside, 0, 0.)))

    def test_y_position_out_of_bounds(self):
        """Check that the magnet successfully tests for the boundaries

        Move particle outside the boundary by the significant figure accuracy in 
        mm
        """
        outside = 10**(-SIG_FIG_ACCURACY) 
        self.assertFalse(self.magnet.is_in_bounds((0., outside, 0.)))
    
    def test_far_right_coor(self):
        nptest.assert_array_equal(
                self.magnet.map_physical_space_to_logic_space((self.x_max/2., 0.,0.)),
                self.magnet.logic_count)

    def test_inbounds(self):
        self.assertTrue(self.magnet.is_in_bounds((0., 0, 0.)))
        self.assertTrue(self.magnet.is_in_bounds((-self.x_max/2., 0, 0.)))
        self.assertTrue(self.magnet.is_in_bounds((self.x_max/2., 0, 0.)))
        self.assertTrue(self.magnet.is_in_bounds((0, -self.y_max, 0.)))




class DetectorTest(unittest.TestCase): 
    @classmethod
    def setUpClass(cls):
        cls.x_max = 2.0
        cls.y_max = 1.0
        field = np.ones((11,6))
        cls.electron = Electron(energy=10**6) 
        cls.distance_outside = 1.0 * 10**(-SIG_FIG_ACCURACY)
        cls.electron.set_position((0., 0., 0.))

        x = np.linspace(-cls.x_max/2., cls.x_max/2., field.shape[1])
        y = np.linspace(0, cls.y_max, field.shape[0])
        xx, yy = np.meshgrid(x, y)
        cls.magnet = Magnet(field, xx, yy)
        cls.detector = Detector(15.,cls.magnet)

    def test_direct_hit(self):
        """Check to see if an electron is captured from a direct hit onto the
        detector.
        """
        # Sanity check, electron is just outside of the magnet
        #self.assertFalse(self.magnet.is_in_bounds(self.electron.position))
        # Set direction with law of sines
        # sin(a)/A=sin(c)/C
        P = self.electron.position
        C = self.detector.distance
        A = np.linalg.norm(self.detector.vector_location - P)
        a = self.detector.theta
        c = np.arcsin(C/A*np.sin(a))
        # Set electron on direct hit trajectory
        self.electron.set_direction([np.cos(c),np.sin(c),0])
        direct_hit_percentage = self.detector.collision_detected(
                                                self.electron,
                                                return_percent=True
                                                )
        nptest.assert_approx_equal(direct_hit_percentage, 1., SIG_FIG_ACCURACY)
        self.assertTrue(self.detector.collision_detected(self.electron))

    def test_analysis(self):
        """Make sure that the analysis recognizes the electron captured
        """
        direction = ( self.detector.vector_location / 
                        np.linalg.norm(self.detector.vector_location) )

        electron = Electron(
                        energy=10**4, 
                        position=(0,0,0),
                        direction=direction
                        )
        #sanity check
        self.assertTrue(self.detector.collision_detected(electron))

        self.assertIsNotNone(self.detector.analyze())


class SimulationTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.phi = np.pi/6. # angle particle should leave the field
        cls.x_max = 2.4
        cls.y_max = 1.0
        cls.electron = Electron(energy=10**4, position=(0., -cls.y_max, 0.))
        cls.B_strength = ( (np.cos(cls.phi)*cls.electron.m*cls.electron.speed) /
                           (cls.y_max*cls.electron.q) )*10**3  # convert to Tesla
        field = np.empty((100,60))
        field.fill(cls.B_strength)

        x = np.linspace(-cls.x_max/2., cls.x_max/2., field.shape[1])
        y = np.linspace(-cls.y_max, 0., field.shape[0])
        xx, yy = np.meshgrid(x, y)
        cls.magnet = Magnet(field, xx, yy)

        cls.detector = Detector(30., cls.magnet)
        cls.r =( (cls.electron.m*cls.electron.speed) /  
                 (cls.electron.q*cls.B_strength*10**(-3)) ) 

    def test_correct_exsit_angle(self):
        """Check that the electron exist the magnet at the expected angle
        """
        expected_direction = np.array([np.cos(self.phi), np.sin(self.phi), 0.])
        electron = simulate_trajectory(self.electron, self.magnet)
        calculated_direction = electron.direction
        nptest.assert_array_almost_equal(
                                calculated_direction, 
                                expected_direction,
                                SIG_FIG_ACCURACY
                                )
        self.assertTrue(self.detector.collision_detected(electron))

    def test_rerunning_simulation_works(self):
        """Make sure that rerunning the same electron through the simulation 
        doesn't have any strange effects. Check that electron is properly 
        initialized 
        """
        old_location = self.electron.position
        electron = simulate_trajectory(self.electron, self.magnet)
        new_location = electron.position
        nptest.assert_array_almost_equal(
                                new_location, 
                                old_location,
                                SIG_FIG_ACCURACY+3
                                )

    def test_geometry(self):
        """Sanity check, make sure my analytical solution checks out
        """
        theta = np.pi/2 - self.phi
        y_calculated = self.r * np.sin(theta)
        nptest.assert_approx_equal(y_calculated, self.y_max)


    def test_correct_exsit_position(self):
        """Check that the electron exist the magnet at the expected position 
        """
        electron = simulate_trajectory(self.electron, self.magnet)
        y = 0.
        x = self.r - self.r * np.cos(np.pi/2 - self.phi)
        expected_position = np.array([x,y,0.])
        calculated_position = electron.position
        nptest.assert_array_almost_equal(
                                calculated_position, 
                                expected_position,
                                SIG_FIG_ACCURACY+3
                                )

    def test_average_field(self):
        """Check trajectory through a field of strength .03 T ~ the average field
        strength of the magnet. Compare to analytical solution 
        """
        x_max = 2.5  # cm
        y_max = 1.4  # cm
        field = np.empty((300,180))
        field_strength = .03 # T    
        field.fill(field_strength)

        x = np.linspace(-x_max/2., x_max/2., field.shape[1])
        y = np.linspace(-y_max, 0, field.shape[0])
        xx, yy = np.meshgrid(x, y)
        magnet = Magnet(field, xx, yy)
        electron = Electron(energy=5*10**5, position=(0., -y_max, 0.))
        r = (electron.m*electron.speed)/(electron.q*field_strength*10**(-3))
        phi = np.arccos(y_max/r)
        electron = simulate_trajectory(electron, magnet)
        nptest.assert_approx_equal(electron.angle*180/np.pi, phi*180./np.pi)


if __name__ == '__main__':
    """run tests
    """
    unittest.main()
