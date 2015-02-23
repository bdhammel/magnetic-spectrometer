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
        speed_for_MeV_e = 2.821257804478669*10**8 # m/s
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

    def test_energy_in_jouls(self):
        """Check that the correct value for energy is returned. 
        Check the conversion of eV to Joules
        """
        expected_energy = 1.6019999999999999 * 10**(-13)
        calculated_energy = self.electron.energy("Joules")
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
        cls.x_max = 20*10**(-3)
        cls.y_max = 10*10**(-3)
        cls.magnet = Magnet(field, (cls.x_max, cls.y_max, 0.))

    def test_map_physical_orgin_to_logic(self):
        """Check that the correct logic location is returned for the origin 
        """
        expected_logic_coor = [5, 0]
        logic_coor = self.magnet.map_physical_space_to_logic_space((0,0,0))
        nptest.assert_array_equal(logic_coor, expected_logic_coor)

    def test_map_negative_physical_to_logic(self):
        """Check that the correct logic location is returned for the minimum x
        distance
        """
        expected_logic_coor = [0, 0]
        pos = (-self.x_max/2,0,0)
        logic_coor = self.magnet.map_physical_space_to_logic_space(pos)
        nptest.assert_array_equal(logic_coor, expected_logic_coor)

    @unittest.skip("Skipping because of even odd rounding")
    def test_position_on_boundary(self):
        """Check that a particle on the boundary is treated as inside
        """
        self.assertTrue(self.magnet.is_in_bounds((self.x_max/2., self.y_max, 0.)))

    def test_x_position_out_of_bounds(self):
        """Check that the magnet successfully tests for the boundaries

        Move particle outside the boundary by the significant figure accuracy in 
        mm
        """
        outside = self.x_max/2 + 10**(-SIG_FIG_ACCURACY) * 10**(-3)
        self.assertFalse(self.magnet.is_in_bounds((outside, 0, 0.)))

    def test_y_position_out_of_bounds(self):
        """Check that the magnet successfully tests for the boundaries

        Move particle outside the boundary by the significant figure accuracy in 
        mm
        """
        outside = self.y_max + 10**(-SIG_FIG_ACCURACY) * 10**(-3)
        self.assertFalse(self.magnet.is_in_bounds((0., outside, 0.)))

class DetectorTest(unittest.TestCase): 
    @classmethod
    def setUpClass(cls):
        cls.x_max = 20*10**(-3)
        cls.y_max = 10*10**(-3)
        field = np.ones((11,6))
        cls.electron = Electron(energy=10**6) 
        cls.distance_outside = 10**(-3) * 10**(-SIG_FIG_ACCURACY)
        cls.electron.set_position((cls.x_max/4, cls.y_max+cls.distance_outside, 0.))
        cls.magnet = Magnet(field, (cls.x_max, cls.y_max, 0.))
        cls.detector = Detector(15.,cls.magnet)

    def test_shift_to_detector_coordinate_system(self):
        """Check that an electron's position is correctly mapped into the 
        coordinate system of the detector
        """
        new_position = self.detector.detector_coor_sys(self.electron.position)
        nptest.assert_array_almost_equal(
                                    new_position,
                                    [self.x_max/4, self.distance_outside, 0.],
                                    SIG_FIG_ACCURACY+3
                                    )

    def test_direct_hit(self):
        """Check to see if an electron is captured from a direct hit onto the
        detector.
        """
        # Sanity check, electron is just outside of the magnet
        self.assertFalse(self.magnet.is_in_bounds(self.electron.position))
        # Set direction with law of sines
        # sin(a)/A=sin(c)/C
        P = self.detector.detector_coor_sys(self.electron.position)
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
                        position=(0,self.y_max,0),
                        direction=direction
                        )
        #sanity check
        self.assertTrue(self.detector.collision_detected(electron))

        self.assertIsNotNone(self.detector.analyze())

    def test_indirect_hit(self):
        """
        """
        pass

class SimulationTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.phi = np.pi/6. # angle particle should leave the field
        cls.x_max = 20*10**(-3)
        cls.y_max = 10*10**(-3)
        cls.electron = Electron(energy=10**4)
        cls.B_strength = ( (np.cos(cls.phi)*cls.electron.m*cls.electron.speed) /
                           (cls.y_max*cls.electron.q) )
        field = np.empty((11,6))
        field.fill(cls.B_strength)
        cls.magnet = Magnet(field, (cls.x_max, cls.y_max, 0.))
        cls.detector = Detector(30.,cls.magnet)
        cls.r =( (cls.electron.m*cls.electron.speed) /  
                 (cls.electron.q*cls.B_strength) ) 

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
        y = self.y_max
        x = self.r - self.r * np.cos(np.pi/2 - self.phi)
        expected_position = np.array([x,y,0.])
        calculated_position = electron.position
        nptest.assert_array_almost_equal(
                                calculated_position, 
                                expected_position,
                                SIG_FIG_ACCURACY+3
                                )

    def test_average_field(self):
        x_max = 25 * 10**(-3) # m
        y_max = 14 * 10**(-3) # m
        field = np.empty((11,6))
        field.fill(.08)
        magnet = Magnet(field, (x_max, y_max, 0.))
        electron = Electron(energy=10**4)




if __name__ == '__main__':
    """run tests
    """
    unittest.main()
