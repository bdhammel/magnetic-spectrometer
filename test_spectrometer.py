import unittest
from numpy import testing as nptest
import numpy as np
from magnetic_spectrometer import *

SIG_FIG_ACCURACY = 4

class ElectronTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.electron = Electron(energy=10**3) # 1 MeV
        pass

    def test_relativistic_speed(self):
        """Check that the calculated speed is correct with units of m/s

        http://hyperphysics.phy-astr.gsu.edu/hbase/relativ/releng.html
        """
        speed_for_MeV_e = 2.821257804478669*10**8 # m/s
        nptest.assert_approx_equal(self.electron.speed, speed_for_MeV_e, SIG_FIG_ACCURACY)

    def test_relativistic_mass(self):
        """Check that the correct value for relativistic mass is returned

        http://hyperphysics.phy-astr.gsu.edu/hbase/relativ/releng.html
        """
        gamma = 2.9567341484994287
        mass_of_MeV_e = gamma*m_e
        nptest.assert_approx_equal(self.electron.m, mass_of_MeV_e, SIG_FIG_ACCURACY)

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

    def test_position_on_boundary(self):
        """Check that a particle on the boundary is treated as inside
        """
        self.assertTrue(self.magnet.is_in_bounds((self.x_max/2., self.y_max, 0.)))

    def test_position_out_of_bounds(self):
        """Check that the magnet successfully tests for the boundaries

        Move particle outside the boundary by the significant figure accuracy in 
        mm
        """
        outside = self.x_max/2 + 10**(-SIG_FIG_ACCURACY) * 10**(-3)
        self.assertFalse(self.magnet.is_in_bounds((outside, 0, 0.)))

class DetectorTest(unittest.TestCase): 
    pass

if __name__ == '__main__':
    """run tests
    """
    unittest.main()
