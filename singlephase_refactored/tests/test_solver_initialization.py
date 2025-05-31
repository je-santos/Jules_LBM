import unittest
import numpy as np
import taichi as ti
from ..LBM_3D_SinglePhase_Solver import LB3D_Solver_Single_Phase

class TestSolverInitialization(unittest.TestCase):

    def generate_simple_channel_geo(self, nx: int, ny: int, nz: int) -> np.ndarray:
        # Creates a simple channel with solid walls at y=0 and y=ny-1
        # Fluid nodes are in between.
        geo_array = np.zeros((nx, ny, nz), dtype=np.int8)
        if ny >= 2:
            geo_array[:, 0, :] = 1  # Wall at y_min
            geo_array[:, ny - 1, :] = 1  # Wall at y_max
        return geo_array

    def test_minimal_solver_init(self):
        ti.reset()
        ti.init(arch=ti.cpu)

        nx, ny, nz = 3, 4, 3 # Small grid, ny=4 for walls + 2 fluid layers

        # Instantiate solver
        solver = LB3D_Solver_Single_Phase(nx, ny, nz, sparse_storage=False)

        # Initialize geometry
        geo_data_np = self.generate_simple_channel_geo(nx, ny, nz)
        solver.solid.from_numpy(geo_data_np)

        # Set viscosity
        solver.set_viscosity(0.1)

        # Initialize simulation (this is the main call we're testing for stability)
        solver.init_simulation()

        # Check if a known fluid node is initialized
        # For nx=3, ny=4, nz=3, node (1,1,1) and (1,2,1) should be fluid.
        # Rho should be approx 1.0 after init.
        rho_at_fluid_node = solver.rho.to_numpy()[1, 1, 1]
        self.assertAlmostEqual(rho_at_fluid_node, 1.0, places=6,
                               msg="Density at known fluid node (1,1,1) not initialized to 1.0")

        rho_at_fluid_node_2 = solver.rho.to_numpy()[1, 2, 1]
        self.assertAlmostEqual(rho_at_fluid_node_2, 1.0, places=6,
                               msg="Density at known fluid node (1,2,1) not initialized to 1.0")

        # Check a solid node (e.g., 1,0,1) - its rho might be 0 or 1 depending on init logic for solids
        # The current init kernel in the original code doesn't explicitly set rho for solid nodes.
        # Let's assume they remain 0 or whatever Taichi defaults them to if not in a dense `else` block.
        # For now, focusing on fluid node initialization.
        # solid_node_rho = solver.rho.to_numpy()[1,0,1]
        # print(f"Density at solid node (1,0,1): {solid_node_rho}")

    def test_minimal_solver_single_step(self):
        ti.reset()
        ti.init(arch=ti.cpu)

        nx, ny, nz = 3, 4, 3 # Small grid

        solver = LB3D_Solver_Single_Phase(nx, ny, nz, sparse_storage=False)

        geo_data_np = self.generate_simple_channel_geo(nx, ny, nz)
        solver.solid.from_numpy(geo_data_np)
        solver.set_viscosity(0.1)
        solver.init_simulation()

        # Attempt a single step
        try:
            solver.step()
            step_completed = True
        except Exception as e:
            step_completed = False
            print(f"Solver step raised an exception: {e}")

        self.assertTrue(step_completed, "solver.step() failed or raised an exception.")
        # We could add more assertions here if needed, e.g., check if rho changed slightly,
        # but for now, just ensuring it runs without crashing is the primary goal.


if __name__ == '__main__':
    unittest.main()
