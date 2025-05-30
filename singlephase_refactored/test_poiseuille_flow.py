import unittest
import numpy as np
import taichi as ti
# Ensure the LBM_3D_SinglePhase_Solver can be imported.
# This might require setting PYTHONPATH or running the test from the directory above singlephase_refactored.
from singlephase_refactored.LBM_3D_SinglePhase_Solver import LB3D_Solver_Single_Phase

# Copied from example_poiseuille_flow.py for self-containment of the test
def calculate_analytical_poiseuille_velocity(fx_body_force_term: float,
                                             kinematic_viscosity: float,
                                             num_y_points: int,
                                             rho_avg: float = 1.0):
    """
    Calculates the analytical solution for Poiseuille flow velocity profile in a channel.
    The flow is driven by a body force fx_body_force_term.
    No-slip boundary conditions are assumed at y_node_idx = 0 and y_node_idx = num_y_points - 1.
    The fluid nodes are from y_node_idx = 1 to y_node_idx = num_y_points - 2.

    Args:
        fx_body_force_term (float): The body force term (e.g., Gx or rho_0 * Gx depending on LBM formulation).
                                   If this is rho_0 * Gx, then rho_avg should be rho_0 for consistency.
        kinematic_viscosity (float): Kinematic viscosity (nu).
        num_y_points (int): Total number of grid points in the y-direction (including walls).
        rho_avg (float): Average fluid density. Used to convert fx_body_force_term to acceleration Gx if needed.
                         The standard formula uses Gx = acceleration. If fx_body_force_term is rho*Gx,
                         then the Gx in the formula is fx_body_force_term / rho_avg.

    Returns:
        numpy.ndarray: A 1D array of the velocity profile u_x(y) across all y-points.
                       Velocities at wall nodes (0 and num_y_points-1) will be zero.
    """
    if num_y_points < 3: # Not enough points for fluid layers
        return np.zeros(num_y_points)

    velocity_profile = np.zeros(num_y_points)

    Gx = fx_body_force_term / rho_avg
    coeff = Gx / (2.0 * kinematic_viscosity)

    for y_node_idx in range(num_y_points):
        velocity_profile[y_node_idx] = coeff * y_node_idx * ( (num_y_points - 1) - y_node_idx )

    return velocity_profile

class TestPoiseuilleFlow(unittest.TestCase):
    def generate_poiseuille_geo_array(self, nx: int, ny: int, nz: int) -> np.ndarray:
        """
        Generates a 3D numpy array for Poiseuille flow geometry.
        Solid walls are at y=0, y=ny-1, z=0, z=nz-1. X-direction is periodic.
        """
        geo_array = np.zeros((nx, ny, nz), dtype=np.int8)
        if ny > 1: # Ensure ny is large enough for walls
            geo_array[:, 0, :] = 1  # Wall at y_min
            geo_array[:, ny - 1, :] = 1  # Wall at y_max
        if nz > 1: # Ensure nz is large enough for walls
            geo_array[:, :, 0] = 1  # Wall at z_min
            geo_array[:, :, nz - 1] = 1  # Wall at z_max
        return geo_array

    def test_poiseuille_flow_comparison(self):
        """
        Tests the LBM solver for Poiseuille flow against the analytical solution.
        """
        ti.reset() # Reset Taichi state for a clean test run
        # Use CPU for potentially faster test initialization and execution for small sims.
        # dynamic_index=False and kernel_profiler=False are good defaults for tests.
        ti.init(arch=ti.cpu, dynamic_index=False, kernel_profiler=False)

        # Simulation parameters (small values for a quick test)
        nx, ny, nz = 3, 10, 3  # Small domain: 3x10x3 (flow in x, channel height in y)
        niu = 0.1              # Kinematic viscosity
        fx_body_force = 1.0e-4 # Body force in x-direction (rho_0 * Gx)
        rho_lbm = 1.0          # Reference density used in LBM setup (often 1.0)

        # Solver setup
        solver = LB3D_Solver_Single_Phase(nx, ny, nz, sparse_storage=False) # Dense storage for simplicity

        # Initialize geometry directly from numpy array
        geo_data_np = self.generate_poiseuille_geo_array(nx, ny, nz)
        solver.solid.from_numpy(geo_data_np) # Direct initialization of the solid field

        solver.set_viscosity(niu)
        solver.set_force([fx_body_force, 0.0, 0.0]) # Force along x-axis

        # Boundary Conditions:
        # X-boundaries are periodic by default in the solver.
        # Y-boundaries (walls) are defined by the solid geometry.
        # Z-boundaries (walls) are defined by the solid geometry.

        solver.init_simulation()

        # Simulation loop - number of iterations to approach steady state
        # This value might need tuning based on parameters and domain size.
        steady_state_iterations = 2000
        for i in range(steady_state_iterations):
            solver.step()
            # Optional: print progress for long tests
            # if i % (steady_state_iterations // 10) == 0:
            #     print(f"Test iteration {i}/{steady_state_iterations}")


        # Extract simulated velocity profile at the center of the domain
        sim_v_np = solver.v.to_numpy()
        slice_x = nx // 2
        slice_z = nz // 2
        # Ensure slices are valid for small dimensions
        if slice_x >= nx: slice_x = nx -1
        if slice_z >= nz: slice_z = nz -1
        sim_vx_profile = sim_v_np[slice_x, :, slice_z, 0]

        # Calculate analytical solution using the same parameters
        analytical_vx_profile = calculate_analytical_poiseuille_velocity(
            fx_body_force, niu, ny, rho_lbm
        )

        # Perform comparison only on the fluid nodes (excluding walls at y=0 and y=ny-1)
        if ny > 2: # Requires at least one fluid layer
            sim_fluid_profile = sim_vx_profile[1:ny-1]
            analytical_fluid_profile = analytical_vx_profile[1:ny-1]

            # Debug prints (can be useful if the test fails)
            # print("\nSimulated Profile (fluid nodes):", sim_fluid_profile)
            # print("Analytical Profile (fluid nodes):", analytical_fluid_profile)
            # diff = sim_fluid_profile - analytical_fluid_profile
            # print("Differences:", diff)
            # print("Max difference:", np.max(np.abs(diff)))
            # print("Relative differences:", diff / (analytical_fluid_profile + 1e-12))


            # Use numpy.testing.assert_allclose for comparing floating-point arrays
            np.testing.assert_allclose(
                sim_fluid_profile,
                analytical_fluid_profile,
                rtol=0.05,  # Relative tolerance: 5%
                atol=1e-5   # Absolute tolerance (important for values close to zero)
            )
        else:
            # If ny <= 2, there are no internal fluid nodes to compare.
            # This case should ideally not occur with valid Poiseuille setup (ny=10 here).
            self.skipTest("Not enough fluid nodes (ny <= 2) to perform Poiseuille flow velocity comparison.")

if __name__ == '__main__':
    unittest.main()
