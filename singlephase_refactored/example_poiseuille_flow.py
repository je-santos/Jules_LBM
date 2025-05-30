import numpy as np
import time
import taichi as ti
from singlephase_refactored.LBM_3D_SinglePhase_Solver import LB3D_Solver_Single_Phase

def generate_poiseuille_geo_dat(nx: int, ny: int, nz: int, filename: str = "geo_poiseuille.dat"):
    """
    Generates a geometry file for Poiseuille flow between two parallel plates.
    The flow is in the x-direction.
    Solid walls are placed at y=0, y=ny-1, z=0, and z=nz-1.
    X-direction is periodic.

    Args:
        nx (int): Number of grid nodes in the x-direction.
        ny (int): Number of grid nodes in the y-direction (channel height).
        nz (int): Number of grid nodes in the z-direction (channel width).
        filename (str): Name of the output geometry file.
    """
    geo_array = np.zeros((nx, ny, nz), dtype=np.int8)

    # Set Y boundaries as solid walls
    geo_array[:, 0, :] = 1  # Bottom wall (y=0)
    geo_array[:, ny - 1, :] = 1  # Top wall (y=ny-1)

    # Set Z boundaries as solid walls
    geo_array[:, :, 0] = 1  # Front wall (z=0)
    geo_array[:, :, nz - 1] = 1  # Back wall (z=nz-1)

    # Reshape for saving: nx rows, ny*nz columns per row
    # This matches the format expected by solver.init_geo, which reshapes with order 'F'
    # The solver expects (nx, ny, nz) order in memory after loading and reshaping.
    # np.savetxt writes row by row.
    # If geo_array is (nx, ny, nz), geo_array.reshape(nx, -1) means each row in the file
    # corresponds to a slice geo_array[i, :, :].flatten().
    np.savetxt(filename, geo_array.reshape(nx, -1), fmt='%d')
    print(f"Generated Poiseuille geometry file: {filename}")

if __name__ == "__main__":
    # Initialize Taichi
    # It's good practice to initialize Taichi early, especially if multiple modules might use it.
    ti.init(arch=ti.gpu, dynamic_index=False, kernel_profiler=False) # Profiler off for example script

    # Simulation Parameters
    nx, ny, nz = 5, 20, 20  # Grid dimensions (short in x for periodic flow driven by force)
    niu = 0.1              # Kinematic viscosity (lattice units)
    fx = 1.0e-5            # Body force in x-direction (lattice units)

    # Generate geometry data file
    geo_file_path = "singlephase_refactored/geo_poiseuille.dat"
    generate_poiseuille_geo_dat(nx, ny, nz, geo_file_path)

    # Instantiate the solver
    solver = LB3D_Solver_Single_Phase(nx=nx, ny=ny, nz=nz)

    # Initialize geometry, viscosity, and force
    solver.init_geo(geo_file_path)
    solver.set_viscosity(niu)
    solver.set_force([fx, 0.0, 0.0]) # Force along x-axis

    # Boundary Conditions:
    # X-boundaries: Periodic (default type 0 in solver, so no explicit call needed if LBM_3D_SinglePhase_Solver defaults to periodic)
    # Y-boundaries (y=0, y=ny-1): Solid walls (defined in geo_poiseuille.dat)
    # Z-boundaries (z=0, z=nz-1): Solid walls (defined in geo_poiseuille.dat)
    # No explicit calls to `set_boundary_condition` are made here, relying on:
    # 1. Default periodic behavior for X faces if not otherwise set.
    # 2. Solid nodes defined in the geometry file for Y and Z faces, which are handled by bounce-back.

    # Initialize the simulation (calculates derived params, sets initial fields)
    solver.init_simulation()

    # Simulation Loop
    max_iterations = 10000  # Example number of iterations
    output_frequency = 1000 # How often to print status and (optionally) save VTK

    print("Starting Poiseuille flow simulation...")
    time_init = time.time()

    for iter_count in range(max_iterations + 1):
        solver.step()

        if iter_count % output_frequency == 0:
            time_now = time.time()
            current_max_v = solver.get_max_v()
            print(f"Iteration: {iter_count:5d}, Max Velocity: {current_max_v:.6e}, "
                  f"Elapsed Time: {time_now - time_init:.2f}s")

            # Optional: Export data for visualization (e.g., with ParaView)
            # solver.export_VTK(iter_count)

    time_end = time.time()
    print(f"Simulation finished. Total elapsed time: {time_end - time_init:.2f}s")

    # Analytical solution
    rho_avg_sim = 1.0 # Assuming average density is 1.0 for simplicity in analytical calculation
    analytical_vy_profile_at_center_z = calculate_analytical_poiseuille_velocity(fx, niu, ny, rho_avg_sim)

    print("\nComparison of Simulated vs. Analytical Velocity Profile (vx at x=nx/2, z=nz/2):")

    sim_v_np = solver.v.to_numpy()
    slice_x = nx // 2
    slice_z = nz // 2
    # Ensure indices are within bounds, especially for small nx or nz
    if slice_x >= sim_v_np.shape[0]: slice_x = sim_v_np.shape[0] -1
    if slice_z >= sim_v_np.shape[2]: slice_z = sim_v_np.shape[2] -1

    sim_vx_profile = sim_v_np[slice_x, :, slice_z, 0]

    print(f"{'Y':>3} | {'Analytical':>12} | {'Simulated':>12} | {'Abs Diff':>12}")
    print("-" * 55)
    for y_idx in range(ny):
        analytical_val = analytical_vy_profile_at_center_z[y_idx]
        simulated_val = sim_vx_profile[y_idx]
        abs_diff = abs(analytical_val - simulated_val)
        print(f"{y_idx:>3} | {analytical_val:>12.6e} | {simulated_val:>12.6e} | {abs_diff:>12.6e}")

    # Calculate RMSE for fluid nodes only (1 to ny-2)
    # Ensure ny > 2 to have fluid nodes
    if ny > 2:
        analytical_fluid_profile = analytical_vy_profile_at_center_z[1:ny-1]
        sim_fluid_profile = sim_vx_profile[1:ny-1]

        # Define RMSE function (can be outside main or locally)
        def calculate_rmse_local(p1, p2):
            return np.sqrt(np.mean((p1 - p2)**2))

        rmse = calculate_rmse_local(analytical_fluid_profile, sim_fluid_profile)
        print(f"\nRMSE between simulated and analytical profiles (fluid nodes): {rmse:.6e}")
    else:
        print("\nNot enough fluid nodes to calculate RMSE (need ny > 2).")

    # Optional: Suggestion for plotting if matplotlib is available
    print("\n# To plot, you can use matplotlib:")
    print("# import matplotlib.pyplot as plt")
    print("# y_coords = np.arange(ny)")
    print("# plt.figure(figsize=(8, 6))")
    print("# plt.plot(analytical_vy_profile_at_center_z, y_coords, label='Analytical', linestyle='--')")
    print("# plt.plot(sim_vx_profile, y_coords, label='Simulated', linestyle='-')")
    print("# plt.xlabel('Velocity (vx)')")
    print("# plt.ylabel('Y-coordinate')")
    print("# plt.title('Poiseuille Flow Velocity Profile')")
    print("# plt.legend()")
    print("# plt.grid(True)")
    print("# plt.show()")

    # Example of how to access some data (demonstration, not part of typical run)
    # rho_data = solver.rho.to_numpy()
    # print(f"\nShape of density data: {rho_data.shape}")
    # print(f"Density at center (approx): {rho_data[nx//2, ny//2, nz//2]}")

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

    # Gx = acceleration = Force_per_unit_mass.
    # If fx_body_force_term is given as rho_0 * Gx (common LBM input for `fx`),
    # then Gx = fx_body_force_term / rho_avg (assuming rho_avg is close to rho_0).
    Gx = fx_body_force_term / rho_avg

    # The formula for u_x(y_node) for nodes y_node = 0, 1, ..., Ny-1,
    # with walls at y_node=0 and y_node=Ny-1 is:
    # u_x(y_node) = (Gx / (2 * nu)) * y_node * ( (Ny-1) - y_node )
    # where Ny is num_y_points.
    coeff = Gx / (2.0 * kinematic_viscosity)

    for y_node_idx in range(num_y_points):
        # This formula gives u=0 at y_node_idx=0 and y_node_idx=num_y_points-1
        velocity_profile[y_node_idx] = coeff * y_node_idx * ( (num_y_points - 1) - y_node_idx )

    return velocity_profile

    # To run this example:
    # 1. Ensure Taichi is installed (`pip install taichi`).
    # 2. Ensure the LBM_3D_SinglePhase_Solver.py file is in the `singlephase_refactored` directory.
    # 3. Run `python example_poiseuille_flow.py` from the parent directory of `singlephase_refactored`.

    # Note on geometry and BCs for Poiseuille:
    # The channel walls are created by setting geo_array elements to 1.
    # The flow is driven by the body force fx.
    # The x-direction is periodic by default in the solver's current BC setup.
    # If the solver's default for x-faces was not periodic, one would need:
    # solver.set_boundary_condition('x_left', 'periodic')
    # solver.set_boundary_condition('x_right', 'periodic')
    # But since they are periodic by default, these calls are omitted for brevity.
