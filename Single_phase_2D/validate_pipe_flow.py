import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from lbm_solver_2d import LB2D_Solver_Single_Phase

# Initialize Taichi
# Try with default first, GPU if available.
try:
    # ti.init(arch=ti.gpu, dynamic_index=False, kernel_profiler=False, print_ir=False) # Old call
    ti.init(arch=ti.gpu) # Simplified init
    print("Taichi initialized on GPU.")
except RuntimeError:
    # ti.init(arch=ti.cpu, dynamic_index=False, kernel_profiler=False, print_ir=False) # Old call
    ti.init(arch=ti.cpu) # Simplified init
    print("Taichi initialized on CPU (GPU not available or failed).")


def create_pipe_geometry(Lx, Ly):
    """
    Creates a 2D pipe geometry as a boolean NumPy array.
    Walls are True (solid), fluid domain is False.
    Walls are at y=0 and y=Ly-1.
    """
    geometry = np.full((Lx, Ly), False, dtype=bool)
    geometry[:, 0] = True  # Bottom wall
    geometry[:, Ly - 1] = True  # Top wall
    return geometry

def calculate_analytical_poiseuille(G_effective, niu, H_channel, y_coords_physical):
    """
    Calculates the analytical velocity profile for 2D Poiseuille flow.
    u_x(y) = (G_effective / (2 * niu)) * y * (H_channel - y)
    G_effective: Effective pressure gradient (fx in LBM for body force driven flow, assuming rho_avg=1)
    niu: Kinematic viscosity.
    H_channel: Total height of the channel (distance between the inner edges of the walls).
    y_coords_physical: Physical y-coordinates ranging from 0 to H_channel.
    """
    # Ensure y_coords are relative to the start of the fluid domain (0 to H_channel)
    # H_channel = Ly - 2 for a pipe defined by Ly grid points with walls at 0 and Ly-1
    # y_physical should go from 0 (just above bottom wall) to H_channel (just below top wall)

    # The formula u_x(y) = G/(2*nu) * y * (H-y) assumes y is from 0 to H (fluid domain height)
    # If y_coords_physical are already 0 to H, then it's direct.

    velocity_profile = (G_effective / (2 * niu)) * y_coords_physical * (H_channel - y_coords_physical)
    return velocity_profile

def run_lbm_simulation_and_get_velocity(Lx, Ly, geometry, niu, fx, total_iterations):
    """
    Initializes and runs the LBM simulation, then returns the velocity profile.
    """
    solver = LB2D_Solver_Single_Phase(nx=Lx, ny=Ly)
    solver.init_geo(geometry)
    solver.set_viscosity(niu)
    solver.set_force([fx, 0.0]) # Apply force in x-direction

    # Boundary conditions:
    # X-direction: Periodic (this is the default if no other BC is set for x-min/x-max planes)
    # Y-direction: No-slip walls are handled by the geometry (solid=True) and bounce-back.

    print("Initializing LBM simulation...")
    solver.init_simulation() # This also initializes fields based on initial rho=1, v=0

    print(f"Running LBM simulation for {total_iterations} iterations...")
    for iter_num in range(total_iterations):
        solver.step()
        if (iter_num + 1) % 1000 == 0:
            max_v = solver.get_max_v()
            print(f"Iteration: {iter_num + 1}/{total_iterations}, Max Velocity: {max_v:.6e}")

    # Extract velocity profile (vx) at the center of the pipe (x = Lx // 2)
    # v is a Taichi field of shape (Lx, Ly, 2) for vx, vy
    # We need vx, which is v[x, y][0]
    v_data = solver.v.to_numpy()
    velocity_profile_lbm = v_data[Lx // 2, :, 0] # vx at x=Lx/2, across all y

    return velocity_profile_lbm, solver # Return solver to access other data if needed

def plot_results(y_coords_lbm, lbm_velocities, analytical_velocities, Ly, H_channel):
    """
    Plots LBM and analytical velocity profiles.
    y_coords_lbm: y-coordinates for LBM data (grid indices, 0 to Ly-1)
    H_channel: actual fluid channel height
    """
    plt.figure(figsize=(8, 6))

    # LBM data: y_coords_lbm are grid indices. Fluid domain is from y=1 to y=Ly-2.
    # Physical y for LBM: (y_grid_index - 0.5) if 0 is wall center.
    # Or, if walls are at 0 and Ly-1, fluid nodes are 1 to Ly-2.
    # y_physical_lbm = (y_coords_lbm[1:-1] - 1 + 0.5) # Centered in channel cells, from 0.5 to H-0.5
    y_physical_lbm_plot = y_coords_lbm[1:Ly-1] - 0.5 # Shift to match physical domain (0 to H_channel)
                                                 # if y_coords_lbm are cell centers from 0.5 to Ly-1.5
                                                 # If y_coords_lbm are nodes 0..Ly-1, then fluid nodes 1..Ly-2
                                                 # Physical coords for these nodes: 0 .. H_channel-1 (if H_channel = Ly-2)
                                                 # So, y_coords_lbm[1:-1] - 1.0 should map to 0 to H_channel-1

    # Create physical y-coordinates for analytical solution (0 to H_channel)
    # H_channel = Ly - 2 (number of fluid cells)
    # y_analytical_plot = np.linspace(0, H_channel, len(analytical_velocities)) #This assumes analytical_vel has H_channel points
    # The analytical solution was calculated on y_coords_physical which should match the LBM fluid nodes' physical locations.

    # For plotting, we want y to go from 0 (bottom wall fluid interface) to H_channel (top wall fluid interface)
    # LBM results are at cell centers. y_coords_lbm[1:-1] are indices of fluid cells.
    # If Ly = 20, H_channel = 18. Fluid cells are 1, ..., 18.
    # y_plot_lbm = (np.arange(1, Ly-1) - 1 + 0.5) # maps to 0.5, 1.5, ..., H_channel-0.5
    y_plot_lbm = (y_coords_lbm[1:Ly-1] - 1.0) + 0.5 # Centered points from 0.5 to H_channel - 0.5

    plt.plot(lbm_velocities[1:Ly-1], y_plot_lbm, 'bo-', label='LBM Simulation', markersize=5)

    # Analytical solution is already calculated on physical coordinates 0 to H_channel
    # We need y points for analytical that match the LBM cell centers.
    # y_analytical_points_for_plot = np.linspace(0.5, H_channel - 0.5, H_channel) # if analytical has H_channel points
    # Or, if analytical_velocities corresponds to y_coords_physical used in its calculation:
    y_analytical_plot_points = np.linspace(0.5, H_channel - 0.5, num=H_channel)


    plt.plot(analytical_velocities, y_analytical_plot_points, 'r--', label='Analytical Solution')

    plt.xlabel("Velocity (u_x)")
    plt.ylabel("y-position (relative to bottom wall)")
    plt.title(f"Poiseuille Flow Validation (Ly={Ly}, H_channel={H_channel})")
    plt.legend()
    plt.grid(True)
    plt.savefig("poiseuille_flow_validation.png")
    print("Plot saved as poiseuille_flow_validation.png")
    plt.show()

def calculate_discrepancy(lbm_velocities_fluid, analytical_velocities_fluid):
    """
    Calculates the Mean Squared Error (MSE) between LBM and analytical velocities.
    Assumes both inputs are 1D NumPy arrays of the same length, corresponding to the fluid domain.
    """
    if len(lbm_velocities_fluid) != len(analytical_velocities_fluid):
        raise ValueError("LBM and analytical velocity profiles must have the same length for discrepancy calculation.")

    mse = np.mean((lbm_velocities_fluid - analytical_velocities_fluid)**2)
    return mse

if __name__ == "__main__":
    # Simulation Parameters
    Ly_grid = 30      # Number of grid points in y (including walls)
    Lx_grid = Ly_grid * 2  # Length of the pipe (for periodic, aspect ratio matters less for fully developed)

    # Channel height H is the number of fluid cells. If walls are at y=0 and y=Ly_grid-1,
    # then there are Ly_grid-2 fluid cells.
    H_actual_channel = Ly_grid - 2

    # LBM parameters
    niu_lbm = 0.05     # Kinematic viscosity in LBM units
    fx_lbm = 1.0e-5   # Body force in x-direction (acts as G_effective if rho_avg=1)
    # G_effective for analytical solution. If using body force fx, G_eff = fx (assuming density is 1)
    # For pressure driven, G_eff = (P_in - P_out) / Lx_grid. Here, fx is direct.
    G_eff = fx_lbm

    total_sim_iterations = 20000 # Number of iterations to reach steady state (adjust as needed)

    # 1. Create Geometry
    pipe_geom = create_pipe_geometry(Lx_grid, Ly_grid)

    # 2. Run LBM Simulation
    lbm_velocity_profile, lbm_solver_instance = run_lbm_simulation_and_get_velocity(
        Lx_grid, Ly_grid, pipe_geom, niu_lbm, fx_lbm, total_sim_iterations
    )
    # lbm_velocity_profile is for all y-nodes, including walls.

    # 3. Calculate Analytical Solution
    # Physical y-coordinates for the fluid domain.
    # These should correspond to the centers of the fluid cells.
    # Fluid cells are from index 1 to Ly_grid-2.
    # Physical y: 0.5, 1.5, ..., H_actual_channel-0.5
    y_physical_coords_for_analytical = np.linspace(0.5, H_actual_channel - 0.5, num=H_actual_channel)

    analytical_velocity_profile = calculate_analytical_poiseuille(
        G_eff, niu_lbm, H_actual_channel, y_physical_coords_for_analytical
    )

    # 4. Plot Results
    # y_coords_for_lbm_plot are grid indices from 0 to Ly_grid-1
    y_grid_indices_lbm = np.arange(Ly_grid)

    plot_results(y_grid_indices_lbm, lbm_velocity_profile, analytical_velocity_profile, Ly_grid, H_actual_channel)

    # 5. Calculate and Print Discrepancy
    # Ensure we are comparing only the fluid parts.
    # lbm_velocity_profile includes walls (indices 0 and Ly_grid-1). Fluid is 1 to Ly_grid-2.
    lbm_fluid_velocities = lbm_velocity_profile[1:Ly_grid-1]

    # analytical_velocity_profile is already for the H_actual_channel fluid points.
    # Double check lengths to be sure:
    if len(lbm_fluid_velocities) != len(analytical_velocity_profile):
        print(f"Warning: Length mismatch for discrepancy calculation. LBM fluid: {len(lbm_fluid_velocities)}, Analytical: {len(analytical_velocity_profile)}")
        # This might happen if y_physical_coords_for_analytical was defined with a different num
        # compared to H_actual_channel, or if slicing lbm_profile is off.
        # H_actual_channel = Ly_grid - 2. Slicing [1:Ly_grid-1] gives Ly_grid-1-1 = Ly_grid-2 elements. Correct.
        # y_physical_coords_for_analytical = np.linspace(0.5, H_actual_channel - 0.5, num=H_actual_channel). Correct.

    mse_discrepancy = calculate_discrepancy(lbm_fluid_velocities, analytical_velocity_profile)
    print(f"Mean Squared Error (MSE) between LBM and analytical velocities: {mse_discrepancy:.6e}")

    print("Validation script finished.")
