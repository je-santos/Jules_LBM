from sympy import inverse_mellin_transform
import taichi as ti
import numpy as np
from pyevtk.hl import gridToVTK
import time

#ti.init(arch=ti.gpu, dynamic_index=False, kernel_profiler=True, print_ir=False) # Example Taichi initialization

@ti.data_oriented
class LB3D_Solver_Single_Phase:
    """
    3D Lattice Boltzmann Method (LBM) solver for single-phase flow using the D3Q19 model
    with a Multiple Relaxation Time (MRT) collision operator.

    The solver supports periodic boundaries, fixed pressure (density), and fixed velocity
    boundary conditions. It can also handle external forces using the Guo-Zheng forcing scheme.
    Sparse storage options are available for memory efficiency in simulations with large void spaces.

    Key steps in the LBM algorithm implemented:
    1. Collision: Distribution functions relax towards an equilibrium state.
    2. Streaming: Distribution functions propagate to neighboring lattice nodes.
    3. Boundary Conditions: Applied at domain edges to simulate different physical scenarios.
    4. Macroscopic Update: Density and velocity are calculated from the distribution functions.
    """
    def __init__(self, nx: int, ny: int, nz: int, sparse_storage: bool = False):
        """
        Initializes the LBM solver with grid dimensions and storage options.

        Args:
            nx (int): Number of grid nodes in the x-direction.
            ny (int): Number of grid nodes in the y-direction.
            nz (int): Number of grid nodes in the z-direction.
            sparse_storage (bool, optional): If True, uses Taichi's sparse data structures
                                             for fields like f, F, rho, v. Defaults to False.
        """
        # --- Core Simulation Parameters ---
        self.nx, self.ny, self.nz = nx, ny, nz  # Grid dimensions
        self.niu = 0.16667  # Kinematic viscosity
        self.fx, self.fy, self.fz = 0.0e-6, 0.0, 0.0  # External force components (default values)

        # --- Taichi Runtime Configuration ---
        self.enable_projection = True # Related to static_init, affects e and w initialization.
        self.sparse_storage = sparse_storage

        # --- Boundary Condition Configuration (Python-side) ---
        # This dictionary stores the configuration for each boundary face.
        # 'type' can be 'periodic', 'pressure', or 'velocity'.
        # 'value' is the corresponding pressure (float) or velocity (list/array of 3 floats).
        self.boundary_configs = {
            'x_left': {'type': 'periodic', 'value': None},
            'x_right': {'type': 'periodic', 'value': None},
            'y_left': {'type': 'periodic', 'value': None},
            'y_right': {'type': 'periodic', 'value': None},
            'z_left': {'type': 'periodic', 'value': None},
            'z_right': {'type': 'periodic', 'value': None},
        }

        # --- Taichi Fields for Boundary Conditions (used by kernels) ---
        # These fields are populated from `boundary_configs` in `init_simulation`.
        # Type: 0 for periodic, 1 for pressure, 2 for velocity.
        self.bc_type_x_left = ti.field(ti.i32, shape=())
        self.bc_value_rho_x_left = ti.field(ti.f32, shape=())
        self.bc_value_vel_x_left = ti.Vector.field(3, ti.f32, shape=())
        self.bc_type_x_right = ti.field(ti.i32, shape=())
        self.bc_value_rho_x_right = ti.field(ti.f32, shape=())
        self.bc_value_vel_x_right = ti.Vector.field(3, ti.f32, shape=())
        self.bc_type_y_left = ti.field(ti.i32, shape=())
        self.bc_value_rho_y_left = ti.field(ti.f32, shape=())
        self.bc_value_vel_y_left = ti.Vector.field(3, ti.f32, shape=())
        self.bc_type_y_right = ti.field(ti.i32, shape=())
        self.bc_value_rho_y_right = ti.field(ti.f32, shape=())
        self.bc_value_vel_y_right = ti.Vector.field(3, ti.f32, shape=())
        self.bc_type_z_left = ti.field(ti.i32, shape=())
        self.bc_value_rho_z_left = ti.field(ti.f32, shape=())
        self.bc_value_vel_z_left = ti.Vector.field(3, ti.f32, shape=())
        self.bc_type_z_right = ti.field(ti.i32, shape=())
        self.bc_value_rho_z_right = ti.field(ti.f32, shape=())
        self.bc_value_vel_z_right = ti.Vector.field(3, ti.f32, shape=())

        # --- Core LBM Taichi Fields ---
        # These fields store the primary simulation data.
        if not self.sparse_storage:
            self.f = ti.Vector.field(19, ti.f32, shape=(nx, ny, nz))  # Distribution functions
            self.F = ti.Vector.field(19, ti.f32, shape=(nx, ny, nz))  # Post-collision distribution functions
            self.rho = ti.field(ti.f32, shape=(nx, ny, nz))          # Macroscopic density
            self.v = ti.Vector.field(3, ti.f32, shape=(nx, ny, nz)) # Macroscopic velocity
        else: # Sparse storage setup
            self.f = ti.Vector.field(19, ti.f32)
            self.F = ti.Vector.field(19, ti.f32)
            self.rho = ti.field(ti.f32)
            self.v = ti.Vector.field(3, ti.f32)
            n_mem_partition = 3  # Example partitioning, can be adjusted based on domain size and memory
            cell_block = ti.root.pointer(ti.ijk, (nx // n_mem_partition + 1, ny // n_mem_partition + 1, nz // n_mem_partition + 1))
            cell_block.dense(ti.ijk, (n_mem_partition, n_mem_partition, n_mem_partition)).place(self.rho, self.v, self.f, self.F)

        self.solid = ti.field(ti.i8, shape=(nx, ny, nz)) # Solid geometry (0 for fluid, 1 for solid node)
        self.max_v = ti.field(ti.f32, shape=())          # For monitoring maximum velocity in the domain
        self.ext_f = ti.Vector.field(3, ti.f32, shape=()) # External force vector (Taichi field)

        # --- D3Q19 Model Constants and Derived Parameters ---
        # These are specific to the LBM model (D3Q19 in this case).
        self.e = ti.Vector.field(3, ti.i32, shape=19)    # Lattice velocities (integer representation)
        self.e_f = ti.Vector.field(3, ti.f32, shape=19)  # Lattice velocities (float representation)
        self.w = ti.field(ti.f32, shape=19)              # Weights for equilibrium distribution
        self.LR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17] # Indices of opposite (reflected) lattice directions

        # MRT (Multiple Relaxation Time) Collision Operator Matrices
        # M is the transformation matrix from population space to moment space.
        # inv_M is its inverse, transforming moments back to populations.
        # These are constant for a given lattice model.
        M_np = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1], # Energy (related to e = rho * cs^2) - this is actually related to momentum flux for MRT
            [1,-2,-2,-2,-2,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1], # Energy squared (related to epsilon)
        [0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
        [0,-2,2,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
        [0,0,0,1,-1,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
        [0,0,0,-2,2,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
        [0,0,0,0,0,1,-1,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
        [0,0,0,0,0,-2,2,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
        [0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
        [0,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
        [0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
        [0,0,0,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1],
        [0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0],
        [0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0],
        [0,0,0,0,0,0,0,-1,1,1,-1,0,0,0,0,1,-1,1,-1],
            [0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,-1,1,1,-1] # Higher order moments
        ])
        inv_M_np = np.linalg.inv(M_np)
        self.M = ti.Matrix.field(19, 19, ti.f32, shape=())
        self.inv_M = ti.Matrix.field(19, 19, ti.f32, shape=())
        self.M[None] = ti.Matrix(M_np)      # Store numpy array in Taichi field
        self.inv_M[None] = ti.Matrix(inv_M_np) # Store numpy array in Taichi field

        self.S_dig = ti.Vector.field(19,ti.f32,shape=()) # Diagonal matrix S for MRT collision step, populated in init_simulation

        # --- Coordinate System (primarily for VTK export) ---
        # These are numpy arrays, not Taichi fields, used by pyevtk.
        self.x = np.linspace(0, nx-1, nx)
        self.y = np.linspace(0, ny-1, ny) # y-coordinates for grid nodes
        self.z = np.linspace(0, nz-1, nz) # z-coordinates for grid nodes

    def init_simulation(self):
        """
        Initializes simulation-specific parameters and Taichi fields.
        This method must be called after setting desired physical parameters (viscosity, force)
        and boundary conditions, and before starting the simulation steps.

        Key actions:
        - Transfers boundary condition settings from Python dictionary to Taichi fields.
        - Calculates derived parameters like relaxation times and rates.
        - Populates the diagonal relaxation matrix `S_dig` for MRT.
        - Sets up external force parameters for the kernels.
        - Calls `static_init()` to set D3Q19 model constants.
        - Calls `init()` to initialize macroscopic fields (rho, v) and distribution functions (f, F).
        """
        # 1. Populate Taichi fields for boundary conditions from the Python `boundary_configs` dictionary.
        # This makes BC information directly accessible to Taichi kernels.
        for face_name, config in self.boundary_configs.items():
            if face_name == 'x_left':
                if config['type'] == 'periodic':
                    self.bc_type_x_left[None] = 0 # 0: Periodic
                elif config['type'] == 'pressure':
                    self.bc_type_x_left[None] = 1 # Pressure
                    self.bc_value_rho_x_left[None] = config['value'] if config['value'] is not None else 1.0 # Default pressure/rho
                elif config['type'] == 'velocity':
                    self.bc_type_x_left[None] = 2 # 2: Velocity
                    self.bc_value_vel_x_left[None] = ti.Vector(config['value'] if config['value'] is not None else [0.0, 0.0, 0.0]) # Default velocity
            elif face_name == 'x_right':
                if config['type'] == 'periodic':
                    self.bc_type_x_right[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_x_right[None] = 1
                    self.bc_value_rho_x_right[None] = config['value'] if config['value'] is not None else 1.0
                elif config['type'] == 'velocity':
                    self.bc_type_x_right[None] = 2
                    self.bc_value_vel_x_right[None] = ti.Vector(config['value'] if config['value'] is not None else [0.0, 0.0, 0.0])
            elif face_name == 'y_left':
                if config['type'] == 'periodic':
                    self.bc_type_y_left[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_y_left[None] = 1
                    self.bc_value_rho_y_left[None] = config['value'] if config['value'] is not None else 1.0
                elif config['type'] == 'velocity':
                    self.bc_type_y_left[None] = 2
                    self.bc_value_vel_y_left[None] = ti.Vector(config['value'] if config['value'] is not None else [0.0, 0.0, 0.0])
            elif face_name == 'y_right':
                if config['type'] == 'periodic':
                    self.bc_type_y_right[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_y_right[None] = 1
                    self.bc_value_rho_y_right[None] = config['value'] if config['value'] is not None else 1.0
                elif config['type'] == 'velocity':
                    self.bc_type_y_right[None] = 2
                    self.bc_value_vel_y_right[None] = ti.Vector(config['value'] if config['value'] is not None else [0.0, 0.0, 0.0])
            elif face_name == 'z_left':
                if config['type'] == 'periodic':
                    self.bc_type_z_left[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_z_left[None] = 1
                    self.bc_value_rho_z_left[None] = config['value'] if config['value'] is not None else 1.0
                elif config['type'] == 'velocity':
                    self.bc_type_z_left[None] = 2
                    self.bc_value_vel_z_left[None] = ti.Vector(config['value'] if config['value'] is not None else [0.0, 0.0, 0.0])
            elif face_name == 'z_right':
                if config['type'] == 'periodic':
                    self.bc_type_z_right[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_z_right[None] = 1
                    self.bc_value_rho_z_right[None] = config['value'] if config['value'] is not None else 1.0
                elif config['type'] == 'velocity':
                    self.bc_type_z_right[None] = 2
                    self.bc_value_vel_z_right[None] = ti.Vector(config['value'] if config['value'] is not None else [0.0, 0.0, 0.0])

        # 2. Calculate derived simulation parameters
        # Relaxation time `tau_f` is derived from kinematic viscosity `niu`.
        # The lattice speed of sound squared, cs^2, is 1/3 for D3Q19.
        # Formula: niu = cs^2 * (tau_f - 0.5), so tau_f = niu / cs^2 + 0.5
        cs2 = 1.0 / 3.0
        self.tau_f = self.niu / cs2 + 0.5

        # Relaxation rates for MRT collision model.
        # s_v corresponds to shear viscosity, s_other to other moments (often related to bulk viscosity or fixed).
        self.s_v = 1.0 / self.tau_f
        # This specific form for s_other is common in some MRT literature, e.g., D'Humieres.
        # It aims to maintain stability and accuracy.
        self.s_other = 8.0 * (2.0 - self.s_v) / (8.0 - self.s_v)

        # 3. Populate the diagonal entries of the relaxation matrix `S_dig` for MRT collision.
        # These values depend on the calculated relaxation rates.
        self.S_dig[None] = ti.Vector([
            0.0, self.s_v, self.s_v, 0.0, self.s_other, 0.0, self.s_other, 0.0, self.s_other,
            self.s_v, self.s_v, self.s_v, self.s_v, self.s_v, self.s_v, self.s_v,
            self.s_other, self.s_other, self.s_other
        ])

        # 4. Set up external force handling.
        self.ext_f[None] = ti.Vector([self.fx, self.fy, self.fz]) # Transfer Python attributes to Taichi field
        # Determine if force needs to be applied in the collision step.
        if abs(self.fx) > 1e-9 or abs(self.fy) > 1e-9 or abs(self.fz) > 1e-9: # Use a small epsilon for float comparison
            self.force_flag = 1
        else:
            self.force_flag = 0

        # Note: `ti.static` calls on `self.inv_M`, `self.M`, `self.S_dig` are not needed here
        # as they are already Taichi fields. `ti.static` is used for compile-time constants or loop unrolling.

        # 5. Call initialization kernels.
        self.static_init() # Initializes D3Q19 model constants (e, w, e_f) using @ti.kernel.
        self.init()        # Initializes macroscopic variables (rho, v) and distribution functions (f, F) using @ti.kernel.

    @ti.func
    def feq(self, k: ti.i32, rho_local: ti.f32, u: ti.template()):
        """
        Calculates the equilibrium distribution function for a given population `k`,
        local density `rho_local`, and local velocity `u`.
        D3Q19 model's second-order equilibrium distribution.
        """
        eu = self.e[k].dot(u) # Projection of u onto e_k
        uv = u.dot(u)         # Magnitude squared of u
        # Standard second-order equilibrium distribution function
        feqout = self.w[k] * rho_local * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)
        return feqout

    @ti.kernel
    def init(self):
        """
        Initializes the macroscopic fields (density `rho`, velocity `v`) and
        the distribution functions (`f`, `F`) to a state of rest (zero velocity)
        and unit density for all fluid nodes. Solid nodes are typically not modified here
        as their state is static (or handled by geometry initialization).
        """
        for i,j,k in self.solid: # Iterates over all grid cells
            if (self.sparse_storage==False or ti.is_active(self.solid.parent(),[i,j,k])) : # Ensure cell is active for sparse storage
                if self.solid[i,j,k] == 0: # Fluid node
                    self.rho[i,j,k] = 1.0
                    self.v[i,j,k] = ti.Vector([0.0, 0.0, 0.0])
                    for s_idx in ti.static(range(19)): # 19 populations for D3Q19
                        eq = self.feq(s_idx, 1.0, self.v[i,j,k])
                        self.f[i,j,k][s_idx] = eq
                        self.F[i,j,k][s_idx] = eq
                # else: solid nodes, rho/v can remain undefined or set to a marker if needed, f/F are irrelevant.

    def init_geo(self, filename: str):
        """
        Initializes the simulation geometry from a file.
        The file should contain a 3D array representing the solid (1) and fluid (0) nodes.
        The array is flattened in Fortran order (column-major).

        Args:
            filename (str): Path to the geometry file.
        """
        in_dat = np.loadtxt(filename)
        in_dat[in_dat > 0] = 1 # Ensure binary: 0 for fluid, 1 for solid
        # Reshape according to Fortran order (column-major)
        loaded_solid_np = np.reshape(in_dat, (self.nx, self.ny, self.nz), order='F')
        self.solid.from_numpy(loaded_solid_np)

        # For sparse storage, ensure solid nodes are deactivated to save memory/computation
        if self.sparse_storage:
            self.deactivate_solid_nodes()

    @ti.kernel
    def deactivate_solid_nodes(self):
        """
        Deactivates Taichi sparse grid blocks that contain only solid nodes.
        This is relevant only when `sparse_storage` is True.
        """
        # This is a placeholder; actual deactivation logic depends on Taichi's sparse API
        # and how `solid` field is used to infer block activity.
        # For instance, if a block is entirely solid, its f, F, rho, v fields could be deactivated.
        # Taichi might do some of this automatically if pointers are not placed for solid-only blocks.
        # A more explicit deactivation would require iterating blocks and checking if all cells in it are solid.
        # For now, this is a conceptual step. Actual Taichi sparse optimization might involve
        # careful structuring of `place` calls or using `ti.deactivate()` on parent pointers if applicable.
        pass # TODO: Implement if finer-grained sparse deactivation is needed and supported easily.


    @ti.kernel
    def static_init(self):
        # This kernel initializes the D3Q19 lattice velocities (e, e_f) and weights (w).
        # These are fundamental constants of the LBM model and are set once.
        # The `ti.static(self.enable_projection)` guard seems to be a remnant;
        # these initializations are essential for D3Q19.
        if ti.static(self.enable_projection):
            # Integer lattice velocities (e)
            self.e[0] = ti.Vector([0,0,0])
            self.e[1] = ti.Vector([1,0,0]); self.e[2] = ti.Vector([-1,0,0])
            self.e[3] = ti.Vector([0,1,0]); self.e[4] = ti.Vector([0,-1,0])
            self.e[5] = ti.Vector([0,0,1]); self.e[6] = ti.Vector([0,0,-1])
            self.e[7] = ti.Vector([1,1,0]); self.e[8] = ti.Vector([-1,-1,0])
            self.e[9] = ti.Vector([1,-1,0]); self.e[10] = ti.Vector([-1,1,0])
            self.e[11] = ti.Vector([1,0,1]); self.e[12] = ti.Vector([-1,0,-1])
            self.e[13] = ti.Vector([1,0,-1]); self.e[14] = ti.Vector([-1,0,1])
            self.e[15] = ti.Vector([0,1,1]); self.e[16] = ti.Vector([0,-1,-1])
            self.e[17] = ti.Vector([0,1,-1]); self.e[18] = ti.Vector([0,-1,1])

            # Float version of lattice velocities (e_f)
            # This conversion is done once and stored.
            for i in ti.static(range(19)):
                self.e_f[i] = self.e[i].cast(float)

            # D3Q19 weights (w)
            self.w[0] = 1.0/3.0
            self.w[1] = 1.0/18.0; self.w[2] = 1.0/18.0 # Directions (+-1, 0, 0) and permutations
            self.w[3] = 1.0/18.0; self.w[4] = 1.0/18.0
            self.w[5] = 1.0/18.0; self.w[6] = 1.0/18.0
            self.w[7] = 1.0/36.0; self.w[8] = 1.0/36.0 # Directions (+-1, +-1, 0) and permutations
            self.w[9] = 1.0/36.0; self.w[10] = 1.0/36.0
            self.w[11] = 1.0/36.0; self.w[12] = 1.0/36.0 # Directions (+-1, 0, +-1) and permutations
            self.w[13] = 1.0/36.0; self.w[14] = 1.0/36.0
            self.w[15] = 1.0/36.0; self.w[16] = 1.0/36.0 # Directions (0, +-1, +-1) and permutations
            self.w[17] = 1.0/36.0; self.w[18] = 1.0/36.0

    #@ti.func
    #def GuoF(self,i,j,k,s,u,f):
    #    out=0.0
    #    for l in ti.static(range(19)):
    #        out += self.w[l]*((self.e_f[l]-u).dot(f)+(self.e_f[l].dot(u)*(self.e_f[l].dot(f))))*self.M[None][s,l]
    #
    #    return out


    @ti.func
    def meq_vec(self, rho_local,u):
        out = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        out[0] = rho_local;             out[3] = u[0];    out[5] = u[1];    out[7] = u[2];
        out[1] = u.dot(u);    out[9] = 2*u.x*u.x-u.y*u.y-u.z*u.z;         out[11] = u.y*u.y-u.z*u.z
        out[13] = u.x*u.y;    out[14] = u.y*u.z;                            out[15] = u.x*u.z
        return out

    @ti.func
    def cal_local_force(self,i,j,k):
        f_vec = ti.Vector([self.fx, self.fy, self.fz])
        return f_vec

    @ti.func
    def _calculate_guo_force_moment_contribution(self, s_moment_index: ti.i32, local_v: ti.template(), local_force_vector: ti.template()) -> ti.f32:
        f_guo_contrib = 0.0
        for l_pop_index in ti.static(range(19)): # 19 is self.e.shape[0] or number of discrete velocities
            term1_dot = (self.e_f[l_pop_index] - local_v).dot(local_force_vector)
            term2_dot_u = self.e_f[l_pop_index].dot(local_v)
            term2_dot_F = self.e_f[l_pop_index].dot(local_force_vector)

            # Guo-Zheng forcing term in moment space for a given moment s_moment_index,
            # summed over all populations l_pop_index. cs^2 = 1/3 is assumed.
            current_l_contrib = self.w[l_pop_index] * ( term1_dot / 3.0 + (term2_dot_u * term2_dot_F) / 9.0 )
            f_guo_contrib += current_l_contrib * self.M[None][s_moment_index, l_pop_index]
        return f_guo_contrib

    @ti.kernel
    def colission(self):
        """
        Performs the MRT collision step, including relaxation to equilibrium and external force application.
        This kernel iterates over all fluid cells.
        """
        for i,j,k in self.rho: # Iterate over all cells (Taichi handles sparse iteration if active)
            if self.solid[i,j,k] == 0: # Process only fluid cells
                # Note: Boundary checks like i < self.nx are implicitly handled by Taichi field access,
                # assuming fields are defined correctly up to nx, ny, nz.

                # 1. Transform distribution functions from population space to moment space
                # F[i,j,k] contains the populations after streaming from the previous step.
                m_f = self.M[None] @ self.F[i,j,k]

                # 2. Calculate equilibrium moments (meq) based on current macroscopic variables.
                current_rho = self.rho[i,j,k]
                current_v = self.v[i,j,k]
                meq = self.meq_vec(current_rho, current_v) # meq is in moment space

                # 3. Apply MRT collision step: relax moments towards their equilibrium values.
                # m_collided = m_f - S * (m_f - meq)
                # where S is the diagonal relaxation matrix (self.S_dig).
                m_collided = ti.Vector([0.0 for _ in range(19)])
                for s_idx in ti.static(range(19)):
                   m_collided[s_idx] = m_f[s_idx] - self.S_dig[None][s_idx] * (m_f[s_idx] - meq[s_idx])

                # 4. Apply Guo-Zheng external force term in moment space (if force is enabled).
                # The force term is added to the already collided moments.
                if ti.static(self.force_flag == 1):
                    f_local_vec = self.cal_local_force(i, j, k) # Get local force vector
                    for s_idx in ti.static(range(19)):
                        # Calculate force contribution for the current moment s_idx
                        guo_force_moment_s = self._calculate_guo_force_moment_contribution(s_idx, current_v, f_local_vec)
                        # Add force contribution, scaled by (I - S/2) factor part of the forcing scheme.
                        m_collided[s_idx] += (1.0 - 0.5 * self.S_dig[None][s_idx]) * guo_force_moment_s

                # 5. Transform moments back to population space to get post-collision, post-forcing distribution functions (self.f).
                self.f[i,j,k] = self.inv_M[None] @ m_collided





    @ti.func
    def periodic_index(self, i_coord_component: ti.i32, limit: ti.i32) -> ti.i32:
        """Applies periodic boundary condition to a single coordinate component."""
        return (i_coord_component + limit) % limit

    @ti.kernel
    def streaming1(self):
        """
        Performs the streaming step using a "pull" scheme with bounce-back for solid boundaries.
        The post-collision distribution functions `self.f` are streamed to `self.F` (which becomes
        the pre-collision distribution for the next step).
        Periodic boundary conditions are handled here implicitly if no other BC is active for a face.
        """
        for i, j, k in self.f: # Iterate over all cells
            if self.solid[i,j,k] == 0: # Process only fluid cells
                for s_idx in ti.static(range(19)):
                    # Determine source coordinates (si, sj, sk) for the "pull" scheme
                    si = i - self.e[s_idx][0]
                    sj = j - self.e[s_idx][1]
                    sk = k - self.e[s_idx][2]

                    # Apply periodic boundary conditions to source coordinates if they fall outside
                    # This check is for non-BC faces or if periodic is explicitly set.
                    # For faces with specific BCs (pressure/velocity), those are handled in Boundary_condition kernel.

                    # Tentative: This periodic handling might conflict if a face has another BC.
                    # It's generally better to handle periodic wrapping only if no other BC is set for that face.
                    # However, the original `periodic_index(i+self.e[s])` was more about where `f[i]` goes to.
                    # Let's stick to the original logic structure first, then refine if needed.
                    # The original logic was: calculate destination `ip = i + self.e[s]`.
                    # If `ip` is fluid, `F[ip] = f[i]`. If `ip` is solid, bounce back at `i`.
                    # This is a "push" like thinking for `F[ip] = f[i]`, but the bounce back is at `i`.

                    # Re-evaluating original logic:
                    # for i in ti.grouped(self.rho): # This means i,j,k
                    #   if (self.solid[i] == 0 ...):
                    #     for s in ti.static(range(19)):
                    #       ip = self.periodic_index(i+self.e[s]) # ip is destination
                    #       if (self.solid[ip]==0): # If destination is fluid
                    #         self.F[ip][s] = self.f[i][s] # PUSH from i to ip
                    #       else: # If destination is solid (bounce-back)
                    #         self.F[i][self.LR[s]] = self.f[i][s] # Store at current 'i', but opposite direction population

                    # This is a "push" scheme for the main streaming part.
                    # The `periodic_index` function in the original code wrapped around the whole domain.

                    # Current cell (i,j,k) is fluid.
                    # We are calculating F[dest_i, dest_j, dest_k][s_idx] or F[i,j,k][LR[s_idx]]

                    dest_i = i + self.e[s_idx][0]
                    dest_j = j + self.e[s_idx][1]
                    dest_k = k + self.e[s_idx][2]

                    # Apply periodic wrapping for destination indices
                    di_w = self.periodic_index(dest_i, self.nx)
                    dj_w = self.periodic_index(dest_j, self.ny)
                    dk_w = self.periodic_index(dest_k, self.nz)

                    if self.solid[di_w, dj_w, dk_w] == 0: # If destination is fluid
                        self.F[di_w, dj_w, dk_w][s_idx] = self.f[i,j,k][s_idx]
                    else: # If destination is solid, bounce-back at the current fluid cell i,j,k
                        self.F[i,j,k][self.LR[s_idx]] = self.f[i,j,k][s_idx]


    @ti.kernel
    def Boundary_condition(self):
        """
        Applies fixed pressure and fixed velocity boundary conditions.
        Periodic boundaries are implicitly handled by `periodic_index` in `streaming1` if
        the corresponding `bc_type_...` is 0 (periodic).
        This kernel overwrites `self.F` at boundary nodes for non-periodic conditions.
        """
        # Example for x_left boundary (i=0)
        if ti.static(self.bc_type_x_left[None] == 1): # 1: Fixed Pressure
            for j,k in ti.ndrange(self.ny, self.nz): # Iterate over the yz-plane at x=0
                if self.solid[0,j,k] == 0: # If it's a fluid node
                    # For pressure BC, velocity of the boundary node itself or its fluid neighbor is used for feq.
                    # If the immediate neighbor (1,j,k) is solid, use v[0,j,k] (extrapolation / bounce-back like).
                    # Otherwise, use v[1,j,k] (value from fluid side). Original logic was:
                    # if (self.solid[1,j,k]>0): self.v[1,j,k] else: self.v[0,j,k]
                    # This seems to refer to the velocity *at* the boundary node for feq calculation.
                    # For Zou-He, rho is known, u is unknown. Here rho is set, u is taken from neighbor or self.
                    # Let's assume v_bc_node is the velocity used for feq at this boundary node.
                    v_bc_node = self.v[0,j,k] # Default to self if neighbor is solid or for simplicity.
                    if self.solid[1,j,k] == 0 : # If inner neighbor is fluid
                         v_bc_node = self.v[1,j,k] # More stable to use fluid neighbor's velocity

                    for s_idx in ti.static(range(19)):
                        self.F[0,j,k][s_idx] = self.feq(s_idx, self.bc_value_rho_x_left[None], v_bc_node)

        elif ti.static(self.bc_type_x_left[None] == 2): # 2: Fixed Velocity
            for j,k in ti.ndrange(self.ny, self.nz):
                if self.solid[0,j,k] == 0:
                    # Density is typically taken as 1.0 or rho of neighbor for Zou-He.
                    # Here, using 1.0 as per original `feq(s,1.0,ti.Vector(self.bc_vel_x_left))`
                    rho_bc_node = 1.0
                    # If inner neighbor is fluid, could use its density:
                    # if self.solid[1,j,k] == 0: rho_bc_node = self.rho[1,j,k]

                    for s_idx in ti.static(range(19)):
                        self.F[0,j,k][s_idx] = self.feq(s_idx, rho_bc_node, self.bc_value_vel_x_left[None])

        # x_right boundary (i = self.nx-1)
        if ti.static(self.bc_type_x_right[None] == 1): # Pressure
            for j,k in ti.ndrange(self.ny, self.nz):
                if self.solid[self.nx-1,j,k] == 0:
                    v_bc_node = self.v[self.nx-1,j,k]
                    if self.solid[self.nx-2,j,k] == 0:
                         v_bc_node = self.v[self.nx-2,j,k]
                    for s_idx in ti.static(range(19)):
                        self.F[self.nx-1,j,k][s_idx] = self.feq(s_idx, self.bc_value_rho_x_right[None], v_bc_node)
        elif ti.static(self.bc_type_x_right[None] == 2): # Velocity
            for j,k in ti.ndrange(self.ny, self.nz):
                if self.solid[self.nx-1,j,k] == 0:
                    rho_bc_node = 1.0
                    # if self.solid[self.nx-2,j,k] == 0: rho_bc_node = self.rho[self.nx-2,j,k]
                    for s_idx in ti.static(range(19)):
                        self.F[self.nx-1,j,k][s_idx] = self.feq(s_idx, rho_bc_node, self.bc_value_vel_x_right[None])

        # y_left boundary (j=0)
        if ti.static(self.bc_type_y_left[None] == 1): # Pressure
            for i,k in ti.ndrange(self.nx, self.nz):
                if self.solid[i,0,k] == 0:
                    v_bc_node = self.v[i,0,k]
                    if self.solid[i,1,k] == 0:
                        v_bc_node = self.v[i,1,k]
                    for s_idx in ti.static(range(19)):
                        self.F[i,0,k][s_idx] = self.feq(s_idx, self.bc_value_rho_y_left[None], v_bc_node)
        elif ti.static(self.bc_type_y_left[None] == 2): # Velocity
            for i,k in ti.ndrange(self.nx, self.nz):
                if self.solid[i,0,k] == 0:
                    rho_bc_node = 1.0
                    # if self.solid[i,1,k] == 0: rho_bc_node = self.rho[i,1,k]
                    for s_idx in ti.static(range(19)):
                        self.F[i,0,k][s_idx] = self.feq(s_idx, rho_bc_node, self.bc_value_vel_y_left[None])

        # y_right boundary (j = self.ny-1)
        if ti.static(self.bc_type_y_right[None] == 1): # Pressure
            for i,k in ti.ndrange(self.nx, self.nz):
                if self.solid[i,self.ny-1,k] == 0:
                    v_bc_node = self.v[i,self.ny-1,k]
                    if self.solid[i,self.ny-2,k] == 0:
                        v_bc_node = self.v[i,self.ny-2,k]
                    for s_idx in ti.static(range(19)):
                        self.F[i,self.ny-1,k][s_idx] = self.feq(s_idx, self.bc_value_rho_y_right[None], v_bc_node)
        elif ti.static(self.bc_type_y_right[None] == 2): # Velocity
            for i,k in ti.ndrange(self.nx, self.nz):
                if self.solid[i,self.ny-1,k] == 0:
                    rho_bc_node = 1.0
                    # if self.solid[i,self.ny-2,k] == 0: rho_bc_node = self.rho[i,self.ny-2,k]
                    for s_idx in ti.static(range(19)):
                        self.F[i,self.ny-1,k][s_idx] = self.feq(s_idx, rho_bc_node, self.bc_value_vel_y_right[None])

        # z_left boundary (k=0)
        if ti.static(self.bc_type_z_left[None] == 1): # Pressure
            for i,j in ti.ndrange(self.nx, self.ny):
                if self.solid[i,j,0] == 0:
                    v_bc_node = self.v[i,j,0]
                    if self.solid[i,j,1] == 0:
                        v_bc_node = self.v[i,j,1]
                    for s_idx in ti.static(range(19)):
                        self.F[i,j,0][s_idx] = self.feq(s_idx, self.bc_value_rho_z_left[None], v_bc_node)
        elif ti.static(self.bc_type_z_left[None] == 2): # Velocity
            for i,j in ti.ndrange(self.nx, self.ny):
                if self.solid[i,j,0] == 0:
                    rho_bc_node = 1.0
                    # if self.solid[i,j,1] == 0: rho_bc_node = self.rho[i,j,1]
                    for s_idx in ti.static(range(19)):
                        self.F[i,j,0][s_idx] = self.feq(s_idx, rho_bc_node, self.bc_value_vel_z_left[None])

        # z_right boundary (k = self.nz-1)
        if ti.static(self.bc_type_z_right[None] == 1): # Pressure
            for i,j in ti.ndrange(self.nx, self.ny):
                if self.solid[i,j,self.nz-1] == 0:
                    v_bc_node = self.v[i,j,self.nz-1]
                    if self.solid[i,j,self.nz-2] == 0:
                        v_bc_node = self.v[i,j,self.nz-2]
                    for s_idx in ti.static(range(19)):
                        self.F[i,j,self.nz-1][s_idx] = self.feq(s_idx, self.bc_value_rho_z_right[None], v_bc_node)
        elif ti.static(self.bc_type_z_right[None] == 2): # Velocity
            for i,j in ti.ndrange(self.nx, self.ny):
                if self.solid[i,j,self.nz-1] == 0:
                    rho_bc_node = 1.0
                    # if self.solid[i,j,self.nz-2] == 0: rho_bc_node = self.rho[i,j,self.nz-2]
                    for s_idx in ti.static(range(19)):
                        self.F[i,j,self.nz-1][s_idx] = self.feq(s_idx, rho_bc_node, self.bc_value_vel_z_right[None])

    @ti.kernel
    def streaming3(self):
        """
        Updates macroscopic variables (density `rho` and velocity `v`) from the
        post-collision distribution functions `self.f`.
        Also applies the second part of the Guo-Zheng force term.
        For solid nodes, rho is set to 1.0 and v to zero.
        """
        for i,j,k in self.f: # Iterate over all cells
            if self.solid[i,j,k] == 0: # Fluid node
                # Summing distribution functions to get density
                self.rho[i,j,k] = 0.0 # Initialize sum for current cell
                for s_idx in ti.static(range(19)):
                    self.rho[i,j,k] += self.f[i,j,k][s_idx]

                # Summing f_k * e_k to get momentum, then velocity
                self.v[i,j,k] = ti.Vector([0.0, 0.0, 0.0]) # Initialize sum for current cell
                for s_idx in ti.static(range(19)):
                    self.v[i,j,k] += self.e_f[s_idx] * self.f[i,j,k][s_idx]

                # Apply external force contribution (second part of Guo-Zheng scheme)
                # This is F_i * dt / (2 * rho), where dt=1. Here, force_vec is F_i.
                force_vec = self.cal_local_force(i,j,k) # Get force at current cell

                # First part of velocity from momentum (rho*u = sum(e*f))
                # Second part from force term before density division
                # v_from_momentum = self.v[i,j,k] / self.rho[i,j,k] (if rho != 0)
                # v_force_adjust = (force_vec / 2.0) / self.rho[i,j,k] (if rho != 0)
                # self.v[i,j,k] = v_from_momentum + v_force_adjust

                if self.rho[i,j,k] > 1e-12: # Avoid division by zero if density is extremely low
                    # Momentum without force: sum(e*f)
                    # Velocity without force: sum(e*f) / rho
                    # Add force contribution: u = sum(e*f)/rho + F_ext/(2*rho) (dt=1)
                    # So, v_field = ( sum(e*f) + F_ext/2 ) / rho
                    self.v[i,j,k] = (self.v[i,j,k] + force_vec / 2.0) / self.rho[i,j,k]
                else: # If rho is effectively zero, set velocity to zero to prevent NaN/Inf
                    self.v[i,j,k] = ti.Vector([0.0, 0.0, 0.0])
            else: # Solid node
                self.rho[i,j,k] = 1.0 # Conventionally, solid rho is 1 or same as fluid initial.
                self.v[i,j,k] = ti.Vector([0.0, 0.0, 0.0]) # No velocity in solid.

    def get_max_v(self):
        """Calculates and returns the maximum velocity magnitude in the fluid domain."""
        self.max_v[None] = -1e10 # Reset before calculation
        self.cal_max_v() # Kernel call to compute max_v
        return self.max_v[None]

    @ti.kernel
    def cal_max_v(self):
        """Taichi kernel to calculate the maximum velocity magnitude."""
        for I in ti.grouped(self.rho): # Iterate over all cells
            if self.solid[I] == 0: # Consider only fluid cells
                ti.atomic_max(self.max_v[None], self.v[I].norm())

    def set_boundary_condition(self, face_name: str, bc_type: str, value=None):
        """
        Sets the boundary condition for a specified face.

        Args:
            face_name (str): The name of the face to configure. Must be one of
                             'x_left', 'x_right', 'y_left', 'y_right', 'z_left', 'z_right'.
            bc_type (str): The type of boundary condition. Must be one of
                           'periodic', 'pressure', or 'velocity'.
            value (optional): The value for the boundary condition.
                              - For 'pressure': a float representing the density.
                              - For 'velocity': a list/tuple/array of 3 floats for vx, vy, vz.
                              - For 'periodic': this value is ignored.
        Raises:
            ValueError: If `face_name` or `bc_type` is invalid, or if `value` is
                        inappropriate for the given `bc_type`.
        """
        if face_name not in self.boundary_configs:
            raise ValueError(f"Invalid face_name: {face_name}. Must be one of {list(self.boundary_configs.keys())}")
        if bc_type not in ['periodic', 'pressure', 'velocity']:
            raise ValueError(f"Invalid bc_type: {bc_type}. Must be 'periodic', 'pressure', or 'velocity'.")

        self.boundary_configs[face_name]['type'] = bc_type
        if bc_type == 'pressure':
            if not isinstance(value, (float, int)):
                raise ValueError("Value for 'pressure' BC must be a float or int.")
            self.boundary_configs[face_name]['value'] = float(value)
        elif bc_type == 'velocity':
            if not (isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3): # type: ignore
                raise ValueError("Value for 'velocity' BC must be a list/tuple/array of 3 floats.")
            self.boundary_configs[face_name]['value'] = [float(v) for v in value]
        else: # 'periodic'
            self.boundary_configs[face_name]['value'] = None

        # Note: After changing boundary conditions, `init_simulation()` should ideally be called
        # again before running steps to ensure Taichi fields for BCs are updated.
        # However, if changed mid-simulation, the new values in `boundary_configs` will be picked up
        # by `init_simulation` if it's called, or directly by kernels if they were adapted to read from it (not current design).

    def set_viscosity(self, niu: float):
        """Sets the kinematic viscosity of the fluid."""
        self.niu = niu

    def set_force(self, force: list[float] | tuple[float,float,float] | np.ndarray): # type: ignore
        """Sets the external force vector (fx, fy, fz)."""
        if not (isinstance(force, (list, tuple, np.ndarray)) and len(force) == 3): # type: ignore
             raise ValueError("Force must be a list/tuple/array of 3 floats.")
        self.fx = float(force[0])
        self.fy = float(force[1])
        self.fz = float(force[2])


    def export_VTK(self, n: int):
        """
        Exports the current simulation state to a VTK file.

        Args:
            n (int): An integer suffix for the filename (e.g., iteration number).
        """
        filename = f"./LB_SingelPhase_{n}" # Consistent naming
        gridToVTK(
                filename,
                self.x, # Grid coordinates
                self.y, # Grid coordinates
                self.z, # Grid coordinates
                pointData={
                    "Solid": np.ascontiguousarray(self.solid.to_numpy()), # Solid geometry
                    "rho": np.ascontiguousarray(self.rho.to_numpy()),     # Density field
                    "velocity": ( # Velocity field (vx, vy, vz components)
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,0]),
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,1]),
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,2])
                    )
                }
            )

    def step(self):
        """Performs one full step of the LBM simulation (collision, streaming, BCs, macroscopic update)."""
        self.colission()
        self.streaming1() # Populates F based on f
        self.Boundary_condition() # Modifies F at boundaries
        self.streaming3() # Updates rho, v based on (now modified) F (which is copied to f internally)

# Example usage (commented out, intended for separate script or interactive use)
'''
# import taichi as ti # Ensure Taichi is imported
# import numpy as np  # For geometry or parameter definitions if needed
# import time         # For timing the simulation
#
# # It's good practice to have Taichi initialization outside the class,
# # typically at the beginning of the script.
# ti.init(arch=ti.gpu) # Initialize Taichi, e.g., on GPU

# # Solver setup
# # Solver setup for a 50x50x50 grid
# solver = LB3D_Solver_Single_Phase(nx=50, ny=50, nz=50)
#
# # Load geometry from a file (assuming 'geo_cavity.dat' is in the correct format and path)
# # The geo_cavity.dat should contain 0s for fluid and 1s for solid, flattened in Fortran order.
# solver.init_geo('./geo_cavity.dat')
#
# # Set physical parameters
# solver.set_viscosity(0.01) # Example viscosity
# # solver.set_force([1e-5, 0.0, 0.0]) # Example: Apply a small force in x-direction (Poiseuille flow)
#
# # Example: Lid-driven cavity setup
# # Top wall (y_right, j=ny-1) moves with u=(0.1, 0, 0)
# # Assuming the geometry file has made the top wall solid (part of the cavity boundary)
# # To make it a lid, that specific wall should be fluid and have a velocity BC.
# # For this example, let's assume 'geo_cavity.dat' defines an open top lid.
# # If the top layer is solid in geo_cavity.dat, this BC won't have an effect unless that layer is fluid.
# # A common way for lid-driven cavity is to have all walls solid, then set velocity on the top layer of *fluid* nodes.
# # Or, ensure the geometry has the lid as fluid nodes and apply velocity BC there.
# # For simplicity, we'll assume the top face (y_right) is meant to be fluid or handled by a specific geometry.
# solver.set_boundary_condition('y_right', 'velocity', [0.1, 0.0, 0.0])
#
# # Set other boundaries if they are not periodic by default or defined by solid geometry
# # For a typical cavity, other walls are no-slip (solid).
# # If the geometry file already defines these walls as solid (1), no further BCs are needed for them.
# # If they are fluid boundaries and need to be no-slip, set them:
# # solver.set_boundary_condition('x_left', 'velocity', [0.0, 0.0, 0.0])
# # solver.set_boundary_condition('x_right', 'velocity', [0.0, 0.0, 0.0])
# # solver.set_boundary_condition('y_left', 'velocity', [0.0, 0.0, 0.0]) # Bottom wall
# # solver.set_boundary_condition('z_left', 'velocity', [0.0, 0.0, 0.0])
# # solver.set_boundary_condition('z_right', 'velocity', [0.0, 0.0, 0.0])
#
# # Finalize setup before starting simulation steps
# solver.init_simulation()
#
# print("Starting simulation...")
# time_init = time.time()
#
# # Simulation loop
# for iter_num in range(5001): # Run for 5000 iterations
#     solver.step()
#
#     if iter_num % 1000 == 0: # Output information every 1000 iterations
#         time_now = time.time()
#         max_vel = solver.get_max_v()
#         print(f"Iteration: {iter_num}, Max Velocity: {max_vel:.4e}, Time: {time_now - time_init:.2f}s")
#
#         if iter_num % 1000 == 0: # Adjust VTK export frequency as needed
#             solver.export_VTK(iter_num)
#
# time_end = time.time()
# print(f"Simulation finished. Total time: {time_end - time_init:.2f}s")
#
# # Taichi profiler information (optional, if kernel_profiler=True in ti.init)
# # ti.profiler.print_kernel_profiler_info()
'''
