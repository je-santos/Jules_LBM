from sympy import inverse_mellin_transform # This import seems unused. Consider removing if not needed.
import taichi as ti
import numpy as np
from pyevtk.hl import gridToVTK
import time

# Example Taichi initialization: ti.init(arch=ti.gpu, dynamic_index=False, kernel_profiler=True)

@ti.data_oriented
class LB3D_Solver_Single_Phase:
    """
    3D Lattice Boltzmann Method (LBM) solver for single-phase flow using the D3Q19 model
    with a Multiple Relaxation Time (MRT) collision operator.

    The solver supports periodic boundaries and fixed pressure (density)
    boundary conditions. It can also handle external forces using the Guo-Zheng forcing scheme.
    Sparse storage options are available for memory efficiency in simulations with large void spaces.
    Solid boundaries are handled by bounce-back on nodes marked as solid in the geometry.

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
        # Default type is 'periodic'. Value is for 'pressure' type.
        self.boundary_configs = {
            'x_left': {'type': 'periodic', 'value': None},
            'x_right': {'type': 'periodic', 'value': None},
            'y_left': {'type': 'periodic', 'value': None},
            'y_right': {'type': 'periodic', 'value': None},
            'z_left': {'type': 'periodic', 'value': None},
            'z_right': {'type': 'periodic', 'value': None},
        }

        # --- Taichi Fields for Boundary Conditions (used by kernels) ---
        # Type: 0 for periodic, 1 for pressure.
        self.bc_type_x_left = ti.field(ti.i32, shape=())
        self.bc_value_rho_x_left = ti.field(ti.f32, shape=())
        self.bc_type_x_right = ti.field(ti.i32, shape=())
        self.bc_value_rho_x_right = ti.field(ti.f32, shape=())
        self.bc_type_y_left = ti.field(ti.i32, shape=())
        self.bc_value_rho_y_left = ti.field(ti.f32, shape=())
        self.bc_type_y_right = ti.field(ti.i32, shape=())
        self.bc_value_rho_y_right = ti.field(ti.f32, shape=())
        self.bc_type_z_left = ti.field(ti.i32, shape=())
        self.bc_value_rho_z_left = ti.field(ti.f32, shape=())
        self.bc_type_z_right = ti.field(ti.i32, shape=())
        self.bc_value_rho_z_right = ti.field(ti.f32, shape=())

        # --- Core LBM Taichi Fields ---
        if not self.sparse_storage:
            self.f = ti.Vector.field(19, ti.f32, shape=(nx, ny, nz))
            self.F = ti.Vector.field(19, ti.f32, shape=(nx, ny, nz))
            self.rho = ti.field(ti.f32, shape=(nx, ny, nz))
            self.v = ti.Vector.field(3, ti.f32, shape=(nx, ny, nz))
        else:
            self.f = ti.Vector.field(19, ti.f32)
            self.F = ti.Vector.field(19, ti.f32)
            self.rho = ti.field(ti.f32)
            self.v = ti.Vector.field(3, ti.f32)
            n_mem_partition = 3
            cell_block = ti.root.pointer(ti.ijk, (nx // n_mem_partition + 1, ny // n_mem_partition + 1, nz // n_mem_partition + 1))
            cell_block.dense(ti.ijk, (n_mem_partition, n_mem_partition, n_mem_partition)).place(self.rho, self.v, self.f, self.F)

        self.solid = ti.field(ti.i8, shape=(nx, ny, nz))
        self.max_v = ti.field(ti.f32, shape=())
        self.ext_f = ti.Vector.field(3, ti.f32, shape=())

        # --- D3Q19 Model Constants ---
        self.e = ti.Vector.field(3, ti.i32, shape=19)
        self.e_f = ti.Vector.field(3, ti.f32, shape=19)
        self.w = ti.field(ti.f32, shape=19)
        self.LR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]

        M_np = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,-2,-2,-2,-2,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1],
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
            [0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,-1,1,1,-1]
        ])
        inv_M_np = np.linalg.inv(M_np)
        self.M = ti.Matrix.field(19, 19, ti.f32, shape=())
        self.inv_M = ti.Matrix.field(19, 19, ti.f32, shape=())
        self.M[None] = ti.Matrix(M_np)
        self.inv_M[None] = ti.Matrix(inv_M_np)

        self.S_dig = ti.Vector.field(19,ti.f32,shape=())

        self.x = np.linspace(0, nx-1, nx)
        self.y = np.linspace(0, ny-1, ny)
        self.z = np.linspace(0, nz-1, nz)

    def init_simulation(self):
        """Initializes simulation parameters and fields."""
        for face_name, config in self.boundary_configs.items():
            if face_name == 'x_left':
                if config['type'] == 'periodic': self.bc_type_x_left[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_x_left[None] = 1
                    self.bc_value_rho_x_left[None] = config['value'] if config['value'] is not None else 1.0
            elif face_name == 'x_right':
                if config['type'] == 'periodic': self.bc_type_x_right[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_x_right[None] = 1
                    self.bc_value_rho_x_right[None] = config['value'] if config['value'] is not None else 1.0
            elif face_name == 'y_left':
                if config['type'] == 'periodic': self.bc_type_y_left[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_y_left[None] = 1
                    self.bc_value_rho_y_left[None] = config['value'] if config['value'] is not None else 1.0
            elif face_name == 'y_right':
                if config['type'] == 'periodic': self.bc_type_y_right[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_y_right[None] = 1
                    self.bc_value_rho_y_right[None] = config['value'] if config['value'] is not None else 1.0
            elif face_name == 'z_left':
                if config['type'] == 'periodic': self.bc_type_z_left[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_z_left[None] = 1
                    self.bc_value_rho_z_left[None] = config['value'] if config['value'] is not None else 1.0
            elif face_name == 'z_right':
                if config['type'] == 'periodic': self.bc_type_z_right[None] = 0
                elif config['type'] == 'pressure':
                    self.bc_type_z_right[None] = 1
                    self.bc_value_rho_z_right[None] = config['value'] if config['value'] is not None else 1.0

        cs2 = 1.0 / 3.0
        self.tau_f = self.niu / cs2 + 0.5
        self.s_v = 1.0 / self.tau_f
        self.s_other = 8.0 * (2.0 - self.s_v) / (8.0 - self.s_v)
        self.S_dig[None] = ti.Vector([
            0.0, self.s_v, self.s_v, 0.0, self.s_other, 0.0, self.s_other, 0.0, self.s_other,
            self.s_v, self.s_v, self.s_v, self.s_v, self.s_v, self.s_v, self.s_v,
            self.s_other, self.s_other, self.s_other
        ])

        self.ext_f[None] = ti.Vector([self.fx, self.fy, self.fz])
        if abs(self.fx) > 1e-9 or abs(self.fy) > 1e-9 or abs(self.fz) > 1e-9:
            self.force_flag = 1
        else:
            self.force_flag = 0

        self.static_init()
        self.init()

    @ti.func
    def feq(self, k: ti.i32, rho_local: ti.f32, u: ti.template()):
        """Equilibrium distribution function."""
        eu = self.e[k].dot(u)
        uv = u.dot(u)
        return self.w[k] * rho_local * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def init(self):
        """Initializes fields to rest state."""
        for i,j,k in self.solid:
            if (self.sparse_storage==False or ti.is_active(self.solid.parent(),[i,j,k])):
                if self.solid[i,j,k] == 0:
                    self.rho[i,j,k] = 1.0
                    self.v[i,j,k] = ti.Vector([0.0, 0.0, 0.0])
                    for s_idx in ti.static(range(19)):
                        eq = self.feq(s_idx, 1.0, self.v[i,j,k])
                        self.f[i,j,k][s_idx] = eq
                        self.F[i,j,k][s_idx] = eq

    def init_geo(self, filename: str):
        """Initializes geometry from file."""
        in_dat = np.loadtxt(filename)
        in_dat[in_dat > 0] = 1
        loaded_solid_np = np.reshape(in_dat, (self.nx, self.ny, self.nz), order='F')
        self.solid.from_numpy(loaded_solid_np)
        if self.sparse_storage:
            self.deactivate_solid_nodes()

    @ti.kernel
    def deactivate_solid_nodes(self):
        """Placeholder for sparse storage optimization."""
        pass

    @ti.kernel
    def static_init(self):
        """Initializes D3Q19 lattice constants."""
        if ti.static(self.enable_projection):
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

            for i in ti.static(range(19)):
                self.e_f[i] = self.e[i].cast(float)

            self.w[0] = 1.0/3.0
            self.w[1] = 1.0/18.0; self.w[2] = 1.0/18.0
            self.w[3] = 1.0/18.0; self.w[4] = 1.0/18.0
            self.w[5] = 1.0/18.0; self.w[6] = 1.0/18.0
            self.w[7] = 1.0/36.0; self.w[8] = 1.0/36.0
            self.w[9] = 1.0/36.0; self.w[10] = 1.0/36.0
            self.w[11] = 1.0/36.0; self.w[12] = 1.0/36.0
            self.w[13] = 1.0/36.0; self.w[14] = 1.0/36.0
            self.w[15] = 1.0/36.0; self.w[16] = 1.0/36.0
            self.w[17] = 1.0/36.0; self.w[18] = 1.0/36.0

    @ti.func
    def meq_vec(self, rho_local,u):
        """Calculates equilibrium moments."""
        out = ti.Vector([0.0 for _ in range(19)]) # Ensure proper initialization
        out[0] = rho_local; out[3] = u[0]; out[5] = u[1]; out[7] = u[2];
        out[1] = u.dot(u); out[9] = 2*u.x*u.x-u.y*u.y-u.z*u.z; out[11] = u.y*u.y-u.z*u.z
        out[13] = u.x*u.y; out[14] = u.y*u.z; out[15] = u.x*u.z
        # Other moments (2,4,6,8,10,12,16,17,18) are often zero at equilibrium or related to higher order,
        # but the M matrix definition implies specific forms if they were non-zero.
        # For this standard MRT, these are sufficient.
        return out

    @ti.func
    def cal_local_force(self,i,j,k):
        f_vec = ti.Vector([self.fx, self.fy, self.fz])
        return f_vec

    @ti.func
    def _calculate_guo_force_moment_contribution(self, s_moment_index: ti.i32, local_v: ti.template(), local_force_vector: ti.template()) -> ti.f32:
        f_guo_contrib = 0.0
        for l_pop_index in ti.static(range(19)):
            term1_dot = (self.e_f[l_pop_index] - local_v).dot(local_force_vector)
            term2_dot_u = self.e_f[l_pop_index].dot(local_v)
            term2_dot_F = self.e_f[l_pop_index].dot(local_force_vector)
            current_l_contrib = self.w[l_pop_index] * ( term1_dot / 3.0 + (term2_dot_u * term2_dot_F) / 9.0 )
            f_guo_contrib += current_l_contrib * self.M[None][s_moment_index, l_pop_index]
        return f_guo_contrib

    @ti.kernel
    def colission(self):
        """MRT collision step with Guo-Zheng forcing."""
        for i,j,k in self.rho:
            if self.solid[i,j,k] == 0:
                m_f = self.M[None] @ self.F[i,j,k]
                current_rho = self.rho[i,j,k]
                current_v = self.v[i,j,k]
                meq = self.meq_vec(current_rho, current_v)

                m_collided = ti.Vector([0.0 for _ in range(19)])
                for s_idx in ti.static(range(19)):
                   m_collided[s_idx] = m_f[s_idx] - self.S_dig[None][s_idx] * (m_f[s_idx] - meq[s_idx])

                if ti.static(self.force_flag == 1):
                    f_local_vec = self.cal_local_force(i, j, k)
                    for s_idx in ti.static(range(19)):
                        guo_force_moment_s = self._calculate_guo_force_moment_contribution(s_idx, current_v, f_local_vec)
                        m_collided[s_idx] += (1.0 - 0.5 * self.S_dig[None][s_idx]) * guo_force_moment_s

                self.f[i,j,k] = self.inv_M[None] @ m_collided

    @ti.func
    def periodic_index(self, i_coord_component: ti.i32, limit: ti.i32) -> ti.i32:
        """Applies periodic boundary condition to a single coordinate component."""
        return (i_coord_component + limit) % limit

    @ti.kernel
    def streaming1(self):
        """Push-scheme streaming with bounce-back."""
        for i, j, k in self.f:
            if self.solid[i,j,k] == 0:
                for s_idx in ti.static(range(19)):
                    dest_i = i + self.e[s_idx][0]
                    dest_j = j + self.e[s_idx][1]
                    dest_k = k + self.e[s_idx][2]

                    di_w = self.periodic_index(dest_i, self.nx)
                    dj_w = self.periodic_index(dest_j, self.ny)
                    dk_w = self.periodic_index(dest_k, self.nz)

                    if self.solid[di_w, dj_w, dk_w] == 0:
                        self.F[di_w, dj_w, dk_w][s_idx] = self.f[i,j,k][s_idx]
                    else:
                        self.F[i,j,k][self.LR[s_idx]] = self.f[i,j,k][s_idx]

    @ti.kernel
    def Boundary_condition(self):
        """Applies fixed pressure boundary conditions."""
        # x_left boundary (i=0)
        if ti.static(self.bc_type_x_left[None] == 1): # 1: Fixed Pressure
            for j,k in ti.ndrange(self.ny, self.nz):
                if self.solid[0,j,k] == 0:
                    v_bc_node = self.v[0,j,k]
                    if self.solid[1,j,k] == 0 :
                         v_bc_node = self.v[1,j,k]
                    for s_idx in ti.static(range(19)):
                        self.F[0,j,k][s_idx] = self.feq(s_idx, self.bc_value_rho_x_left[None], v_bc_node)

        # x_right boundary (i = self.nx-1)
        if ti.static(self.bc_type_x_right[None] == 1):
            for j,k in ti.ndrange(self.ny, self.nz):
                if self.solid[self.nx-1,j,k] == 0:
                    v_bc_node = self.v[self.nx-1,j,k]
                    if self.solid[self.nx-2,j,k] == 0:
                         v_bc_node = self.v[self.nx-2,j,k]
                    for s_idx in ti.static(range(19)):
                        self.F[self.nx-1,j,k][s_idx] = self.feq(s_idx, self.bc_value_rho_x_right[None], v_bc_node)

        # y_left boundary (j=0)
        if ti.static(self.bc_type_y_left[None] == 1):
            for i,k in ti.ndrange(self.nx, self.nz):
                if self.solid[i,0,k] == 0:
                    v_bc_node = self.v[i,0,k]
                    if self.solid[i,1,k] == 0:
                        v_bc_node = self.v[i,1,k]
                    for s_idx in ti.static(range(19)):
                        self.F[i,0,k][s_idx] = self.feq(s_idx, self.bc_value_rho_y_left[None], v_bc_node)

        # y_right boundary (j = self.ny-1)
        if ti.static(self.bc_type_y_right[None] == 1):
            for i,k in ti.ndrange(self.nx, self.nz):
                if self.solid[i,self.ny-1,k] == 0:
                    v_bc_node = self.v[i,self.ny-1,k]
                    if self.solid[i,self.ny-2,k] == 0:
                        v_bc_node = self.v[i,self.ny-2,k]
                    for s_idx in ti.static(range(19)):
                        self.F[i,self.ny-1,k][s_idx] = self.feq(s_idx, self.bc_value_rho_y_right[None], v_bc_node)

        # z_left boundary (k=0)
        if ti.static(self.bc_type_z_left[None] == 1):
            for i,j in ti.ndrange(self.nx, self.ny):
                if self.solid[i,j,0] == 0:
                    v_bc_node = self.v[i,j,0]
                    if self.solid[i,j,1] == 0:
                        v_bc_node = self.v[i,j,1]
                    for s_idx in ti.static(range(19)):
                        self.F[i,j,0][s_idx] = self.feq(s_idx, self.bc_value_rho_z_left[None], v_bc_node)

        # z_right boundary (k = self.nz-1)
        if ti.static(self.bc_type_z_right[None] == 1):
            for i,j in ti.ndrange(self.nx, self.ny):
                if self.solid[i,j,self.nz-1] == 0:
                    v_bc_node = self.v[i,j,self.nz-1]
                    if self.solid[i,j,self.nz-2] == 0:
                        v_bc_node = self.v[i,j,self.nz-2]
                    for s_idx in ti.static(range(19)):
                        self.F[i,j,self.nz-1][s_idx] = self.feq(s_idx, self.bc_value_rho_z_right[None], v_bc_node)

    @ti.kernel
    def streaming3(self):
        """Updates macroscopic variables and applies second part of force term."""
        for i,j,k in self.f:
            if self.solid[i,j,k] == 0:
                self.rho[i,j,k] = 0.0
                for s_idx in ti.static(range(19)):
                    self.rho[i,j,k] += self.f[i,j,k][s_idx]

                self.v[i,j,k] = ti.Vector([0.0, 0.0, 0.0])
                for s_idx in ti.static(range(19)):
                    self.v[i,j,k] += self.e_f[s_idx] * self.f[i,j,k][s_idx]

                force_vec = self.cal_local_force(i,j,k)

                if self.rho[i,j,k] > 1e-12:
                    self.v[i,j,k] = (self.v[i,j,k] + force_vec / 2.0) / self.rho[i,j,k]
                else:
                    self.v[i,j,k] = ti.Vector([0.0, 0.0, 0.0])
            else:
                self.rho[i,j,k] = 1.0
                self.v[i,j,k] = ti.Vector([0.0, 0.0, 0.0])

    def get_max_v(self):
        """Calculates and returns the maximum velocity magnitude."""
        self.max_v[None] = -1e10
        self.cal_max_v()
        return self.max_v[None]

    @ti.kernel
    def cal_max_v(self):
        """Taichi kernel to calculate the maximum velocity magnitude."""
        for I in ti.grouped(self.rho):
            if self.solid[I] == 0:
                ti.atomic_max(self.max_v[None], self.v[I].norm())

    def set_boundary_condition(self, face_name: str, bc_type: str, value=None):
        """
        Sets the boundary condition for a specified face.
        Args:
            face_name (str): e.g., 'x_left', 'x_right', etc.
            bc_type (str): 'periodic' or 'pressure'.
            value (optional): For 'pressure', the density value. Ignored for 'periodic'.
        """
        if face_name not in self.boundary_configs:
            raise ValueError(f"Invalid face_name: {face_name}. Must be one of {list(self.boundary_configs.keys())}")
        if bc_type not in ['periodic', 'pressure']:
            raise ValueError(f"Invalid bc_type: {bc_type}. Must be 'periodic' or 'pressure'.")

        self.boundary_configs[face_name]['type'] = bc_type
        if bc_type == 'pressure':
            if not isinstance(value, (float, int)):
                raise ValueError("Value for 'pressure' BC must be a float or int.")
            self.boundary_configs[face_name]['value'] = float(value)
        else: # 'periodic'
            self.boundary_configs[face_name]['value'] = None

    def set_viscosity(self, niu: float):
        """Sets the kinematic viscosity."""
        self.niu = niu

    def set_force(self, force: list[float] | tuple[float,float,float] | np.ndarray): # type: ignore
        """Sets the external force vector."""
        if not (isinstance(force, (list, tuple, np.ndarray)) and len(force) == 3): # type: ignore
             raise ValueError("Force must be a list/tuple/array of 3 floats.")
        self.fx = float(force[0])
        self.fy = float(force[1])
        self.fz = float(force[2])

    def export_VTK(self, n: int):
        """Exports current state to VTK file."""
        filename = f"./LB_SingelPhase_{n}"
        gridToVTK(
                filename, self.x, self.y, self.z,
                pointData={
                    "Solid": np.ascontiguousarray(self.solid.to_numpy()),
                    "rho": np.ascontiguousarray(self.rho.to_numpy()),
                    "velocity": (
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,0]),
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,1]),
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,2])
                    )
                }
            )

    def step(self):
        """Performs one full LBM step."""
        self.colission()
        self.streaming1()
        self.Boundary_condition()
        self.streaming3()

# Example usage (commented out)
'''
# import taichi as ti
# import numpy as np
# import time
#
# ti.init(arch=ti.gpu)
#
# solver = LB3D_Solver_Single_Phase(nx=50, ny=50, nz=50)
# solver.init_geo('./geo_cavity.dat')
# solver.set_viscosity(0.01)
# # solver.set_boundary_condition('y_right', 'pressure', 1.02) # Example for pressure
# solver.init_simulation()
#
# print("Starting simulation...")
# time_init = time.time()
# for iter_num in range(5001):
#     solver.step()
#     if iter_num % 1000 == 0:
#         max_vel = solver.get_max_v()
#         print(f"Iteration: {iter_num}, Max Velocity: {max_vel:.4e}, Time: {time.time() - time_init:.2f}s")
#         if iter_num % 1000 == 0:
#             solver.export_VTK(iter_num)
# time_end = time.time()
# print(f"Simulation finished. Total time: {time_end - time_init:.2f}s")
'''
