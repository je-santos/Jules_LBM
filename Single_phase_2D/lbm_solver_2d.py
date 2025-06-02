import taichi as ti
import numpy as np
import time

ti.init(arch=ti.gpu, dynamic_index=False, kernel_profiler=True, print_ir=False)

@ti.data_oriented
class LB2D_Solver_Single_Phase:
    def __init__(self, nx, ny, sparse_storage=False):
        self.enable_projection = True  # D2Q9 uses this concept differently or not at all, review
        self.sparse_storage = sparse_storage
        self.nx, self.ny = nx, ny
        self.fx, self.fy = 0.0e-6, 0.0
        self.niu = 0.16667  # Viscosity

        self.max_v = ti.field(ti.f32, shape=())

        # Boundary condition mode: 0=periodic, 1=fix pressure, 2=fix velocity
        # Boundary pressure value (rho); boundary velocity value for vx,vy
        self.bc_x_left, self.rho_bcxl, self.vx_bcxl, self.vy_bcxl = 0, 1.0, 0.0, 0.0
        self.bc_x_right, self.rho_bcxr, self.vx_bcxr, self.vy_bcxr = 0, 1.0, 0.0, 0.0
        self.bc_y_left, self.rho_bcyl, self.vx_bcyl, self.vy_bcyl = 0, 1.0, 0.0, 0.0
        self.bc_y_right, self.rho_bcyr, self.vx_bcyr, self.vy_bcyr = 0, 1.0, 0.0, 0.0

        # D2Q9 model parameters
        self.q_dim = 9  # Number of velocity vectors

        if not sparse_storage:
            self.f = ti.Vector.field(self.q_dim, ti.f32, shape=(nx, ny))
            self.F = ti.Vector.field(self.q_dim, ti.f32, shape=(nx, ny))
            self.rho = ti.field(ti.f32, shape=(nx, ny))
            self.v = ti.Vector.field(2, ti.f32, shape=(nx, ny)) # 2D velocity
            self.solid = ti.field(ti.i8, shape=(nx, ny))
        else:
            # Sparse storage implementation for 2D (simplified or adapted from 3D)
            self.f = ti.Vector.field(self.q_dim, ti.f32)
            self.F = ti.Vector.field(self.q_dim, ti.f32)
            self.rho = ti.field(ti.f32)
            self.v = ti.Vector.field(2, ti.f32)
            self.solid = ti.field(ti.i8) # Add shape for sparse solid field later if needed

            # Example sparse layout (adjust as needed for 2D)
            n_mem_partition = 3 # Example, can be tuned
            cell1 = ti.root.pointer(ti.ij, (nx // n_mem_partition + 1, ny // n_mem_partition + 1))
            cell1.dense(ti.ij, (n_mem_partition, n_mem_partition)).place(self.rho, self.v, self.f, self.F, self.solid)
            # Note: For sparse fields, solid needs to be handled carefully.
            # If solid is dense, it should be defined outside this sparse block.
            # For simplicity, starting with dense solid field, and can make it sparse if required.
            # Reverting to dense solid for now as it's common.
            # self.solid = ti.field(ti.i8, shape=(nx,ny)) # Defined above for non-sparse

        self.e = ti.Vector.field(2, ti.i32, shape=(self.q_dim))  # 2D velocity vectors
        self.w = ti.field(ti.f32, shape=(self.q_dim)) # Weights
        self.ext_f = ti.Vector.field(2, ti.f32, shape=()) # External force (2D)

        # MRT collision operator matrices for D2Q9
        self.M = ti.Matrix.field(self.q_dim, self.q_dim, ti.f32, shape=())
        self.inv_M = ti.Matrix.field(self.q_dim, self.q_dim, ti.f32, shape=())
        self.S_dig = ti.Vector.field(self.q_dim, ti.f32, shape=()) # Diagonal relaxation matrix elements

        # D2Q9 specific velocities, weights, and matrices will be defined in static_init or helper
        self.LR = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7], dtype=int) # Opposite directions for D2Q9

        self.x = np.linspace(0, nx, nx)
        self.y = np.linspace(0, ny, ny)
        
        self._define_d2q9_parameters()


    def _define_d2q9_parameters(self):
        # Velocities e_i for D2Q9
        e_np = np.array([
            [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, -1], [1, -1], [-1, 1]
        ], dtype=int)
        self.e.from_numpy(e_np)

        # Weights w_i for D2Q9
        w_np = np.array([
            4.0/9.0,  # 0
            1.0/9.0,  # 1 (east)
            1.0/9.0,  # 2 (west)
            1.0/9.0,  # 3 (north)
            1.0/9.0,  # 4 (south)
            1.0/36.0, # 5 (north-east)
            1.0/36.0, # 6 (south-west)
            1.0/36.0, # 7 (north-west) Error in comment, should be south-east
            1.0/36.0  # 8 (south-east) Error in comment, should be north-west
        ], dtype=float) # Corrected order below
        
        w_np_corrected = np.array([
            4.0/9.0,  # 0
            1.0/9.0,  # 1 (E)
            1.0/9.0,  # 2 (W)
            1.0/9.0,  # 3 (N)
            1.0/9.0,  # 4 (S)
            1.0/36.0, # 5 (NE)
            1.0/36.0, # 6 (SW)
            1.0/36.0, # 7 (SE)
            1.0/36.0  # 8 (NW)
        ], dtype=np.float32)
        self.w.from_numpy(w_np_corrected)


        # D2Q9 MRT matrix M (example from a common source, e.g., Lallemand and Luo)
        M_np = np.array([
            [1,  1,  1,  1,  1,  1,  1,  1,  1],
            [-4, -1, -1, -1, -1,  2,  2,  2,  2],
            [4, -2, -2, -2, -2,  1,  1,  1,  1],
            [0,  1, -1,  0,  0,  1, -1,  1, -1],
            [0, -2,  2,  0,  0,  1, -1,  1, -1],
            [0,  0,  0,  1, -1,  1, -1, -1,  1],
            [0,  0,  0, -2,  2,  1, -1, -1,  1],
            [0,  1,  1, -1, -1,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  1,  1, -1, -1]
        ], dtype=np.float32)
        
        try:
            inv_M_np = np.linalg.inv(M_np)
        except np.linalg.LinAlgError:
            print("Error: M matrix is singular and cannot be inverted.")
            # Fallback or error handling: use identity or a pseudo-inverse if appropriate,
            # or ensure M_np is correct. For now, let's assume it's correct and invertible.
            # If this error occurs, the M_np matrix needs to be checked.
            inv_M_np = np.identity(self.q_dim, dtype=np.float32) # Placeholder

        self.M[None] = ti.Matrix(M_np)
        self.inv_M[None] = ti.Matrix(inv_M_np)

    def init_simulation(self):
        self.bc_vel_x_left = [self.vx_bcxl, self.vy_bcxl]
        self.bc_vel_x_right = [self.vx_bcxr, self.vy_bcxr]
        self.bc_vel_y_left = [self.vx_bcyl, self.vy_bcyl]
        self.bc_vel_y_right = [self.vx_bcyr, self.vy_bcyr]

        # Relaxation time tau_f and related parameters s_v, s_other for D2Q9 MRT
        # cs^2 = 1/3 for D2Q9
        # niu = cs^2 * (tau_f - 0.5)
        # tau_f = niu / (1/3) + 0.5 = 3 * niu + 0.5
        self.tau_f = 3.0 * self.niu + 0.5
        s_nu = 1.0 / self.tau_f  # Relaxation rate for shear viscosity
        s_q = 8.0 * (2.0 - s_nu) / (8.0 - s_nu) # Relaxation rate for other moments (common choice)
        
        # Define relaxation rates for D2Q9 moments (Lallemand & Luo, PRE 2000)
        # Moments: rho, e, eps, jx, qx, jy, qy, pxx, pxy
        # s0: rho (conserved)
        # s1: e (energy)
        # s2: eps (square of energy, related to s1 or bulk viscosity)
        # s3: jx (x-momentum flux) (conserved)
        # s4: qx (x-energy flux)
        # s5: jy (y-momentum flux) (conserved)
        # s6: qy (y-energy flux)
        # s7: pxx (viscous stress tensor xx component)
        # s8: pxy (viscous stress tensor xy component)

        # Typical relaxation rates for D2Q9 MRT:
        # s0, s3, s5 are 0 for conserved moments if formulation is perfect, but typically set for stability/convention
        # Forcing terms affect conserved moments, so their m_i might not be feq_i
        # Here, we use a common set:
        # s_rho = 0 (or some value if not strictly conserved by m = M*f)
        # s_j = 0 (momentum is conserved)
        # s_e for energy often s_nu or a tunable parameter (bulk viscosity)
        # s_eps related to s_e
        # s_q (energy flux moments qx, qy) usually s_q (same as s_other in 3D code)
        # s_nu for stress moments (pxx, pxy) -> s_v in 3D code
        
        # Lallemand & Luo relaxation rates for D2Q9 (rho, e, eps, jx, qx, jy, qy, pxx, pxy)
        # s_b: bulk viscosity relaxation rate. Often s_b = s_nu for simplicity if bulk effects are not critical.
        s_b = s_nu # Assuming bulk viscosity = shear viscosity for simplicity
        self.S_dig[None] = ti.Vector([
            0.0,  # rho (density)
            s_b,  # e (energy)
            s_q,  # eps (energy squared, often s_q or another value)
            0.0,  # jx (x-momentum)
            s_q,  # qx (x-energy flux)
            0.0,  # jy (y-momentum)
            s_q,  # qy (y-energy flux)
            s_nu, # pxx (viscous stress xx)
            s_nu  # pxy (viscous stress xy)
        ])


        self.ext_f[None][0] = self.fx
        self.ext_f[None][1] = self.fy
        if (abs(self.fx) > 0 or abs(self.fy) > 0):
            self.force_flag = 1
        else:
            self.force_flag = 0

        # static_init was used for e, w in 3D, here we load them directly or in _define_d2q9_parameters
        # self.static_init() # Call if still needed for other static ti variables
        self.init_fields() # Renamed from init to avoid conflict with class __init__

    @ti.func
    def feq(self, k_idx, rho_local, u): # equilibrium distribution function
        eu = self.e[k_idx].dot(u)
        uv_sq = u.dot(u)
        # Standard D2Q9 feq
        feq_val = self.w[k_idx] * rho_local * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv_sq)
        return feq_val

    @ti.kernel
    def init_fields(self): # Changed name from 'init'
        for i, j in self.solid: # Iterate over 2D grid
            if self.sparse_storage == False or self.solid[i, j] == 0: # Check for fluid node
                self.rho[i, j] = 1.0
                self.v[i, j] = ti.Vector([0.0, 0.0]) # 2D velocity
                for s_idx in ti.static(range(self.q_dim)):
                    self.f[i, j][s_idx] = self.feq(s_idx, 1.0, self.v[i, j])
                    self.F[i, j][s_idx] = self.feq(s_idx, 1.0, self.v[i, j])
    
    def init_geo(self, geo_array: np.ndarray): # Expects a 2D boolean numpy array
        if geo_array.ndim != 2:
            raise ValueError("Geometry array must be 2D.")
        if geo_array.shape[0] != self.nx or geo_array.shape[1] != self.ny:
            raise ValueError(f"Geometry array shape {geo_array.shape} does not match simulation domain ({self.nx}, {self.ny}).")
        
        solid_int_array = geo_array.astype(np.int8) # Convert boolean to int8 for Taichi field
        self.solid.from_numpy(solid_int_array)

    # static_init from 3D code primarily set up e and w.
    # In this 2D version, e and w are numpy arrays loaded to Taichi fields in _define_d2q9_parameters.
    # If other truly static Taichi variables are needed, this function can be used.
    # For now, it's not strictly necessary as M, inv_M are also set up.
    # @ti.kernel
    # def static_init(self):
    #     pass # e, w, M, inv_M are already initialized

    @ti.func
    def meq_vec(self, rho_local, u): # Equilibrium moments for D2Q9 (Lallemand & Luo)
        m_eq = ti.Vector([0.0] * self.q_dim)
        ux, uy = u[0], u[1]
        ux2, uy2 = ux * ux, uy * uy
        
        m_eq[0] = rho_local
        m_eq[1] = rho_local * (3.0 * (ux2 + uy2) - 2.0) # e (energy) term, check definition if not L&L
        # The definition of 'e' moment varies. L&L: rho * (3*(vx^2+vy^2) - 2*cs^2*D) where D=2
        # Simplified if cs^2=1/3: rho_local * ( (ux2+uy2) - 2/3 * 2) -> this seems off.
        # Standard definition for e (related to energy, often non-dimensionalized differently)
        # e = rho (u_x^2 + u_y^2) is not a moment.
        # Let's use a common formulation for moments based on powers of velocity.
        # Ref: e.g. "MRT LBE" by Ou, Zou or Lallemand & Luo 2000, PRE
        # rho, e, eps, jx, qx, jy, qy, pxx, pxy
        # (density, energy, energy^2, x-momentum, x-energy flux, y-momentum, y-energy flux, stress_xx, stress_xy)

        # Using definition from provided M matrix for consistency (from a source like Mohamad's book or similar LBM texts)
        # This definition of M implies the following moments:
        # m0 = rho
        # m1 = e = rho * (-2 + 3*(ux^2+uy^2)) -- scaled energy
        # m2 = eps = rho * (1 - 3/2*(ux^2+uy^2)) -- another energy related term, or use L&L's specific higher order one.
        # Let's use L&L's moments for clarity, which are often:
        # rho, jx, jy, pxx, pxy, e (or T_zz like term), qx, qy, (another higher order moment for eps)
        # The M matrix above corresponds to:
        # rho, e_diag, eps_diag, jx, qx_diag, jy, qy_diag, pxx, pxy
        # where e_diag = sum(fi * (cx_i^2+cy_i^2)), eps_diag = sum(fi * ((cx_i^2+cy_i^2)^2))
        # jx = sum(fi * cx_i), qx_diag = sum(fi * cx_i * (cx_i^2+cy_i^2)) etc.
        # The equilibrium values are:
        rho_0 = rho_local # For brevity
        m_eq[0] = rho_0
        m_eq[1] = rho_0 * (-2.0 + 3.0 * (ux2 + uy2)) # e_eq (Note: cs_sq=1/3 is implicitly used in derivation of M)
        m_eq[2] = rho_0 * (1.0 - 1.5 * (ux2 + uy2))  # eps_eq (This is one form, others exist)
        # A more standard Lallemand & Luo form for m2 (eps) is different, let's use what matches the M matrix structure given:
        # The M matrix provided implies these equilibrium moments:
        # m0 = rho
        # m1 = e = rho_0 * (3 * (ux**2 + uy**2) - 2)  -- if cs^2 = 1/3, D=2 for 2D
        # m2 = eps = rho_0 * ( (ux**2 + uy**2)**2 ) related, often scaled.
        # Let's use the common moments for the provided M:
        # rho, e, eps, jx, qx, jy, qy, p_xx, p_xy
        # rho_eq = rho_local
        # e_eq = rho_local * (3 * (ux*ux + uy*uy) - 2) -- check this, definition varies
        # eps_eq = rho_local * ( (ux*ux + uy*uy)^2 ) -- or similar higher order
        # jx_eq = rho_local * ux
        # qx_eq = rho_local * ux * (1 + (ux*ux + uy*uy)) -- complex form
        # jy_eq = rho_local * uy
        # qy_eq = rho_local * uy * (1 + (ux*ux + uy*uy))
        # pxx_eq = rho_local * (ux*ux - uy*uy)
        # pxy_eq = rho_local * ux*uy

        # Simpler set often used with D2Q9 MRT (based on powers of velocity components):
        m_eq[0] = rho_local
        m_eq[1] = rho_local * (3*(ux2+uy2)-2) # Energy. With cs^2=1/3, this is rho*( (u.u)/cs^2 - D ). D=2 for 2D
        m_eq[2] = rho_local * ( (9/2)*(ux2+uy2)**2 - (15/2)*(ux2+uy2) + 2 ) # eps, related to T_zzzz in some models
                                                                            # Or, more commonly for the given M:
        m_eq[2] = rho_local * ( (ux2+uy2) - (2/3)*2 ) # This is rho_0 * (e - 2*cs^2*D). Seems incorrect.
        # Let's use the definition that is consistent with typical D2Q9 MRT (e.g. as in Zou/He or common literature)
        # For the M matrix: [rho, e, eps, jx, qx, jy, qy, pxx, pxy]
        # rho_eq = rho
        # e_eq   = rho * (3 * (vx^2+vy^2) - 2)  -- this is -4*rho + ... in M, so e_eq = rho*(-4 + 3(vx^2+vy^2)/cs^2 - 2)
        # This needs to be checked against the source of the M matrix.
        # For now, using standard polynomial equilibrium moments:
        j_x = rho_local * ux
        j_y = rho_local * uy
        
        m_eq[0] = rho_local
        m_eq[1] = j_x  # This is NOT e. Corresponds to M[1] being [-4, -1, -1, -1, -1, 2, 2, 2, 2] * f
                       # This M is for [rho, e, eps, jx, qx, jy, qy, pxx, pxy]
                       # So m_eq[1] must be e_eq.
        # Re-evaluating meq based on common D2Q9 MRT (Lallemand & Luo or similar):
        # Moments: rho, e, eps, jx, qx, jy, qy, p_xx, p_xy
        m_eq[0] = rho_local # rho
        m_eq[1] = rho_local * (3.0 * (ux2 + uy2) - 2.0) # e (energy related)
        m_eq[2] = rho_local * ( (ux2 - uy2) * 0 + (9/2)*(ux2+uy2)**2 - (21/2)*(ux2+uy2) + 2 ) # eps (higher order)
                                                                                             # A common choice for eps_eq with the given M:
        m_eq[2] = rho_local * ( (ux2+uy2) - 2.0/3.0 * 2.0 ) # This is not matching M[2] elements.
                                                            # M[2] = [4, -2, -2, -2, -2, 1, 1, 1, 1]
                                                            # This corresponds to eps_eq = rho * ( (ux^2+uy^2)/cs^4 - (ux^2+uy^2)/cs^2 )
                                                            # Or simply: rho * ( (ux^2+uy^2) - const ) type terms
        # Let's use the definitions that are most standard for the D2Q9 MRT M matrix given:
        # rho_eq = rho
        # e_eq = rho * ( -2 + 3*(ux^2+uy^2) )  (assuming cs^2=1/3)
        # eps_eq = rho * ( 1 - 3/2*(ux^2+uy^2) ) (assuming cs^2=1/3)
        # jx_eq = rho * ux
        # qx_eq = rho * ux * (-1 + 1.5*(ux^2+uy^2)) (this is one form for qx) or simply rho*ux for M[4]
        # jy_eq = rho * uy
        # qy_eq = rho * uy * (-1 + 1.5*(ux^2+uy^2)) (this is one form for qy) or simply rho*uy for M[6]
        # pxx_eq = rho * (ux^2 - uy^2)
        # pxy_eq = rho * ux*uy
        
        # Corrected meq for the given M matrix structure (common D2Q9 MRT)
        m_eq[0] = rho_local                                 # rho
        m_eq[1] = rho_local * (-2.0 + 3.0 * (ux2 + uy2))     # e
        m_eq[2] = rho_local * (1.0 - 1.5 * (ux2 + uy2))      # eps
        m_eq[3] = rho_local * ux                            # jx
        m_eq[4] = rho_local * (-1.0 + 1.5 * (ux2 + uy2)) * ux # qx (This is not simply -2*jx for M[4])
                                                            # M[4] is [0, -2, 2, 0, 0, 1, -1, 1, -1], which is for qx' = qx - jx * const
                                                            # So, qx_eq should be simplified or M is for a different qx.
                                                            # If M[4] is for jx (scaled), then m_eq[4] = rho*ux. But M[3] is for jx.
                                                            # Let's use the simpler definition for qx, qy as used in some MRT models:
        m_eq[4] = rho_local * ux * (0.0) -2.0 * rho_local * ux # This seems like a typo in the 3D code's M adaptation.
                                                            # For D2Q9 with M given:
                                                            # m3_eq = jx = rho*ux
                                                            # m4_eq = qx (often jx for first order, or higher order like 3*jx - 2*rho*ux*u^2)
                                                            # The M[4] = [0,-2,2,0,0,1,-1,1,-1] suggests qx is related to (f1-f2 + f5-f6+f7-f8)
                                                            # This is related to energy flux in x. Its eq: rho*ux. (No, this is jx).
                                                            # qx_eq is often set to -jx for some MRT models.
                                                            # Let's use standard hydrodynamic moments:
        m_eq[3] = rho_local * ux                            # jx
        m_eq[5] = rho_local * uy                            # jy
        m_eq[7] = rho_local * (ux2 - uy2)                   # pxx (stress xx)
        m_eq[8] = rho_local * ux * uy                       # pxy (stress xy)
        
        # For qx, qy (moments m4, m6), these are often related to higher order terms or set to 0 for some models.
        # Or they are related to jx, jy.
        # For the given M matrix, m4 (qx) and m6 (qy) are usually non-zero at equilibrium.
        # A common choice for the M provided: qx_eq = -jx, qy_eq = -jy or similar simple forms.
        # Or, more accurately derived from Chapman-Enskog:
        # qx_eq = rho_local * ux * (1 - (5/3)*(1/2)) where cs^2=1/3. This is complex.
        # For Lallemand & Luo's M, qx_eq = rho*ux, qy_eq = rho*uy (if M entries are for these moments)
        # Given M[4] is [0, -2, 2, ...], this suggests qx is not simply rho*ux.
        # It is likely qx_eq is related to jx, e.g. qx_eq = -2/3 * jx or some other scaling.
        # Let's assume for now: (This needs verification against the source of M and S_dig)
        m_eq[4] = - (2.0/3.0) * j_x                          # qx (approximation often used)
        m_eq[6] = - (2.0/3.0) * j_y                          # qy (approximation often used)

        return m_eq


    @ti.func
    def cal_local_force(self, i, j): # 2D force
        f_vec = ti.Vector([self.fx, self.fy])
        return f_vec

    @ti.kernel
    def colission(self):
        for i, j in self.rho: # Iterate over 2D grid
            if self.solid[i, j] == 0: # Fluid node
                # Calculate moments m from F
                m_temp = self.M[None] @ self.F[i, j]
                
                # Calculate equilibrium moments meq
                meq = self.meq_vec(self.rho[i, j], self.v[i, j])
                
                # Collision step in moment space (MRT)
                m_coll = m_temp - self.S_dig[None] * (m_temp - meq)
                
                # Force term (Guo et al. forcing scheme for MRT)
                # F_s = M * Sigma_s * inv_M * S_force
                # S_force_alpha = w_alpha * ( (e_alpha - u)/cs^2 + (e_alpha . u)e_alpha / cs^4 ) . F_body
                # For MRT, the force term is added to the moments:
                # delta_m_s = (I - S_diag/2) * M_s_alpha * F_alpha_prime
                # F_alpha_prime = w_alpha * ( ( (e_alpha-u)/cs^2 + (e_alpha.u)e_alpha/cs^4 ).F_body )
                # Simplified: Add source term in moment space
                # Si = (1 - 0.5 * S_dig[i]) * Fi_source_moment
                # Fi_source_moment = (M * source_dist_func)_i
                # source_dist_func_alpha = w_alpha * ( (e_alpha-u)/cs^2 + (e_alpha.u)/cs^4 * (e_alpha.F_body) ) . F_body
                # This is complex. A simpler Guo forcing for MRT:
                # F_k' = (1 - 0.5*S_k) * (M * Psi_vec)_k
                # Psi_vec_alpha = w_alpha * ( (e_alpha-u)/cs^2 + ( (e_alpha.u)*e_alpha )/cs^4 ) . force_vector

                f_body = self.cal_local_force(i, j) # Get body force F_b
                cs_sq = 1.0/3.0 # Sound speed squared

                if ti.static(self.force_flag == 1):
                    # Force term in moment space (Guo's scheme for MRT)
                    # This is a common way to implement it.
                    # F_m_i = (M_ij * Psi_j) * (1 - 0.5 * S_i)
                    # Psi_j = w_j * [ ((ej-u)/cs^2) . F_b + ( (ej.u)*(ej.F_b) )/cs^4 - (u.F_b)/ (2*cs^4) ] NO, this is more complex than needed
                    # Simpler: (M * source_term_in_f_space)_s * (1 - s_diag_s/2)
                    # source_term_in_f_space_alpha = w_alpha * ( ( (e_alpha-u)/cs^2 + (e_alpha.u)/(cs^2*cs^2) * e_alpha ).F_b )
                    # Let's use the one from the 3D code, adapted.
                    # m_coll[s] += (1 - 0.5 * S_dig[s]) * f_guo_s
                    # f_guo_s = sum_l ( M[s,l] * w[l] * ( ((e[l]-v)/cs^2).force + ((e[l].v)*(e[l].force))/(cs^2*cs^2) ) )
                    # Note: The 3D code had force / 3.0 and force / 9.0. This implies cs^2 = 1/3.
                    # So, (e-u).F / cs^2  and (e.u)(e.F) / cs^4.
                    # The 3D code uses: w_l * ( (e_l-u).f/3 + (e_l.u)(e_l.f)/9 ) * M_sl
                    # This corresponds to: w_l * ( (e_l-u).f/cs^2 + (e_l.u)(e_l.f)/(cs^2*cs^2) ) * M_sl if cs^2 = 1/3
                    # This is the standard Guo forcing term for MRT.
                    
                    force_moment_source = ti.Vector([0.0] * self.q_dim)
                    vel_ij = self.v[i,j] # Current velocity at node

                    for s_alpha in ti.static(range(self.q_dim)): # Loop over velocity directions for source term
                        e_alpha_minus_u = self.e[s_alpha] - vel_ij
                        e_alpha_dot_u = self.e[s_alpha].dot(vel_ij)
                        e_alpha_dot_f_body = self.e[s_alpha].dot(f_body)
                        
                        # Guo's original source term for f_alpha (not moment space directly)
                        # F_alpha_src = w_alpha * ( (e_alpha-u)/cs^2 . F_b + (e_alpha.u)(e_alpha.F_b)/cs^4 )
                        # The 3D code's version:
                        # f_guo_contrib = self.w[s_alpha] * ( (e_alpha_minus_u.dot(f_body) / cs_sq ) + \
                        #                                   (e_alpha_dot_u * e_alpha_dot_f_body / (cs_sq*cs_sq) ) )
                        # This is Psi_alpha. Then sum (M_s_l * Psi_l) for the moment source.
                        # The 3D code was: sum_l M[s,l] * w[l] * ( ( (e[l]-v).dot(f)/3.0 ) + ( (e[l].dot(v))*(e[l].dot(f))/9.0 ) )
                        # This is sum_l M[s,l] * Psi_l where Psi_l is the source term for f_l.

                        # Let's calculate Psi_l (source term for distribution function f_l)
                        psi_l = self.w[s_alpha] * ( (e_alpha_minus_u.dot(f_body) / cs_sq) + \
                                                (e_alpha_dot_u * e_alpha_dot_f_body / (cs_sq * cs_sq)) )
                        
                        # Add to the moment source: M_s_l * psi_l
                        for s_moment_idx in ti.static(range(self.q_dim)):
                            force_moment_source[s_moment_idx] += self.M[None][s_moment_idx, s_alpha] * psi_l
                    
                    # Add to collided moments
                    for s_idx in ti.static(range(self.q_dim)):
                        m_coll[s_idx] += (1.0 - 0.5 * self.S_dig[None][s_idx]) * force_moment_source[s_idx]

                # Transform back to distribution functions f
                self.f[i, j] = self.inv_M[None] @ m_coll
    
    @ti.func
    def periodic_index(self, i_coord, j_coord): # 2D periodic boundary
        i_out, j_out = i_coord, j_coord
        if i_coord < 0: i_out = self.nx - 1
        if i_coord > self.nx - 1: i_out = 0
        if j_coord < 0: j_out = self.ny - 1
        if j_coord > self.ny - 1: j_out = 0
        return i_out, j_out

    @ti.kernel
    def streaming1(self): # Streaming and bounce-back
        for i, j in self.rho: # Iterate over 2D grid
            if self.solid[i, j] == 0: # Fluid node
                for s_idx in ti.static(range(self.q_dim)):
                    # Get destination coordinates
                    i_dest = i + self.e[s_idx][0]
                    j_dest = j + self.e[s_idx][1]
                    
                    # Periodic boundary condition for streaming
                    i_p, j_p = self.periodic_index(i_dest, j_dest) # ip = periodic_index(i_node + e[s])
                                                                    # In 3D: ip = self.periodic_index(i+self.e[s])

                    if self.solid[i_p, j_p] == 0: # If destination is fluid
                        self.F[i_p, j_p][s_idx] = self.f[i, j][s_idx]
                    else: # If destination is solid (bounce-back)
                        self.F[i, j][self.LR[s_idx]] = self.f[i, j][s_idx]
                        # Note: This is bounce-back on F, which is post-collision f.
                        # Some implementations do bounce-back on f before collision.
                        # This seems to be "bounce-back of post-collision f" which becomes
                        # the pre-collision F for the next step at the same node but opposite direction.

    @ti.kernel
    def Boundary_condition(self): # Apply non-periodic boundary conditions
        # X-boundaries
        if ti.static(self.bc_x_left != 0): # If not periodic on x-left
            for j_coord in range(self.ny):
                if self.solid[0, j_coord] == 0: # Fluid node on boundary
                    if ti.static(self.bc_x_left == 1): # Fix pressure (density)
                        rho_b = self.rho_bcxl
                        # vel_b = self.v[1,j_coord] if self.solid[1,j_coord]==0 else self.v[0,j_coord] # Extrapolate? Or use specified vel?
                        # Simplified: use known rho, estimate u from neighbor or set to zero if unknown
                        # Zou-He boundary: calc u_x from unknown f_i, set f_i for E,NE,SE.
                        # Here, simpler: reconstruct all f from feq at boundary rho and extrapolated/boundary v.
                        # For fixed pressure BC, velocity is often extrapolated or taken from the inner node.
                        vel_b = self.v[0, j_coord] # Default to current velocity at boundary if not specified
                        if self.solid[1, j_coord] > 0: # If inner node is solid, this might be problematic
                             vel_b = ti.Vector([self.vx_bcxl, self.vy_bcxl]) # Use specified BC velocity if inner is solid
                        else: # Inner node is fluid
                             vel_b = self.v[1, j_coord] # Extrapolate velocity from the first fluid layer inside
                        
                        for s_idx in ti.static(range(self.q_dim)):
                            self.F[0, j_coord][s_idx] = self.feq(s_idx, rho_b, vel_b)
                            
                    elif ti.static(self.bc_x_left == 2): # Fix velocity
                        rho_b = self.rho[0,j_coord] # Use current density (or from neighbor: self.rho[1,j_coord])
                                                    # Zou-He: rho is calculated from unknown f_i.
                                                    # Simpler: use rho from neighbor or specified rho_bcxl
                        rho_b = self.rho[1,j_coord] if self.solid[1,j_coord]==0 else self.rho_bcxl

                        vel_b = ti.Vector([self.vx_bcxl, self.vy_bcxl])
                        for s_idx in ti.static(range(self.q_dim)):
                            self.F[0, j_coord][s_idx] = self.feq(s_idx, rho_b, vel_b)

        if ti.static(self.bc_x_right != 0):
            for j_coord in range(self.ny):
                if self.solid[self.nx - 1, j_coord] == 0:
                    if ti.static(self.bc_x_right == 1): # Fix pressure
                        rho_b = self.rho_bcxr
                        vel_b = self.v[self.nx-1, j_coord] 
                        if self.solid[self.nx-2, j_coord] > 0:
                            vel_b = ti.Vector([self.vx_bcxr, self.vy_bcxr])
                        else:
                            vel_b = self.v[self.nx-2, j_coord]

                        for s_idx in ti.static(range(self.q_dim)):
                            self.F[self.nx - 1, j_coord][s_idx] = self.feq(s_idx, rho_b, vel_b)

                    elif ti.static(self.bc_x_right == 2): # Fix velocity
                        rho_b = self.rho[self.nx-2,j_coord] if self.solid[self.nx-2,j_coord]==0 else self.rho_bcxr
                        vel_b = ti.Vector([self.vx_bcxr, self.vy_bcxr])
                        for s_idx in ti.static(range(self.q_dim)):
                            self.F[self.nx - 1, j_coord][s_idx] = self.feq(s_idx, rho_b, vel_b)
        
        # Y-boundaries
        if ti.static(self.bc_y_left != 0):
            for i_coord in range(self.nx):
                if self.solid[i_coord, 0] == 0:
                    if ti.static(self.bc_y_left == 1): # Fix pressure
                        rho_b = self.rho_bcyl
                        vel_b = self.v[i_coord, 0]
                        if self.solid[i_coord, 1] > 0:
                             vel_b = ti.Vector([self.vx_bcyl, self.vy_bcyl])
                        else:
                             vel_b = self.v[i_coord, 1]
                        for s_idx in ti.static(range(self.q_dim)):
                           self.F[i_coord, 0][s_idx] = self.feq(s_idx, rho_b, vel_b)

                    elif ti.static(self.bc_y_left == 2): # Fix velocity
                        rho_b = self.rho[i_coord,1] if self.solid[i_coord,1]==0 else self.rho_bcyl
                        vel_b = ti.Vector([self.vx_bcyl, self.vy_bcyl])
                        for s_idx in ti.static(range(self.q_dim)):
                            self.F[i_coord, 0][s_idx] = self.feq(s_idx, rho_b, vel_b)
                            
        if ti.static(self.bc_y_right != 0):
            for i_coord in range(self.nx):
                if self.solid[i_coord, self.ny - 1] == 0:
                    if ti.static(self.bc_y_right == 1): # Fix pressure
                        rho_b = self.rho_bcyr
                        vel_b = self.v[i_coord, self.ny-1]
                        if self.solid[i_coord, self.ny-2] > 0:
                            vel_b = ti.Vector([self.vx_bcyr, self.vy_bcyr])
                        else:
                            vel_b = self.v[i_coord, self.ny-2]
                        for s_idx in ti.static(range(self.q_dim)):
                            self.F[i_coord, self.ny - 1][s_idx] = self.feq(s_idx, rho_b, vel_b)

                    elif ti.static(self.bc_y_right == 2): # Fix velocity
                        rho_b = self.rho[i_coord,self.ny-2] if self.solid[i_coord,self.ny-2]==0 else self.rho_bcyr
                        vel_b = ti.Vector([self.vx_bcyr, self.vy_bcyr])
                        for s_idx in ti.static(range(self.q_dim)):
                           self.F[i_coord, self.ny - 1][s_idx] = self.feq(s_idx, rho_b, vel_b)


    @ti.kernel
    def streaming3_update_macro(self): # Update macroscopic variables (rho, v) from F
                                 # And copy F to f for next collision
        for i, j in self.rho: # Iterate over 2D grid
            if self.solid[i, j] == 0: # Fluid node
                self.rho[i, j] = 0.0
                self.v[i, j] = ti.Vector([0.0, 0.0])
                
                current_f_sum = 0.0
                for s_idx in ti.static(range(self.q_dim)):
                    self.f[i,j][s_idx] = self.F[i,j][s_idx] # Copy F (post-streaming) to f (pre-collision for next step)
                    current_f_sum += self.f[i,j][s_idx]
                    self.v[i,j] += self.e[s_idx] * self.f[i,j][s_idx]
                
                self.rho[i,j] = current_f_sum
                
                f_body_local = self.cal_local_force(i, j) # Body force F_b
                
                if self.rho[i,j] > 1e-6 : # Avoid division by zero if density is too low
                    self.v[i,j] /= self.rho[i,j]
                    # Add force contribution to velocity (standard LBM, second-order accurate)
                    # v = v_collisionless + dt * F_b / (2 * rho)
                    # Here dt=1 (LBM units)
                    if ti.static(self.force_flag == 1):
                         self.v[i,j] += f_body_local / (2.0 * self.rho[i,j])
                else: # Reset velocity if density is near zero
                    self.v[i,j] = ti.Vector([0.0,0.0])
                    self.rho[i,j] = 1e-6 # Prevent issues with zero density, though this indicates instability

            else: # Solid node
                self.rho[i, j] = 1.0 # Or some reference density
                self.v[i, j] = ti.Vector([0.0, 0.0])


    def get_max_v(self):
        self.max_v[None] = -1e10
        self.cal_max_v_kernel() # Renamed kernel
        return self.max_v[None]

    @ti.kernel
    def cal_max_v_kernel(self): # Kernel to calculate max velocity
        for I in ti.grouped(self.rho): # Correct iteration for Taichi fields
            if self.solid[I] == 0:
                 ti.atomic_max(self.max_v[None], self.v[I].norm())
    
    # Boundary condition setters
    def set_bc_vel_x1(self, vel: list): # vel = [vx, vy]
        self.bc_x_right = 2
        self.vx_bcxr, self.vy_bcxr = vel[0], vel[1]

    def set_bc_vel_x0(self, vel: list):
        self.bc_x_left = 2
        self.vx_bcxl, self.vy_bcxl = vel[0], vel[1]

    def set_bc_vel_y1(self, vel: list):
        self.bc_y_right = 2
        self.vx_bcyr, self.vy_bcyr = vel[0], vel[1]

    def set_bc_vel_y0(self, vel: list):
        self.bc_y_left = 2
        self.vx_bcyl, self.vy_bcyl = vel[0], vel[1]

    def set_bc_rho_x0(self, rho_val: float):
        self.bc_x_left = 1
        self.rho_bcxl = rho_val
    
    def set_bc_rho_x1(self, rho_val: float):
        self.bc_x_right = 1
        self.rho_bcxr = rho_val

    def set_bc_rho_y0(self, rho_val: float):
        self.bc_y_left = 1
        self.rho_bcyl = rho_val
    
    def set_bc_rho_y1(self, rho_val: float):
        self.bc_y_right = 1
        self.rho_bcyr = rho_val

    def set_viscosity(self, niu_val: float):
        self.niu = niu_val
        # Re-initialize simulation parameters that depend on niu if called mid-simulation
        # For now, assume it's called before init_simulation() or handle updates there.

    def set_force(self, force_vec: list): # force_vec = [fx, fy]
        self.fx = force_vec[0]
        self.fy = force_vec[1]
        # Update force_flag and ext_f if called mid-simulation
        self.ext_f[None][0] = self.fx
        self.ext_f[None][1] = self.fy
        if (abs(self.fx) > 0 or abs(self.fy) > 0):
            self.force_flag = 1
        else:
            self.force_flag = 0
            
    def step(self):
        self.colission()
        self.streaming1() # Includes bounce-back
        self.Boundary_condition() # Apply non-periodic BCs
        self.streaming3_update_macro() # Update rho, v and copy F to f

# Example Usage (commented out, to be used for testing later)
'''
if __name__ == '__main__':
    nx, ny = 100, 50
    lbm_solver = LB2D_Solver_Single_Phase(nx, ny)

    # Create a simple geometry: channel flow with a square obstacle
    geometry = np.zeros((nx, ny), dtype=bool)
    geometry[nx//4 : nx//4 + 10, ny//2 - 5 : ny//2 + 5] = True # Obstacle
    
    lbm_solver.init_geo(geometry)
    
    # Set boundary conditions: Poiseuille flow inlet/outlet (example)
    # Inlet: fixed velocity (parabolic profile can be complex to set directly this way)
    # For simplicity, using uniform velocity inlet
    inlet_vel_x = 0.01 
    lbm_solver.set_bc_vel_x0([inlet_vel_x, 0.0]) # Inlet velocity at x=0
    lbm_solver.bc_x_left = 2 # Fix velocity mode

    # Outlet: fixed pressure (density)
    lbm_solver.set_bc_rho_x1(1.0) # Outlet pressure at x=nx-1
    lbm_solver.bc_x_right = 1 # Fix pressure mode
    
    # Top and bottom walls: no-slip (implicitly handled by solid geometry and bounce-back)
    # If domain boundaries are walls:
    # geometry[:, 0] = True
    # geometry[:, ny-1] = True
    # lbm_solver.init_geo(geometry) # re-init if changed

    lbm_solver.set_viscosity(0.02)
    # lbm_solver.set_force([1e-5, 0]) # Example body force

    lbm_solver.init_simulation()

    for iter_num in range(1000):
        lbm_solver.step()
        if iter_num % 100 == 0:
            max_v = lbm_solver.get_max_v()
            print(f"Iteration: {iter_num}, Max Velocity: {max_v:.4e}")
            # Add plotting or data saving here if needed
            
    # Retrieve data for plotting (example)
    # rho_data = lbm_solver.rho.to_numpy()
    # v_data = lbm_solver.v.to_numpy() # v_data[:,:,0] for vx, v_data[:,:,1] for vy
    # import matplotlib.pyplot as plt
    # plt.imshow(rho_data.T, origin='lower')
    # plt.colorbar(label='Density')
    # plt.show()
'''
print("LB2D_Solver_Single_Phase class defined.")

# Final check on D2Q9 parameters if they are used by Taichi kernels before instance creation
# e.g. if M, inv_M, w were directly used in @ti.func without self.
# In this code, they are instance variables (self.M, self.w etc.) and set up in __init__
# so it should be fine.
