"""This code was written to implement the Haldane model ribbon part of the
paper arXiv:2604.10190; please cite it if you use this code extensively in
your project. See https://github.com/maaks12/ribbon-LCM for more details.

maks.repse@fmf.uni-lj.si
"""


import numpy as np
import kwant
from scipy import linalg, fft


class Haldane:

    def __init__(self, Nx, Ny, delta=0, seed=None, y_wall=None):
        """Implements a Haldane model and methods for calculating dispersion,
        ground state density matrix, local Chern marker, local electron
        density and conductance.

        Units: a = hbar = e0 = 1.

        Possible boundary conditions (x direction - y direction) are
        periodic-open and periodic-periodic. Possible bases are
        momentum-position and position-position.

        Parameters
        ----------
        Nx : int
            Number of unit cells in the x direction. This is assumed to be
            the periodic direction when paired with use_kx=True.
        Ny : int
            Number of unit cells in the y direction. Always in the local
            basis.
        delta : float
            Disorder magnitude.
        seed : int or None
            RNG seed.
        y_wall : float or None
            Position of the domain wall (if needed).
        """
        self.Nx, self.Ny = Nx, Ny

        # domain wall position (kwant y coordinate); only used when M or phi is a (below, above) pair
        # default cuts the vertical nn bonds between A(Ny//2) and B(Ny//2)
        self.y_wall = np.sqrt(3)/2 * (Ny//2 + 1/3) if y_wall is None else y_wall

        self.delta = delta # disorder amplitude
        self.rng = np.random.default_rng(seed=seed)
        self.disorder = self.rng.uniform(low=-0.5, high=0.5, size=3*self.Ny)
        self.disorder2D = self.rng.uniform(low=-0.5, high=0.5, size=(self.Nx, 3*self.Ny))

        self.t1 = 1.0
        self.t2 = 1/3

        # lead parameters
        self.mu_lead = -7.5
        self.t_lead = 5.0

        self.BZx = self.BrillouinZone(self.Nx)

    def BrillouinZone(self, N):
        """Creates a Brillouin zone."""
        BZ = np.arange(N) * 2*np.pi/N
        for i in range(N):
            if BZ[i] > np.pi:
                BZ[i] -= 2*np.pi
        return np.sort(BZ)

    def makeKwantSystem(self, y_bc='open', disorder_type='1d', use_kx=True, leads=False, delta=None):
        """Makes the kwant system for the Haldane model.

        Parameters
        ----------
        y_bc : {'open', 'periodic'}
            Sets boundary condition in the y direction.
        disorder_type : {'1d', '2d'}
            Sets disorder type: '1d' -- stripe-like, '2d' -- point-like.
        use_kx : bool
            Should a hybrid basis (True) or fully local basis (False) be used.
        leads : bool
            Should conducting leads be attached for kwant conductance
            calculations.
        """

        if delta is None:
            delta = self.delta

        assert y_bc in ['open', 'periodic'], "y_bc should be 'open' or 'periodic'"
        assert disorder_type in ['1d', '2d'], "disorder_type should be '1d' (stripe-like) or '2d' (point-like)"
        assert not ((y_bc == 'periodic') and leads), "leads require an open boundary in y"
        assert not ((disorder_type == '2d') and use_kx), "2D disorder is incompatible with kx basis"

        # make lattice
        lattice = kwant.lattice.honeycomb(a=1, norbs=1, name=('A', 'B'))
        A, B = lattice.sublattices
        a1, a2 = lattice.prim_vecs

        def ribbon(pos):
            """Helper function for defining ribbon shape."""
            _, y = pos
            # return 0 <= x and x < self.Nx
            # return -np.sqrt(3) * self.Ny / 4 < y <= np.sqrt(3) * self.Ny / 4
            return 0 < y <= np.sqrt(3) * self.Ny / 2
            # W = np.sqrt(3)/2 * (self.Ny-2/3)
            # return -W/2 <= y <= W/2

        def calc_symm_y(site):
            """Helper function that calculates y coordinate with 0 set in the
            middle of the ribbon, to be used in Peierls phase calculation.
            """
            _, j = site.tag
            if self.Ny % 2 == 0:
                y = np.sqrt(3)/2 * (j - self.Ny/2) # sredina oc
                if site.family is A:
                    y -= np.sqrt(3)/6
                else:
                    y += np.sqrt(3)/6
            else:
                if site.family is A:
                    y = 1/(np.sqrt(3)*4) + (j-(self.Ny//2+1))*np.sqrt(3)/2
                else:
                    y = -1/(np.sqrt(3)*4) + (j-self.Ny//2)*np.sqrt(3)/2
            return y

        def peierlsPhase(site1, site2, B):
            """Calculates Peierls phase for hopping between sites 1 and 2.

            B is in 2pi/phi_0 = 1 units.
            """
            x1, _ = site1.pos
            x2, _ = site2.pos
            y1 = calc_symm_y(site1)
            y2 = calc_symm_y(site2)
            dx = x2 - x1
            y_avg = 0.5 * (y1 + y2)
            phase = -B * dx * y_avg
            return np.exp(-1j * phase)

        Ly = np.sqrt(3) * self.Ny / 2  # period in y

        def above_wall(y):
            """Is position y in the second domain?

            For periodic y there are necessarily two walls, placed at y_wall
            and y_wall + Ly/2.
            """
            if y_bc == 'periodic':
                return (y - self.y_wall) % Ly < Ly / 2
            return y > self.y_wall

        def local(val, y):
            """val is a scalar (homogeneous) or a pair (below wall, above wall)."""
            if np.ndim(val) == 0:
                return val
            return val[1] if above_wall(y) else val[0]

        def onsite(site, M):
            """Defines onsite energy."""
            M = local(M, site.pos[1])
            i, j = site.tag
            if disorder_type == '1d':
                d1 = delta*self.disorder[2*j]
                d2 = delta*self.disorder[2*j+1]
            else:
                # print(i,j)
                d1 = delta*self.disorder2D[i, 2*j]
                d2 = delta*self.disorder2D[i, 2*j+1]
            M1 = M + d1
            M2 = M + d2
            return M1 if site.family is A else -M2

        def nn_hopping(site1, site2, B):
            """Defines nearest neighbor hopping."""
            return self.t1 * peierlsPhase(site1, site2, B)

        def nnn_hopping(site1, site2, phi, B):
            """Defines next nearest neighbor hopping."""
            phi = local(phi, 0.5*(site1.pos[1] + site2.pos[1]))  # bond belongs to the domain of its midpoint
            return self.t2*np.exp(-1j*phi) * peierlsPhase(site1, site2, B)

        # nnn_hopping structure (NOTE: hopping direction is important)
        nnn_hoppings_a = (((-1, 0), A, A), ((0, 1), A, A), ((1, -1), A, A))
        nnn_hoppings_b = (((1, 0), B, B), ((0, -1), B, B), ((-1, 1), B, B))

        if y_bc == 'periodic':
            if not use_kx:
                # create a kwant system with a Nx x Ny supercell; after wraparound, keep only kx=0 block
                syst = kwant.Builder(kwant.TranslationalSymmetry(self.Nx * a1, self.Ny * a2))
                for i in range(self.Nx):
                    for j in range(self.Ny):
                        syst[A(i, j)] = onsite
                        syst[B(i, j)] = onsite
            else:
                syst = kwant.Builder(kwant.TranslationalSymmetry(a1, self.Ny * a2))
                for j in range(self.Ny):
                    syst[A(0, j)] = onsite
                    syst[B(0, j)] = onsite
        else:
            # no translational symmetry in the a2 direction
            if not use_kx:
                syst = kwant.Builder(kwant.TranslationalSymmetry(self.Nx*a1))
            else:
                syst = kwant.Builder(kwant.TranslationalSymmetry(a1))
            syst[lattice.shape(ribbon, (0, 0))] = onsite

        # pass hoppings to kwant
        syst[lattice.neighbors()] = nn_hopping
        syst.eradicate_dangling()
        syst[
            [kwant.builder.HoppingKind(*hopping) for hopping in nnn_hoppings_a]
        ] = nnn_hopping
        syst[
            [kwant.builder.HoppingKind(*hopping) for hopping in nnn_hoppings_b]
        ] = nnn_hopping

        syst_wrapped = kwant.wraparound.wraparound(syst)

        if not leads:
            return syst_wrapped.finalized()

        def lead_shape(pos):
            return True

        def lead_onsite(site):
            return self.mu_lead

        def lead_hopping(site1, site2, B):
            return self.t_lead * peierlsPhase(site1, site2, B)

        if not use_kx:
            lead_bottom = kwant.Builder(kwant.TranslationalSymmetry(self.Nx*a1,-a2))
            lead_top = kwant.Builder(kwant.TranslationalSymmetry(self.Nx*a1,+a2))
        else:
            lead_bottom = kwant.Builder(kwant.TranslationalSymmetry(a1,-a2))
            lead_top = kwant.Builder(kwant.TranslationalSymmetry(a1,+a2))
        lead_bottom[lattice.shape(lead_shape, (0,0))] = lead_onsite
        lead_bottom[lattice.neighbors()] = lead_hopping
        lead_top[lattice.shape(lead_shape, (0,0))] = lead_onsite
        lead_top[lattice.neighbors()] = lead_hopping

        lead_bottom_wrapped = kwant.wraparound.wraparound(lead_bottom, keep=1)
        lead_top_wrapped = kwant.wraparound.wraparound(lead_top, keep=1)
        syst_wrapped.attach_lead(lead_bottom_wrapped)
        syst_wrapped.attach_lead(lead_top_wrapped)

        return syst_wrapped.finalized()

    def HamiltonianMatrix(self, params, y_bc='open', disorder_type='1d', use_kx=True, delta=None):
        """Creates the Hamiltonian matrix from kwant system.

        Parameters
        ----------
        params : (M, phi, B)
            M and phi can be numbers or pairs of numbers -> values on either
            side of the domain wall.
        y_bc : {'open', 'periodic'}
            Boundary condition in the y direction (x direction is always
            periodic).
        disorder_type : {'1d', '2d'}
            Stripe-like or point-like (see paper).
        use_kx : bool
            If True uses hybrid basis, else position basis.
        delta : float or None
            Disorder amplitude; if None, uses self.delta.
        """
        M, phi, B = params

        if use_kx:
            syst = self.makeKwantSystem(y_bc=y_bc, disorder_type=disorder_type, use_kx=use_kx, leads=False, delta=delta)
            perm = np.argsort([site.pos[1] for site in syst.sites]) # kwant does not return hamiltonian_submatrix sorted by site position
            H = np.empty((self.Nx,2*self.Ny,2*self.Ny), dtype=np.complex128)
            for i, k in enumerate(self.BZx):
                pars = dict(k_x=k, M=M, phi=phi, B=B)
                if y_bc == 'periodic':
                    pars['k_y'] = 0
                Hi = syst.hamiltonian_submatrix(params=pars)
                # sorting
                Hi = Hi[perm][:, perm]
                H[i,:,:] = Hi
        else:
            syst = self.makeKwantSystem(y_bc=y_bc, disorder_type=disorder_type, use_kx=use_kx, leads=False, delta=delta)
            pars = dict(k_x=0, k_y=0, M=M, phi=phi, B=B)
            H = syst.hamiltonian_submatrix(params=pars)
            # kwant does not return hamiltonian_submatrix sorted by site position
            # sort by lattice column i (not real x = i + j/2) to match the (Nx, Ny, 2) layout used in projector
            tags_i = np.array([site.tag[0] for site in syst.sites])
            ys = np.array([site.pos[1] for site in syst.sites])
            perm = np.lexsort((ys, tags_i))  # primary key: lattice column i (a1 index), tie-broken by y
            H = H[perm][:,perm]

        return H

    def dispersion(self, params, y_bc='open', disorder_type='1d', use_kx=True, delta=None):
        """Calculates dispersion of the system.

        Parameters
        ----------
        params : (M, phi, B)
            M and phi can be numbers or pairs of numbers -> values on either
            side of the domain wall.
        y_bc : {'open', 'periodic'}
            Boundary condition in the y direction (x direction is always
            periodic).
        disorder_type : {'1d', '2d'}
            Stripe-like or point-like (see paper).
        use_kx : bool
            If True uses hybrid basis, else position basis.
        delta : float or None
            Disorder amplitude; if None, uses self.delta.

        Returns
        -------
        energies : ndarray
            If use_kx=True, energy bands indexed as (kx, n), else sorted
            eigenvalues.
        """
        if use_kx:
            H = self.HamiltonianMatrix(params, y_bc=y_bc, disorder_type=disorder_type, use_kx=use_kx, delta=delta)
            energies = np.zeros((self.Nx,2*self.Ny))
            for i, _ in enumerate(self.BZx):
                Hi = H[i,:,:]
                energies[i,:] = linalg.eigh(Hi, eigvals_only=True)
        else:
            H = self.HamiltonianMatrix(params, y_bc=y_bc, disorder_type=disorder_type, use_kx=use_kx, delta=delta)
            energies = linalg.eigh(H, eigvals_only=True)
        return energies

    def fermiEnergy(self, params):
        """Half-filling Fermi energy of a clean system with open y_bc.

        Returns midpoint between highest filled and lowest empty levels.

        Parameters
        ----------
        params : (M, phi, B)
            M and phi can be numbers or pairs of numbers -> values on either
            side of the domain wall.
        """
        disp = self.dispersion(params, y_bc='open', disorder_type='1d', use_kx=True, delta=0)
        return np.mean(np.sort(disp.reshape(-1))[self.Nx*self.Ny-1:self.Nx*self.Ny+1])

    def groundStateWavefunction(self, params, fermiEnergy, y_bc='open', disorder_type='1d', use_kx=True, delta=None):
        """Calculates ground state wavefunction.

        Parameters
        ----------
        params : (M, phi, B)
            M and phi can be numbers or pairs of numbers -> values on either
            side of the domain wall.
        y_bc : {'open', 'periodic'}
            Boundary condition in the y direction (x direction is always
            periodic).
        disorder_type : {'1d', '2d'}
            Stripe-like or point-like (see paper).
        use_kx : bool
            If True uses hybrid basis, else position basis.
        delta : float or None
            Disorder amplitude; if None, uses self.delta.
        """
        if use_kx:
            H = self.HamiltonianMatrix(params, y_bc=y_bc, disorder_type=disorder_type, use_kx=use_kx, delta=delta)
            psi = np.zeros((self.Nx,2*self.Ny,2*self.Ny), dtype=np.complex128)
            for i, _ in enumerate(self.BZx):
                Hi = H[i,:,:]
                psii = linalg.eigh(Hi, subset_by_value=(-np.inf, fermiEnergy))[1]
                _, n = psii.shape
                psi[i,:,:n] = psii
        else:
            H = self.HamiltonianMatrix(params, y_bc=y_bc, disorder_type=disorder_type, use_kx=use_kx, delta=delta)
            _, psi = linalg.eigh(H, subset_by_value=(-np.inf, fermiEnergy))
        return psi

    def projector(self, psi):
        """Given wavefunction psi, calculates |psi><psi|.

        Shape of the projector is (Nx, Ny, 2, Ny, 2) if use_kx=True, else
        (Nx, Ny, 2, Nx, Ny, 2).
        """
        n = len(psi.shape)
        assert (n==2) or (n==3), "psi must be 2- or 3-dimensional"
        if n == 2:
            P = (psi @ np.conjugate(psi.T)).reshape(self.Nx,self.Ny,2,self.Nx,self.Ny,2) # P = sum_n |psi_n><psi_n|
        elif n == 3:
            P = (psi @ np.conjugate(psi.transpose(0,2,1))).reshape(self.Nx,self.Ny,2,self.Ny,2)
        return P

    def groundStateProjector(self, params, fermiEnergy, y_bc='open', disorder_type='1d', use_kx=True, delta=None):
        """Wrapper for calculating ground state projector.

        Parameters
        ----------
        params : (M, phi, B)
            M and phi can be numbers or pairs of numbers -> values on either
            side of the domain wall.
        y_bc : {'open', 'periodic'}
            Boundary condition in the y direction (x direction is always
            periodic).
        disorder_type : {'1d', '2d'}
            Stripe-like or point-like (see paper).
        use_kx : bool
            If True uses hybrid basis, else position basis.
        delta : float or None
            Disorder amplitude; if None, uses self.delta.
        """
        psi = self.groundStateWavefunction(params, fermiEnergy, y_bc=y_bc, disorder_type=disorder_type, use_kx=use_kx, delta=delta)
        return self.projector(psi)

    def derivk(self, P):
        """Helper function for calculating local Chern marker.

        Calculates dP/dkx using spectral derivative.
        """
        axis = 0 # always periodic direction
        N = P.shape[axis]
        f = 1j*N*fft.fftfreq(N)
        if N % 2 == 0:
            f[N//2] = 0
        res = np.einsum('i, ijakb -> ijakb', f, fft.fft(P, axis=axis))
        return fft.ifft(res, axis=axis)

    def derivx(self, P, axis=1, y_bc='open'):
        """Helper function for calculating local Chern marker.

        Calculates [x_i, P] for open or periodic bc.

        Parameters
        ----------
        axis : {0, 1}
            0 for x, 1 for y.
        """
        assert y_bc in ['open', 'periodic'], "y_bc should be 'open' or 'periodic'"
        n = len(P.shape)
        assert (n==6) or (n==5), "Projector not in correct form, see projector method"

        if (axis == 1) or (n==5):
            N = self.Ny
        else:
            N = self.Nx

        y = np.arange(N, dtype=float)
        if y_bc == 'periodic': # this is the same as the fftfreq trick
            # Delta = ((y[:,None] - y[None,:] + N//2) % N) - N//2
            y = np.roll(y-N//2, (N+1)//2)
            if N % 2 == 0:
                y[N//2] = 0.
        Delta = linalg.toeplitz(y, -y)
        # else:
            # Delta = y[:,None] - y[None,:]

        if n == 5:
            comm = np.einsum('ik, jiakb -> jiakb', Delta, P)
        else:
            if axis == 0:
                comm = np.einsum('ik, ijaklb -> ijaklb', Delta, P)
            else:
                comm = np.einsum('jl, ijaklb -> ijaklb', Delta, P)
        return -1j * comm

    def prod(self, A, B):
        """Helper function, calculates matrix product of two operators."""
        nA = len(A.shape)
        nB = len(B.shape)
        shape = A.shape
        assert nA == nB, "Dimension of A different from dimension of B"
        assert (nA==6) or (nA==5), "Operators not in correct form, see projector method"
        if nA == 6:
            A = A.reshape(2*self.Nx*self.Ny,2*self.Nx*self.Ny)
            B = B.reshape(2*self.Nx*self.Ny,2*self.Nx*self.Ny)
        else:
            A = A.reshape(self.Nx,2*self.Ny,2*self.Ny)
            B = B.reshape(self.Nx,2*self.Ny,2*self.Ny)
        AB = A @ B
        return AB.reshape(shape)

    def localChernMarker(self, P, y_bc='open'):
        """Calculates local Chern marker from projector (density matrix) P.

        Parameters
        ----------
        y_bc : {'open', 'periodic'}
            Boundary condition in the y direction (x direction is always
            periodic).
        """
        assert y_bc in ['open', 'periodic'], "y_bc should be 'open' or 'periodic'"
        n = len(P.shape)
        assert (n==6) or (n==5), "Projector not in correct form, see projector method"
        if n == 6:
            dPx = self.derivx(P, axis=0, y_bc='periodic')
        else:
            dPx = self.derivk(P)
        dPy = self.derivx(P, axis=1, y_bc=y_bc)
        comm = self.prod(dPx, dPy) - self.prod(dPy, dPx)
        if n == 6:
            result = np.einsum('ijaklb, klbija -> ij', P, comm)
        else:
            result = np.einsum('jiakb, jkbia -> ji', P, comm)
            result = np.sum(result, axis=0) / self.Nx # sum over kx
            result = np.repeat([result], self.Nx, axis=0)
        return 2 * np.pi * result.imag

    def electronDensity(self, P):
        """Calculates electron density from projector (density matrix) P.

        Used for the local Streda marker.
        """
        n = len(P.shape)
        assert (n==6) or (n==5), "Projector not in correct form, see projector method"
        if n == 6:
            density = np.einsum('ijaija -> ij', P).real / (np.sqrt(3)/2) # per unit cell
        else:
            density = np.einsum('kiaia -> ki', P).real / (np.sqrt(3)/2)
        return density

    def calculateConductance(self, M, phi, E, disorder_type='1d', use_kx=True):
        """Uses kwant.smatrix to calculate conductance at given parameters
        M, phi and energy E.
        """
        assert disorder_type in ['1d', '2d'], "disorder_type should be '1d' (stripe-like) or '2d' (point-like)"
        syst = self.makeKwantSystem(y_bc='open', disorder_type=disorder_type, use_kx=use_kx, leads=True)
        if use_kx:
            T = np.zeros(self.Nx)
            for j, k in enumerate(self.BZx):
                pars = dict(k_x=k, M=M, phi=phi, B=0)
                T[j] = kwant.smatrix(syst, energy=E, params=pars).transmission(1,0)
            T = np.sum(T)
        else:
            pars = dict(k_x=0, M=M, phi=phi, B=0)
            T = kwant.smatrix(syst, energy=E, params=pars).transmission(1,0)
        return T


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    Nx = 30
    Ny = 20
    haldane = Haldane(Nx, Ny)
    M = (0.2, 0.2)
    phi = (-np.pi/2, np.pi/2)
    params = (M, phi, 0)

    P_1d = haldane.groundStateProjector(params, 0, use_kx=True)
    lcm_1d = haldane.localChernMarker(P_1d)
    P_2d = haldane.groundStateProjector(params, 0, use_kx=False)
    lcm_2d = haldane.localChernMarker(P_2d)

    lcm = [lcm_1d, lcm_2d]
    name = [r'Hybrid $(x,k_y)$ basis', r'Real space $(x,y)$ basis']
    ylabel = [r'$c_{1d}$', r'$c_{2d}$']
    x = np.arange(Ny)

    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(3, 5, fig, width_ratios=[1,1,0.2,1,1], height_ratios=[1,0.2,1])
    for i, lcmi in enumerate(lcm):
        ax = fig.add_subplot(gs[0,2*i+i:2*i+2+i])
        ax.plot(x, lcmi[0,:])
        ax.set_title(name[i])
        ax.set_xlabel('$x$')
        ax.set_ylabel(ylabel[i])

    ax = fig.add_subplot(gs[2,1:4])
    ax.plot(x, lcm_2d[0,:]-lcm_1d[0,:])
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$c_{2d}-c_{1d}$')
    plt.show()
