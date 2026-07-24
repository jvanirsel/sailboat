from . import _Q_ELEM
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import math

from matplotlib import pyplot as plt


def plasma_parameters(
        file: Path,
        add_noise: bool = True,
        ) -> None:
    
    from . import write

    rpa_plot_filename = file.parent / f'{file.stem.replace("data", "rpa_fit")}.png'
    idm_plot_filename = file.parent / f'{file.stem.replace("data", "idm_fit")}.png'
    pars_filename = file.parent / file.name.replace('config', 'measured')

    with h5py.File(file) as h5f:
        n_true = float(np.array(h5f['plasma/density'])) # millimeter^-3
        u1_true = float(np.array(h5f['plasma/beam/velocity'])[0]) # millimeter / microsecond
        u2_true = float(np.array(h5f['plasma/beam/velocity'])[1]) # millimeter / microsecond
        u3_true = float(np.array(h5f['plasma/beam/velocity'])[2]) # millimeter / microsecond
        t_true = float(np.array(h5f['plasma/temperature/ion'])) # electronvolt

        mass = float(np.array(h5f['plasma/mass'])) # electronvolt microsecond^2 / millimeter^2
        charge = float(np.array(h5f['plasma/charge'])) # femtocoulomb

        area = float(np.array(h5f['rpa/geometry/aperture_area'])) # millimeter^2
        radius = float(np.array(h5f['rpa/geometry/aperture_size'])) / 2.0 # millimeter
        drift_length = float(np.array(h5f['rpa/geometry/depth'])) # millimeter
        num_screens = int(np.array(h5f['rpa/screens/number']))
        opacity = float(np.array(h5f['rpa/screens/opacity']))
        Aeff = area * opacity**num_screens

        voltages = np.array(h5f['iv_curve/voltages']) # volt
        currents = np.array(h5f['iv_curve/currents']) # nanoampere
        if add_noise:
            noise_level = float(np.array(h5f['rpa/noise_level'])) # nanoampere
            currents += np.random.normal(0.0, noise_level / 2, currents.shape)

    total_currents = np.sum(currents, axis=1) # nanoampere
    total_currents_theoretical = get_theoretical_currents(voltages, n_true, u3_true, t_true, mass, charge, Aeff)

    sat_current_ids = total_currents > np.quantile(total_currents, 0.95)
    sat_current = float(np.mean(total_currents[sat_current_ids]))
    sat_current_error = float(2 * np.std(total_currents[sat_current_ids]))

    didv = -np.gradient(total_currents, voltages) # nanoampere / volt
    didv_actual = -np.gradient(total_currents_theoretical, voltages) # nanoampere / volt

    didv_fit, popt, perr, fit_info, didv_start = fit_linearly_modulated_gaussian(voltages, didv, mass)
    rpa_pars = compute_rpa_params(popt, perr, mass, charge, Aeff, (sat_current, sat_current_error))
    idm_pars = compute_idm_params(currents, rpa_pars["U3"], radius, drift_length, idm_plot_filename)

    n1_meas = rpa_pars['N1'] + (rpa_pars['N1'][0] - n_true, 100 * (rpa_pars['N1'][0] - n_true) / n_true)
    n2_meas = rpa_pars['N2'] + (rpa_pars['N2'][0] - n_true, 100 * (rpa_pars['N2'][0] - n_true) / n_true)
    u1_meas = idm_pars['U1'] + (idm_pars['U1'][0] - u1_true, 100 * (idm_pars['U1'][0] - u1_true) / u1_true)
    u2_meas = idm_pars['U2'] + (idm_pars['U2'][0] - u2_true, 100 * (idm_pars['U2'][0] - u2_true) / u2_true)
    u3_meas = rpa_pars['U3'] + (rpa_pars['U3'][0] - u3_true, 100 * (rpa_pars['U3'][0] - u3_true) / u3_true)
    t_meas = rpa_pars['T'] + (rpa_pars['T'][0] - t_true, 100 * (rpa_pars['T'][0] - t_true) / t_true)

    print(f'N1 = {n1_meas[0]:6.2f} \u00b1 {n1_meas[1]:4.2f}\tError: {n1_meas[2]:6.2f} ({n1_meas[3]:6.2f} %) mm^-3')
    print(f'N2 = {n2_meas[0]:6.2f} \u00b1 {n2_meas[1]:4.2f}\tError: {n2_meas[2]:6.2f} ({n2_meas[3]:6.2f} %) mm^-3')
    print(f'U1 = {u1_meas[0]:6.2f} \u00b1 {u1_meas[1]:4.2f}\tError: {u1_meas[2]:6.2f} ({u1_meas[3]:6.2f} %) km/s')
    print(f'U2 = {u2_meas[0]:6.2f} \u00b1 {u2_meas[1]:4.2f}\tError: {u2_meas[2]:6.2f} ({u2_meas[3]:6.2f} %) km/s')
    print(f'U3 = {u3_meas[0]:6.2f} \u00b1 {u3_meas[1]:4.2f}\tError: {u3_meas[2]:6.2f} ({u3_meas[3]:6.2f} %) km/s')
    print(f'Ti = {t_meas[0]*1e3:6.2f} \u00b1 {t_meas[1]*1e3:4.2f}\tError: {(t_meas[2])*1e3:6.2f} ({t_meas[3]:6.2f} %) meV')
    print(f'R2 = {fit_info["r2"]:.2f}')

    title = f'n\u2081 = {n1_meas[0]:.1f} \u00b1 {n1_meas[1]:.1f} mm\u207b\u00b3,  \u0394 = {n1_meas[2]:.1f} ({n1_meas[3]:+.2f}%)'
    title += f',      u\u2081 = {u1_meas[0]:.2f} \u00b1 {u1_meas[1]:.2f} mm \u03bcs\u207b\u00b9,  \u0394 = {u1_meas[2]:.2f} ({u1_meas[3]:+.2f}%)'
    title += f'\nn\u2082 = {n2_meas[0]:.1f} \u00b1 {n2_meas[1]:.1f} mm\u207b\u00b3,  \u0394 = {n2_meas[2]:.1f} ({n2_meas[3]:+.2f}%)'
    title += f',      u\u2082 = {u2_meas[0]:.2f} \u00b1 {u2_meas[1]:.2f} mm \u03bcs\u207b\u00b9,  \u0394 = {u2_meas[2]:.2f} ({u2_meas[3]:+.2f}%)'
    title += f'\nT\u1d62 = {t_meas[0]*1e3:.2f} \u00b1 {t_meas[1]*1e3:.2f} meV,  \u0394 = {t_meas[2]*1e3:.2f} ({t_meas[3]:+.2f}%)'
    title += f',      u\u2083 = {u3_meas[0]:.2f} \u00b1 {u3_meas[1]:.2f} mm \u03bcs\u207b\u00b9,  \u0394 = {u3_meas[2]:.2f} ({u3_meas[3]:+.2f}%)'
    plt.style.use('dark_background')
    plt.figure(figsize=(9,6))
    plt.scatter(voltages, total_currents, color='w', label='IV data (nA)')
    plt.plot(voltages, total_currents_theoretical, 'm', label='IV theor. (nA)')
    plt.scatter(voltages[sat_current_ids], total_currents[sat_current_ids], color='g', marker='x', label='Sat. current (nA)')
    plt.plot(voltages, [sat_current + sat_current_error] * len(sat_current_ids), 'g:')
    plt.plot(voltages, [sat_current - sat_current_error] * len(voltages), 'g:')
    plt.scatter(voltages, didv, color='r', label='dI/dV data (nA / V)')
    plt.plot(voltages, didv_actual, 'w:', label='dI/dV theor. (nA / V)')
    plt.plot(voltages, didv_fit, 'y:', label='dI/dV data fit (nA / V)')
    plt.legend()
    plt.grid()
    plt.xlabel('Bias voltage (V)')
    plt.title(title)
    plt.savefig(rpa_plot_filename, dpi=600)
    plt.close()

    pars = {'N1': n1_meas, 'N2': n2_meas, 'U1': u1_meas, 'U2': u2_meas, 'U3': u3_meas, 'T': t_meas}
    write.measured_parameters(pars_filename, pars, fit_info)


def fit_linearly_modulated_gaussian(
        voltages: np.ndarray,
        didv: np.ndarray,
        mass: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:

    u = np.sign(voltages) * np.sqrt(2 * np.abs(voltages) / mass) # millimeter / microsecond
    dvdu = mass * u # electronvolt microsecond / millimeter
    didu = dvdu * didv # nanoampere microsecond / millimeter

    mu0 = u[np.argmax(didu)] # millimeter / microsecond
    a0 = np.max(didu) / mu0 # nanoampere microsecond^2 / millimeter^2
    indices = np.where(didu > a0 * mu0 / 2)[0]
    if len(indices) >= 2:
        fwhm_est = u[indices[-1]] - u[indices[0]] # millimeter / microsecond
        betasq0 = 2 / (fwhm_est**2) # microsecond^2 / millimeter^2
    else:
        betasq0 = 100 / (np.max(u) - np.min(u))**2
    p0 = [a0, betasq0, mu0]

    def model(x, a, betasq, mu):
        return a * x * np.exp(-betasq * (x - mu)**2)

    popt, pcov = curve_fit(model, u, didu, p0=p0)

    didu_fit = model(u, *popt)
    didu_start = model(u, *p0)
    dvdu[dvdu == 0.0] = np.nan
    didv_start = didu_start / dvdu
    didv_fit = didu_fit / dvdu

    perr = 2 * np.sqrt(np.diag(pcov)) # error = 2 * sigma
    residuals = didu - didu_fit
    sigma = didu * 0.01 # estimate a 1% error on dI/dV measurements
    ids = sigma != 0.0
    chi2 = np.sum(residuals[ids]**2 / sigma[ids]**2)
    dof = max(1, len(didu) - len(popt))
    reduced_chi2 = chi2 / dof
    squared_sum_residuals = np.sum(residuals**2)
    squared_sum_total = np.sum((didu - np.mean(didu))**2)
    r2 = 1 - squared_sum_residuals / squared_sum_total
    fit_info = {'chi2': chi2, 'red_chi2': reduced_chi2, 'r2': r2, 'cov': pcov, 'dof': dof}

    return didv_fit, popt, perr, fit_info, didv_start


def compute_rpa_params(
        popt: np.ndarray,
        perr: np.ndarray,
        mass: float, # electronvolt microsecond^2 / millimeter^2
        charge: float, # femtocoulomb
        Aeff: float, # millimeter^2
        imax: tuple[float, float], # nanoampere
        ) -> dict[str, tuple[float, float]]:

    if Aeff == 0.0 or charge == 0.0:
        raise ValueError(f'Invalid value(s): Aeff = {Aeff}, q = {charge}')
    if 0.0 in popt:
        raise ValueError(f'Invalide optimal parameters: popt = {popt}')

    u = popt[2] # millimeter / microsecond
    du = perr[2]

    e = mass * u**2 / 2 # electronvolt
    de = mass * np.abs(u) * du

    beta = popt[1]**0.5 # microsecond / millimeter
    dbeta = perr[1] / (2 * beta)

    t = mass / (beta**2) # electronvolt
    dt = 2 * t * dbeta / beta

    n1 = popt[0] * np.sqrt(np.pi) / (Aeff * charge * beta) # millimeter^-3
    dn1 = n1 * np.sqrt((dbeta / beta)**2 + (perr[0] / popt[0])**2)

    n2 = imax[0] / (Aeff * charge * u) # millimeter^-3
    dn2 = n2 * np.sqrt((du / u)**2 + (imax[1] / imax[0])**2)

    return {'N1': (n1, dn1), 'N2': (n2, dn2), 'U3': (u, du), 'E': (e, de), 'T': (t, dt)}


def compute_idm_params(
        currents: np.ndarray,
        u3: tuple[float, float],
        R: float,
        D: float,
        plot_filename: Path,
        factor: float = 0.99,
        ) -> dict[str, tuple[float, float]]:
    
    total_currents = np.sum(currents, axis=1) # nanoampere
    ids = total_currents > factor * np.max(total_currents)

    r1s = (currents[ids, 0] + currents[ids, 3] - currents[ids, 1] - currents[ids, 2]) / total_currents[ids]
    r2s = (currents[ids, 0] + currents[ids, 1] - currents[ids, 2] - currents[ids, 3]) / total_currents[ids]
    r1 = float(np.mean(r1s))
    r2 = float(np.mean(r2s))
    dr1 = float(2 * np.std(r1s))
    dr2 = float(2 * np.std(r2s))

    plt.plot(r1s, 'k', label='r1')
    plt.plot([r1] * len(r1s), 'k--', label='r1 avg.')
    plt.plot([r1+dr1] * len(r1s), 'k:', label='r1 2\u03c3')
    plt.plot([r1-dr1] * len(r1s), 'k:')
    plt.plot(r2s, 'r', label='r2')
    plt.plot([r2] * len(r2s), 'r--', label='r2 avg.')
    plt.plot([r2+dr2] * len(r2s), 'r:', label='r2 2\u03c3')
    plt.plot([r2-dr2] * len(r2s), 'r:')
    plt.legend()
    plt.grid()
    plt.xlabel('index')
    plt.savefig(plot_filename, dpi=400)

    c = np.pi * R / D / 4

    u1 = c * r1 * u3[0]
    du1 = np.abs(u1) * np.sqrt((dr1 / r1)**2 + (u3[1] / u3[0])**2)

    u2 = c * r2 * u3[0]
    du2 = np.abs(u2) * np.sqrt((dr2 / r2)**2 + (u3[1] / u3[0])**2)

    return {'U1': (u1, du1), 'U2': (u2, du2)}


def get_theoretical_currents(
        voltages: np.ndarray,
        n: float,
        u: float,
        t: float,
        m: float,
        q: float,
        A: float,
        ) -> np.ndarray:
    
    v = np.sqrt(2 * voltages / m)
    vth = np.sqrt(t / m)
    chi = (u - v) / vth
    erfchi = np.array([math.erf(c) for c in chi])
    i = 1 + erfchi + np.exp(-chi**2) * vth / u / np.sqrt(np.pi)
    i *= A * n * q * u / 2
    return i


if __name__ == '__main__':
    import sys
    file = sys.argv[1]
    assert isinstance(file, Path)
    plasma_parameters(file)