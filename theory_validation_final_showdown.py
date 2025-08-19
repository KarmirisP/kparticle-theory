# theory_validation_final_showdown.py
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================
DATA_FOLDER = 'sparc_data'
CSV_FILE = 'SPARC_Lelli2016_masses_cleaned.csv'
EPSILON = 1e-10
G_ASTRO = 4.3009e-6  # kpc (km/s)² / M_sun
A0_MOND = 1.2e-10 * (3.086e16)**2 * 1e3  # Corrected MOND constant
c_LIGHT = 3e5
H0 = 0.07

# ============================================================
# DATA LOADING (FROM WORKING VERSION)
# ============================================================


def load_sparc_data(filepath):
    """Load SPARC data with automatic corruption handling"""
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    # Check for corruption patterns
    is_corrupted = any(x in content for x in ["994634", "2614594", "1648158"])
    is_8_column = "SBdisk" in content and "SBbul" in content

    if is_8_column:
        df = pd.read_csv(
            filepath, sep=r'\s+', comment='#',
            names=['R', 'V_obs', 'e_V_obs', 'V_gas',
                   'V_disk', 'V_bulge', 'SBdisk', 'SBbulge'],
            na_values=['-9999', 'NaN', 'nan']
        )
    elif is_corrupted:
        data = []
        for line in content.split('\n'):
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        row = [
                            float(parts[0]),
                            float(parts[1])/10000,
                            float(parts[2])/10000,
                            float(parts[3])/10000,
                            float(parts[4])/10000,
                            float(parts[5])/10000
                        ]
                        data.append(row)
                    except:
                        continue
        df = pd.DataFrame(
            data, columns=['R', 'V_obs', 'e_V_obs', 'V_gas', 'V_disk', 'V_bulge'])
    else:
        df = pd.read_csv(
            filepath, sep=r'\s+', comment='#',
            names=['R', 'V_obs', 'e_V_obs', 'V_gas', 'V_disk', 'V_bulge'],
            na_values=['-9999', 'NaN', 'nan']
        )

    # Clean data
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Convert to absolute values and check scale
    for col in ['V_obs', 'V_gas', 'V_disk', 'V_bulge']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: abs(
                float(x)) if pd.notna(x) else 0)

    # Scale check
    v_max = df['V_obs'].max()
    if v_max < 20:  # Too low
        scale = 10
        for col in ['V_obs', 'e_V_obs', 'V_gas', 'V_disk', 'V_bulge']:
            if col in df.columns:
                df[col] *= scale

    return df


def load_and_prepare_data():
    """Full data loading with feature engineering"""
    metadata = pd.read_csv(CSV_FILE)
    metadata['Name_std'] = metadata['Name'].str.strip(
    ).str.replace(r'\s+', '', regex=True).str.upper()
    metadata.set_index('Name_std', inplace=True)

    all_galaxies = []

    for fname in os.listdir(DATA_FOLDER):
        if not fname.endswith(('.dat', 'rotmod.dat')):
            continue

        name_std = "".join(fname.split('_')[0].strip().split()).upper()

        if name_std not in metadata.index:
            continue

        try:
            meta = metadata.loc[name_std]
            filepath = os.path.join(DATA_FOLDER, fname)

            df = load_sparc_data(filepath)

            if len(df) < 3:
                continue

            # Get Hubble type
            try:
                hubble_type = float(meta['Type'])
            except:
                hubble_type = 5.0  # Default to Sc

            # M/L ratios
            if hubble_type <= -2:
                M_L_disk, M_L_bulge = 0.8, 0.9
            elif hubble_type <= 0:
                M_L_disk, M_L_bulge = 0.7, 0.8
            elif hubble_type <= 3:
                M_L_disk, M_L_bulge = 0.5, 0.7
            elif hubble_type <= 6:
                M_L_disk, M_L_bulge = 0.3, 0.5
            else:
                M_L_disk, M_L_bulge = 0.2, 0.3

            # Calculate V_bary
            v_gas_sq = df['V_gas']**2
            v_disk_sq = df['V_disk']**2 * M_L_disk
            v_bulge_sq = df['V_bulge']**2 * M_L_bulge
            df['v_bary'] = np.sqrt(np.maximum(
                0, v_gas_sq + v_disk_sq + v_bulge_sq))

            # Variable M/L for UC
            M_L_disk_var = 0.5 if hubble_type >= 7 else 1.0
            v_disk_sq_var = df['V_disk']**2 * M_L_disk_var
            df['v_bary_uc'] = np.sqrt(np.maximum(
                0, v_gas_sq + v_disk_sq_var + v_bulge_sq))

            # Calculate features
            stellar_mass = meta['L3.6'] * 1e9 * M_L_disk
            gas_mass = meta.get('MHI', 0) * 1e9
            if pd.isna(gas_mass):
                gas_mass = 0
            total_mass = stellar_mass + gas_mass

            R_disk = meta.get('Rdisk', 10)
            if pd.isna(R_disk) or R_disk <= 0:
                R_disk = 10

            log_E_g = np.log10(np.abs(G_ASTRO * total_mass **
                               2 / (R_disk + EPSILON)) + EPSILON)
            log_Sigma = np.log10(
                total_mass / (2 * np.pi * R_disk**2 + EPSILON))
            gas_fraction = gas_mass / (total_mass + EPSILON)
            log_mstar = np.log10(stellar_mass + EPSILON)

            galaxy = {
                'name': meta['Name'],
                'df': df,
                'features': {
                    'hubble_type': hubble_type,
                    'inclination': meta.get('i', 60),
                    'log_E_g': log_E_g,
                    'log_Sigma': log_Sigma,
                    'log_mstar': log_mstar,
                    'total_mass': total_mass,
                    'stellar_mass': stellar_mass,
                    'gas_mass': gas_mass,
                    'gas_fraction': gas_fraction,
                    'R_disk': R_disk
                }
            }
            all_galaxies.append(galaxy)

        except Exception as e:
            continue

    print(f"Successfully loaded {len(all_galaxies)} galaxies")
    # return all_galaxies
    # FIX: Convert the list of galaxies to a NumPy array to allow advanced indexing
    return np.array(all_galaxies, dtype=object)

# ============================================================
# ALL MODEL DEFINITIONS INCLUDING HYBRID
# ============================================================


def safe_sqrt(x):
    """Safe square root."""
    return np.sqrt(np.maximum(0, x))

# --- BASELINE MODELS ---


def model_baryonic(df, params, features):
    """Pure baryonic."""
    return df['v_bary']


def model_simple_scaling(df, params, features):
    """Simple universal scaling."""
    p0 = np.clip(params[0], 0.1, 10)
    return df['v_bary'] * safe_sqrt(p0)

# --- DARK MATTER MODELS ---


def model_lcdm_nfw(df, params, features):
    """ΛCDM with NFW halo."""
    v200 = np.clip(params[0], 50, 300)
    c = np.clip(params[1], 5, 20)

    r200 = v200 / (10 * H0)
    rs = r200 / c
    r = df['R'].values
    x = r / (rs + EPSILON)

    v_halo_sq = np.zeros_like(r)
    mask = x > 0

    A_x = np.log(1 + x[mask]) - x[mask]/(1 + x[mask])
    A_c = np.log(1 + c) - c/(1 + c)

    v_halo_sq[mask] = v200**2 * A_x / (x[mask] * (A_c + EPSILON))

    return safe_sqrt(df['v_bary']**2 + v_halo_sq)


def model_lcdm_dc14(df, params, features):
    """DC14 profile with baryonic feedback."""
    v200 = np.clip(params[0], 50, 300)
    c = np.clip(params[1], 5, 20)

    log_mstar = features['log_mstar']

    # DC14 parameters
    alpha = 2.94 - np.log10((10**log_mstar)**0.035 + 0.1)
    beta = 4.23 - 1.34 * log_mstar + 0.26 * log_mstar**2
    gamma = -0.06 + np.log10((10**log_mstar)**-0.04 + 0.1)

    r200 = v200 / (10 * H0)
    rs = r200 / c
    r = df['R'].values
    x = r / (rs + EPSILON)

    f_trans = np.tanh(x)
    v_inner = v200 * safe_sqrt(x) * (1 - np.exp(-x))

    A_x = np.log(1 + x) - x/(1 + x)
    A_c = np.log(1 + c) - c/(1 + c)
    v_outer = v200 * safe_sqrt(A_x / (x * (A_c + EPSILON)))

    v_dm = f_trans * v_outer + (1 - f_trans) * v_inner

    gas_fraction = features['gas_fraction']
    feedback_factor = 1.0 - 0.3 * gas_fraction
    v_dm *= feedback_factor

    return safe_sqrt(df['v_bary']**2 + v_dm**2)


def model_burkert(df, params, features):
    """Burkert profile."""
    rho_0 = np.clip(params[0], 1e5, 1e9)
    r_0 = np.clip(params[1], 1, 20)
    r = df['R'].values

    x = r / (r_0 + EPSILON)
    M_enc = 4 * np.pi * rho_0 * r_0**3 * (
        0.5 * np.log(1 + x**2) + np.log(1 + x) - np.arctan(x)
    )
    v_halo_sq = G_ASTRO * M_enc / (r + EPSILON)

    return safe_sqrt(df['v_bary']**2 + v_halo_sq)

# --- MODIFIED GRAVITY MODELS ---


def model_mond(df, params, features):
    """MOND with simple interpolation."""
    a0 = np.clip(params[0], 3000, 4500)

    r = df['R'].values + EPSILON
    g_bary = df['v_bary']**2 / r

    # Numerically stable MOND
    x = g_bary / (a0 + EPSILON)
    g_mond = safe_sqrt(g_bary**2 + a0**2)

    return safe_sqrt(np.clip(g_mond * r, 0, 1e6))


def model_rar(df, params, features):
    """Radial Acceleration Relation."""
    a0 = np.clip(params[0], 3000, 4500)

    r = df['R'].values + EPSILON
    g_bary = df['v_bary']**2 / r

    # RAR function
    x = g_bary / (a0 + EPSILON)
    g_obs = g_bary / (1 - np.exp(-safe_sqrt(x)))

    return safe_sqrt(np.clip(g_obs * r, 0, 1e6))

# --- UC MODELS ---


def model_uc_fixed(df, params, features):
    """UC with fixed M/L."""
    p0 = np.clip(params[0], 0.8, 1.5)
    p1 = np.clip(params[1], -0.1, 0.1)

    T = features['hubble_type']
    scaling = np.clip(p0 + p1 * T, 0.5, 2.0)

    return df['v_bary'] * safe_sqrt(scaling)


def model_uc_variable(df, params, features):
    """UC with variable M/L."""
    p0 = np.clip(params[0], 0.8, 1.5)
    p1 = np.clip(params[1], -0.1, 0.1)

    T = features['hubble_type']
    scaling = np.clip(p0 + p1 * T, 0.5, 2.0)

    return df['v_bary_uc'] * safe_sqrt(scaling)


def model_uc_continuous(df, params, features):
    """UC with continuous morphology."""
    p0 = np.clip(params[0], 0.8, 1.5)
    p1 = np.clip(params[1], -0.1, 0.1)
    p2 = np.clip(params[2], -0.05, 0.05)

    T_norm = (features['hubble_type'] - 5.0) / 5.0
    scaling = np.clip(p0 + p1 * T_norm + p2 * T_norm**2, 0.5, 2.0)

    return df['v_bary'] * safe_sqrt(scaling)

# --- PARTANEN MODELS ---


def model_partanen_energy(df, params, features):
    """Partanen energy-dependent."""
    alpha = np.clip(params[0], -1, 1)
    beta = np.clip(params[1], -0.5, 0.5)
    gamma = np.clip(params[2], -0.2, 0.2)

    log_E = features['log_E_g']
    denominator = 1 + abs(gamma * log_E) + EPSILON
    scaling = 1 + (alpha + beta * log_E) / denominator
    scaling = np.clip(scaling, 0.5, 2.0)

    return df['v_bary'] * safe_sqrt(scaling)


def model_partanen_surface(df, params, features):
    """Partanen surface density."""
    alpha = np.clip(params[0], -1, 1)
    beta = np.clip(params[1], -0.5, 0.5)
    gamma = np.clip(params[2], -0.2, 0.2)

    log_S = features['log_Sigma']
    denominator = 1 + abs(gamma * log_S) + EPSILON
    scaling = 1 + (alpha + beta * log_S) / denominator
    scaling = np.clip(scaling, 0.5, 2.0)

    return df['v_bary'] * safe_sqrt(scaling)


def model_partanen_unified(df, params, features):
    """Partanen unified."""
    p0 = np.clip(params[0], 0.8, 1.2)
    p1 = np.clip(params[1], -0.1, 0.1)
    p2 = np.clip(params[2], -0.1, 0.1)

    log_E = features['log_E_g']
    log_S = features['log_Sigma']
    scaling = p0 + p1 * log_E + p2 * log_S
    scaling = np.clip(scaling, 0.5, 2.0)

    return df['v_bary'] * safe_sqrt(scaling)

# --- NEW PHYSICS MODELS ---


def model_biphasic_gravity(df, params, features):
    """Biphasic gravity."""
    g_rep = np.clip(params[0], 10, 100)
    r_rep = np.clip(params[1], 5, 50)

    r = df['R'].values + EPSILON
    g_repulsive = g_rep * np.exp(-r/r_rep) / r
    g_attractive = df['v_bary']**2 / r
    g_total = np.maximum(0, g_attractive - g_repulsive)

    return safe_sqrt(np.clip(g_total * r, 0, 1e6))


def model_scalar_field(df, params, features):
    """Scalar field fifth force."""
    phi0 = np.clip(params[0], 1, 10)
    m_phi = np.clip(params[1], 0.01, 0.5)
    beta = np.clip(params[2], 1, 10)

    r = df['R'].values + EPSILON
    phi = phi0 * np.exp(-m_phi * r)
    g_scalar = beta * phi / r
    g_total = df['v_bary']**2 / r + g_scalar

    return safe_sqrt(np.clip(g_total * r, 0, 1e6))


def model_vector_field(df, params, features):
    """Vector field theory."""
    g_v = np.clip(params[0], 1, 10)
    m_v = np.clip(params[1], 0.01, 0.5)
    lambda_v = np.clip(params[2], 0.1, 1)

    r = df['R'].values + EPSILON
    A_mu = g_v * np.exp(-m_v * r) / r
    g_vector = A_mu * (1 + lambda_v * abs(A_mu))
    g_total = df['v_bary']**2 / r + g_vector

    return safe_sqrt(np.clip(g_total * r, 0, 1e6))


def model_emergent_gravity(df, params, features):
    """Verlinde's emergent gravity."""
    a0_eg = np.clip(params[0], 2000, 4000)
    c_H = np.clip(params[1], 0.1, 2.0)

    r = df['R'].values + EPSILON
    a_D = c_H * c_LIGHT**2 * H0 / 6
    g_emergent = safe_sqrt(a0_eg * max(a_D, EPSILON)) * \
        (1 + r / (features['R_disk'] + EPSILON))
    g_total = df['v_bary']**2 / r + g_emergent

    return safe_sqrt(np.clip(g_total * r, 0, 1e6))


def model_superfluid_dm(df, params, features):
    """Superfluid dark matter."""
    Lambda_sf = np.clip(params[0], 10, 500)
    alpha = np.clip(params[1], 0.5, 5)

    r = df['R'].values + EPSILON
    g_bary = df['v_bary']**2 / r
    a0_eff = max(Lambda_sf * G_ASTRO, EPSILON)
    x = g_bary / (a0_eff + EPSILON)
    nu = 1 / (1 + np.exp(-alpha * (x - 1)))
    g_sf = g_bary * (1 + nu)

    return safe_sqrt(np.clip(g_sf * r, 0, 1e6))


def model_kparticle(df, params, features):
    """K-particle ultra-light dark matter."""
    rho_0 = np.clip(params[0], 1e6, 1e9)
    r_c = np.clip(params[1], 5, 20)
    n = np.clip(params[2], 0.5, 1.5)

    r = df['R'].values + EPSILON
    rho = rho_0 / (1 + (r/(r_c + EPSILON))**2)**n
    M_enc = 4 * np.pi * rho * r**3 / 3
    g_k = G_ASTRO * M_enc / (r**2 + EPSILON)
    g_total = df['v_bary']**2 / r + g_k

    return safe_sqrt(np.clip(g_total * r, 0, 1e6))


def model_quantum_temporal(df, params, features):
    """Quantum temporal field."""
    tau_0 = np.clip(params[0], 0.1, 1)
    omega = np.clip(params[1], 0.1, 1)
    phi = np.clip(params[2], 0, 6.28)

    r = df['R'].values + EPSILON
    quantum_factor = 1 + tau_0 * np.sin(omega * r + phi)
    g_quantum = df['v_bary']**2 / r * quantum_factor

    return safe_sqrt(np.clip(g_quantum * r, 0, 1e6))


def model_unified_interaction(df, params, features):
    """Unified morphology-energy interaction."""
    p0 = np.clip(params[0], -1, 1)
    p1 = np.clip(params[1], -0.1, 0.1)
    q0 = np.clip(params[2], -1, 1)
    q1 = np.clip(params[3], -0.1, 0.1)

    T = features['hubble_type']
    log_E = features['log_E_g']
    morph_term = p0 + p1 * T
    energy_term = q0 + q1 * log_E
    scaling = 1 + morph_term * energy_term
    scaling = np.clip(scaling, 0.5, 2.0)

    return df['v_bary'] * safe_sqrt(scaling)

# ============================================================
# FIXED: HYBRID RAR + UC MORPHOLOGY MODEL
# ============================================================


def model_hybrid_rar_uc(df, params, features):
    """Hybrid RAR with UC morphology correction - FIXED VERSION."""
    a0 = np.clip(params[0], 3000, 4500)
    p0 = np.clip(params[1], 0.5, 2.0)  # Wider range
    p1 = np.clip(params[2], -0.1, 0.1)  # Wider range

    # Standard RAR first
    r = df['R'].values + EPSILON
    g_bary = df['v_bary']**2 / r

    # Apply RAR transformation
    x = g_bary / (a0 + EPSILON)
    g_rar = g_bary / (1 - np.exp(-safe_sqrt(x)))
    v_rar = safe_sqrt(g_rar * r)

    # THEN apply morphology correction to the OUTPUT
    T = features['hubble_type']
    morph_correction = np.clip(p0 + p1 * T, 0.5, 2.0)

    return v_rar * safe_sqrt(morph_correction)

# Alternative implementation that scales acceleration differently


def model_hybrid_rar_uc_v2(df, params, features):
    """Alternative: Apply morphology to acceleration scale."""
    a0_base = np.clip(params[0], 3000, 4500)
    morph_factor = np.clip(params[1], 0.8, 1.2)
    type_coef = np.clip(params[2], -0.05, 0.05)

    # Morphology-dependent acceleration scale
    T = features['hubble_type']
    a0_effective = a0_base * (morph_factor + type_coef * T)

    # Standard RAR with modified a0
    r = df['R'].values + EPSILON
    g_bary = df['v_bary']**2 / r
    x = g_bary / (a0_effective + EPSILON)
    g_obs = g_bary / (1 - np.exp(-safe_sqrt(x)))

    return safe_sqrt(np.clip(g_obs * r, 0, 1e6))

# ============================================================
# EVALUATION
# ============================================================


def evaluate_theory_robust(train, test, name, model, bounds):
    """Evaluate with robust metrics."""

    print(f"\nTesting: {name}")

    # Optimize on training
    if bounds:
        def objective(params):
            chi2_total = 0
            n_points = 0

            for g in train:
                try:
                    pred = model(g['df'], params, g['features'])
                    obs = g['df']['V_obs'].values

                    # Check validity
                    if np.any(~np.isfinite(pred)) or np.any(pred < 0):
                        return 1e10

                    # Use robust errors
                    err = np.maximum(0.1 * obs + 5, 5)

                    chi2 = np.sum(((obs - pred) / err)**2)
                    chi2_total += chi2
                    n_points += len(obs)

                except:
                    return 1e10

            return chi2_total / max(n_points, 1)

        try:
            result = differential_evolution(
                objective, bounds,
                seed=42, maxiter=100,
                popsize=15, tol=1e-6
            )
            params = result.x
            print(f"  Optimized: χ²/n = {result.fun:.2f}, params = {params}")
        except:
            params = [np.mean(b) for b in bounds]
            print(f"  Using defaults: {params}")
    else:
        params = []

    # Test phase
    chi2_list = []
    all_obs = []
    all_pred = []

    for g in test:
        try:
            pred = model(g['df'], params, g['features'])
            obs = g['df']['V_obs'].values

            # Skip invalid predictions
            if np.any(~np.isfinite(pred)) or np.any(pred < 0):
                continue

            # Robust errors
            err = np.maximum(0.1 * obs + 5, 5)

            chi2_red = np.sum(((obs - pred) / err)**2) / len(obs)

            if np.isfinite(chi2_red) and chi2_red < 1e6:
                chi2_list.append(chi2_red)
                all_obs.extend(obs)
                all_pred.extend(pred)

        except:
            continue

    # Calculate metrics
    if chi2_list:
        success_rate = np.mean(np.array(chi2_list) < 2.0) * 100
        median_chi2 = np.median(chi2_list)

        if len(all_obs) > 30:
            correlation = pearsonr(all_obs, all_pred)[0]
        else:
            correlation = 0
    else:
        success_rate = 0
        median_chi2 = np.inf
        correlation = 0

    print(
        f"  → Success: {success_rate:.1f}%, χ²: {median_chi2:.2f}, r: {correlation:.3f}")

    return {
        'name': name,
        'success': success_rate,
        'chi2': median_chi2,
        'r': correlation,
        'params': params
    }

# ============================================================
# MAIN
# ============================================================


def main():
    print("\n🏆 FINAL SHOWDOWN - ALL 22 MODELS INCLUDING HYBRID")
    print("="*60)

    # Load data
    galaxies = load_and_prepare_data()

    if len(galaxies) < 50:
        print("❌ Insufficient data")
        return

    # Split data
    train, test = train_test_split(galaxies, test_size=0.25, random_state=42)
    print(f"Training: {len(train)}, Testing: {len(test)}")

    # ALL 22 models (21 original + 1 hybrid)
    models = [
        # Baseline
        ("Baryonic", model_baryonic, []),
        ("Simple Scaling", model_simple_scaling, [(1.5, 2.5)]),

        # Dark Matter (3 models)
        ("ΛCDM-NFW", model_lcdm_nfw, [(100, 250), (8, 15)]),
        ("ΛCDM-DC14", model_lcdm_dc14, [(100, 250), (8, 15)]),
        ("Burkert", model_burkert, [(1e6, 1e8), (2, 10)]),

        # Modified Gravity (2 models)
        ("MOND", model_mond, [(3000, 4500)]),
        # Modified Gravity
        ("RAR", model_rar, [(3000, 4500)]),

        # FIXED Hybrid models with better bounds
        ("★ Hybrid RAR+UC (Fixed)", model_hybrid_rar_uc,
         [(3000, 4500), (0.8, 1.5), (-0.1, 0.1)]),  # Much wider bounds

        ("★ Hybrid RAR+UC v2", model_hybrid_rar_uc_v2,
         [(3000, 4500), (0.9, 1.1), (-0.02, 0.02)]),

        # UC Models (3 models)
        ("UC Fixed M/L", model_uc_fixed, [(1.0, 1.5), (-0.05, 0.05)]),
        ("UC Variable M/L", model_uc_variable, [(1.0, 1.5), (-0.05, 0.05)]),
        ("UC Continuous", model_uc_continuous, [
         (1.0, 1.5), (-0.05, 0.05), (-0.02, 0.02)]),

        # Partanen Models (3 models)
        ("Partanen Energy", model_partanen_energy,
         [(-0.5, 0.5), (-0.2, 0.2), (-0.1, 0.1)]),
        ("Partanen Surface", model_partanen_surface,
         [(-0.5, 0.5), (-0.2, 0.2), (-0.1, 0.1)]),
        ("Partanen Unified", model_partanen_unified,
         [(0.8, 1.2), (-0.1, 0.1), (-0.1, 0.1)]),

        # New Physics (8 models)
        ("Biphasic Gravity", model_biphasic_gravity, [(20, 80), (10, 40)]),
        ("Scalar Field", model_scalar_field, [(2, 8), (0.05, 0.3), (2, 8)]),
        ("Vector Field", model_vector_field,
         [(2, 8), (0.05, 0.3), (0.2, 0.8)]),
        ("K-Particle", model_kparticle, [(1e7, 1e8), (8, 15), (0.8, 1.2)]),
        ("Emergent Gravity", model_emergent_gravity,
         [(2000, 4000), (0.5, 1.5)]),
        ("Superfluid DM", model_superfluid_dm, [(50, 200), (1, 3)]),
        ("Quantum Temporal", model_quantum_temporal,
         [(0.3, 0.7), (0.3, 0.7), (0, 6.28)]),
        ("Unified Interaction", model_unified_interaction, [
         (-0.5, 0.5), (-0.05, 0.05), (-0.5, 0.5), (-0.05, 0.05)]),

    ]

    # Test all models
    results = []
    for name, model, bounds in models:
        result = evaluate_theory_robust(train, test, name, model, bounds)
        results.append(result)

    # Sort by correlation (most important metric)
    results_sorted = sorted(results, key=lambda x: x['r'], reverse=True)

    print("\n" + "="*60)
    print("FINAL RESULTS - ALL 22 MODELS")
    print("="*60)
    print(f"{'Model':<22} {'Success':>8} {'χ²':>10} {'r':>8}")
    print("-"*60)

    for r in results_sorted:
        if r['chi2'] < 1e5:
            # Highlight the hybrid model
            if "Hybrid" in r['name']:
                print(
                    f">>> {r['name']:<18} {r['success']:>7.1f}% {r['chi2']:>10.2f} {r['r']:>8.3f} <<<")
            else:
                print(
                    f"{r['name']:<22} {r['success']:>7.1f}% {r['chi2']:>10.2f} {r['r']:>8.3f}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Best overall
    best = results_sorted[0]
    print(f"\n🏆 BEST MODEL: {best['name']}")
    print(f"   Success Rate: {best['success']:.1f}%")
    print(f"   Median χ²: {best['chi2']:.2f}")
    print(f"   Correlation: {best['r']:.3f}")

    # Check if hybrid is best
    hybrid_result = next((r for r in results if "Hybrid" in r['name']), None)
    if hybrid_result:
        rank = results_sorted.index(hybrid_result) + 1
        print(f"\n🌟 HYBRID MODEL RANK: #{rank} of 22")
        print(f"   Success Rate: {hybrid_result['success']:.1f}%")
        print(f"   Median χ²: {hybrid_result['chi2']:.2f}")
        print(f"   Correlation: {hybrid_result['r']:.3f}")
        print(
            f"   Parameters: a0={hybrid_result['params'][0]:.1f}, scale={hybrid_result['params'][1]:.3f}, morph={hybrid_result['params'][2]:.4f}")

    # Top 5
    print("\n📊 TOP 5 MODELS:")
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"   {i}. {r['name']}: r={r['r']:.3f}, χ²={r['chi2']:.2f}")

    # Category winners
    print("\n🏅 CATEGORY WINNERS:")

    # Dark Matter
    dm_models = [r for r in results if 'CDM' in r['name']
                 or 'Burkert' in r['name']]
    if dm_models:
        best_dm = max(dm_models, key=lambda x: x['r'])
        print(f"   Dark Matter: {best_dm['name']} (r={best_dm['r']:.3f})")

    # Modified Gravity
    mg_models = [r for r in results if 'MOND' in r['name']
                 or 'RAR' in r['name'] or 'Hybrid' in r['name']]
    if mg_models:
        best_mg = max(mg_models, key=lambda x: x['r'])
        print(f"   Modified Gravity: {best_mg['name']} (r={best_mg['r']:.3f})")

    # UC Models
    uc_models = [r for r in results if 'UC' in r['name']
                 and 'Hybrid' not in r['name']]
    if uc_models:
        best_uc = max(uc_models, key=lambda x: x['r'])
        print(f"   UC Models: {best_uc['name']} (r={best_uc['r']:.3f})")

    # New Physics
    new_physics = [r for r in results if any(x in r['name'] for x in
                   ['Scalar', 'Vector', 'Biphasic', 'Emergent', 'Superfluid', 'Quantum', 'K-Particle'])]
    if new_physics:
        best_new = max(new_physics, key=lambda x: x['r'])
        print(f"   New Physics: {best_new['name']} (r={best_new['r']:.3f})")


if __name__ == "__main__":
    main()
