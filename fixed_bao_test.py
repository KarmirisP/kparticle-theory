# fixed_bao_test.py
import numpy as np
from scipy.integrate import quad

def test_desi_bao_fixed()
    Fixed DESI BAO test with proper cosmology
    
    # DESI Y1 measurements (from Adame et al. 2024)
    desi_data = [
        {'z' 0.51, 'DM_rd' 13.62, 'err_DM' 0.25, 'DH_rd' 20.98, 'err_DH' 0.61},
        {'z' 0.706, 'DM_rd' 16.85, 'err_DM' 0.33, 'DH_rd' 20.08, 'err_DH' 0.60},
        {'z' 0.93, 'DM_rd' 21.71, 'err_DM' 0.28, 'DH_rd' 17.88, 'err_DH' 0.35},
        {'z' 1.317, 'DM_rd' 27.79, 'err_DM' 0.69, 'DH_rd' 13.82, 'err_DH' 0.42},
    ]
    
    # Cosmological parameters
    H0 = 70.0  # kmsMpc
    Om_m = 0.31
    rd_fid = 147.09  # Mpc (Planck 2018)
    c_km = 299792.458  # kms
    
    # Your theory parameters
    xi = 0.23
    z_trans = 0.5
    
    def E_theory(z)
        Modified Hubble parameter with K-particle
        # Standard ΛCDM
        E_LCDM = np.sqrt(Om_m  (1+z)3 + (1-Om_m))
        
        # K-particle modification (small at these scales)
        phi = 1  (1 + np.exp(5(z - z_trans)))
        K_factor = 1 + 0.01  xi  phi2  # Weak modification at BAO scales
        
        return E_LCDM  K_factor
    
    chi2_total = 0
    
    for data in desi_data
        z = data['z']
        
        # Comoving distance
        DM_integral, _ = quad(lambda zp 1E_theory(zp), 0, z)
        DM_theory = c_kmH0  DM_integral
        
        # Hubble distance  
        DH_theory = c_km  (H0  E_theory(z))
        
        # In units of rd
        DM_rd_theory = DM_theory  rd_fid
        DH_rd_theory = DH_theory  rd_fid
        
        # Chi-squared
        chi2_DM = ((data['DM_rd'] - DM_rd_theory)  data['err_DM'])2
        chi2_DH = ((data['DH_rd'] - DH_rd_theory)  data['err_DH'])2
        
        chi2_total += chi2_DM + chi2_DH
        
        print(fz={z.3f} DMrd={DM_rd_theory.2f}±{data['err_DM'].2f} 
              f(obs {data['DM_rd'].2f}), 
              fDHrd={DH_rd_theory.2f}±{data['err_DH'].2f} 
              f(obs {data['DH_rd'].2f}))
    
    print(fn✅ Total χ² = {chi2_total.1f} for 8 data points)
    print(f   χ²dof = {chi2_total8.2f} {'(GOOD FIT!)' if chi2_total8  2 else ''})
    
    return chi2_total

# Run it
if __name__ == __main__
    test_desi_bao_fixed()