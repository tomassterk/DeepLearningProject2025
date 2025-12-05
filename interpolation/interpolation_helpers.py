import numpy as np 

a = 6378137.0  # semi-major axis [m]
f = 1 / 298.257223563 # flattening
e2 = 2*f - f**2 # eccentricity squared

# https://en.wikipedia.org/wiki/Earth_radius
def meridional_radius(phi_rad):
    sinphi = np.sin(phi_rad)
    denom = (1 - e2 * sinphi**2) ** 1.5
    return a * (1 - e2) / denom

def meters_per_degree_lat_exact(phi_deg):
    phi_rad = np.deg2rad(phi_deg)
    M = meridional_radius(phi_rad) # m per radian
    return M * np.pi / 180.0    # m per degree

def prime_vertical_radius(phi_deg):
    denom2 = np.sqrt(1-e2*np.sin(phi_deg)**2)
    return a / denom2

def meters_per_degree_long_exact(theta_rad):
    theta = np.deg2rad(theta_rad)
    N = prime_vertical_radius(theta)
    return (N*np.cos(theta)) * np.pi / 180

sample_lats = np.array([0.0, 30.0, 45.0]) 
sample_longs = np.array([0.0, 30.0])

y = np.array([meters_per_degree_lat_exact(phi) for phi in sample_lats])
phi_rad = np.deg2rad(sample_lats)
X = np.column_stack([
    np.ones_like(phi_rad), # A
    -np.cos(2 * phi_rad), # -B cos(2φ)
    np.cos(4 * phi_rad) # +C cos(4φ)
])

z = np.array([meters_per_degree_long_exact(phi) for phi in sample_longs])
theta_rad = np.deg2rad(sample_longs)
Z = np.column_stack([
    np.cos(theta_rad),
    -np.cos(3*theta_rad)
])

coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
A, B, C = coeffs

coeffs_long, *_ = np.linalg.lstsq(Z,z,rcond=None)
D, E = coeffs_long

# print("Fitted coefficients for meters/degree of latitude (WGS-84):")
# print(f"A = {A:.6f}")
# print(f"B = {B:.6f}")
# print(f"C = {C:.6f}")
# print(f"D = {D:.6f}")
# print(f"E = {E:.6f}")
def meters_per_degree(lat_deg):
    """Return (m_per_deg_lat, m_per_deg_lon) at given latitude (deg).
    local approximation for Denmark."""
    lat = np.radians(lat_deg)
    m_per_deg_lat = A - B*np.cos(2*lat) + C*np.cos(4*lat)
    m_per_deg_lon = D*np.cos(lat) - E*np.cos(3*lat)
    return m_per_deg_lat, m_per_deg_lon

def wrap_angle_diff_deg(a_next, a_now):
    """Minimal absolute diff in degrees, accounting for wrap at 360."""
    d = np.abs(a_next - a_now)
    d = np.where(d > 180.0, 360.0 - d, d)
    return d
