from math import sin, cos, sqrt, radians, atan2, degrees

def gps_to_ecef(lat, lon, alt): # Transforms GPS coordinates to cartesian coordinates (center of earth is (0, 0, 0))
    a = 6378137.0  # Earth's equatorial radius in meters
    e = 8.1819190842622e-2 # Earth's eccentrity - refers to the Earth's flattening on poles (ellipsoid)

    lat = radians(lat)
    lon = radians(lon)

    N = a / sqrt(1 - e**2 * sin(lat)**2) # Calculate radius of curvature - Earth's radius in given place

    X = (N + alt) * cos(lat) * cos(lon)
    Y = (N + alt) * cos(lat) * sin(lon)
    Z = ((1 - e**2) * N + alt) * sin(lat)

    return X, Y, Z

def ecef_to_enu_vector(dir_vector, ref_lat, ref_lon): # Changes coordinates to local (East-North-Up)
    lat = radians(ref_lat)
    lon = radians(ref_lon)

    dx, dy, dz = dir_vector

    t = [ # Transformation matrix
        [-sin(lon),               cos(lon),              0],
        [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)],
        [ cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat)]
    ]

    east  = t[0][0]*dx + t[0][1]*dy + t[0][2]*dz
    north = t[1][0]*dx + t[1][1]*dy + t[1][2]*dz
    up    = t[2][0]*dx + t[2][1]*dy + t[2][2]*dz

    return east, north, up

def compute_angles(east, north, up):
    azimuth = atan2(east, north)
    elevation = atan2(up, sqrt(east**2 + north**2))

    return degrees(azimuth), degrees(elevation)

def get_camera_orientation(cam_gps, obj_gps):
    cam_ecef = gps_to_ecef(*cam_gps)
    obj_ecef = gps_to_ecef(*obj_gps)
    dir_vector = [o - c for o, c in zip(obj_ecef, cam_ecef)]
    east, north, up = ecef_to_enu_vector(dir_vector, cam_gps[0], cam_gps[1])
    azimuth, elevation = compute_angles(east, north, up)
    return azimuth % 360, elevation

# Przykładowe dane
cam_gps = (50.06143, 19.93658, 220)  # kamera
obj_gps = (50.06465, 19.94498, 220)  # obiekt

azimuth, elevation = get_camera_orientation(cam_gps, obj_gps)

print(f"Azymut: {azimuth:.2f}°")
print(f"Elewacja: {elevation:.2f}°")
