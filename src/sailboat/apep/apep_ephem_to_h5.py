import numpy as np
from scipy.interpolate import interp1d
import h5py

min_alt = 40e3 # m
start_times = [np.datetime64('2023-10-14T16:00:00'), 
               np.datetime64('2023-10-14T16:35:00'), 
               np.datetime64('2023-10-14T17:10:00')]
h5f = h5py.File('../apep/2023/ephemeris.h5', 'w')

for rid in range(386, 389):
    filename = f'../apep/2023/ephemeris/Apep1_x{rid}.txt'
    print(f'Processing {filename}...')
    data = np.loadtxt(filename, delimiter='\t', dtype=np.float64)
    time = start_times[rid - 386] + (data[:, 0]*1e6).astype('timedelta64[us]')
    day0 = time[0].astype('datetime64[D]')
    us_of_day = (time - day0).astype(np.int64)

    lat = data[:, 1]
    lon = data[:, 2] % 360
    alt = data[:, 3] * 1e3
    
    ds = h5f.create_dataset(f'/36.{rid}/raw/time', data=us_of_day, dtype=np.int64)
    ds.attrs['description'] = f'Microsecond of {day0}'
    ds.attrs['units'] = 'microseconds'

    ds = h5f.create_dataset(f'/36.{rid}/raw/latitude', data=lat, dtype=np.float64)
    ds.attrs['description'] = 'Geographic latitude'
    ds.attrs['units'] = 'degrees'

    ds = h5f.create_dataset(f'/36.{rid}/raw/longitude', data=lon, dtype=np.float64)
    ds.attrs['description'] = 'Geographic longitude'
    ds.attrs['units'] = 'degrees'

    ds = h5f.create_dataset(f'/36.{rid}/raw/altitude', data=alt, dtype=np.float64)
    ds.attrs['description'] = 'Geographic altitude'
    ds.attrs['units'] = 'meters'

    is_above_min_alt = alt > min_alt
    is_before_max_time = time < np.datetime64('2023-10-15')
    ids = is_above_min_alt & is_before_max_time
    
    dt = np.median(np.diff(us_of_day[ids]))
    us_of_day_interp = np.arange(np.min(us_of_day[ids]), np.max(us_of_day[ids]) + dt/2, dt, dtype=np.int64)

    flat = interp1d(us_of_day[ids], lat[ids])
    flon = interp1d(us_of_day[ids], lon[ids])
    falt = interp1d(us_of_day[ids], alt[ids])

    lat_interp = flat(us_of_day_interp)
    lon_interp = flon(us_of_day_interp)
    alt_interp = falt(us_of_day_interp)

    ds = h5f.create_dataset(f'/36.{rid}/interpolated/time', data=us_of_day_interp, dtype=np.int64)
    ds.attrs['description'] = f'Interpolated microsecond of {day0}'
    ds.attrs['units'] = 'microseconds'

    ds = h5f.create_dataset(f'/36.{rid}/interpolated/latitude', data=lat_interp, dtype=np.float64)
    ds.attrs['description'] = 'Interpolated geographic latitude'
    ds.attrs['units'] = 'degrees'

    ds = h5f.create_dataset(f'/36.{rid}/interpolated/longitude', data=lon_interp, dtype=np.float64)
    ds.attrs['description'] = 'Interpolated geographic longitude'
    ds.attrs['units'] = 'degrees'

    ds = h5f.create_dataset(f'/36.{rid}/interpolated/altitude', data=alt_interp, dtype=np.float64)
    ds.attrs['description'] = 'Interpolated geographic altitude'
    ds.attrs['units'] = 'meters'

h5f.close()
