# ----------------------------------------------------------------------------- #
# This script for saving 5-year averages of depth-integrated momentum budget terms
# for vorticity budget analysis for the depth-integrated flow
# ----------------------------------------------------------------------------- #

# Import modules
import xarray as xr
import numpy as np
from xgcm import Grid
import glob
import dask.distributed
from dask.distributed import Client
from dask.distributed import LocalCluster

#if __name__ == '__main__':
#    cluster = LocalCluster()
#    client = Client(cluster)
#    print(client)

# Define relevant variables to save
get_var = ['hf_CAu_2d', 'hf_CAv_2d', 'hf_PFu_2d', 'hf_PFv_2d', 'hf_diffu_2d', 'hf_diffv_2d', 'hf_du_dt_visc_2d', \
'hf_dudt_2d', 'hf_dv_dt_visc_2d', 'hf_dvdt_2d', 'hf_gKEu_2d', 'hf_gKEv_2d', 'hf_rvxu_2d', \
'hf_rvxv_2d', 'hf_u_BT_accel_2d', 'hf_v_BT_accel_2d','pbo','taux', 'taux_bot', 'tauy', 'tauy_bot',\
'ubt_dt','vbt_dt','wfo','col_height','intz_CAu_2d','intz_CAv_2d','intz_diffu_2d','intz_diffv_2d',\
'intz_gKEu_2d', 'intz_gKEv_2d', 'intz_PFu_2d', 'intz_PFv_2d', 'intz_rvxu_2d', 'intz_rvxv_2d',\
'intz_u_BT_accel_2d', 'intz_v_BT_accel_2d']

# Path to dataset, create filelist, read grid info
path = "/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20210308/CM4_piControl_c192_OM4p25/" + \
"gfdl.ncrc4-intel18-prod-openmp/pp/ocean_monthly/ts/monthly/5yr/"

filelist = glob.glob(path + "*intz_CAu_2d.nc")
filelist.sort()

path_ssh = "/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20210308/CM4_piControl_c192_OM4p25/" + \
"gfdl.ncrc4-intel18-prod-openmp/pp/ocean_daily/ts/daily/5yr/"

filelist_ssh = glob.glob(path_ssh + "*.zos.nc")
filelist_ssh.sort()

path_vmo = "/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20210308/CM4_piControl_c192_OM4p25/" + \
"gfdl.ncrc4-intel18-prod-openmp/pp/ocean_annual_z/ts/annual/10yr/"

filelist_vmo = glob.glob(path_vmo + "*.vmo.nc")
filelist_vmo.sort()
filelist_umo = glob.glob(path_vmo + "*.umo.nc")
filelist_umo.sort()

save_path = "/archive/Hemant.Khatri/MOM_Budget/"

path_grid = "/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20210308/CM4_piControl_c192_OM4p25/" + \
"gfdl.ncrc4-intel18-prod-openmp/pp/ocean_monthly/"
ds_grid = xr.open_dataset(path_grid + "ocean_monthly.static.nc")

# Loop for reading data and saving 5-yr averages
for i in range(0, 16):  #len(filelist)):

    tim_yr = filelist[i][-28:-15]
    print("Year Running = ", tim_yr)

    # First extract momentum budget terms
    filename = "ocean_monthly." + tim_yr + "*.nc"
    ds_full = xr.open_mfdataset(path + filename, decode_times=False)
    ds_full = ds_full.get(get_var)

    ds_full = ds_full.isel(xq = slice(1,1441), yq=slice(1,1081))
    ds_grid1 = ds_grid.isel(xq = slice(1,1441), yq=slice(1,1081))

    ds_full = xr.merge([ds_full, ds_grid1])

    # get d(ssh)/dt from daily-averages of ssh
    ds_ssh = xr.open_dataset(filelist_ssh[i], decode_times=False)
    dt = ds_ssh.time.shape[0] * 3600. * 24.
    eta_dt = (ds_ssh.zos.isel(time=ds_ssh.time.shape[0]-1) - ds_ssh.zos.isel(time=0)) / dt

    # Compute 5-yr averages of umo and vmo. note that each of the original files have 10 years of data
    ind = int(i / 2)
    ds_vmo = xr.open_dataset(filelist_vmo[ind], decode_times=False)
    ds_umo = xr.open_dataset(filelist_umo[ind], decode_times=False)

    if(i%2 == 0):
        vmo_2d = (ds_vmo['vmo'].isel(yq=slice(1,1081),time=slice(0, 5)).mean('time')).sum('z_l')
        umo_2d = (ds_umo['umo'].isel(xq=slice(1,1441),time=slice(0, 5)).mean('time')).sum('z_l')
    else:
        vmo_2d = (ds_vmo['vmo'].isel(yq=slice(1,1081),time=slice(5, 10)).mean('time')).sum('z_l')
        umo_2d = (ds_umo['umo'].isel(xq=slice(1,1441),time=slice(5, 10)).mean('time')).sum('z_l')

    # Save data
    ds_save = (ds_full).mean('time')
    ds_save = xr.merge([ds_save, vmo_2d, umo_2d, eta_dt])

    save_file = save_path + "OM4p25/OM4p25_" + tim_yr[0:4] + "_" + tim_yr[7:11] + ".nc"

    ds_save = ds_save.load()
    ds_save.to_netcdf(save_file)
