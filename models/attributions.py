import hvplot.xarray
import panel as pn
import cartopy.crs as ccrs
import wandb
import xarray as xr 
from xarrayutils import xr_linregress
import numpy as np


def log_attributions_quantiles(ds_attr, wandb_logger, input_variable):
    ds_attr_per_season = ds_attr.groupby('time.season').mean()
    html_file = f"/Data/gfi/users/rogui7909/wanbd_logs/{wandb_logger.experiment.id}_attributions.html"
    season = pn.widgets.RadioButtonGroup(name="Season",description='Season',options=['DJF','MAM','JJA','SON'])
    attr_method = pn.widgets.Select(description="Attribution method",options=[ str(k) for k in ds_attr_per_season.attr_method.values])

    ds_attr_per_season = ds_attr_per_season.astype('float32')
    def make_plot_attrs(var_name,season, attr_method, num_var=1):
        return ds_attr_per_season.attributions.isel(var_name=var_name).sel(season=season, attr_method=attr_method)\
                    .hvplot.quadmesh(x='longitude', y='latitude', geo=True, coastline=True,
                                        project=True, projection=ccrs.PlateCarree(0), crs=ccrs.PlateCarree(),
                                        cmap='PuOr_r', symmetric=True,frame_width=200, colorbar=(True if var_name==num_var-1 else False), 
                                        title=ds_attr_per_season.var_name.isel(var_name=var_name).values+' attributions')

    def make_plot_anomaly(var_name,season,num_var=1):
        return ds_attr_per_season.data.isel(var_name=var_name).sel(season=season)\
                    .hvplot.quadmesh(x='longitude', y='latitude', geo=True, coastline=True,
                                     project=True, projection=ccrs.PlateCarree(0), crs=ccrs.PlateCarree(), 
                                     cmap='RdBu_r', symmetric=True,frame_width=200, colorbar=(True if var_name==num_var-1 else False),
                                     title=ds_attr_per_season.var_name.isel(var_name=var_name).values+' anomalies')

    def plot_ones_season_method(season, method):
        plot_anomaly = make_plot_anomaly(0,season, num_var=len(input_variable))
        plot_attrs = make_plot_attrs(0,season,method,num_var=len(input_variable))
        for k in range(1, ds_attr_per_season.data.var_name.size):
            plot_anomaly = plot_anomaly + make_plot_anomaly(k,season, num_var=len(input_variable))
            plot_attrs =  plot_attrs +make_plot_attrs(k,season, method,num_var=len(input_variable))
        all_plots = (plot_anomaly+plot_attrs).cols(len(input_variable))  
        return all_plots 

    interactive_plot = pn.bind(plot_ones_season_method, season, attr_method)
    
    pn_layout = pn.Column(
                    pn.WidgetBox(season, attr_method, horizontal=True),  
                    interactive_plot
                      ).servable()
    pn_layout.save(html_file,embed=True)
    wandb.log({"Attribution/AttributingClass9": wandb.Html(html_file)})

def log_attribution_regression(ds_attr, wandb_logger, input_variable):
    ds_attr_per_season = ds_attr.groupby('time.season').mean()
    areas_1pct = np.abs(ds_attr.attributions.stack(space=['latitude','longitude','var_name'])).rank('space', pct=True).unstack()
    important_areas = areas_1pct.where(areas_1pct>0.99).count('time')/ds_attr.time.size*100
    important_areas_per_season = areas_1pct.where(areas_1pct>0.99).groupby('time.season').count('time')/ds_attr.time.groupby('time.season').count()*100
    important_areas_per_season = important_areas_per_season.where(important_areas_per_season>0)
    quantiles_attr = ds_attr.attributions.quantile([0.05,0.5,0.95], dim='time')
    ranks = (ds_attr.attributions.isel(time=0, attr_method=0).count()+1 - np.abs(ds_attr.attributions.stack(space = ['latitude','longitude', 'var_name'])).rank('space').unstack())
    all_ranks = xr.DataArray(np.arange(1,ranks.max()+1), dims=['rank'], coords=dict(rank=np.arange(1,ranks.max()+1)))
    predic_first_n_points = ds_attr.attributions.where(ranks<=all_ranks).sum(['longitude','latitude','var_name'])
    regr = xr_linregress(ds_attr.attributions.where(ranks<all_ranks).sum(['longitude','latitude','var_name']).isel(attr_method=0), ds_attr.pred)

    def make_plot_attrs(var_name,season, attr_method, num_var=1):
        return ds_attr_per_season.attributions.isel(var_name=var_name).sel(season=season, attr_method=attr_method)\
                    .hvplot.quadmesh(x='longitude', y='latitude', geo=True, coastline=True,
                                        project=True, projection=ccrs.PlateCarree(0), crs=ccrs.PlateCarree(),
                                        cmap='PuOr_r', symmetric=True,frame_width=200, colorbar=(True if var_name==num_var-1 else False), 
                                        title=ds_attr_per_season.var_name.isel(var_name=var_name).values+' attributions')

    def make_plot_anomaly(var_name,season,num_var=1):
        return ds_attr_per_season.data.isel(var_name=var_name).sel(season=season)\
                    .hvplot.quadmesh(x='longitude', y='latitude', geo=True, coastline=True,
                                     project=True, projection=ccrs.PlateCarree(0), crs=ccrs.PlateCarree(), 
                                     cmap='RdBu_r', symmetric=True,frame_width=200, colorbar=(True if var_name==num_var-1 else False),
                                     title=ds_attr_per_season.var_name.isel(var_name=var_name).values+' anomalies')
    
    def make_plot_importance_map(var_name,season,attr_method,num_var=1):
        return important_areas_per_season.isel(var_name=var_name).sel(season=season, attr_method=attr_method)\
                    .hvplot.quadmesh(x='longitude', y='latitude', geo=True, coastline=True,
                                     project=True, projection=ccrs.PlateCarree(0), crs=ccrs.PlateCarree(), 
                                     cmap='magma_r', symmetric=False,frame_width=200, colorbar=(True if var_name==num_var-1 else False),
                                     clim=(0,100),
                                     title=ds_attr_per_season.var_name.isel(var_name=var_name).values+r' % of time in top 1% features')
    
    def plot_ones_season_method(season, method):
        plot_anomaly = make_plot_anomaly(0,season, num_var=len(input_variable))
        plot_attrs = make_plot_attrs(0,season,method,num_var=len(input_variable))
        plot_importance = make_plot_importance_map(0,season,method,num_var=len(input_variable) )
        for k in range(1, ds_attr_per_season.data.var_name.size):
            plot_anomaly = plot_anomaly + make_plot_anomaly(k,season, num_var=len(input_variable))
            plot_attrs =  plot_attrs +make_plot_attrs(k,season, method,num_var=len(input_variable))
            plot_importance =  plot_importance +make_plot_importance_map(k,season, method,num_var=len(input_variable))

        all_plots = (plot_anomaly+plot_attrs+plot_importance).cols(len(input_variable))  
        return all_plots 
    
    season = pn.widgets.RadioButtonGroup(name="Season",description='Season',options=['DJF','MAM','JJA','SON'])
    attr_method = pn.widgets.Select(description="Attribution method",options=[ str(k) for k in ds_attr_per_season.attr_method.values])
    interactive_plot = pn.bind(plot_ones_season_method, season, attr_method)
    html_file = f"/Data/gfi/users/rogui7909/wanbd_logs/{wandb_logger.experiment.id}_attributions.html"
    # netcdf_file = f"/Data/gfi/users/rogui7909/wanbd_logs/{wandb_logger.experiment.id}_attributions.netcdf"
    # ds_attr.to_netcdf(netcdf_file)
    pn_layout = pn.Column(
                    pn.WidgetBox(season,attr_method, horizontal=True),  
                    interactive_plot
                      ).servable()
    pn_layout.save(html_file,embed=True)
    wandb.log({f"Attribution/AttributingOver20mm": wandb.Html(html_file)})