import xarray as xr 
from captum.attr import IntegratedGradients
from .preprocessings import add_timesteps, filter_by_season, preprocess_rain, get_loader_from_ds, get_expanded_ds
import numpy as np

class MyDataLoader:
    def __init__(self,config, load_atmos=True):
        self.config = config
        self.load_rain_data()
        self.load_atmospheric_features(load=load_atmos)
        self.harmonize_time()
        self.make_train_val_test_split_datasets()
        self.make_data_loaders()


    def load_rain_data(self):
        ds_rain = xr.open_dataset(self.config['file_name_data_out']).tp
        ds_rain = add_timesteps(ds_rain, num_timesteps_predicted=self.config['num_timesteps_predicted'])
        ds_rain = filter_by_season(ds_rain,season=self.config['season'])
        self.rain = ds_rain
        self.targets = preprocess_rain(ds_rain, 
                                  type_predictions=self.config['type_prediction'], 
                                  quantile_extreme=self.config['quantile_extreme'],
                                  quantile_extreme_based_on_rainy_days=self.config['quantile_extreme_based_on_rainy_days'])
        match self.config['type_prediction']:
            case 'boolean':
                self.config['prediction_per_timestep'] = 1
            case 'three_classes':
                self.config['prediction_per_timestep'] = 3
            case 'regression':
                self.config['prediction_per_timestep'] = 1
            case 'quantiles':
                self.config['prediction_per_timestep'] = 10
            case _:
                raise ValueError("type_predictions must be 'boolean', 'three_classes', 'quantiles' or 'regression'")
        self.config['num_classes'] = self.config['prediction_per_timestep']*self.config['num_timesteps_predicted']

    def load_atmospheric_features(self,load=True):
        input_variables = self.config['inputs'].split(' ')
        self.config['input_variables'] = input_variables 
        ds_atm = xr.open_zarr(self.config['file_name_data_in']).data_normed
        ds_atm = ds_atm.sel(var_name = input_variables)
        lon_min, lon_max, lat_min, lat_max = self.config['spatial_extent']
        if ds_atm.latitude.diff('latitude')[0]<0:
            lat_min, lat_max = lat_max, lat_min
        ds_atm = ds_atm.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
        
        if load:
            self.features = ds_atm.astype('float32').load()
        else:
            self.features = ds_atm.astype('float32')
        
        self.feature_height = self.features.latitude.size
        self.feature_width = self.features.longitude.size
        self.feature_image_size = self.feature_height*self.feature_width
        self.config['num_channels'] = len(self.config['input_variables'])
        self.config['feature_height'] = self.feature_height
        self.config['feature_width'] = self.feature_width
        self.config['feature_image_size'] = self.feature_image_size
        
    def harmonize_time(self):
        common_time = [time for time in self.features.time.values if time in self.rain.time.values]
        if len(common_time) == 0:
            raise ValueError('No common time between inputs and outputs')
        self.features = self.features.sel(time=common_time)
        self.rain = self.rain.sel(time=common_time)
        self.targets = self.targets.sel(time=common_time)
        self.n_samples = self.targets.time.size

    def make_train_val_test_split_datasets(self, ratio=[.7,.15], shuffle_train=True):
        all_indices = np.arange(self.rain.time.size)
        total_size = all_indices.size
        indices_train = all_indices[:int(total_size*ratio[0])]
        indices_val = all_indices[int(total_size*ratio[0]):int(total_size*(ratio[0]+ratio[1]))]
        indices_test = all_indices[int(total_size*(ratio[0]+ratio[1])):] 
        if shuffle_train:
            np.random.shuffle(indices_train)
        self.indices_train = indices_train 
        self.indices_val = indices_val
        self.indices_test = indices_test
        self.ds_train = xr.Dataset(dict(features=self.features.isel(time=self.indices_train),
                               targets=self.targets.isel(time=self.indices_train),
                               rain=self.rain.isel(time=self.indices_train),
                               ))
        if self.config['augment_training_with_noise']:
            self.ds_train = get_expanded_ds(self.ds_train, 
                                            noisy_samples=self.config['num_noisy_samples'], 
                                            noise_scale=self.config['augment_noise_amplitude'], 
                                            shuffle_after_noise=shuffle_train)
        self.ds_val = xr.Dataset(dict(features=self.features.isel(time=self.indices_val),
                               targets=self.targets.isel(time=self.indices_val),
                               rain=self.rain.isel(time=self.indices_val),
                               ))
        self.ds_test = xr.Dataset(dict(features=self.features.isel(time=self.indices_test),
                               targets=self.targets.isel(time=self.indices_test),
                               rain=self.rain.isel(time=self.indices_test),
                               ))
            
    def make_data_loaders(self):
        self.train_loader = get_loader_from_ds(self.ds_train, batch_size=self.config['batch_size'])
        self.val_loader = get_loader_from_ds(self.ds_val, batch_size=self.config['batch_size'])
        self.test_loader = get_loader_from_ds(self.ds_test, batch_size=self.config['batch_size'])
    
    
    def print_infos(self):
        print(f'Data ready:')
        print(f"    Image size: {self.feature_image_size} ({self.feature_width}x{self.feature_height})")
        print(f"    Input variables: " + ', '.join(self.config['input_variables']))
        print(f"    {self.ds_train.time.size} samples in train")
        print(f"    {self.ds_val.time.size} samples in validation")
        print(f"    {self.ds_test.time.size} samples in test")
        print(f"    {self.config['num_timesteps_predicted']} Predicted timesteps for future rainfall")
        print(f"    Prediction type: {self.config['type_prediction']}")
        print(f"    What quantile is considered extreme: {self.config['quantile_extreme']*100:.0f}th of {'all' if not self.config['quantile_extreme_based_on_rainy_days']  else 'rainy'} days")

    def attribute_integrated_gradients(self, model, ds_features, predictions, targets, samples_to_attribute='TP'):
        ds_predictions = xr.DataArray(predictions, dims=['time','timestep'], coords = dict(time=ds_features.time[:predictions.shape[0]], timestep=range(1,5)))
        ds_targets = xr.DataArray(targets, dims=['time','timestep'], coords = dict(time=ds_features.time[:targets.shape[0]], timestep=range(1,5)))
        ds_results = xr.Dataset(dict(predictions=ds_predictions, targets=ds_targets))                             
        self.ds_results = ds_results

        series_targets = ds_targets.to_series()
        series_predictions = ds_predictions.to_series()

        all_preds = ((series_predictions == series_targets)&(series_targets==2)).reset_index()
        all_preds.columns = ['time_of_prediction','timestep','TP_extreme']
        all_preds['time_of_event'] = all_preds.time_of_prediction + all_preds.timestep*pd.Timedelta(1,'D')
        extreme_events_predictions = all_preds.groupby('time_of_event').TP_extreme.sum()

        all_targets = ((series_targets==2)).reset_index()
        all_targets.columns = ['time_of_prediction','timestep','extreme']
        all_targets['time_of_event'] = all_targets.time_of_prediction + all_targets.timestep*pd.Timedelta(1,'D')
        extreme_events = all_targets.groupby('time_of_event').extreme.sum()

        extreme_events_predictions = extreme_events_predictions.loc[extreme_events[extreme_events==4].index]
        
        match samples_to_attribute:
            case 'TP':
                time_TP_extreme = extreme_events_predictions.loc[extreme_events_predictions==4].index
                self.time_of_TP_extremes = time_TP_extreme
            case 'all_extr':
                pass 
            case 'all':
                time_TP_extreme = extreme_events_predictions.index
        # else:
        #     time_TP_extreme = extreme_events_predictions.loc[extreme_events_predictions==4].index
        #     self.time_of_TP_extremes = time_TP_extreme
        steps = 4
        start_times = time_TP_extreme -  pd.Timedelta(steps-1,'D')
        self.first_time_prediction_of_TP_extremes = start_times

        method = IntegratedGradients(model.model)
        all_multi_attrs = []
        all_multi_sens = []
        print('Attributing true positive extremes ...')
        for k in tqdm(range(len(time_TP_extreme))):
            start = start_times[k].strftime("%Y-%m-%d %H:%M:%S")
            end = time_TP_extreme[k].strftime("%Y-%m-%d %H:%M:%S")
            # print(start,end)
            ds_test_extract = ds_features.sel(time=slice(start ,end))
            tensor_in = torch.Tensor(ds_test_extract.features.values)
            tensor_out = torch.Tensor(ds_test_extract.targets.values)
            attrs = method.attribute(tensor_in, baselines=torch.Tensor([0]),target=[3*(steps-k)-1 for k in range(steps)]) 
            da_attrs = xr.DataArray(attrs, dims = ['timestep','var_name','latitude','longitude'], 
                                    coords= dict(timestep=np.arange(1,steps+1),var_name=ds_test_extract.var_name, latitude=ds_test_extract.latitude, longitude=ds_test_extract.longitude))
            all_multi_attrs.append(da_attrs.assign_coords(time=time_TP_extreme[k]))
            sens = da_attrs/ds_test_extract.features.rename(time='timestep').assign_coords(timestep=da_attrs.timestep)
            all_multi_sens.append(sens.assign_coords(time=time_TP_extreme[k]))
        ds_attributions = xr.concat(all_multi_attrs, dim='time')
        ds_sens = xr.concat(all_multi_sens, dim='time')
        final_ds = xr.Dataset(dict(attributions = ds_attributions, sensitivity=ds_sens))
        self.ds_attribution = final_ds


