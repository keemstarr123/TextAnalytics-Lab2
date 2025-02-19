import warnings
warnings.filterwarnings('ignore')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rio
from matplotlib.cm import RdYlGn,jet,RdBu
import pystac_client
import planetary_computer 
from odc.stac import stac_load
import pandas as pd
from datetime import datetime
import time
from geopy.distance import distance
from geopy.point import Point
import dask.array as da
import concurrent.futures



def load_data(bounds):
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search1 = stac.search(
    bbox = bounds,
    datetime = "2021-06-01T15:00:00Z/2021-12-31T16:00:00Z",
    collections = ['sentinel-2-l2a'],
    query={"eo:cloud_cover":{"lt":10}}
    )
    
    search2= stac.search(
    bbox = bounds,
    datetime = "2021-06-01T15:00:00Z/2021-12-31T16:00:00Z",
    collections = ['landsat-c2-l2'],
    query={"eo:cloud_cover":{"lt":10},"platform": {"in": ["landsat-8"]}}
    )

    resolution = 10 
    scale = resolution / 111320.0 

    data1, data2= search1_func(search1, scale , bounds), search2_func(search2, scale , bounds)

    return data1, data2

def search1_func(search, scale , bounds):
    items = list(search.get_items())
    datas = stac_load(
        items,
        bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
        crs="EPSG:4326", # Latitude-Longitude
        resolution=scale, # Degrees
        chunks={"x": 512, "y": 512},
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox= bounds
    )
    return datas.chunk({'time': -1, 'latitude': 512, 'longitude': 512}).persist()

def search2_func(search, scale, bounds):
    items = list(search.get_items())
    # Scale Factors for the Surface Temperature band
    
    print(list(items[0].assets.keys()))  # ✅ Correct (accessing as an object)
    datas = stac_load(
        items,
        bands=["lwir11"],
        crs="EPSG:4326", # Latitude-Longitude
        resolution=scale, # Degrees
        chunks={"x": 512, "y": 512},
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox= bounds
    )
    scale2 = 0.00341802 
    offset2 = 149.0 
    kelvin_celsius = 273.15 
    datas = datas.astype(float) * scale2 + offset2 - kelvin_celsius

    return datas.chunk({'time': -1, 'latitude': 512, 'longitude': 512}).persist() 

def calculate_median(datas):
    median = datas.median(dim="time").compute()
    return median

def compute_ndvi(median):
    return (median.B08-median.B04)/(median.B08+median.B04)

def compute_ndbi(median):
    return (median.B11-median.B08)/(median.B11+median.B08)

def compute_ndwi(median):
    return (median.B03-median.B08)/(median.B03+median.B08)

def landsat_plot(info):
    fig, ax = plt.subplots(figsize=(6,6))
    print("Variables in info before plotting:", list(info.data_vars))
    info['lwir11'].plot.imshow(vmin=20.0, vmax=45.0, cmap="jet",add_colorbar = False)
    plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.title("")
    plt.savefig("lst_0.png", dpi=300, bbox_inches="tight", pad_inches=0)

def plot_graph(info,vmin, vmax, color, type, index):
    fig, ax = plt.subplots(figsize=(6,6))
    info.plot.imshow(vmin=vmin, vmax=vmax, cmap = color, add_colorbar = False)
    plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.title("")
    plt.savefig(f"{type}_{index}.png", dpi=300, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    start = time.time()
    data = pd.read_csv("ProcessedTrainingData.csv")
    x = data.iloc[:,1:5]
    y = data[['UHI Index']]
    
    bounds = tuple(map(float, x['Bounds'][0][1:-1].split(",")))
    
    loaded_data, surface_data = load_data(bounds)
    

    with concurrent.futures.ProcessPoolExecutor() as Executor:
        future_median = Executor.submit(calculate_median,loaded_data)
        future_lst = Executor.submit(calculate_median,surface_data)
        median = future_median.result()
        lst = future_lst.result()

    print("Calculated mean successfully.")
    ndvi = compute_ndvi(median)
    plot_graph(ndvi, 0.0, 1.0 , "RdYlGn", "ndvi", 0)
    ndbi = compute_ndbi(median)
    plot_graph(ndbi, -0.1, 0.1 , "jet", "ndbi", 0)
    ndwi = compute_ndwi(median)
    plot_graph(ndwi, -0.3, 0.3 ,"RdBu", "ndwi", 0)
    landsat_plot(lst)

    end = time.time()

    print (f"{end-start:.4f}")
    