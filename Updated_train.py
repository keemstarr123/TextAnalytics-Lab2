#Import all of the third-party library with existing feature that can be utilised in fetching data.
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
import rasterio


#Function to load data (satelite images) from Microsoft's planetary computer 
def load_data(bounds):
    #Establish the API (Communication link) to perform fetching action later 
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    #Perform searching on sentinel 2 satelite 
    search1 = stac.search(
    bbox = bounds, #Desired area
    datetime = "2021-06-01T15:00:00Z/2021-12-31T16:00:00Z", #Date and time range 
    collections = ['sentinel-2-l2a'],
    query={"eo:cloud_cover":{"lt":10}} #Condition to filter images with less than 10% of cloud coverage
    )
    
    #Perform searching on landsat c2 satelite 
    search2= stac.search(
    bbox = bounds,
    datetime = "2021-06-01T15:00:00Z/2021-12-31T16:00:00Z",
    collections = ['landsat-c2-l2'],
    query={"eo:cloud_cover":{"lt":10},"platform": {"in": ["landsat-8"]}} #Condition to filter images with less than 10% of cloud coverage
    )

    resolution = 30 #Meter that represents each pixel in the images
    scale = resolution / 111320.0 #Convert into degree (Format to be used for loading the image later)

    data1, data2= search1_func(search1, scale , bounds), search2_func(search2, scale , bounds)

    return data1, data2

#Load the search result (image) on the sentinel-l2 
def search1_func(search, scale , bounds):
    items = list(search.get_items())
    datas = stac_load(
        items,
        bands=["B03", "B04", "B08", "B11"], #Bands represent different color of images. B03 = green, B04 = red, B08 = Near Infrared, B11 = Shortwave Infrared
        crs="EPSG:4326", # Latitude-Longitude
        resolution=scale, # Degrees
        chunks={"x": 512, "y": 512}, #chunks of each images
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox= bounds
    )
    return datas.persist() #load quicker, we use persist

def search2_func(search, scale, bounds):
    items = list(search.get_items())
    # Scale Factors for the Surface Temperature band
    
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

    return datas.persist() 


#Since we have multiple result in each search, we combine them by calculating the mean, which will be used later for generating the images.
def calculate_median(datas):
    median = datas.median(dim="time").compute()
    return median

#Calculating Vegetation degree
def compute_ndvi(median):
    return (median.B08-median.B04)/(median.B08+median.B04)

#Calculating building density
def compute_ndbi(median):
    return (median.B11-median.B08)/(median.B11+median.B08)

#Calculating water surface area
def compute_ndwi(median):
    return (median.B03-median.B08)/(median.B03+median.B08)

#Ploting graph for land surface area
def landsat_plot(info, index):
    fig, ax = plt.subplots(figsize=(6,6))
    info['lwir11'].plot.imshow(vmin=20.0, vmax=45.0, cmap="jet",add_colorbar = False)
    plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.title("")
    plt.savefig(f"lst_{index}.png", dpi=300, bbox_inches="tight", pad_inches=0)

#Plotting graph for NDVI, NDBI, NDWI
def plot_graph(info,vmin, vmax, color, type, index):
    fig, ax = plt.subplots(figsize=(6,6))
    info.plot.imshow(vmin=vmin, vmax=vmax, cmap = color, add_colorbar = False)
    plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.title("")
    plt.savefig(f"{type}_{index}.png", dpi=300, bbox_inches="tight", pad_inches=0)


#Main flow of the system
if __name__ == "__main__":
    data = pd.read_csv("ProcessedTrainingData.csv") #Read the excel file that contains the boundaries of each coordinate and their relative UHI value
    x = data.iloc[:,0:4] #x is the variable that affects the UHI, in this case we have longitude, latitude, boundaries, and time
    y = data[['UHI Index']] #y is the target or the value we want to predict
    counter = 10000 #Used to print the row number
    
    for row in x.values[9999]: #convert the x into a list for better retrieval
        print(f"Row {counter}:\n")
        start = time.time() 
        bounds = tuple(map(float, row[-1][1:-1].split(","))) #To formulate the boundary of each row, from string values to list of two coordinate that can be processed by system
        
        loaded_data, surface_data = load_data(bounds) #perform searching and load satelite data into a variable
        

        with concurrent.futures.ProcessPoolExecutor() as Executor: #To increase the speed of execution, we use multiple CPU by specifying this code
            future_median = Executor.submit(calculate_median,loaded_data) #calculate median for NDBI, NDVI, and NDWI
            future_lst = Executor.submit(calculate_median,surface_data) #calculate median for LST
            median = future_median.result() #store them into a proper variable 
            lst = future_lst.result() #store them into a proper variable 

        print("Calculated mean successfully.")
        ndvi = compute_ndvi(median) #compute ndvi used to generate image later
        plot_graph(ndvi, 0.0, 1.0 , "RdYlGn", "ndvi", counter)
        ndbi = compute_ndbi(median) #compute ndbi used to generate image later
        plot_graph(ndbi, -0.1, 0.1 , "jet", "ndbi", counter)
        ndwi = compute_ndwi(median) #compute ndwi used to generate image later
        plot_graph(ndwi, -0.3, 0.3 ,"RdBu", "ndwi", counter) #plot all three ndvi, ndbi and ndwi
        landsat_plot(lst, counter) #plot lst 

        end = time.time()
        counter+=1
        print (f"{end-start:.4f}")
    