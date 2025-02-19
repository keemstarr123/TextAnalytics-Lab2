# Supress Warnings 
import warnings
warnings.filterwarnings('ignore')

# Import common GIS tools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio
from matplotlib.cm import RdYlGn,jet,RdBu

# Import Planetary Computer tools
import stackstac
import pystac_client
import planetary_computer 
from odc.stac import stac_load
import pandas as pd
from datetime import datetime
import time

start = time.time()

data = pd.read_csv(r"C:\Users\haoho\Downloads\Training_data_uhi_index 2025-02-04.csv")
data['datetime'] = data['datetime'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y %H:%M"))

from geopy.distance import distance
from geopy.point import Point

radius = 5  # Radius in kilometers
bounds = []

for i in data.values:  # Assuming data.values contains [(lat, lon), (lat, lon), ...]
    center_point = Point(i[0], i[1])  # Create a geopy Point (latitude, longitude)

    # Compute bounding box (NW and SE corners)
    nw = distance(kilometers=radius).destination(center_point, 45)  # Northwest
    se = distance(kilometers=radius).destination(center_point, 225)  # Southeast

    # Store bounds in (min_lat, min_lon, max_lat, max_lon) format
    bounds.append((se.latitude, se.longitude, nw.latitude, nw.longitude))

data.insert(3, "Bounds", bounds)
x = data.iloc[:,0:4]
y = data[['UHI Index']]
x_in_array = x.to_numpy()
stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

search = stac.search(
    bbox = x['Bounds'][0],
    datetime = "2021-06-01T15:00:00Z/2021-12-31T16:00:00Z",
    collections = ['sentinel-2-l2a'],
    query={"eo:cloud_cover":{"lt":10}}
)

items = list(search.get_items())
signed_items = [planetary_computer.sign(item).to_dict() for item in items]
resolution = 10 
scale = resolution / 111320.0 

datas = stac_load(
    items,
    bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    crs="EPSG:4326", # Latitude-Longitude
    resolution=scale, # Degrees
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox= x['Bounds'][0]
)

# Plot sample images from the time series
plot_data = datas[["B04","B03","B02"]].to_array()

median = datas.median(dim="time").compute()

ndvi_median = (median.B08-median.B04)/(median.B08+median.B04)

fig, ax = plt.subplots(figsize=(7,6))
ndvi_median.plot.imshow(vmin=0.0, vmax=1.0, cmap="RdYlGn", add_colorbar = False)
plt.title("Median NDVI")
plt.axis('off')
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)
plt.title("")
plt.savefig("median_ndvi_square.png", dpi=300, bbox_inches="tight", pad_inches=0)

plt.show()

end = time.time()

print (f"{start-end:.4f}")