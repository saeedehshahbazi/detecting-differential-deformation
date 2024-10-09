'''
File:      Splitor.py
Author:     saeedeh shahbazi 
Date:       March 2023
Summary of File: number 2
    This file contains code which read EGMS basic products, then change thier CRS to 32630 and finally split PS points into inside / outside of buildings. 
    we used DSM and buidings height.
    Both groups are saving as CSV files.

'''

import shapely.wkt
from base64 import encode
from PIL import Image
from os import sep
import pandas as pd
import numpy as np
import copy
import time
import fiona
# import rasterio
import sys
import geopandas as gpd
from shapely import wkt
from datetime import date
from fiona.crs import from_epsg
import csv
import os

print('spliting is starting ....')
t0 = time.time()
# arginput = sys.argv[1]
# PS_Point = str(arginput)
# # name = PS_Point
dir_name = 'F:\\secondPart\\zone1\\NEW_Inside_osm_inspire\\'
Building = 'F:\\secondPart\\Buildings_OSM_Zone1\\Documents\\Zone1_Spain_DTM_osm_inspire3035.shp'


print("Reading building data from {} process started ....".format(Building))
df2 = gpd.read_file(Building) 
df3 = copy.copy(df2)
geodf2= gpd.GeoDataFrame(df3 , geometry="geometry")
print("Reading building process finished succesfully.")
geodf2 = geodf2.set_geometry('geometry')  
# geodf4326 = geodf2.set_crs("EPSG:4326")   
# geodf3035 = geodf4326.to_crs(epsg= 3035) 
# set CRS to 3035 if it is None
if geodf2.crs is None:
    geodf3035 = geodf2.set_crs("EPSG:3035")
    # geodf3035= geodf3035.to_crs(epsg= 3035)
else:
    # if CRS is not 3035, change it to 3035
    if geodf2.crs.to_epsg() != 3035:
        geodf3035 = geodf2.to_crs(epsg= 3035)
    else:
        geodf3035 = geodf2
                     
geodf32630 = geodf3035.to_crs(epsg= 32630)
df4 = copy.copy(geodf32630)
myPolygons= gpd.GeoDataFrame(df4 , geometry="geometry") 

poly3035= gpd.GeoDataFrame(geodf3035 , geometry="geometry") 
myPolygons['PGnoBf3035'] = poly3035.geometry 

# myPolygons['geometry'] = myPolygons.geometry.buffer(-5.6)                   #Creat Inner Buffer around Building with size of 5.6 m
myPolygons['geometry'] = myPolygons.geometry.buffer(5.6)  



print('reading EGMS points......')
for item in os.listdir(dir_name): # loop through items in dir
    if item.endswith('_name.txt'):
        data_name = item 

with open(data_name, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        PS_Point = row[0]
        # PS_Point = "..\\EGMS_L2a_103_0223_IW1_VV.csv"
        name = PS_Point[21:-4]
        print("Reading PS data from {} process started ....".format(name))
        if (".shp" in PS_Point):
            df = gpd.read_file(PS_Point)
            data1 = copy.copy(df)
        elif (".csv" in PS_Point):                                                    
            df = pd.read_csv(PS_Point , sep=',', usecols=["easting", "northing","height","mean_velocity"] )
            # df['geometry'] = df['geometry'].apply(wkt.loads)
            df['geometry']=gpd.points_from_xy(df["easting"], df["northing"])
            geodf= gpd.GeoDataFrame(df , geometry="geometry")
            geodataf = geodf.set_geometry('geometry')  
            data1 = copy.copy(geodataf)
        elif (".txt" in PS_Point):
            df = pd.read_csv(PS_Point , sep='\t')
            df['geometry']=gpd.points_from_xy(df["easting"], df["northing"])
            geodf= gpd.GeoDataFrame(df , geometry="geometry")
            geodataf = geodf.set_geometry('geometry') 
            data1 = copy.copy(geodataf)
        else:
            (print(" PS data is not readable..."))                                                                          # Reading PS data
        print("Reading PS data process finished succesfully.")
        data2 = data1.set_crs(epsg = 3035)
        data32630 = data2.to_crs(epsg= 32630)
        data3 = copy.copy(data32630)
        # data2['geometry'] = data2['geometry'].apply(wkt.loads)
        testPoints = gpd.GeoDataFrame(data3 , geometry="geometry" ) 
        testPoints['E'] = testPoints['geometry'].apply(lambda p:p.x)
        testPoints['N'] = testPoints['geometry'].apply(lambda p:p.y)

        # myPolygons.to_file('Buffer' +str(PS_Point[0:-4]) +'.shp')                 # saving Buffer polygons as shapefile
        print(" Seprating PS points process started ....")
        t2 = time.time()
        # ############# Spatial join #################
        print(testPoints.crs)
        print(myPolygons.crs)
        # spatial join between PS points and polygons without adding polygon attributes to PS points


        pointInPolys = gpd.tools.sjoin(testPoints, myPolygons, op="within" , how ='left' )
        print("Spatial join finished succesfully.")
        inside=pointInPolys[~np.isnan(pointInPolys.index_right)]                               # inside points

        includedPolygons = pd.DataFrame(columns=["index_right", "PGnoBf3035"])
        #concatenate inside["PGnoBf3035"] to includedPolygons
        includedPolygons = pd.concat([includedPolygons, inside[["index_right","PGnoBf3035"]]], axis=0)
        #drop duplicates
        includedPolygons = includedPolygons.drop_duplicates(subset=["index_right","PGnoBf3035"])
        inside = inside.drop(columns=["PGnoBf3035"])

        inside['height'] =inside['height'].astype(float)
        inside['DTM_max'] = inside['DTM_max'].astype(float)
        inside['DTM_min'] = inside['DTM_min'].astype(float)
        inside['DTM_Ave'] = inside['DTM_mean'].astype(float) 

        height = copy.deepcopy(inside[(inside['DTM_min'] +4 <= inside['height'])])
        # Add the 'geometry' column to the DataFrame as the last column
        geometry_column = height.pop('geometry')
        height['geometry'] = geometry_column
        #save the dataframe as a shapefile
        print("Saving inside points process started ....")
        insideDataFile=  open('_inside_SpainC_'+ str(name) +'_new.csv', "w" ,newline='', encoding="utf-8")
        height.to_csv(insideDataFile,  sep=',' )
        insideDataFile.close()
        print("Saving inside points process finished succesfully.")

        print("Saving included polygons process started ....")
        # save includedPolygons as a csv file
        includedPolygonsCsvFile = open('includedPolygons_'+ str(name) +'_new.csv', "w" ,newline='', encoding="utf-8")
        includedPolygons.to_csv(includedPolygonsCsvFile,  sep=',' )
        includedPolygonsCsvFile.close()
        print("Saving included polygons process finished succesfully.")
        # outside=pointInPolys[np.isnan(pointInPolys.index_right ,where=True)]                        # outside points
        # outsideDataFile=  open('outside'+ str(name[0:-4]) +'.csv', "w" ,newline='', encoding="utf-8")
        # outside.to_csv(outsideDataFile,  sep=',' )
        # outsideDataFile.close()

        t3 = time.time()
        timimng2 = t3-t2
        print("Timing step2: " + str(timimng2/60) +" (min)")
        print(" seperated PS point finished succesfully.")
