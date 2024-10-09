import fiona
import sys
import os
import shapely.wkt
from shapely.geometry import Point
from shapely.ops import transform
import geopandas as gpd
import pandas as pd
import tkinter as tk
from tkinter import messagebox, LabelFrame, filedialog 
import copy
from shapely import wkt
import time
import numpy as np
from scipy.spatial import KDTree
import traceback
import math
import richdem as rd
import multiprocessing as mp
import subprocess
import pyproj
import datetime




def install(requirementsPath):
    subprocess.check_call([sys.executable, "-m", "pip", "install","-r", requirementsPath])


def getFileWindow(title):
   scriptDirectory = os.path.dirname(os.path.abspath(sys.argv[0]))
   filepath = ""
   file = filedialog.askopenfile(title=title, mode='r', filetypes=[('All files', '*.shp *.csv'), ('Shapefile', '*.shp'), ('csv file', '*.csv')],
                                initialdir=scriptDirectory)
   if file:       
      filepath = os.path.abspath(file.name)

   return filepath

def getEntryVAlue( myCrs,e1,window):
   myCrs.append(e1.get())
   window.destroy()

def getEntryParams( params,e1,e2,window):
   params.append(e1.get())
   params.append(e2.get())
   window.destroy()
# input dialogbox to get single value
def showInputBox( title, sentenceLabel ,inputLabel):
   myCrs = []
   window = tk.Tk()
   window.configure(width=600, height=400)
   window.geometry("+40+40")
   window.title(title)
   frame = LabelFrame(window, text=title, padx=20, pady=20, font= ('Helvetica 14'))
   frame.pack(pady=20, padx=10)
   tk.Label(frame, 
         text=sentenceLabel).grid(row=0)
   tk.Label(frame, 
         text=inputLabel).grid(row=1)
   e1 = tk.Entry(frame)

   e1.grid(row=1, column=1)

   tk.Button(frame, text='Cancel', command=window.quit).grid(row=4, column=0, sticky=tk.W, pady=4, padx=4)
   tk.Button(frame, text='Ok', command=lambda:getEntryVAlue(myCrs,e1, window)).grid(row=4, column=1, sticky=tk.W, pady=4,padx=4)

   tk.mainloop()
   if myCrs!=[]:
      return myCrs[0]
   else:
      return None

def returnSelectedValue(checkboxes, selectedValues,window):
   for var, value in checkboxes.items():
      if value.get():
         selectedValues.append(var)
   window.destroy()

def showCheckBox(values, title):
   window = tk.Tk()
   # Set the window width and height using the configure method
   window.configure(width=800, height=500)
   window.geometry("+40+40")
   window.title(title)
   # Dictionary to store the checkbox variables
   checkboxes = {}
   # Create a LabelFrame
   frame = LabelFrame(window, text=title, padx=20, pady=20, font= ('Helvetica 14'))
   frame.pack(pady=20, padx=10)
   # Create checkboxes for each value
   col = 0
   row = 0
   for value in values:
      if (value.find("20")==-1):
         var = None
         if value in ["E", "N","index_right","mean_velocity","geometry","osm_id",'height' , 'DTM_min','PGnoBf3035']:
               var = tk.IntVar(value=1)
         else:
               var = tk.IntVar()
         checkboxes[value] = var
         checkbox = tk.Checkbutton(frame, text=value ,variable=var,onvalue=1, 
                                       offvalue=0,anchor="w",justify="left",padx=10, font= ('Helvetica 12'))
         checkbox.grid(row=row,column=col,pady=5, padx=(2,2),sticky='w')
         col += 1
         if (col > 4):
               row += 1
               col = 0

   # Button to display selected values
   selectedValues = []
   button = tk.Button(window, text="Ok",command=lambda:returnSelectedValue(checkboxes, selectedValues,window), height = 1, width = 8)
   button.pack(pady=(1,20))
   
   # Run the main event loop
   window.mainloop()
   return (selectedValues)

def readCsvLineByLine(filePath:str, sep=",", logfile=None) -> pd.DataFrame:
   print("Reading data line-By-line from file\n {}".format(filePath))
   tmpLine = ''
   cnt = 0
   with open(filePath, 'r', errors="ignore") as file:
      while True:
         try:
            line = file.readline()
            if not line:
                  break
            if '\n' not in line:
                  tmpLine += line
                  cnt += 1
            else:
                  tmpLine += line
                  
                  if cnt == 0:
                     tmpLine = "myID" + tmpLine
                     tmpLine = tmpLine.replace(',,', '')
                     #empty dataframe with column names tmpLine split by comma
                     df = pd.DataFrame(columns=tmpLine.split(','))
                     tmpLine = ''
                     cnt += 1
                  else:
                     try:
                        cnt += 1
                        tmpLine = tmpLine.replace(',,,,', '')
                        tmpLine = tmpLine.replace(', ', '+')
                        myRow = tmpLine.split(',')
                        for idx, item in enumerate(myRow):
                              if '+' in item:
                                 myRow[idx] = item.replace('+', ', ')
                        
                        # append to dataframe
                        df = df.append(pd.Series(myRow, index=df.columns), ignore_index=True)
                        tmpLine = ''
                     except Exception as ex:
                        print(traceback.format_exc(),flush=True)
                        logfile.write("[{}] error: {}\n".format(datetime.datetime.now(), traceback.format_exc()))
                        print("Line: {}".format(cnt))
                        print(len(tmpLine.split(',')))
                        print(tmpLine.split(','))
                        tmpLine = ''
                        sys.exit(2)
         except Exception as ex:
            print(traceback.format_exc())
            logfile.write("[{}] error: {}\n".format(datetime.datetime.now(), traceback.format_exc()))
            print("Line: {}".format(cnt))
            print(len(tmpLine.split(',')))
            print(tmpLine.split(','))
            tmpLine = ''
            sys.exit(2)
   return(df)

def checkFile(filePath:str) -> bool:
   tmpLine = ''
   cnt = 0
   with open(filePath, 'r') as file:
      line = file.readline()
      if ',,,' in line:
         print("file {} is corrupted.".format(filePath))
         return False
      else:
         return True

def readData(filePath:str, dataFormat="shp", sep=",", logfile=None, headerList=None) -> gpd.GeoDataFrame:
   if  (checkFile(filePath)):
      df = None
      gdf = None
      givenCrs = None
      try:
         if (dataFormat=="shp"):
            print("Reading data using Pandas from file\n {}".format(filePath))
            gdf = gpd.read_file(filePath)
            print("Reading data is completed successfully.")
         elif (dataFormat=="csv"):
            # Read just the header row
            header = pd.read_csv(filePath, nrows=1)
            if (headerList==None):
               headerList =  ["E", "N","index_right","mean_velocity","geometry","osm_id",'height' , 'DTM_min']
            # headerList = showCheckBox(header,"Select Columns")
            time.sleep(1)
            if (headerList==[]):
                  print("Process is finished with error")
                  print("You should select proper headers to continue")
                  sys.exit(2)
            else:
                  print("Reading data from file\n {}".format(filePath))

                  df = pd.read_csv(filePath,sep=sep,usecols=headerList)
                  print("Reading data is completed successfully.")
      except Exception as ex:
         print(traceback.format_exc())
         logfile.write("[{}] error: {}\n".format(datetime.datetime.now(), traceback.format_exc()))
         sys.exit(2)
   else:
      df = readCsvLineByLine(filePath, sep=sep, logfile=logfile)

   return(df)

   if (dataFormat=="csv"):
      headerLowerCase = list(map(lambda x: x.lower(), headerList))
      if "geometry" in headerLowerCase:
         gdf = df2gdf(df, fileName=filePath)
      else:
         print("Dataset has no geometry Column.\n Select Coordinates columns to build a Geometry.")
         coordinateHeaders = showCheckBox(headerList,"Select Coordinate Columns (e.g. E,N)")
   else:
      if "geometry" not in gdf.keys():
         print("Dataset has no geometry Column.\n Select Coordinates columns to build a Geometry.")

   if(gdf.crs == None):
      print("GeoDatafram has no CRS.")
      print("Setting CRS to GeoDatafram.")
      try:   
         # givenCrs = showInputBox( title="Set CRS",sentenceLabel="Please enter the dataset CRS.",inputLabel="CRS (only digits without epsg)")
         givenCrs = 3035
         gdf.set_crs("EPSG:{}".format(givenCrs))
         print("CRS is assigned to GeoDatafram successfully.")
         print("CRS: EPSG:{}".format(givenCrs))
      except Exception as ex:
         print(traceback.format_exc())
         sys.exit(2)
   
   # return(gdf)

def df2gdf(df, fileName=""):
   print("Converting Pandas dataframe to GeoPandas")
   errorLine=0
   funcName=""
   data = copy.deepcopy(df)
   try:
      # change the geometry column to be of type Point
      data['geometry'] = data['geometry'].apply(shapely.wkt.loads)
      gdf = gpd.GeoDataFrame(data , geometry="geometry" )
   except Exception as ex:
      print(data['geometry'])
      print(traceback.format_exc())
      sys.exit(2)
   print("Dataframe is converted successfully.")
   return gdf

def setGdfCrs(gdf,myCrs):
   return gdf.set_crs("EPSG:{}".format(myCrs))    

def convertCrs(gdf, epsg):
   if (isinstance(epsg, str)):
      if ("EPSG" not in epsg):
         epsg = "EPSG:"+epsg
   print("Converting CRS from {} to {}".format(gdf.crs,epsg))
   try:
      tmp = gdf.to_crs(epsg)
   except Exception as ex:
      print(traceback.format_exc())
      sys.exit(2)
   print("CRS is converted successfully.")
   return tmp

def getRandomParameters():
   title="Gaussian random parameters"
   myParameters = []
   window = tk.Tk()
   window.configure(width=600, height=400)
   window.geometry("+40+40")
   window.title(title)
   frame = LabelFrame(window, text=title, padx=20, pady=20, font= ('Helvetica 14'))
   frame.pack(pady=20, padx=10)
   tk.Label(frame, 
         text="Please enter Gaussian random parameters.").grid(row=0)
   tk.Label(frame, 
         text=u"Mean (\u03bc)").grid(row=1)
   e1 = tk.Entry(frame)

   e1.grid(row=1, column=1)
   tk.Label(frame, 
         text=u"STD (\u03C3)").grid(row=2)
   e2 = tk.Entry(frame)

   e2.grid(row=2, column=1)


   tk.Button(frame, text='Cancel', command=window.quit).grid(row=4, column=0, sticky=tk.W, pady=4, padx=4)
   tk.Button(frame, text='Ok', command=lambda:getEntryParams(myParameters,e1,e2, window)).grid(row=4, column=1, sticky=tk.W, pady=4,padx=4)

   tk.mainloop()
   if ((myParameters[0]!="") & (myParameters[1]!="")):
      if((myParameters[0].replace(".", "",1).isdecimal()) & (myParameters[1].replace(".", "",1).isdecimal())):
         print("Mean(\u03bc): {} ,  STD(\u03C3): {}".format(myParameters[0], myParameters[1]))
         return float(myParameters[0]), float(myParameters[1])
      else:
         print("Error")
         print("Values for Mean and STD are not valid.")
         print("Mean(\u03bc): {} ,  STD(\u03C3): {}".format(myParameters[0], myParameters[1]))
         sys.exit(2)
   else:
      print("Error")
      print("Values for Mean and STD are not valid.")
      print("Mean(\u03bc): {} ,  STD(\u03C3): {}".format(myParameters[0], myParameters[1]))
      sys.exit(2)
   

def filterOutlier(pointsArray,indexE, indexN, indexVel, numOfNeighbour=20, threshold=2):
   # print("number of points before filter: {}".format(len(pointsArray)), flush=True)
   # print("Removing the Outliers.", flush=True) 
   num_neighbor = 4
   outlierIndx = []
   points = np.c_[pointsArray[:,indexE], pointsArray[:,indexN]]
   tree = KDTree(points)
   newPoints = None
   for idx,p in enumerate(pointsArray):
      try:
         if(len(pointsArray[:,0])< numOfNeighbour):
               numOfNeighbour = len(pointsArray[:,0])
         pp = np.c_[p[indexE], p[indexN]]
         dist, ind = tree.query(pp, k=numOfNeighbour)
         tmp_Z = pointsArray[ind,indexVel]
         Q1 = np.percentile(tmp_Z,25)
         Q3 = np.percentile(tmp_Z,75)
         IQR = Q3-Q1
         lowerBound = Q1 - 1.5*IQR
         upperBound = Q3 + 1.5*IQR                
         outliersIdx = (tmp_Z[0]<lowerBound) | (tmp_Z[0]>upperBound)
         if (len(tmp_Z[0])<num_neighbor):
               num_neighbor = len(tmp_Z[0])
         if(np.std(tmp_Z[0])==0):
            z_scores = [0]* len(tmp_Z[0][0:num_neighbor+1])
         else:
            z_scores = (tmp_Z[0][0:num_neighbor+1] - np.mean(tmp_Z[0][0:num_neighbor+1])) / np.std(tmp_Z[0])

         outliers_zscore = np.abs(z_scores) > threshold

         if(outliers_zscore[0]):
               outlierIndx.append(copy.copy(idx))

         if len(outliersIdx)>0:
               if(outliersIdx[0]):
                  if(idx not in outlierIndx):
                     outlierIndx.append(copy.copy(idx))
         newPoints = copy.deepcopy(np.delete(pointsArray, outlierIndx, axis=0) )
      except Exception as ex:
         print(traceback.format_exc(),flush=True)
         sys.exit(2)  
   print('Outliers are removed successfully.',flush=True)
   # print("number of points after filter: {}".format(len(newPoints)),flush=True)
   return newPoints
def removeIsolatedPoints(pointsArr, pointsX, pointsY, thresholdDistance):
   # print("number of points before filter: {}".format(len(pointsArr)),flush=True)
   # print("Removing Isolated Points.",flush=True)
   filterdOutput = None
   try:
      points = np.c_[pointsX, pointsY]
      kdtree = KDTree(points)
      min_distance_threshold = thresholdDistance
      num_isolatedPoints = 2
      # Find the indices and distances of the nearest neighbors for each point
      distances, indices = kdtree.query(points, k=num_isolatedPoints + 1,distance_upper_bound=float(thresholdDistance) )
      isolatedPointsIndices = []
      
      for i in range(len(pointsX)):
         if (np.Inf in distances[i]):
               isolatedPointsIndices.append(copy.copy(indices[i][0]))

      isolatedPointsIndices = np.array(isolatedPointsIndices)
      if (len(isolatedPointsIndices) == 0):
         filterdOutput = pointsArr
      else:
         iso = pointsArr[isolatedPointsIndices]
         filterdOutput = np.delete(pointsArr, isolatedPointsIndices, axis=0)
      outPut = copy.deepcopy(filterdOutput)
   except Exception as ex:
      print(traceback.format_exc(),flush=True)
      sys.exit(2)
   print('Isolated points are removed successfully.',flush=True)
   # print("number of points after filter: {}".format(len(outPut)),flush=True)
   return(outPut)


def outsideRect(x1,y1,x2,y2,x,y):
   if((x >= x1) and (x <= x2) and (y <= y1) and (y >= y2)):
      return False
   else:
      return True

def idwPointsInside(xx,yy,zz,xz,yz,p,w_pixel, h_pixel):                             #calculate IDW weight based on distance and p value
   w_list=[]
   for j in range(len(xx)):
      if (outsideRect(xz,yz,xz+w_pixel,yz-h_pixel,xx[j],yy[j])==True):
         z_idw = np.nan
         w_list.append(-1)
      else:
         w_list.append(0)   #if meet this condition, it means d<=0, weight is set to 0 #check if there is 0 in weight list
   w_check=0 in w_list
   if w_check==True:
      if (w_list.count(0) < 2):
         idx=w_list.index(0) # find index for weight=0
         z_idw=copy.copy(zz[idx]) # set the value to the current sample value
      else:
         tmpSum = 0.0
         tmpCnt = 0.0
         for idx,item in enumerate(w_list):
               if (item == 0):
                  tmpSum += zz[idx]
                  tmpCnt += 1.0
         z_idw=copy.copy(tmpSum/tmpCnt)       
   return z_idw

def idwPixel(pixel_i, pixel_j, myGrid, p, pixelSize):
   z_idw = 0
   w_list=[]
   zz = []
   for i,item in enumerate(myGrid):
      for j,element in enumerate(item):
         if (np.isnan(element)):
               pass
         else:
               zz.append(element)
               d=pixelSize*math.sqrt((i-pixel_i)**2+(j-pixel_j)**2)
               w=1.0/(d**p)
               w_list.append(copy.deepcopy(w))
   
   zz = np.array(zz)
   wt=np.transpose(w_list)

   z_idw=np.dot(zz,wt)/np.float64(sum(w_list))

   return z_idw

def interPolationOfPoints(myPoints,pixelSize,colE,colN,colVelocity,colpolygongeom, processRank):
   print("Interpolating data.",flush=True)
   try:
      Geom_B = myPoints[0,colpolygongeom]
      
      x_coordinates = []
      y_coordinates = []
      if (type(Geom_B) is str):
         Geom_B = shapely.wkt.loads(Geom_B)

      Geom_B_3035 = copy.deepcopy(Geom_B)
      epsg3035 = pyproj.CRS('EPSG:3035')
      epsg32630 = pyproj.CRS('EPSG:32630')

      project = pyproj.Transformer.from_crs(epsg3035, epsg32630, always_xy=True).transform
      Geom_B_32630 = transform(project, Geom_B_3035)
      if (type(Geom_B_32630) is shapely.geometry.polygon.Polygon):
         
         # Geom_B_32630 = Geom_B_32630.buffer(5.6)
         for point in Geom_B_32630.exterior.coords:
            x_coordinates.append(point[0])
            y_coordinates.append(point[1])
      else:
         # Geom_B_32630 = Geom_B_32630.buffer(5.6)
         for polyitem in Geom_B_32630.geoms:
            for point in polyitem.exterior.coords:
               x_coordinates.append(point[0])
               y_coordinates.append(point[1])  

      gridArray = None
      x=myPoints[:,colE]
      y=myPoints[:,colN]
      z=myPoints[:,colVelocity]
      xmin = np.min(myPoints[:,colE])                                      
      xmax = np.max(myPoints[:,colE])
      ymin = np.min(myPoints[:,colN])
      ymax = np.max(myPoints[:,colN])
      deltaX = xmax - xmin                                                           
      deltaY = ymax - ymin
      

      xBmin = min(x_coordinates)
      xBmax = max(x_coordinates)
      yBmin = min(y_coordinates)
      yBmax = max(y_coordinates)
      deltaX = xBmax - xBmin                                                           
      deltaY = yBmax - yBmin
      
      linespaceRangeX = 0
      linespaceRangeY = 0
      if (deltaX > deltaY):
         linespaceRangeY = abs(deltaX- deltaY)
         deltaY = deltaX
      elif (deltaX < deltaY):
         linespaceRangeX = abs(deltaX- deltaY) 
         deltaX = deltaY

      deltaX = deltaX + (pixelSize - (deltaX%pixelSize))
      deltaY = deltaY + (pixelSize - (deltaY%pixelSize))

      if (deltaX <pixelSize ):
         nx = int(deltaX/pixelSize) +1
      else:
         nx = int(deltaX/pixelSize)
      
      if (deltaY <pixelSize ):
         ny = int(deltaY/pixelSize)+1
      else:
         ny = int(deltaY/pixelSize)

      tx = np.linspace(xBmin,xBmax + linespaceRangeX, nx)
      ty = np.linspace(yBmax + linespaceRangeY,yBmin,ny)

      widthPixel = abs(xBmax - xBmin)
      heightPixel = abs(yBmax - yBmin )

      # tx = np.linspace(myPoints[:,colE].min(),myPoints[:,colE].max() + linespaceRangeX, nx)
      # ty = np.linspace(myPoints[:,colN].max() + linespaceRangeY,myPoints[:,colN].min(),ny)

      # widthPixel = abs(myPoints[:,colE].min()-myPoints[:,colE].max())
      # heightPixel = abs(myPoints[:,colN].max()-myPoints[:,colN].min())

      if(nx>1 ):
         widthPixel = tx[1]-tx[0]
      if(ny>1):
         heightPixel = ty[0]-ty[1]
      
      XI, YI = np.meshgrid(tx, ty) 
      ZI = list()
      x_coordinates = np.array(XI)
      y_coordinates = np.array(YI)
      
      print("ny: {} , nx: {}".format(ny,nx),flush=True)
      for i in range(ny):
         tmpList = list()
         for j in range(nx):                          
            IDW=idwPointsInside(x,y,z,XI[i,j],YI[i,j], 5,widthPixel,heightPixel)
            tmpList.append(copy.deepcopy(IDW))
         ZI.append(copy.copy(tmpList))
      ZI = np.array(ZI)
      knownGrid = copy.deepcopy(ZI)
      print("row: {} , col: {}".format(len(knownGrid),len(knownGrid[0])),flush=True)
      for rowIdx,row in enumerate(knownGrid):
         for colIdx,item in enumerate(row):
            if (np.isnan(item)):
                  IDW=idwPixel(rowIdx, colIdx, knownGrid,5, pixelSize)
                  ZI[rowIdx, colIdx]=copy.copy(IDW)
                  
         print("please wait... row: {} | cpu: {}".format(rowIdx,processRank),flush=True)
      print("line: {}".format(555))
      gridArray = copy.deepcopy(ZI)
   except Exception as ex:
      print("line: {}".format(557), flush=True)
      print(traceback.format_exc(),flush=True)
      sys.exit(2)
   print("Data interpolation is completed successfully.",flush=True)
   return gridArray , x_coordinates, y_coordinates


def classify (MAX_slope , ave_slope , STDvalue):
    slop_intens = 0
    if ( (MAX_slope >= 0.35) ):
        slop_intens  = 5
    elif (( 0.25< MAX_slope <= 0.35) ):
        slop_intens = 4
    elif (( 0.15< MAX_slope <= 0.25) ):
        slop_intens = 3
    elif (( 0.05< MAX_slope <= 0.15)):
        slop_intens =2
    elif (( MAX_slope <= 0.05)):
        slop_intens = 1  
    else :
        slop_intens = -1
    return slop_intens


def slopeAspectCalc(psPointArr,pixelSize,numActivePoint,buildingIndex,colE,colN,
                    colVelocity,colIndexRight,colgeometry,processRank, nProcess,
                    gdfPolygon,psPointColumn, colHeight,colDTMmin, colOsmId, queue): 

   logfile = open("log_{}.txt".format(processRank), "a")
   print("Process #{} is start to work".format(processRank), flush=True)

   logfile.write("[{}] Process #{} is start to work\n".format(datetime.datetime.now(), processRank))
   resultDict = {}
   allSlope = []
   allAspect = []
   buildingID = []
   slopeSTD = []
   maximumSlope = []
   slopeMean = []
   eastAspect = []
   northAspect = []
   numPs = []
   Xcor = []
   Ycor=[]
   latX = []
   latY = []
   classification = []
   buildingGeometry= []
   buildingAreas = []
   pointDataFrame = []

   # colosm_id = [436]

   
      

   for id in range(processRank,len(buildingIndex), nProcess):
      try:
         idx = buildingIndex[id]
         tmpInside = copy.deepcopy(psPointArr[np.where( psPointArr[:,colIndexRight] == idx)])    
         unique_keys, indices = np.unique(np.array(tmpInside[:,[colE, colN]], dtype=float), return_index=True,axis=0)
         tmpInside= tmpInside[indices]
         # gdfPolygon structure ["Unnamed: 0", index_right,"PGnoBf3035"]
         gdfPolygon_arr = np.array(gdfPolygon)
         geomet = copy.deepcopy(gdfPolygon_arr[np.where( np.array(gdfPolygon_arr[:,gdfPolygon.columns.get_loc("index_right")]) == idx)])
         geomet = geomet[0][gdfPolygon.columns.get_loc("PGnoBf3035")]
      except Exception as ex:
         logfile.write("[{}] Error: {}\n".format(datetime.datetime.now(), traceback.format_exc()))
         print(traceback.format_exc(), flush=True)
         return None
      try:
         myPoly = shapely.wkt.loads(geomet)
         centerLat = myPoly.centroid 
      except Exception as ex:
         logfile.write("[{}] Error: {}\n".format(datetime.datetime.now(), traceback.format_exc()))
         print(traceback.format_exc(), flush=True)
         return None
      
      tmpInsidePoint = copy.deepcopy(tmpInside)

      # concatenate geomet column to tmpInsidePoint
      tmpInsidePoint = np.concatenate((tmpInsidePoint, np.array([geomet]*len(tmpInsidePoint)).reshape(-1,1)), axis=1)
      # set thr colpolygongeom to the last column of tmpInsidePoint
      colpolygongeom = tmpInsidePoint.shape[1]-1

      # check if Points are inside Polygon, Uncomment if required for Monte Carlo
      # try:
      #    pointdf = pd.DataFrame(tmpInside , columns = psPointColumn)
      #    gdfp = gpd.GeoDataFrame(pointdf , geometry='geometry')
      #    gdfp['geometry'] = gdfp.apply(lambda row: Point(row['E'], row['N']), axis=1)
      #    within_poly = gdfp[gdfp.geometry.within(myPoly)]
      #    tmpInsidePoint = copy.deepcopy(within_poly.to_numpy())
      # except Exception as ex:
      #    print(traceback.format_exc(), flush=True)
      
      ## Filtering ##
      try:
         velocityArr = tmpInsidePoint[:,colVelocity]
         if(len(tmpInsidePoint) >= 14):
            filterdPoints = filterOutlier(tmpInsidePoint,colE,colN,colVelocity)
            
            filterdPoints = removeIsolatedPoints(filterdPoints, filterdPoints[:,colE], filterdPoints[:,colN],thresholdDistance=25)
            outlier = np.logical_and(np.array(filterdPoints[:,colVelocity]>2) ,np.array(filterdPoints[:,colVelocity]< 6))    
            number_outlier = outlier.sum()
            if ((number_outlier/len(tmpInsidePoint)) < 0.3):
               filterdPoints =np.array(filterdPoints[~outlier])                        #removing outliers 
            else:
               filterdPoints = copy.deepcopy(np.array(filterdPoints))
            #based on Unwarpping error we have this condition
            countActive = 0
            countActive = (filterdPoints[:,colVelocity] >= 2.9).sum()  + (filterdPoints[:,colVelocity] <= -2.9).sum()
            # print test
            
            if(countActive > numActivePoint):
               print("cpu: {} | polygonID: {} | #Acitve: {}".format(processRank,idx, countActive), flush=True)
               gridArray,x_coordinates ,y_coordinates = copy.deepcopy(interPolationOfPoints(filterdPoints,pixelSize,colE,colN,colVelocity,colpolygongeom, processRank))
               if (~np.isnan(gridArray[0][0])):

                  gridArray  = rd.rdarray(gridArray, no_data=-9999)

                  
                  for i in range(len(tmpInsidePoint)):
                     pointdata=tmpInsidePoint[i]
                     mypoint = {'E': pointdata[colE],'N':pointdata[colN],'id':pointdata[colIndexRight],'Velocity':pointdata[colVelocity],\
                                'Geometry':pointdata[colgeometry],'ID':pointdata[colOsmId] , 'Height':pointdata[colHeight] , 'DSM':pointdata[colDTMmin]}
                     pointDataFrame.append(copy.deepcopy(mypoint))

                  buildingGeometry.append(copy.deepcopy(myPoly))
                  latX.append(centerLat.x)
                  latY.append(centerLat.y)
                  numPs.append(copy.deepcopy(len(tmpInsidePoint)))

                  buildingAreas.append(copy.deepcopy(myPoly.area))   

                  buildingID.append(copy.copy(idx))
                  slope =rd.TerrainAttribute(gridArray, attrib='slope_riserun',zscale=(1.0/pixelSize))
                  slopeArr = np.array(slope)
                  allSlope.append(copy.deepcopy(slopeArr))
                  slopeStdVal = np.std(slope)
                  slopeSTD.append(copy.copy(slopeStdVal))
                  slopeMaxValue = slope.max()      
                  maximumSlope.append(copy.copy(slopeMaxValue))
                  slopeMeanVal = slope.mean()
                  slopeMean.append(copy.copy(slopeMeanVal))

                  aspect = rd.TerrainAttribute(gridArray, attrib='aspect',zscale=(1.0/pixelSize))
                  aspectArr = np.array(aspect)
                  allAspect.append(copy.deepcopy(aspectArr))

                  result = np.argwhere(slopeArr == slopeMaxValue)                                                # Find the indices of maxslope from slope array in order to extract aspect
                  try:
                     rowmaxSlp =  result[0][0]
                     ColmaxSLP= result[0][1]
                  except Exception as ex:
                     logfile.write("[{}] Error: {}\n".format(datetime.datetime.now(), traceback.format_exc()))
                     print(traceback.format_exc(), flush=True) 
                     return None
                  
                  northAspVal = aspectArr[rowmaxSlp][ColmaxSLP]
                  x = x_coordinates [rowmaxSlp ,ColmaxSLP]
                  y = y_coordinates [rowmaxSlp ,ColmaxSLP]
                  Xcor.append(x)
                  Ycor.append(y)
                  eastAsp =northAspVal -90
                  if eastAsp < 0:
                     eastAsp = 360 + eastAsp

                  northAspect.append(copy.deepcopy(northAspVal))
                  eastAspect.append(copy.deepcopy(eastAsp))
                  slope_class = copy.deepcopy(classify (slopeMaxValue,slopeMeanVal, slopeStdVal))
                  classification.append(copy.deepcopy(slope_class))
            else:
                  # print("cpu: {} | polygonID: {} | #Acitve: {}".format(processRank,idx, countActive), flush=True)
                  # print("Number of points with Veolocity value in (-inf, -3] and [+3, +inf) is {}".format(countActive),flush=True)      
                  # print("which is less than {}".format(numActivePoint),flush=True)
                  pass
      except Exception as ex:
         logfile.write("[{}] Error: {}\n".format(datetime.datetime.now(), traceback.format_exc()))
         print(traceback.format_exc(), flush=True) 
         return None
      
      # print progress bar
      
      # if (counter%(int(len(buildingIndex)/len(progressBar))) == 0):
      #    progressBar[int(counter/int(len(buildingIndex)/len(progressBar)))] = "#"
      #    # print progess bar
      #    print("cpu: {} | progress: [{}]".format(processRank, "".join(progressBar)))#, flush=True)
      # counter += 1   
   try:
      resultDict["slopeSTD"]=copy.deepcopy(slopeSTD)
      resultDict["maximumSlope"]=copy.deepcopy(maximumSlope)
      resultDict["slopeMean"]=copy.deepcopy(slopeMean)

      resultDict["northAspect"]=copy.deepcopy(northAspect)
      resultDict["eastAspect"]=copy.deepcopy(eastAspect)
      resultDict["buildingID"]=copy.deepcopy(buildingID)
      resultDict["numPs"]=copy.deepcopy(numPs)
      resultDict["X"]=copy.deepcopy(Xcor)
      resultDict["Y"]=copy.deepcopy(Ycor)
      resultDict["class"] = copy.deepcopy(classification)
      resultDict["X_lat_cord"] = copy.deepcopy(latX)
      resultDict["Y_lat_cord"] = copy.deepcopy(latY)
      resultDict["buildingGeometry"] = copy.deepcopy(buildingGeometry)
      resultDict["buildingAreas"] = copy.deepcopy(buildingAreas)
      resultDict["pointDataFrame"] = copy.deepcopy(pointDataFrame)
   except Exception as ex:
      logfile.write("[{}] Error: {}\n".format(datetime.datetime.now(), traceback.format_exc()))
      print(traceback.format_exc(), flush=True) 
      return None

   logfile.write("[{}] Process #{} is finished\n".format(datetime.datetime.now(), processRank))
   logfile.close()
   queue.put(resultDict)



def gradientCalc(gdfPs, gdfPolygon, buildingIndex=[],numActivePoint=3, pixelSize=8, deprecatedIndexRight=None,logfile=None): 
   
   colE = 0
   colN = 1
   colVelocity = 3 
   colBIndex = 7
   colIndexRight = 2
   colDTMmin = 7
   colOsmId = 5
   errorLine = None
   bufferSize= 5.6
   allSlope = []

   colE = gdfPs.keys().get_loc("E")
   colN = gdfPs.keys().get_loc("N")
   colIndexRight = gdfPs.keys().get_loc("index_right")
   colVelocity = gdfPs.keys().get_loc("mean_velocity")
   colgeometry = gdfPs.keys().get_loc("geometry")
   # colpolygongeom = gdfPs.keys().get_loc("PGnoBf3035")
   colDTMmin = gdfPs.keys().get_loc("DTM_min")
   colOsmId  = gdfPs.keys().get_loc("osm_id")
   colHeight = gdfPs.keys().get_loc("height")
   psPointArr = gdfPs.to_numpy()
   psPointColumn = list(gdfPs.columns)
   if buildingIndex==[]:
      try:
         indexID = psPointArr[:,colIndexRight].astype(int)
         if (deprecatedIndexRight != None):
            deprecatedIndexRightSet = set(deprecatedIndexRight)   
         else:
            deprecatedIndexRightSet = set()
         buildingIndex =  list(set(indexID)-set(deprecatedIndexRight))

      except Exception as ex:
         print(traceback.format_exc(),flush=True)
         sys.exit(2)
   nCores = mp.cpu_count()
   nProcess = 4 #nCores - 2
   queueList = []
   allSlope = []
   resultList = []
   
   
   # set the fork start method
   try:
      mp.set_start_method('spawn')
   except RuntimeError:
      pass
   
   try:
      print("Number of processes: {}".format(nProcess))
      print("Trying to create the pool of workers.")
      # create the manager
      with mp.Manager() as  manager:
         # create the shared queue
         queue = manager.Queue()
         # create the pool of workers
         pool = mp.Pool(processes=nProcess)
         for processRank in range(nProcess):
            # print("Process #{} is start to work".format(processRank))
            # create a list of arguments, one for each call


            pool.apply_async(slopeAspectCalc,args=(psPointArr,pixelSize,numActivePoint,
                                                      buildingIndex,colE,colN,colVelocity,colIndexRight,colgeometry,
                                                      processRank, nProcess, gdfPolygon,psPointColumn, colHeight,colDTMmin,
                                                      colOsmId, queue, ))
         print("Pool of workers is created successfully.")
         for i in range(nProcess):
            print("Waiting for process #{} to finish.".format(i))
            # get item from queue
            resultList.append(copy.deepcopy(queue.get()))
            print("Process #{} is finished.".format(i))
         
         # Close the pool to prevent any more tasks from being submitted
         pool.close()
         # Wait for all the tasks to complete
         pool.join()
   except Exception as ex:
         logfile.write("[{}] Error: {}\n".format(datetime.datetime.now(), traceback.format_exc()))
         print(traceback.format_exc(),flush=True) 
         sys.exit(2)

   
   
   

   print("All processes have finished.")
   resultDictMerged = {"buildingID":[], "eastAspect":[], "northAspect":[], \
                       "slopeMean":[], "maximumSlope":[], "slopeSTD":[] , \
                       "numPs":[] ,"X":[], "Y":[],"class":[], "X_lat_cord":[], \
                       "Y_lat_cord":[], "buildingGeometry":[], "buildingAreas":[],\
                        "pointDataFrame":[]}
   for i, item in enumerate(resultList):
      resultDictMerged = {key: resultDictMerged[key] + item[key] for key in set(resultDictMerged) & set(item)}
   return(resultDictMerged)




def runSimulation(gdfPS, buildingIndexList):
   result = gradientCalc(gdfPS, buildingIndex=buildingIndexList)
   return(result)


def makeRandomData(meanGauss, sigmaGauss, gdfPS , sigmaE , sigmaN):
   print("Building Monte Carlo data.")
   try:
      colE = gdfPS.keys().get_loc("E")
      colN = gdfPS.keys().get_loc("N")
      colVelocity = gdfPS.keys().get_loc("mean_velocity")
      randomError = np.random.normal(loc=meanGauss, scale=sigmaGauss,size=(gdfPS.shape[0]))
      randomErrorE = np.random.normal(loc=meanGauss, scale=sigmaE,size=(gdfPS.shape[0]))
      randomErrorN = np.random.normal(loc=meanGauss, scale=sigmaN,size=(gdfPS.shape[0]))
      # randomError = np.random.normal(loc=meanGauss, scale=sigmaGauss,size=(1,))
      gdfPS["mean_velocity"] = gdfPS["mean_velocity"] + randomError
      gdfPS["E"] = gdfPS["E"]  + randomErrorE
      gdfPS["N"] = gdfPS["N"]  + randomErrorN

   except Exception as ex:
      print(traceback.format_exc())
      sys.exit(2)
   print("Data is created successfully.")
   return gdfPS

def MonteCarloSim():
   
   # psFilePath = getFileWindow(title="Select PS points file")
   psFilePath = "inside_sim_10.6B.csv"
   monteCarloPath = "testmonteCarlo_combination_pullotion_XY.csv"

   # meanGauss, sigmaGauss = getRandomParameters()
   meanGauss, sigmaGauss = 0.0, 1.0
   sigmaE , sigmaN = 2.0 , 5.0
   # numberOfSimulation = int(showInputBox(title="Simulation",sentenceLabel="Please enter the number of simulation",inputLabel="#Simulation"))
   numberOfSimulation =100

   gdfPS = readData(psFilePath, dataFormat="csv", sep=",")
   print(gdfPS.columns)
   realDataResult = gradientCalc(gdfPS)
   buildingIndexList = realDataResult["buildingID"]
   realMaxSlope = realDataResult["maximumSlope"]
   with open(monteCarloPath, "w") as myFile:
      for item in buildingIndexList:
         myFile.write("slope_max_"+str(item))
         myFile.write(",")
         myFile.write("slope_std_"+str(item))
         myFile.write(",")
         myFile.write("slope_mean_"+str(item))
         myFile.write(",")
         myFile.write("numPs"+str(item))
         myFile.write(",")
         myFile.write("X"+str(item))
         myFile.write(",")
         myFile.write("Y"+str(item))
         myFile.write(",")
      myFile.write("\n")
      numberOfColumns = 6
      tmpResultList = [0]*numberOfColumns*len(buildingIndexList)
      for idx, idxBuilding in enumerate(realDataResult["buildingID"]):            
         tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns] = realDataResult["maximumSlope"][idx]
         tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns+1] = realDataResult["slopeSTD"][idx]
         tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns+2] = realDataResult["slopeMean"][idx]
         tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns+3] = realDataResult["numPs"][idx]
         tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns+4] = realDataResult["X"][idx]
         tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns+5] = realDataResult["Y"][idx]
      myFile.write(",".join(map(str, tmpResultList)))
      myFile.write("\n")
      for run in range(numberOfSimulation):
         tmpResultList = [0]*numberOfColumns*len(buildingIndexList)
         tmpgdf = copy.deepcopy(gdfPS)
         newGdfPS = makeRandomData(meanGauss, sigmaGauss, tmpgdf , sigmaE,sigmaN)
         simResultDict = runSimulation(newGdfPS, buildingIndexList)
         
         for idx, idxBuilding in enumerate(simResultDict["buildingID"]):            
            tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns] = simResultDict["maximumSlope"][idx]
            tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns+1] = simResultDict["slopeSTD"][idx]
            tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns+2] = simResultDict["slopeMean"][idx]
            tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns+3] = simResultDict["numPs"][idx]
            tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns+4] = simResultDict["X"][idx]
            tmpResultList[buildingIndexList.index(idxBuilding)*numberOfColumns+5] = simResultDict["Y"][idx]
         myFile.write(",".join(map(str, tmpResultList)))
         myFile.write("\n")

   

def writeResults(filePath, dataDict):
   print("Writing results as a shape file...")

   tmpGeoDataFrame = pd.DataFrame({ "Poly_ID":np.array(dataDict["buildingID"], dtype="str"),"Easting": np.array(dataDict["X_lat_cord"], dtype="float32") ,\
                                    "Northing":np.array(dataDict["Y_lat_cord"], dtype="float32") ,"Slp_Intens":np.array(dataDict["class"] ,dtype="int32"),"Mean_slope":np.array(dataDict["slopeMean"], dtype="float32"),\
                                    "Maxslope":np.array(dataDict["maximumSlope"], dtype="float32") ,"Aspect_east":np.array(dataDict["eastAspect"], dtype="float32"),\
                                    "STD_slope":np.array(dataDict["slopeSTD"], dtype="float32") ,'Aspect_north':np.array(dataDict["northAspect"]), \
                                    "geometry": dataDict["buildingGeometry"] ,"CORDX":np.array(dataDict["X"]),"CORDY":np.array(dataDict["Y"]) \
                                    ,"Num_PS":np.array(dataDict["numPs"], dtype="int32"),"Area":np.array(dataDict["buildingAreas"], dtype="float32")})
   # check of dataframe is empty or not
   if (tmpGeoDataFrame.empty):
      print("Dataframe is empty.")
      print("File is not created.")
   else:
      Slope_GEOdataframe =gpd.GeoDataFrame(tmpGeoDataFrame)
      Slope_GEOdataframe = Slope_GEOdataframe.round(decimals=3)
      psPointDataframe = dataDict["pointDataFrame"]
      # psPointDataframe list of dictionary to dataframe
      psPointDataframe = pd.DataFrame(psPointDataframe)
      # save ps points as a csv file
      csvFile = open(filePath+"_psPoints.csv", "w" ,newline='')   
      psPointDataframe.to_csv(csvFile, sep=',')

      Slope_GEOdataframe.to_file(filePath+".shp") 
      print("Writing results is finished.")

def insidePoints2SlopeAspect(psFilePath, polygonFilePath, logfile):

   numActivePoint = 3 
   pixelSize = 8 # meters

   
   # slopeAspectFile = "TotalBDD_Sentinel_V{}_fixed_{}_CT".format(numActivePoint, pixelSize)
   slopeAspectFile = None
   if ((slopeAspectFile==None)):
      myInx = psFilePath.rfind("\\") 
      if (myInx != "-1"):
         slopeAspectFile = "gradient_" + psFilePath[myInx+1:-5]
      else:
         slopeAspectFile = "gradient_" + psFilePath[:-5]

   
   gdfPS = readData(psFilePath, dataFormat="csv", sep=",", logfile=logfile)
   gdfPolygon = readData(polygonFilePath, dataFormat="csv", sep=",", logfile=logfile, headerList=["index_right","PGnoBf3035"]) 
   
   # print the number of point of the polygon in each row of gdfPolygon["PGnoBf3035"]
   deprecatedIndexRight = []
   # for loop over rows of pandas dataframe
   for idx, row in gdfPolygon.iterrows():
      tmpPoly = shapely.wkt.loads(row["PGnoBf3035"])
      if ((len(tmpPoly.exterior.coords) > 6000) or (tmpPoly.area > 100000)):
         deprecatedIndexRight.append(int(row["index_right"]))


   

   # concatinate two dataframes
   logfile.write("[{}] Data is read successfully.\n".format(datetime.datetime.now()))
   logfile.write("[{}] file: {}\n".format(datetime.datetime.now(), psFilePath))
   logfile.write("[{}] file: {}\n".format(datetime.datetime.now(), polygonFilePath))
   insideDataGradientDict = gradientCalc(gdfPs=gdfPS,gdfPolygon=gdfPolygon, numActivePoint=numActivePoint, pixelSize=pixelSize,deprecatedIndexRight=deprecatedIndexRight, logfile=logfile)
   logfile.write("[{}] calculation finished successfully.\n".format(datetime.datetime.now()))
   writeResults(slopeAspectFile,insideDataGradientDict)
   logfile.write("[{}] Data is written successfully.\n\n".format(datetime.datetime.now()))
   



if __name__=="__main__":
   
   logFilePath = "log.txt"
   logfile = open(logFilePath, "a")

   startTime = time.time()
   # install("requirements.txt")
   # psPointFile = "../insideDataPoints.csv"
   # MonteCarloSim()
   # get ps file path from python script arguments
   psFilePath = os.path.join("..\\" , sys.argv[1])
   polygonFilePath = os.path.join("..\\" , sys.argv[2])

   insidePoints2SlopeAspect(psFilePath, polygonFilePath,logfile)

   logfile.close()

   endTime = time.time()
   totalTime = endTime - startTime
   print("Total running time: {:.2f} secs".format(totalTime))