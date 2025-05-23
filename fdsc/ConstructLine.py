#! /usr/bin/python
# -*- coding: utf-8 -*-

##########################################################################
# Fault Displacement Slip-Curve (FDSC) v1.0                              #
#                                                                        #
# Copyright (C) (2013-2014) Ana-Maria Rosu ana-maria.rosu@ign.fr         #
# IPGP-ENSG/IGN project financed by TOSCA/CNES                           #
#                                                                        #
#                                                                        #
# This software is governed by the CeCILL-B license under French law and #
#abiding by the rules of distribution of free software.  You can  use,   #
#modify and/ or redistribute the software under the terms of the CeCILL-B#
#license as circulated by CEA, CNRS and INRIA at the following URL       #
#"http://www.cecill.info".                                               #
##########################################################################

import matplotlib.pyplot as plt
from osgeo import gdal


class ConstructLine:
  """ Construct line class """
  def __init__(self,fig,ax, filepath_im, filepath_out):
    self.fig=fig
    self.ax=ax
    self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)
    self.coords=[]
    self.filepath_im=filepath_im
    self.importIm()
    self.filepath_out=filepath_out

  def importIm(self):
    ds = gdal.Open(self.filepath_im, gdal.GA_ReadOnly)
    nb_col=ds.RasterXSize #number of columns of image
    nb_lines=ds.RasterYSize #number of lines of image
    nb_b=ds.RasterCount #number of bands of image
    print(f'{nb_col=}   {nb_lines=}   {nb_b=}')
    data_im=ds.ReadAsArray()#!! in gdal : data[line][col]
    plt.imshow(data_im, cmap = plt.cm.Greys_r, interpolation=None)
    plt.colorbar()
    self.ax.set_autoscale_on(False)

  def recoverLine(self,filepath_line):
    self.coords=[]
    reading_status=0 #0: begin , 1: getting polyline points, 2: polyline finished
    with open(filepath_line, 'r', encoding='utf-8') as fline:
      for line in fline:
        if (reading_status==0):
          if (line[0:7]=="#image "):
            print("Try to load image *",(line[7:]).strip(),"*")
            self.filepath_im=(line[7:]).strip()
            continue
        if (line.strip()=="#begin polyline X Y"):
          reading_status=1
          continue
        if (line.strip()=="#end polyline"):
          reading_status=2
          continue
        if (reading_status==1):
          (x,y)=line.split()
          self.coords.append((float(x),float(y)))
          continue

    print(self.coords)
    self.drawAllSeg(0.7,'red')

  def drawAllSeg(self, alpha_s, color_s):
    print("drawAllSeg")
    if len(self.coords)>0:
      (list_x,list_y)=zip(*self.coords)
      self.ax.plot(list_x, list_y, color=color_s, alpha=alpha_s)
    self.ax.figure.canvas.draw()

  def drawLastSeg(self, alpha_s, color_s):
    print("drawLastSeg")
    if len(self.coords)>1:
      self.ax.plot((self.coords[-2][0], self.coords[-1][0]),
                   (self.coords[-2][1], self.coords[-1][1]),
                   color=color_s, alpha=alpha_s)
      self.ax.figure.canvas.draw()

  def saveTrace(self):
    with open(self.filepath_out, 'w', encoding='utf-8') as file:
      file.write(f"#image '{self.filepath_im}'\n")
      file.write("#begin polyline X Y\n")
      for (x,y) in self.coords:
        file.write(f"{x} {y}\n")
      file.write("#end polyline\n")

  def redraw(self):
    print("redraw")
    self.fig.clf()
    self.ax = self.fig.add_subplot(111)
    if (self.filepath_im != ""):
      self.importIm()
    self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self)
    self.drawAllSeg(0.7,'red')

  def __call__(self, event):
    print('click', event)
    print('button=', event.button)
    if event.inaxes!=self.ax: return

    if event.button==1: # draw line segment
      if event.dblclick: # 2nd click of a double-click is discarded
        return
      if len(self.coords) > 0:
        # check if clicked point is not a duplicate of the previous point
        if self.coords[-1] == (event.xdata, event.ydata):
          print("WARNING: Same point clicked twice!! Second occurrence is discarded")
          return
      self.coords.append((event.xdata, event.ydata))
      print(self.coords)
      self.drawLastSeg(0.7,'red')

    elif event.button==2: # delete line segment between last and second to last points
      print('Delete wanted')
      #self.drawLastSeg(0.5, 'white')
      if len(self.coords)>=1:
        del self.coords[-1]
        print(self.coords)
      else: print("No more points")
      if len(self.coords)==1:
        del self.coords[-1]
      self.redraw()

    elif event.button==3: # save the (poly)line
      print(f"Save points to file '{self.filepath_out}'")
      self.saveTrace()
      plt.close()
