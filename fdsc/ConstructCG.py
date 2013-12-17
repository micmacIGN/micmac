#! /usr/bin/python
# -*- coding: utf-8 -*-

##########################################################################
#FDSC v0.9                                                              #
#Fault Displacement Slip-Curve                                           #
#                                                                        #
#Copyright (C) (2013) Ana-Maria Rosu, IPGP-IGN project financed by CNES  #
#am.rosu@laposte.net                                                     #
#                                                                        #
#This software is governed by the CeCILL-B license under French law and  #
#abiding by the rules of distribution of free software.  You can  use,   #
#modify and/ or redistribute the software under the terms of the CeCILL-B#
#license as circulated by CEA, CNRS and INRIA at the following URL       #
#"http://www.cecill.info".                                               #
##########################################################################

import re
import os
import sys

from scipy import *
from pylab import *
from osgeo import gdal
from ConstructLine import *



# ConstructCG
# 4 points (self.pt), to draw before and after part of a gap in a profile
class ConstructCG:
  def __init__(self,filename_Px,filename_weights,im_resol,pow_weight, midLength_stack, midWidth_stack, dist_profCen, stack_calcMethod,fig,ax,profil_absc,profil_ordo,tab_ecart,label_fig,color_fig, xlabel_fig, ylabel_fig, title_fig, name_root_fig, num_stack, num_CenProfile, xCen_stack, yCen_stack, filepath_polyline,filepath_out, showErrBar=False):
    self.coords=[]

    self.filename_imPx=filename_Px
    self.filename_imWeights=filename_weights
    self.resol=im_resol
    self.pow_weight=pow_weight
    self.length_stack=2*midLength_stack+1 ## !!!!
    self.width_stack=2*midWidth_stack+1 ### !!!!
    self.dist_profCen=dist_profCen #number of pixel between central profiles of stacks
    self.stack_calcMethod=stack_calcMethod #weighted median or weighted mean

    self.stack_calcMethod_name=""

    if self.stack_calcMethod.func_name=="calc_profil_mediane_pond" or self.stack_calcMethod.func_name=="calc_profil_mediane_pond_m":
      if len(self.filename_imWeights)==0:
        self.stack_calcMethod_name="median"
      else:
        self.stack_calcMethod_name="weighted median"
    if self.stack_calcMethod.func_name=="calc_profil_coef_correl" or self.stack_calcMethod.func_name=="calc_profil_coef_correl_m" :
      if len(self.filename_imWeights)==0:
        self.stack_calcMethod_name="mean"
      else:
        self.stack_calcMethod_name="weighted mean"

    self.num_stack=num_stack
    self.num_CenProfile=num_CenProfile
    self.colCen_stack=xCen_stack
    self.ligCen_stack=yCen_stack

    self.filepath_poly=filepath_polyline
    self.filepath_out=filepath_out
    self.showErrorBar=showErrBar

    self.profil_absc=profil_absc
    self.profil_ordo=profil_ordo
    self.tab_ecart=tab_ecart
    self.label_fig=label_fig
    self.color_fig=color_fig
    self.xlabel_fig=xlabel_fig
    self.ylabel_fig=ylabel_fig
    self.title_fig=title_fig
    self.nameRoot_fig=name_root_fig

    self.fig=fig
    self.ax=ax
    self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)
    self.pt=[[0,0],[0,0],[0,0],[0,0]] #the 4 points
    self.index_ptModif=-1 #number of the selected point (-1 if none)


    #points initialization
    #print "points initialization"
    #print self.profil_absc
    #print self.profil_ordo
    #self.pt[0][0]=self.profil_absc[0]+0.5
    #self.pt[1][0]=(self.profil_absc[-1]+self.profil_absc[0])/2
    #self.pt[2][0]=(self.profil_absc[-1]+self.profil_absc[0])/2
    #self.pt[3][0]=self.profil_absc[-1]-0.5
    #y_min=min(self.profil_ordo)
    #y_max=max(self.profil_ordo)
    #self.pt[0][1]=y_min+0.01
    #self.pt[1][1]=y_min+0.01
    #self.pt[2][1]=y_max-0.01
    #self.pt[3][1]=y_max-0.01
    #end init

    #point initialization using least squares
    X1,X2=self.leastSq()
    #line 1 equation: y=X1[0][0]*x+X1[1][0]
    #line 2 equation: y=X2[0][0]*x+X2[1][0]

    self.pt[0][0]=self.profil_absc[0]+0.5 #x 1st point
    self.pt[0][1]=X1[0][0]*self.pt[0][0]+X1[1][0] #y 1st point

    self.pt[1][0]=(self.profil_absc[-1]+self.profil_absc[0])/2 #x 2nd point
    self.pt[1][1]=X1[0][0]*self.pt[1][0]+X1[1][0] #y 2nd point

    self.pt[2][0]=(self.profil_absc[-1]+self.profil_absc[0])/2 #x 3rd point
    self.pt[2][1]=X2[0][0]*self.pt[2][0]+X2[1][0] #y 3rd point

    self.pt[3][0]=self.profil_absc[-1]-0.5 #x 4th point
    self.pt[3][1]=X2[0][0]*self.pt[3][0]+X2[1][0] #y 4th point
    #end init

    self.showText='click on a point to change its position'
    self.xText='distance along profile (px)'
    self.yText='offset (px)'
    self.ax.set_title(self.showText)
    self.ax.set_xlabel(self.xText)
    self.ax.set_ylabel(self.yText)

    self.redraw('k')


  def drawLines(self, alpha_s, color_s):
    #~ #draw line before gap
    self.ax.plot((self.pt[0][0],self.pt[1][0]), (self.pt[0][1],self.pt[1][1]), marker='o', color=color_s, alpha=alpha_s)#, picker=1)
    #draw line after gap
    self.ax.plot((self.pt[2][0],self.pt[3][0]), (self.pt[2][1],self.pt[3][1]), marker='o', color=color_s, alpha=alpha_s)#, picker=1)

    self.ax.figure.canvas.draw()

  def redraw(self, color_redraw):
    self.fig.clf()
    self.ax = self.fig.add_subplot(111)
    self.ax.set_title(self.showText)
    self.ax.set_xlabel(self.xText)
    self.ax.set_ylabel(self.yText)
    self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self)
    grid(True)

    if self.showErrorBar:
      self.ax.errorbar(self.profil_absc,self.profil_ordo,self.tab_ecart,label=self.label_fig, color=self.color_fig, ecolor='g')
    else:
      self.ax.plot(self.profil_absc,self.profil_ordo,label=self.label_fig, color=self.color_fig)
    self.drawLines(1.0,color_redraw)
    print "redraw",self.showText

    if self.nameRoot_fig!="":
      #~ self.showText="stack " + " - " + self.stack_calcMethod_name+" method"
      #~ self.ax.set_title(self.showText)
      nameFig=self.nameRoot_fig+'_cenProf'+str(self.num_CenProfile)+'_col'+str(int(round(self.colCen_stack)))+'_lig'+str(int(round(self.ligCen_stack)))
        #cenProf: number of the central profile, (col,lig) - coordinates of the central point on the central profile of the stack
      print nameFig
      savefig(nameFig)


  def redraw_pt(self, pt_x, pt_y):
    self.redraw('black')
    self.ax.plot(pt_x, pt_y, marker='D', color='r', alpha=1.0, picker=True)
    self.ax.figure.canvas.draw()

  def pointToBeModified(self, event):
    index_modif=-1
    for i in range(len(self.pt)):
      #~ if (abs(event.xdata-self.pt[i][0])<=0.5 and abs(event.ydata-self.pt[i][1])<=0.01) :
      if (abs(event.xdata-self.pt[i][0])<=0.8 and abs(event.ydata-self.pt[i][1])<=0.01) :
        print 'in the zone of the point ',i
        self.showText='click to define the new position of the point'
        self.ax.set_xlabel(self.xText)
        self.ax.set_ylabel(self.yText)
        self.redraw_pt(self.pt[i][0], self.pt[i][1])
        index_modif=i
    return index_modif

  def onpick(self,event):
    print 'onpick start'
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    print 'onpick points:', zip(xdata[ind], ydata[ind])


  def estimOffset(self):
   offset_val=self.pt[2][1]-self.pt[1][1]
   #~ print 'offset value: ', offset_val
   return offset_val

  def saveInfosOffset(self, offset_val):
    if os.path.exists(self.filepath_out):
      with open(self.filepath_out, 'a') as file:
        file.write("  {}    {}    {}  {}  {}\n".format(self.num_stack, self.num_CenProfile, self.colCen_stack,self.ligCen_stack, offset_val))
    else:
      #~ print "no ",self.filepath_out
      with open(self.filepath_out, 'w') as file:
        for line in open(self.filepath_poly,'r').readlines():
          file.write(line)
        file.write("\n")
        file.write("#begin info stack\n")
        file.write("#image Px: {}\n".format(self.filename_imPx))
        file.write("#image of weights: {}\n".format(self.filename_imWeights))
        file.write("#image resolution: {}\n".format(self.resol))
        file.write("#method used for stacks: {}\n".format(self.stack_calcMethod_name))
        file.write("#power of weights: {}\n".format(self.pow_weight))
        file.write("#length: {} px\n".format(int(self.length_stack)))
        file.write("#width: {} px\n".format(int(self.width_stack)))
        file.write("#distance between consecutive central profiles for stacks: {} px\n".format(self.dist_profCen))
        file.write("#end info stack\n")
        file.write("\n")
        file.write("#no.stack  no.profile   X(px)   Y(px)   offset(px)\n")
        file.write("  {}    {}    {}  {}  {}\n".format(self.num_stack, self.num_CenProfile, self.colCen_stack,self.ligCen_stack, offset_val))

  #Least Mean Square fonction :
  #   equation y=ax+b
  # x constants (distance along the profile)
  # y observations: parallax values
  # a,b parameters of the line
  # AX=B
  # X=(At*P*A)**(-1)*At*P*B
  def leastSq(self):
    #~ buff=5 #a buffer zone is taken into account when computing LMS - we suppose that when drawing the polilyne (fault), errors may occur on choosing the pixels that define it
    buff=3 #a buffer zone is taken into account when computing LMS - we suppose that when drawing the polilyne (fault), errors may occur on choosing the pixels that define it

    ### for the 1st line segment
    val_absc1=self.profil_absc[:len(self.profil_absc)/2-buff]
    A=[]
    for x in val_absc1:
      A.append((x,1)) # A: design matrix
    Am=matrix(A)
    P=eye(len(val_absc1))*self.tab_ecart[:len(self.profil_absc)/2-buff] # P: weights matrix
    #~ print 'len(Am): ', len(Am)
    #~ print 'tab_ecart: ', self.tab_ecart
    #~ print 'P: ', P
    #~ print len(P)
    B=matrix(self.profil_ordo[:len(self.profil_absc)/2-buff])
    X1=linalg.solve(Am.T*P*Am,Am.T*P*B.T) #parameters vector
    #resid=Am*X-B.T
    v1=B.T-Am*X1 #errors vector
    sig1=sqrt((v1.T*P*v1)/(len(val_absc1)-len(X1)))

    ### for the 2nd line segment
    val_absc2=self.profil_absc[len(self.profil_absc)/2+buff:]
    A=[]
    for x in val_absc2:
      A.append((x,1))
    Am=matrix(A)
    #~ w=self.tab_ecart[len(self.profil_absc)/2+buff:]
    P=eye(len(val_absc2))*self.tab_ecart[len(self.profil_absc)/2+buff:]
    B=matrix(self.profil_ordo[len(self.profil_absc)/2+buff:])
    X2=linalg.solve(Am.T*P*Am,Am.T*P*B.T)
    v2=B.T-Am*X2
    sig2=sqrt((v2.T*P*v2)/(len(val_absc2)-len(X2)))

    return array(X1),array(X2)


  def __call__(self, event):
    print 'click', event
    if event.inaxes!=self.ax: return
    self.fig.canvas.mpl_connect('pick_event', self.onpick)
    if event.button==1:
      print 'click on a point to change its position'
      if (self.index_ptModif==-1): #no point selected
        # try to select a point
        #~ self.coords.append((event.xdata, event.ydata))
        #~ print self.coords
        print 'click button 1'
        self.index_ptModif=self.pointToBeModified(event)
        if self.index_ptModif>=0:
          print 'The point to be modified: ',self.index_ptModif
      else: #one point already selected
        #move the point
        self.showText='click on a point to change its position'
        #self.pt[self.index_ptModif][0]=event.xdata
        self.pt[self.index_ptModif][1]=event.ydata
        self.redraw('black')
        self.index_ptModif=-1
    #~ if event.button==2:
     #~ print 'click button 2 : estimation of offset'
     #~ offset_value=self.estimOffset()
    if event.button==3: #save the offset's value and other info
      offset_value=self.estimOffset()
      print 'Offset value (px):', offset_value
      print 'Save offsets to file'
      self.saveInfosOffset(offset_value)
      close()



#~ if __name__ == '__main__':
  #fig=figure()
  #ax = fig.add_subplot(111)

