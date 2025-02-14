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

import os
from scipy import array, matrix, linalg, eye
import matplotlib.pyplot as plt


class ConstructCG:
  """ Construct slip-curve class
      4 points (self.pt), to draw before and after part of a dist in a profile """
  def __init__(self,filename_Px1,filename_Px2,filename_weights,im_resol,pow_weight, midLength_stack, midWidth_stack, dist_profCen, stack_calcMethod,fig,ax,profile_absc,profile_ordo,tab_dist,label_fig,color_fig, xlabel_fig, ylabel_fig, title_fig, name_root_fig, num_stack, num_CenProfile, xCen_stack, yCen_stack, filepath_polyline,filepath_out, type_dir_output, need_validation, showErrBar=False):
    self.need_valid=need_validation
    self.coords=[]
    self.filename_imPx1=filename_Px1
    self.filename_imPx2=filename_Px2
    self.filename_imWeights=filename_weights
    self.resol=im_resol
    self.pow_weight=pow_weight
    self.length_stack=2*midLength_stack+1
    self.width_stack=2*midWidth_stack+1
    self.dist_profCen=dist_profCen #number of pixels between central profiles of stacks
    self.stack_calcMethod=stack_calcMethod #mean, median,weighted median or weighted mean
    self.stack_calcMethod_name=""
    if self.stack_calcMethod.__name__=="calc_profile_weighted_median": #or self.stack_calcMethod.__name__=="calc_profile_weighted_median_m":
      if len(self.filename_imWeights)==0:
        self.stack_calcMethod_name="median"
      else:
        self.stack_calcMethod_name="weighted median"
    if self.stack_calcMethod.__name__=="calc_profile_coef_correl": #or self.stack_calcMethod.__name__=="calc_profile_coef_correl_m":
      if len(self.filename_imWeights)==0:
        self.stack_calcMethod_name="mean"
      else:
        self.stack_calcMethod_name="weighted mean"
    self.num_stack=num_stack
    self.num_CenProfile=num_CenProfile
    self.colCen_stack=xCen_stack
    self.lineCen_stack=yCen_stack
    self.filepath_poly=filepath_polyline
    self.filepath_out=filepath_out
    self.showErrorBar=showErrBar
    self.type_dirStack=type_dir_output #stacks offsets direction: col, lines, parallel, perpendicular direction
    self.type_dirStack_name=("","column","line","parallel","perpendicular")[self.type_dirStack]
    #~ print("** type_dirStack_name:", self.type_dirStack_name)
    self.profile_absc=profile_absc
    self.profile_ordo=profile_ordo
    self.tab_dist=tab_dist
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
    #point initialization using least squares
    X1,X2=self.leastSq()
    #~ print(f"{X1=}, {X2=}")
    #line 1 equation: y=X1[0][0]*x+X1[1][0]
    #line 2 equation: y=X2[0][0]*x+X2[1][0]
    self.pt[0][0]=self.profile_absc[0]+0.5 #x 1st point
    self.pt[0][1]=X1[0][0]*self.pt[0][0]+X1[1][0] #y 1st point
    #~ self.pt[1][0]=(self.profile_absc[-1]+self.profile_absc[0])/2 #x 2nd point
    self.pt[1][0]=(self.profile_absc[-1]+self.profile_absc[0])*0.5 #x 2nd point
    self.pt[1][1]=X1[0][0]*self.pt[1][0]+X1[1][0] #y 2nd point
    #~ self.pt[2][0]=(self.profile_absc[-1]+self.profile_absc[0])/2 #x 3rd point
    self.pt[2][0]=(self.profile_absc[-1]+self.profile_absc[0])*0.5 #x 3rd point
    self.pt[2][1]=X2[0][0]*self.pt[2][0]+X2[1][0] #y 3rd point
    self.pt[3][0]=self.profile_absc[-1]-0.5 #x 4th point
    self.pt[3][1]=X2[0][0]*self.pt[3][0]+X2[1][0] #y 4th point
    self.showText='click on a point to change its position'
    self.xText='distance along profile (px)'
    self.yText='offset (m)'
    self.ax.set_title(self.showText)
    self.ax.set_xlabel(self.xText)
    self.ax.set_ylabel(self.yText)

    self.redraw('k')
    if (not self.need_valid):
      self.saveInfoOffset()
      plt.close()

  def drawLines(self, alpha_s, color_s):
    """ Draw lines along slip """
    #draw line before dist
    self.ax.plot((self.pt[0][0],self.pt[1][0]), (self.pt[0][1],self.pt[1][1]), marker='o', color=color_s, alpha=alpha_s)#, picker=1)
    #draw line after dist
    self.ax.plot((self.pt[2][0],self.pt[3][0]), (self.pt[2][1],self.pt[3][1]), marker='o', color=color_s, alpha=alpha_s)#, picker=1)
    self.ax.figure.canvas.draw()

  def drawCenLine(self):
    """ Draw center line in plot """
    #~ self.ax.vlines(self.length_stack*0.5, min(self.profile_ordo), max(self.profile_ordo), color='b', linestyles='dashed')
    #~ self.ax.vlines(int(self.length_stack*0.5), min(self.profile_ordo), max(self.profile_ordo), color='b', linestyles='dashed')
    self.ax.axvline(int(self.length_stack*0.5), color='b', ls='--')
    self.ax.figure.canvas.draw()

  def redraw(self, color_redraw):
    """ Redraw & save plot """
    self.fig.clf()
    self.ax = self.fig.add_subplot(111)
    self.ax.set_title(self.showText)
    self.ax.set_xlabel(self.xText)
    self.ax.set_ylabel(self.yText)
    self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self)
    plt.grid(True)
    if self.showErrorBar:
      self.ax.errorbar(self.profile_absc,self.profile_ordo,self.tab_dist,label=self.label_fig, color=self.color_fig, ecolor='g')
    else:
      self.ax.plot(self.profile_absc,self.profile_ordo,label=self.label_fig, color=self.color_fig)
    self.drawLines(1.0,color_redraw)
    self.drawCenLine()
    print("redraw",self.showText)
    if self.nameRoot_fig!="":
      #self.showText="stack " + " - " + self.stack_calcMethod_name+" method"
      #self.ax.set_title(self.showText)
      nameFig=self.nameRoot_fig+'_cenProf'+str(self.num_CenProfile)+'_col'+str(int(round(self.colCen_stack)))+'_lines'+str(int(round(self.lineCen_stack)))+'.svg'
        #cenProf: number of the central profile, (col,lines) - coordinates of the central point on the central profile of the stack
      print("Figure: ", nameFig)
      plt.savefig(nameFig)

  def redraw_pt(self, pt_x, pt_y):
    """ Redraw point """
    self.redraw('black')
    self.ax.plot(pt_x, pt_y, marker='D', color='r', alpha=1.0, picker=True)
    self.ax.figure.canvas.draw()

  def pointToBeModified(self, event):
    """ Change position of point """
    index_modif=-1
    for i in range(len(self.pt)):
      if self.resol<=1:
        buff_y=0.05
      else:
        buff_y=0.01
      if (abs(event.xdata-self.pt[i][0])<=0.8 and abs(event.ydata-self.pt[i][1])<=buff_y):
        print('in the zone of the point ', i)
        self.showText='click to define the new position of the point'
        self.ax.set_xlabel(self.xText)
        self.ax.set_ylabel(self.yText)
        self.redraw_pt(self.pt[i][0], self.pt[i][1])
        index_modif=i
    return index_modif

  def onpick(self,event):
    """ Pick points """
    print('onpick start')
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    print('onpick points:', zip(xdata[ind], ydata[ind]))

  def estimOffset(self):
   offset_val=self.pt[2][1]-self.pt[1][1]
   #print('offset value: ', offset_val)
   return offset_val

  def saveInfoPlot(self, fich):
    """ Write plot information into file """
    fich.write(f"#begin info plot for stack nb.{self.num_stack}\n")
    for x,y in zip(self.profile_absc, self.profile_ordo):
      fich.write(f"{x} {y}\n")
    fich.write("#end info plot\n")

  def saveInfoOffset(self):
    """ Write offsets information into file """
    offset_val=self.estimOffset()
    print("Offsets direction: ", self.type_dirStack_name)
    print('Offset value (m):', offset_val)
    print('Save offsets to file:',self.filepath_out)
    if os.path.isfile(self.filepath_out): # append to existing file
      with open(self.filepath_out, 'a', encoding='utf-8') as file:
        file.write(f"  {self.num_stack}    {self.num_CenProfile}    {self.colCen_stack}  {self.lineCen_stack}  {offset_val}\n")
        #~ self.saveInfoPlot(file)
    else: # create file
      with open(self.filepath_out, 'w', encoding='utf-8') as file:
        file.write(f"#fault trace file '{self.filepath_poly}'\n")
        with open(self.filepath_poly, 'r', encoding='utf-8') as fpoly:
          for line in fpoly:
            file.write(line)
          file.write("\n")
          file.write("#begin info stack\n")
          file.write(f"#image Px1: '{self.filename_imPx1}'\n")
          file.write(f"#image Px2: {self.filename_imPx2}\n")
          file.write(f"#image of weights: '{self.filename_imWeights}'\n")
          file.write(f"#image resolution (1px=?m): {self.resol}\n")
          file.write(f"#method used for stacks: {self.stack_calcMethod_name}\n")
          file.write(f"#power of weights: {self.pow_weight}\n")
          file.write(f"#length: {int(self.length_stack)} px\n")
          file.write(f"#width: {int(self.width_stack)} px\n")
          file.write(f"#distance between consecutive central profiles for stacks: {self.dist_profCen} px\n")
          file.write(f"#offsets direction: {self.type_dirStack_name}\n")
          file.write("#end info stack\n\n")
          file.write("#no.stack  no.profile   X(px)   Y(px)   offset(m)\n")
          file.write(f"  {self.num_stack}    {self.num_CenProfile}    {self.colCen_stack}  {self.lineCen_stack}  {offset_val}\n")
        #~ self.saveInfoPlot(file)

  def leastSq(self):
    """ Least Mean Square function:
       Equation y=ax+b, where
     :x: constants (distance along the profile)
     :y: observations: parallax values
     :a,b: parameters of the line

        AX=B
        X=(At*P*A)**(-1)*At*P*B

    A buffer zone is taken into account when computing LMS, as we suppose that,when drawing
    the polilyne (fault), errors may occur on choosing the pixels that define it
    """
    #~ buff=5
    buff=3
    ### for the 1st line segment
    if (len(self.profile_absc)<=9):
      buff=2
    if (len(self.profile_absc)<=7):
      buff=1
    if (len(self.profile_absc)<=5):
      buff=0
    if (len(self.profile_absc)<=3):
      X1=[[0],[self.profile_ordo[0]]]
      X2=[[0],[self.profile_ordo[-1]]]
      return array(X1),array(X2)

    val_absc1=self.profile_absc[:int(len(self.profile_absc)/2-buff)]
    #~ print("val_absc1,len(self.profile_absc),buff,self.profile_absc: ",val_absc1,len(self.profile_absc),buff,self.profile_absc)
    A=[]
    for x in val_absc1:
      A.append((x,1)) # A: design matrix
    Am=matrix(A)
    #~ print("A1:", Am)
    P=eye(len(val_absc1))#*self.tab_dist[:len(self.profile_absc)/2-buff] # P: weights matrix
    #~ print('len(Am): ', len(Am))
    #~ print('tab_dist: ', self.tab_dist)
    #~ print('P1: ', P)
    #~ print(len(P))
    B=matrix(self.profile_ordo[:int(len(self.profile_absc)/2-buff)])
    #~ print("B1: ", B)
    #~ print("A.T*P*A: ",Am.T*P*Am)
    X1=linalg.solve(Am.T*P*Am,Am.T*P*B.T) #parameters vector
    # resid=Am*X-B.T
    # v1=B.T-Am*X1 #errors vector
    #~ sig1=sqrt((v1.T*P*v1)/(len(val_absc1)-len(X1)))
    #~ print("sig1: ", sig1[0][0])
    ### for the 2nd line segment
    val_absc2=self.profile_absc[int((len(self.profile_absc)+1)/2+buff):]
    #~ print("val_absc2,len(self.profile_absc),buff,self.profile_absc: ",val_absc2,len(self.profile_absc),buff,self.profile_absc)
    A=[]
    for x in val_absc2:
      A.append((x,1))
    Am=matrix(A)
    #~ w=self.tab_dist[len(self.profile_absc)/2+buff:]
    P=eye(len(val_absc2))#*self.tab_dist[(len(self.profile_absc)+1)/2+buff:]
    B=matrix(self.profile_ordo[int((len(self.profile_absc)+1)/2+buff):])
    #~ print("A2:", Am)
    #~ print()'P2: ', P)
    #~ print("B2: ", B)
    #~ print("Am.T*P*Am:", Am.T*P*Am)
    #~ print("Am.T*P*B.T:", Am.T*P*B.T)

    X2=linalg.solve(Am.T*P*Am,Am.T*P*B.T)
    # v2=B.T-Am*X2
    #~ sig2=sqrt((v2.T*P*v2)/(len(val_absc2)-len(X2)))
    #~ print("sig2: ", sig2[0][0])
    return array(X1),array(X2)

  def __call__(self, event):
    print('click', event)
    if event.inaxes!=self.ax: return
    self.fig.canvas.mpl_connect('pick_event', self.onpick)
    if event.button==1:
      print('click on a point to change its position')
      if (self.index_ptModif==-1): #no point selected
        # try to select a point
        print('click button 1')
        self.index_ptModif=self.pointToBeModified(event)
        if self.index_ptModif>=0:
          print('The point to be modified: ', self.index_ptModif)
      else: #one point already selected
        #move the point
        self.showText='click on a point to change its position'
        #self.pt[self.index_ptModif][0]=event.xdata
        self.pt[self.index_ptModif][1]=event.ydata
        self.redraw('black')
        self.index_ptModif=-1
    if event.button==3: #right-click to save the offset's value and other info
      self.saveInfoOffset()
      plt.close()
