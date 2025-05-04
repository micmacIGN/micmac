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
import matplotlib.pyplot as plt
from scipy import ones, sqrt
from osgeo import gdal
from perp import create_all_perp, interpol_bilin
from ConstructCG import ConstructCG


def stackPerp(filename_Px1, filename_Px2, filename_weights,im_resolution, pow_weights,
              length_profile, width, dist_profiles, stack_calcMethod,filename_polyline,
              filename_info_out, type_output,label_fig, color_fig, xlabel_fig, ylabel_fig,
              title_fig, basename_fig, showErrBar, showFig=False):
  """ Compute stacks of perpendicular profiles
    params:
      :filename_Px: filename of the disparity image
      :filename_weights: filename of weights (correlation coefficients image)
      :pow_weights: power(exponent) of weights when calculating the weighted median
      :length_profile: the total length of a profile is 2*length_profile+1
      :width: the total width of a stack is 2*width_mean+1
      :dist_profiles: number of pixels between central profiles of stacks
      :stack_calcMethod: defines the function to use for computing the stacks (calc_profile_weighted_median for weighted median 
      or calc_profile_coef_correl for weighted mean)
      :filename_polyline: filename of the polyline which describes the fault
      :filename_info_out: filename out
      :type_output: offsets direction
    information needed for plotting:
      :label_fig: label of the plot
      :color_fig: color of the plot
      :xlabel_fig, ylabel_fig: labels of plot axes
      :title_fig: title of the figure
      :basename_fig: common part of the name when saving figure """
  print("-------------------")
  print("Save offsets to file: ", filename_info_out)
  step_perp=1
  step=1
  length_profile=(length_profile-1)/2+0.0001
  width_mean=int((width-1)/2)
  if (os.path.isfile(filename_info_out)):
    os.remove(filename_info_out)
  if (os.path.isfile(filename_polyline)):
    tab_pts=TraceFromFile(filename_polyline)
    #print("Trace points: ", tab_pts)
    tab_all_perp=create_all_perp(tab_pts, step,length_profile, step_perp)
    #print("Nb of perpendicular profiles: ", len(tab_all_perp))
  if (os.path.isfile(filename_Px1)):
    ds_Px1=gdal.Open(filename_Px1, gdal.GA_ReadOnly)
    nb_col_Px1=ds_Px1.RasterXSize #number of columns
    nb_lines_Px1=ds_Px1.RasterYSize #number of lines
    #nb_b_Px1=ds_Px1.RasterCount #number of bands
    #print('nb_col: ',nb_col_Px1,' nb_lines: ',nb_lines_Px1,' nb_b:',nb_b_Px1)
    data_Px1=ds_Px1.ReadAsArray()#!! attention: gdal data's structure is data[lines][col]
  else:
    print('Error! ', str(filename_Px1), 'does not exist!')
  if (os.path.isfile(filename_Px2)):
    ds_Px2=gdal.Open(filename_Px2, gdal.GA_ReadOnly)
    # nb_col_Px2=ds_Px2.RasterXSize #number of columns
    # nb_lines_Px2=ds_Px2.RasterYSize #number of lines
    #nb_b_Px2=ds_Px2.RasterCount #number of bands
    #print('nb_col: ', nb_col_Px2,' nb_lines: ', nb_lines_Px2,' nb_b:', nb_b_Px2)
    data_Px2=ds_Px2.ReadAsArray() #!! attention: gdal data's structure is data[lines][col]
  else:
    print('Error! ', str(filename_Px2), 'does not exist!')
  if (len((filename_weights))!=0):
    if (os.path.isfile(filename_weights)):
      ds_weights=gdal.Open(filename_weights, gdal.GA_ReadOnly)
      nb_col_weights=ds_weights.RasterXSize #number of columns
      nb_lines_weights=ds_weights.RasterYSize #number of lines
      #nb_b_weights=ds_weights.RasterCount #number of bands
      #print('nb_col_weights: ', nb_col_weights,' nb_lines_weights: ', nb_lines_weights)
      data_weights=ds_weights.ReadAsArray()#!! attention: with gdal, data's structure is data[lines][col]
      if nb_col_weights!=nb_col_Px1 and nb_lines_weights!=nb_lines_Px1:
        print("!!!Different image size for weights ("+str(filename_weights)+") and parallax ("+str(filename_Px1)+")!!!")
    else:
      print('Error! ',str(filename_weights), 'does not exist!')
  else:
    if (os.path.isfile(filename_Px1)): #!!!!!!!!!
      data_weights=ones((nb_lines_Px1,nb_col_Px1))
      #print(len(data_weights), len(data_weights[0]))
  count_profilemean=0
  print(f'DBG {width_mean=}  {len(tab_all_perp)-width_mean=}  {dist_profiles=}')
  for num_profile_central in range(width_mean,len(tab_all_perp)-width_mean,dist_profiles):
    #coord of central point on the central profile of the stack:
    col_cen_num_profile_central=tab_all_perp[num_profile_central][int(length_profile+1)][0]
    line_cen_num_profile_central=tab_all_perp[num_profile_central][int(length_profile+1)][1]
    print("Central profile number:", num_profile_central)
    #compute orientation vector of the fault for this profile
    firstProfilePoint=tab_all_perp[num_profile_central][0]
    lastProfilePoint=tab_all_perp[num_profile_central][-1]
    vectAB=(lastProfilePoint[0]-firstProfilePoint[0],lastProfilePoint[1]-firstProfilePoint[1])
    normAB=sqrt(vectAB[0]**2+vectAB[1]**2)
    u=(vectAB[0]/normAB,vectAB[1]/normAB) #unit vector parallel to fault
    v=(u[1],-u[0]) #unit vector perp to fault
    #~ print("Profile ", num_profile_central,": parall ",u,"; perp ",v)
    fig = plt.figure(1)
    #~ profile_absc,profile_ordo,tab_dist=stack_calcMethod(data_Px1,data_Px2, data_weights,width_mean,tab_all_perp,num_profile_central,step_perp,interpol_bilin,pow_weights, u, v, type_output)
    profile_absc, profile_ordo, tab_dist = stack_calcMethod(data_Px1 * im_resolution,
                                                           data_Px2*im_resolution,
                                                           data_weights, width_mean,
                                                           tab_all_perp, num_profile_central,
                                                           step_perp, interpol_bilin, pow_weights,
                                                           u, v, type_output)
    count_profilemean+=1
    ax = fig.add_subplot(111)
    #~ ax = plt.subplot2grid((1,2),(0, 0))
    #~ print("**** type_output: ", type_output)
    ConstructCG(filename_Px1, filename_Px2, filename_weights, im_resolution, pow_weights,
                length_profile, width_mean, dist_profiles, stack_calcMethod, fig, ax,
                profile_absc, profile_ordo, tab_dist, label_fig,color_fig, xlabel_fig,
                ylabel_fig, title_fig, basename_fig, count_profilemean, num_profile_central,
                col_cen_num_profile_central, line_cen_num_profile_central, filename_polyline,
                filename_info_out,type_output,showFig,showErrBar)

    #~ g=figure(2)
    #~ ds = gdal.Open(filename_Px1, gdal.GA_ReadOnly)
    #~ data_im=ds.ReadAsArray()
    #~ imshow(data_im,cmap=cm.Greys_r, interpolation=None)

    if showFig:
      plt.show()

def TraceFromFile(filepath_trace):
  """ Retrieve the coordinates of the fault (polyline) from file """
  coords_trace=[]
  reading_status=0 #0: begin , 1: getting polyline points, 2: polyline finished
  with open(filepath_trace, 'r', encoding='utf-8') as f:
    for line in f:
      if (line.strip()=="#begin polyline X Y"):
        reading_status=1
        continue
      if (line.strip()=="#end polyline"):
        reading_status=2
        continue
      if (reading_status==1):
        (x,y)=line.split()
        coords_trace.append((float(x),float(y)))
        continue
  return coords_trace


def drawCG(filepath_infoOffsets, nom_fig):
  """ Draw slip-curve """
  nrProf_offsets=[]
  coords_fault=[]
  reading_status=0 #0: begin , 1: getting info on offsets, 2: start getting info on fault polyline, 3: end polyline, 4: getting resolution value
  if (os.path.isfile(filepath_infoOffsets)):
    with open(filepath_infoOffsets, 'r', encoding='utf-8') as finfo:
      for line in finfo:
        #~ if (line.strip()=="#no.stack  no.profile   X(px)   Y(px)   offset(px)"):
        if (line.strip()=="#no.stack  no.profile   X(px)   Y(px)   offset(m)"):
          reading_status=1
          continue
        if (reading_status==1):
          l=line.split()
          nrProf_offsets.append((int(l[1]),float(l[-1])))
          continue
        if (line.strip()=="#begin polyline X Y"):
          reading_status=2
          continue
        if (line.strip()=="#end polyline"):
          reading_status=3
          continue
        if (reading_status==2):
          (x,y)=line.split()
          coords_fault.append((float(x),float(y)))
          continue
        #~ if (line[0:27]=="#image resolution (1px=?m):"):
          #~ resol=float((line[27:]).strip())
          #~ continue
    fig=plt.figure()
    ax = fig.add_subplot(111)
    if len(nrProf_offsets)>0:
      (list_x,list_y)=zip(*nrProf_offsets)
      #~ print('list_y', list_y)
      #~ ym=[]
      #~ for y in list_y:
        #~ ym.append(y*resol)
      #~ ax.plot(list_x,ym,color='k', marker='o')
      ax.plot(list_x,list_y,color='k', marker='o')
    plt.xlabel('distance along the fault (px)')
    plt.ylabel('offset(m)')
    plt.title('Slip-curve')
    plt.savefig(nom_fig)
    plt.show()
