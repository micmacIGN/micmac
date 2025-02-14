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

from scipy import sqrt, isnan
import matplotlib.pyplot as plt


def set_range(col_min, col_max, line_min, line_max, marge=20):
  """ set dimensions for window """
  plt.plot([col_min-marge,col_max+marge],[-line_max-marge,-line_min+marge],visible=False)

def set_range_tab(tab_points, marge=20):
  line_min=tab_points[0][1]
  line_max=tab_points[0][1]
  col_min=tab_points[0][0]
  col_max=tab_points[0][0]
  for (col,line) in tab_points:
    if line_min>line:
      line_min=line
    if line_max<line:
      line_max=line
    if col_min>col:
      col_min=col
    if col_max<col:
      col_max=col
  set_range(col_min, col_max, line_min, line_max,marge)
  return (col_min, col_max, line_min, line_max)

def calc_pts_intermed(ptA,ptB,dist,shift):
  """ compute intermediate points between A et B every dist """
  vectAB=(ptB[0]-ptA[0],ptB[1]-ptA[1])
  normAB=sqrt(vectAB[0]**2+vectAB[1]**2)
  nb_pts_intermed=int((normAB+shift)/dist)+1

  tab_pts_intermed=[0]*nb_pts_intermed
  vect_d=(dist*vectAB[0]/normAB,dist*vectAB[1]/normAB)

  pt_start=(ptA[0]-shift*(vectAB[0]/normAB),ptA[1]-shift*(vectAB[1]/normAB))

  for i in range(nb_pts_intermed):
    tab_pts_intermed[i]=(pt_start[0]+i*vect_d[0],pt_start[1]+i*vect_d[1])

  shift_next=normAB+shift-nb_pts_intermed*dist
  return tab_pts_intermed,shift_next

def calc_pts_intermed_tab(tab_pts,dist):
  """ compute intermediate points every dist for all the points in tab_pts """
  pt_prec=tab_pts[0]
  tab_pts_intermed=[]
  shift=0
  for pt in tab_pts[1:]:
    tab,shift=calc_pts_intermed(pt_prec,pt,dist,shift)
    tab_pts_intermed+=tab
    pt_prec=pt
  return tab_pts_intermed

def create_perp(length,ptA,ptB,ptP):
  """ compute start and end points of a perpendicular going through P """
  vectAB=(ptB[0]-ptA[0],ptB[1]-ptA[1])
  normAB=sqrt(vectAB[0]**2+vectAB[1]**2)
  v=(vectAB[0]/normAB,vectAB[1]/normAB) #unit vector of (AB)
  u=(v[1],-v[0]) #perp unit vector
  pt_start=(ptP[0]+length*u[0],ptP[1]+length*u[1])
  pt_end=(ptP[0]-length*u[0],ptP[1]-length*u[1])
  return pt_start,pt_end

def create_all_perp(tab_pts,step,length_profile,step_perp):
  """ create a perpendicular every "step" and the perpendiculars are trimmed every "step_perp" """
  tab_pts_intermed=calc_pts_intermed_tab(tab_pts,step)
  tab_all_perp=[0]*(len(tab_pts_intermed)) #all the perpendiculars (1 per intermediary point of the polyline)

  i=0
  #1st perp
  ptStart,ptEnd=create_perp(length_profile,tab_pts_intermed[0],tab_pts_intermed[1],tab_pts_intermed[0])
  perpendicular=[ptStart,ptEnd]
  tab_pts_intermed_perp=calc_pts_intermed_tab(perpendicular,step_perp)
  tab_all_perp[i]=tab_pts_intermed_perp
  i+=1
  #the rest of the perp
  pt_prec=tab_pts_intermed[0]
  for p in tab_pts_intermed[1:]:
    ptStart,ptEnd=create_perp(length_profile,pt_prec,p,p)
    perpendicular=[ptStart,ptEnd]
    #draw_polyline(perpendicular,1)
    tab_pts_intermed_perp=calc_pts_intermed_tab(perpendicular,step_perp)
    tab_all_perp[i]=tab_pts_intermed_perp
    #draw_polyline(tab_pts_intermed_perp,1)
    pt_prec=p
    i+=1

  return tab_all_perp

def interpol_ppv(data, col, line):
  # compute nearest-neighbor interpolation (with gdal, data[line][column]!!)
  return data[int(round(line))][int(round(col))]

def interpol_bilin(data, col, line):
  """ compute bilinear interpolation (with gdal, data[line][column]!!) """
  line_before=int(line)
  line_after=int(line)+1
  col_before=int(col)
  col_after=int(col)+1
  #start by interpolating in columns
  dist_a_col_before=col-col_before
  dist_a_col_after=1-dist_a_col_before
  #interpolation on "before" line
  val_interp_line_before=data[line_before][col_before]*dist_a_col_after+data[line_before][col_after]*dist_a_col_before
  #interpolation on "after" line
  val_interp_line_after=data[line_after][col_before]*dist_a_col_after+data[line_after][col_after]*dist_a_col_before
  #interpolation in lines
  dist_a_line_before=line-line_before
  dist_a_line_after=1-dist_a_line_before
  #interpolated value
  val_interp_final=val_interp_line_before*dist_a_line_after+val_interp_line_after*dist_a_line_before
  return val_interp_final

def draw_profile_interpol(data, tab_all_perp, num_profile, step_perp, interpol):
  """ plot the perpendicular profile with the interpolated points (giving the array
  with all the points of all the perpendiculars, the number of the perpendicular profile
  wanted, the interpolation functor """
  ordinate_profile=[0]*(len(tab_all_perp[num_profile])) #ordinate: parallax values
  abscissa_profile=[0]*(len(tab_all_perp[num_profile])) #abscissa: distance along profile
  i=0
  for (col,line) in tab_all_perp[num_profile]:
    ordinate_profile[i]=interpol(data,col,line)
    abscissa_profile[i]=i*step_perp
    i+=1
  print("Points of the profile ",num_profile,": ", tab_all_perp[num_profile])
  print("Ordinate : ", ordinate_profile)
  print("Abscissa: ", abscissa_profile)
  plt.plot(abscissa_profile,ordinate_profile)

def calc_profile_coef_correl(data_px1, data_px2,coef_correl, width_mean,tab_all_perp,
                             num_profile, step_perp,interpol, pow_weights, vect_perp,
                             vect_paral, type_output):
  """ compute stack of perpendicular profiles using the weighted mean method
     (mean on 2*width_mean+1 profiles; weights=correlation coefficients)
     vect_perp,vect_paral : unit vectors of the fault
     type_output : 1=px1, 2=px2, 3=parallel, 4=perp """
  ordinate_profile=[0]*(len(tab_all_perp[num_profile])) #ordinate: parallax values
  abscissa_profile=[0]*(len(tab_all_perp[num_profile])) #abscissa: distance along profile
  tab_sig=[0]*(len(tab_all_perp[num_profile])) #sigma in ordinate
  for i in range(len(tab_all_perp[num_profile])): #for every point of the central profile
    #computing the mean value using the "before" and "after" profiles
    values=[] #all the parallax on j
    weights=[] #all the weights on j
    divisor=0
    abscissa_profile[i]=i*step_perp #abscissa of the i point
    for j in range(num_profile-width_mean, num_profile+width_mean+1):#for every profile we search for the i point
      (col,line)=tab_all_perp[j][i]
      if type_output==1:
        if not isnan(interpol(data_px1,col,line)) :
          values.append(interpol(data_px1,col,line))
          weights.append(interpol(coef_correl,col,line))
          ordinate_profile[i]+=values[-1]*weights[-1]**int(pow_weights)
          divisor+=weights[-1]**int(pow_weights)
      if type_output==2:
        if not isnan(interpol(data_px2,col,line)) :
          values.append(interpol(data_px2,col,line))
          weights.append(interpol(coef_correl,col,line))
          ordinate_profile[i]+=values[-1]*weights[-1]**int(pow_weights)
          divisor+=weights[-1]**int(pow_weights)
      if type_output==3:
        if not (isnan(interpol(data_px1,col,line)) or isnan(interpol(data_px2,col,line))) :
          val_px1=interpol(data_px1,col,line)
          val_px2=interpol(data_px2,col,line)
          # total=(val_px1, val_px2) #vector of total parallax
          px_projection_paral=val_px1*vect_paral[0]+val_px2*vect_paral[1]
          values.append(px_projection_paral)
          weights.append(interpol(coef_correl,col,line))
          ordinate_profile[i]+=values[-1]*weights[-1]**int(pow_weights)
          divisor+=weights[-1]**int(pow_weights)
      if type_output==4:
        if not (isnan(interpol(data_px1,col,line)) or isnan(interpol(data_px2,col,line))) :
          val_px1=interpol(data_px1,col,line)
          val_px2=interpol(data_px2,col,line)
          # total=(val_px1, val_px2) #vector of total parallax
          px_projection_perp=val_px1*vect_perp[0]+val_px2*vect_perp[1]
          values.append(px_projection_perp)
          weights.append(interpol(coef_correl,col,line))
          ordinate_profile[i]+=values[-1]*weights[-1]**int(pow_weights)
          divisor+=weights[-1]**int(pow_weights)
    ordinate_profile[i]/=divisor
    #sigma computation: generalized function for absolute deviation (average of abs of value-mean)
    tmp_val_sigma=0
    div_sigma=0
    for j in range(len(values)):
      tmp_val_sigma+=abs(values[j]-ordinate_profile[i])*weights[j]**int(pow_weights)
      div_sigma+=weights[j]**int(pow_weights)
    tab_sig[i]=tmp_val_sigma/div_sigma

  return abscissa_profile, ordinate_profile, tab_sig

def calc_weighted_median(val, weights, pow_weights):
  """ compute weighted median value """
  if len(val)!=len(weights):
    print('Error in weighted median computation!')
    return None
  valp=[] #creating a list (valp) gathering the values (val) and their corresponding weights (weights)
  for i in range(len(val)):
    valp.append([val[i],weights[i]])
  valp.sort() #ordering the values of valp (in place algorithm)
  #~ print('valp: ',valp)
  sum_po=[0]*len(valp)  #cumulated sum of weights
  sum_po[0]=valp[0][1]
  for i in range(1,len(valp)):
    sum_po[i]=valp[i][1]+sum_po[i-1]
  #~ print('sum_po: ', sum_po)
  min_diff_weights=10000000 #initialization
  sp2=sum_po[-1]/2.0
  for i in range(len(sum_po)):
    diff_weights=abs(sum_po[i]-sp2)
    if (diff_weights<=min_diff_weights):
      min_diff_weights=diff_weights
      weighted_median=valp[i][0]
    else:
      weighted_median=valp[i-1][0]
      break
  #sigma computation: generalized function for absolute deviation (average of abs of value-median)
  tmp_val_sigma=0
  div_sigma=0
  for i in range(len(val)):
    tmp_val_sigma+=abs(val[i]-weighted_median)*weights[i]**int(pow_weights)
    div_sigma+=weights[i]**int(pow_weights)
  sigma=tmp_val_sigma/div_sigma
  return weighted_median, sigma

def calc_profile_weighted_median(data_px1, data_px2, coef_correl, width_mean, tab_all_perp,
                                 num_profile,step_perp,interpol, pow_weights, vect_perp,
                                 vect_paral, type_output):
  """ compute stack of perpendicular profiles when using the weighted median method
    (median on 2*width_mean+1 profiles; weights=correlation coefficients)
    vect_perp,vect_paral : unit vectors of the fault
    type_output : 1=px1, 2=px2, 3=parallel, 4=perp """
  ordinate_profile=[0]*(len(tab_all_perp[num_profile])) #ordinate: parallax values
  abscissa_profile=[0]*(len(tab_all_perp[num_profile])) #abscissa: distance along profile
  tab_sig=[0]*(len(tab_all_perp[num_profile])) #sigma in ordinate
  for i in range(len(tab_all_perp[num_profile])): #for every point of the central profile
    values=[] #all the parallax on j
    weights=[] #all the weights on j
    abscissa_profile[i]=i*step_perp #abscissa of the i point
    for j in range(num_profile-width_mean, num_profile+width_mean+1): #for every profile we search for the i point
      (col,line)=tab_all_perp[j][i]
      if type_output==1:
        if not isnan(interpol(data_px1,col,line)) :
          values.append(interpol(data_px1,col,line))
          weights.append(interpol(coef_correl,col,line))
      if type_output==2:
        if not isnan(interpol(data_px2,col,line)):
          values.append(interpol(data_px2,col,line))
          weights.append(interpol(coef_correl,col,line))
      if type_output==3:
        if not (isnan(interpol(data_px1,col,line)) or isnan(interpol(data_px2,col,line))):
          weights.append(interpol(coef_correl,col,line))
          val_px1=interpol(data_px1,col,line)
          val_px2=interpol(data_px2,col,line)
          # total=(val_px1, val_px2) #vector of total parallax
          px_projection_paral=val_px1*vect_paral[0]+val_px2*vect_paral[1]
          values.append(px_projection_paral)
      if type_output==4:
        if not (isnan(interpol(data_px1,col,line)) or isnan(interpol(data_px2,col,line))):
          weights.append(interpol(coef_correl,col,line))
          val_px1=interpol(data_px1,col,line)
          val_px2=interpol(data_px2,col,line)
          # total=(val_px1, val_px2) #vector of total parallax
          #~ print("values: ",val_px1," ",val_px2, " ",total)
          #~ print("vect_perp: ",vect_perp)
          px_projection_perp=val_px1*vect_perp[0]+val_px2*vect_perp[1]
          values.append(px_projection_perp)
    ordinate_profile[i],tab_sig[i]=calc_weighted_median(values,weights,pow_weights)
  return abscissa_profile, ordinate_profile, tab_sig


def calc_absdev_profile(data_px1, data_px2,coef_correl, width_mean, tab_all_perp, num_profile,
                        interpol, mes_tendanceCentrale, pow_weights, vect_perp, vect_paral,
                        type_output):
  """ generalized function for absolute deviation (measurement of the central tendency: weighted
  median or weighted mean)
    vect_perp, vect_paral : unit vectors of the fault
    type_output : 1=px1, 2=px2, 3=parallel, 4=perp """
  tab_sig_profile=[0]*(len(tab_all_perp[num_profile]))
  for i in range(len(tab_all_perp[num_profile])): #for every point of the central profile
    divisor=0
    for j in range(num_profile-width_mean, num_profile+width_mean+1): #for every profile we search for the i point
      (col,line)=tab_all_perp[j][i]
      if type_output==1:
        if not isnan(interpol(data_px1,col,line)) :
          tab_sig_profile[i]+=abs(interpol(data_px1,col,line)-mes_tendanceCentrale[i])*interpol(coef_correl,col,line)**int(pow_weights)
      if type_output==2:
        if not isnan(interpol(data_px2,col,line)):
          tab_sig_profile[i]+=abs(interpol(data_px2,col,line)-mes_tendanceCentrale[i])*interpol(coef_correl,col,line)**int(pow_weights)
      if type_output==3:
        if not (isnan(interpol(data_px1,col,line)) or isnan(interpol(data_px2,col,line))):
          val_px1=interpol(data_px1,col,line)
          val_px2=interpol(data_px2,col,line)
          # total=(val_px1, val_px2) #vector of total parallax
          px_projection_paral=val_px1*vect_paral[0]+val_px2*vect_paral[1]
          px_paral=(px_projection_paral*vect_paral[0],px_projection_paral*vect_paral[1]) #parallel part
          px_paral_norm=sqrt(px_paral[0]**2+px_paral[1]**2)
          tab_sig_profile[i]+=abs(interpol(px_paral_norm,col,line)-mes_tendanceCentrale[i])*interpol(coef_correl,col,line)**int(pow_weights)
      if type_output==4:
        if not (isnan(interpol(data_px1,col,line)) or isnan(interpol(data_px2,col,line))):
          val_px1=interpol(data_px1,col,line)
          val_px2=interpol(data_px2,col,line)
          # total=(val_px1, val_px2) #vector of total parallax
          px_projection_perp=val_px1*vect_perp[0]+val_px2*vect_perp[1]
          px_perp=(px_projection_perp*vect_perp[0],px_projection_perp*vect_perp[1]) #perp part
          px_perp_norm=sqrt(px_perp[0]**2+px_perp[1]**2)
          tab_sig_profile[i]+=abs(interpol(px_perp_norm,col,line)-mes_tendanceCentrale[i])*interpol(coef_correl,col,line)**int(pow_weights)
      divisor+=interpol(coef_correl,col,line)**int(pow_weights)
    tab_sig_profile[i]/=divisor
  return tab_sig_profile

