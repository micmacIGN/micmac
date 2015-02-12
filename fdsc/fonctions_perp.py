#! /usr/bin/python
# -*- coding: utf-8 -*-

##########################################################################
# Fault Displacement Slip-Curve (FDSC) v1.0                              #
#                                                                        #
# Copyright (C) (2013-2014) Ana-Maria Rosu am.rosu@laposte.net           #
# IPGP-ENSG/IGN project financed by TOSCA/CNES                           #
#                                                                        #
#                                                                        #
# This software is governed by the CeCILL-B license under French law and #
#abiding by the rules of distribution of free software.  You can  use,   #
#modify and/ or redistribute the software under the terms of the CeCILL-B#
#license as circulated by CEA, CNRS and INRIA at the following URL       #
#"http://www.cecill.info".                                               #
##########################################################################

from scipy import *
from pylab import *


#set dimensions for window
def set_range(col_min, col_max, lig_min, lig_max, marge=20):
  plt.plot([col_min-marge,col_max+marge],[-lig_max-marge,-lig_min+marge],visible=False)

def set_range_tab(tab_points, marge=20):
  lig_min=tab_points[0][1]
  lig_max=tab_points[0][1]
  col_min=tab_points[0][0]
  col_max=tab_points[0][0]
  for (col,lig) in tab_points:
    if lig_min>lig:
      lig_min=lig
    if lig_max<lig:
      lig_max=lig
    if col_min>col:
      col_min=col
    if col_max<col:
      col_max=col
  set_range(col_min, col_max, lig_min, lig_max,marge)
  return (col_min, col_max, lig_min, lig_max)


#compute intermediate points between A et B every dist
def calc_pts_intermed(ptA,ptB,dist,decal):
  vectAB=(ptB[0]-ptA[0],ptB[1]-ptA[1])
  normeAB=sqrt(vectAB[0]**2+vectAB[1]**2)
  nb_pts_intermed=int((normeAB+decal)/dist)+1

  tab_pts_intermed=[0]*nb_pts_intermed
  vect_d=(dist*vectAB[0]/normeAB,dist*vectAB[1]/normeAB)

  pt_debut=(ptA[0]-decal*(vectAB[0]/normeAB),ptA[1]-decal*(vectAB[1]/normeAB))

  for i in range(nb_pts_intermed):
    tab_pts_intermed[i]=(pt_debut[0]+i*vect_d[0],pt_debut[1]+i*vect_d[1])

  decal_suiv=normeAB+decal-nb_pts_intermed*dist
  return tab_pts_intermed,decal_suiv

#compute intermediate points every dist for all the points in tab_pts
def calc_pts_intermed_tab(tab_pts,dist):
  pt_prec=tab_pts[0]
  tab_pts_intermed=[]
  decal=0
  for pt in tab_pts[1:]:
    tab,decal=calc_pts_intermed(pt_prec,pt,dist,decal)
    tab_pts_intermed+=tab
    pt_prec=pt
  return tab_pts_intermed


#compute start and end points of a perpendicular going through P
def create_perp(longueur,ptA,ptB,ptP):
  vectAB=(ptB[0]-ptA[0],ptB[1]-ptA[1])
  normeAB=sqrt(vectAB[0]**2+vectAB[1]**2)
  v=(vectAB[0]/normeAB,vectAB[1]/normeAB) #unit vector of (AB)
  u=(v[1],-v[0]) #perp unit vector
  pt_deb=(ptP[0]+longueur*u[0],ptP[1]+longueur*u[1])
  pt_fin=(ptP[0]-longueur*u[0],ptP[1]-longueur*u[1])
  return pt_deb,pt_fin

#create a perpendicular every "pas_droit" and the perpendiculars are trimmed every "pas_perp"
def create_all_perp(tab_pts,pas_droit,longueur_profil,pas_perp):
  tab_pts_intermed=calc_pts_intermed_tab(tab_pts,pas_droit)
  tab_toutes_perp=[0]*(len(tab_pts_intermed)) #all the perpendiculars (1 per intermediary point of the polyline)

  i=0
  #1st perp
  ptDeb,ptFin=create_perp(longueur_profil,tab_pts_intermed[0],tab_pts_intermed[1],tab_pts_intermed[0])
  perprendiculaire=[ptDeb,ptFin]
  tab_pts_intermed_perp=calc_pts_intermed_tab(perprendiculaire,pas_perp)
  tab_toutes_perp[i]=tab_pts_intermed_perp
  i+=1
  #the rest of the perp
  pt_prec=tab_pts_intermed[0]
  for p in tab_pts_intermed[1:]:
    ptDeb,ptFin=create_perp(longueur_profil,pt_prec,p,p)
    perprendiculaire=[ptDeb,ptFin]
    #dessine_polylig(perprendiculaire,1)
    tab_pts_intermed_perp=calc_pts_intermed_tab(perprendiculaire,pas_perp)
    tab_toutes_perp[i]=tab_pts_intermed_perp
    #dessine_polylig(tab_pts_intermed_perp,1)
    pt_prec=p
    i+=1

  return tab_toutes_perp

#compute nearest-neighbor interpolation (with gdal, data[line][column]!!)
def interpol_ppv(data,col,lig):
  return data[int(round(lig))][int(round(col))]

##compute bilinear interpolation (with gdal, data[line][column]!!)
def interpol_bilin(data,col,lig):
  lig_avant=int(lig)
  lig_apres=int(lig)+1
  col_avant=int(col)
  col_apres=int(col)+1
  #start by interpolating in columns
  dist_a_col_avant=col-col_avant
  dist_a_col_apres=1-dist_a_col_avant
  #interpolation on "before" line
  val_interp_lig_avant=data[lig_avant][col_avant]*dist_a_col_apres+data[lig_avant][col_apres]*dist_a_col_avant
  #interpolation on "after" line
  val_interp_lig_apres=data[lig_apres][col_avant]*dist_a_col_apres+data[lig_apres][col_apres]*dist_a_col_avant
  #interpolation in lines
  dist_a_lig_avant=lig-lig_avant
  dist_a_lig_apres=1-dist_a_lig_avant
  #interpolated value
  val_interp_finale=val_interp_lig_avant*dist_a_lig_apres+val_interp_lig_apres*dist_a_lig_avant
  return val_interp_finale


#plot the perpendicular profile with the interpolated points (giving the array with all the points of all the perpendiculars, the number of the perpendicular profile wanted, the interpolation functor
def dessine_profil_interpol(data,tab_toutes_perp,num_profil,pas_perp,interpol):
  ordonnees_profil=[0]*(len(tab_toutes_perp[num_profil])) #ordinate: parallax values
  abscisses_profil=[0]*(len(tab_toutes_perp[num_profil])) #abscissa: distance along profile
  i=0
  for (col,lig) in tab_toutes_perp[num_profil]:
    ordonnees_profil[i]=interpol(data,col,lig)
    abscisses_profil[i]=i*pas_perp
    i+=1
  print "Points of the profile ",num_profil,": ", tab_toutes_perp[num_profil]
  print "Ordinate : ", ordonnees_profil
  print "Abscissa: ", abscisses_profil
  plot(abscisses_profil,ordonnees_profil)


#compute stack of perpendicular profiles using the weighted mean method (mean on 2*larg_moy+1 profiles; weights=correlation coefficients)
#vect_perp,vect_paral : unit vectors of the fault
#type_output : 1=px1, 2=px2, 3=parallel, 4=perp
def calc_profil_coef_correl(data_px1, data_px2,coef_correl,larg_moy,tab_toutes_perp,num_profil,pas_perp,interpol, puiss_poids, vect_perp,vect_paral, type_output):
  ordonnees_profil=[0]*(len(tab_toutes_perp[num_profil])) #ordinate: parallax values
  abscisses_profil=[0]*(len(tab_toutes_perp[num_profil])) #abscissa: distance along profile
  tab_ecarts=[0]*(len(tab_toutes_perp[num_profil])) #sigma in ordinate
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    #computing the mean value using the "before" and "after" profiles
    valeurs=[] #all the parallax on j
    poids=[] #all the weights on j
    diviseur=0
    abscisses_profil[i]=i*pas_perp #abscissa of the i point
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1):#for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      if type_output==1:
        if not isnan(interpol(data_px1,col,lig)) :
          valeurs.append(interpol(data_px1,col,lig))
          poids.append(interpol(coef_correl,col,lig))
          ordonnees_profil[i]+=valeurs[-1]*poids[-1]**int(puiss_poids)
          diviseur+=poids[-1]**int(puiss_poids)
      if type_output==2:
        if not isnan(interpol(data_px2,col,lig)) :
          valeurs.append(interpol(data_px2,col,lig))
          poids.append(interpol(coef_correl,col,lig))
          ordonnees_profil[i]+=valeurs[-1]*poids[-1]**int(puiss_poids)
          diviseur+=poids[-1]**int(puiss_poids)
      if type_output==3:
        if not (isnan(interpol(data_px1,col,lig)) or isnan(interpol(data_px2,col,lig))) :
          val_px1=interpol(data_px1,col,lig)
          val_px2=interpol(data_px2,col,lig)
          total=(val_px1, val_px2) #vector of total parallax
          px_projection_paral=val_px1*vect_paral[0]+val_px2*vect_paral[1]
          valeurs.append(px_projection_paral)
          poids.append(interpol(coef_correl,col,lig))
          ordonnees_profil[i]+=valeurs[-1]*poids[-1]**int(puiss_poids)
          diviseur+=poids[-1]**int(puiss_poids)
      if type_output==4:
        if not (isnan(interpol(data_px1,col,lig)) or isnan(interpol(data_px2,col,lig))) :
          val_px1=interpol(data_px1,col,lig)
          val_px2=interpol(data_px2,col,lig)
          total=(val_px1, val_px2) #vector of total parallax
          px_projection_perp=val_px1*vect_perp[0]+val_px2*vect_perp[1]
          valeurs.append(px_projection_perp)
          poids.append(interpol(coef_correl,col,lig))
          ordonnees_profil[i]+=valeurs[-1]*poids[-1]**int(puiss_poids)
          diviseur+=poids[-1]**int(puiss_poids)
    ordonnees_profil[i]/=diviseur
    #sigma computation: generalized function for absolute deviation (average of abs of value-mean)
    tmp_val_sigma=0
    div_sigma=0
    for j in range(len(valeurs)):
      tmp_val_sigma+=abs(valeurs[j]-ordonnees_profil[i])*poids[j]**int(puiss_poids)
      div_sigma+=poids[j]**int(puiss_poids)
    tab_ecarts[i]=tmp_val_sigma/div_sigma

  return abscisses_profil,ordonnees_profil,tab_ecarts


#compute weighted median value
def calc_mediane_pond(val,poids,puiss_poids):
  if len(val)!=len(poids):
    print 'Error in weighted median computation!'
    return None
  valp=[] #creating a list (valp) gathering the values (val) and their corresponding weights (poids)
  for i in range(len(val)):
    valp.append([val[i],poids[i]])
  valp.sort() #ordering the values of valp (in place algorithm)
  #~ print 'valp: ',valp
  sum_po=[0]*len(valp)  #cumulated sum of weights
  sum_po[0]=valp[0][1]
  for i in range(1,len(valp)):
    sum_po[i]=valp[i][1]+sum_po[i-1]
  #~ print 'sum_po: ', sum_po
  min_diff_poids=10000000 #initialization
  sp2=sum_po[-1]/2.0
  for i in range(len(sum_po)):
    diff_poids=abs(sum_po[i]-sp2)
    if (diff_poids<=min_diff_poids):
      min_diff_poids=diff_poids
      mediane_pond=valp[i][0]
    else:
      mediane_pond=valp[i-1][0]
      break
  #sigma computation: generalized function for absolute deviation (average of abs of value-median)
  tmp_val_sigma=0
  div_sigma=0
  for i in range(len(val)):
    tmp_val_sigma+=abs(val[i]-mediane_pond)*poids[i]**int(puiss_poids)
    div_sigma+=poids[i]**int(puiss_poids)
  sigma=tmp_val_sigma/div_sigma
  return mediane_pond, sigma


#compute stack of perpendicular profiles when using the weighted median method (median on 2*larg_moy+1 profiles; weights=correlation coefficients)
#vect_perp,vect_paral : unit vectors of the fault
#type_output : 1=px1, 2=px2, 3=parallel, 4=perp
def calc_profil_mediane_pond(data_px1, data_px2, coef_correl,larg_moy,tab_toutes_perp,num_profil,pas_perp,interpol, puiss_poids, vect_perp,vect_paral, type_output):
  ordonnees_profil=[0]*(len(tab_toutes_perp[num_profil])) #ordinate: parallax values
  abscisses_profil=[0]*(len(tab_toutes_perp[num_profil])) #abscissa: distance along profile
  tab_ecarts=[0]*(len(tab_toutes_perp[num_profil])) #sigma in ordinate
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    valeurs=[] #all the parallax on j
    poids=[] #all the weights on j
    abscisses_profil[i]=i*pas_perp #abscissa of the i point
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1): #for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      if type_output==1:
        if not isnan(interpol(data_px1,col,lig)) :
          valeurs.append(interpol(data_px1,col,lig))
          poids.append(interpol(coef_correl,col,lig))
      if type_output==2:
        if not isnan(interpol(data_px2,col,lig)):
          valeurs.append(interpol(data_px2,col,lig))
          poids.append(interpol(coef_correl,col,lig))
      if type_output==3:
        if not (isnan(interpol(data_px1,col,lig)) or isnan(interpol(data_px2,col,lig))):
          poids.append(interpol(coef_correl,col,lig))
          valeurs_px1=interpol(data_px1,col,lig)
          valeurs_px2=interpol(data_px2,col,lig)
          total=(valeurs_px1, valeurs_px2) #vector of total parallax
          px_projection_paral=valeurs_px1*vect_paral[0]+valeurs_px2*vect_paral[1]
          valeurs.append(px_projection_paral)
      if type_output==4:
        if not (isnan(interpol(data_px1,col,lig)) or isnan(interpol(data_px2,col,lig))):
          poids.append(interpol(coef_correl,col,lig))
          valeurs_px1=interpol(data_px1,col,lig)
          valeurs_px2=interpol(data_px2,col,lig)
          total=(valeurs_px1, valeurs_px2) #vector of total parallax
          #~ print "valeurs: ",valeurs_px1," ",valeurs_px2, " ",total
          #~ print "vect_perp: ",vect_perp
          px_projection_perp=valeurs_px1*vect_perp[0]+valeurs_px2*vect_perp[1]
          valeurs.append(px_projection_perp)
    ordonnees_profil[i],tab_ecarts[i]=calc_mediane_pond(valeurs,poids,puiss_poids)
  return abscisses_profil,ordonnees_profil,tab_ecarts


#generalized function for absolute deviation (measurement of the central tendency: weighted median or weighted mean)
#vect_perp,vect_paral : unit vectors of the fault
#type_output : 1=px1, 2=px2, 3=parallel, 4=perp
def calc_ecartMoy_profil(data_px1, data_px2,coef_correl,larg_moy,tab_toutes_perp,num_profil,interpol,mes_tendanceCentrale, puiss_poids, vect_perp,vect_paral, type_output):
  tab_ecartMoy_profil=[0]*(len(tab_toutes_perp[num_profil]))
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    diviseur=0
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1): #for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      if type_output==1:
        if not isnan(interpol(data_px1,col,lig)) :
          tab_ecartMoy_profil[i]+=abs(interpol(data_px1,col,lig)-mes_tendanceCentrale[i])*interpol(coef_correl,col,lig)**int(puiss_poids)
      if type_output==2:
        if not isnan(interpol(data_px2,col,lig)):
          tab_ecartMoy_profil[i]+=abs(interpol(data_px2,col,lig)-mes_tendanceCentrale[i])*interpol(coef_correl,col,lig)**int(puiss_poids)
      if type_output==3:
        if not (isnan(interpol(data_px1,col,lig)) or isnan(interpol(data_px2,col,lig))):
          val_px1=interpol(data_px1,col,lig)
          val_px2=interpol(data_px2,col,lig)
          total=(val_px1, val_px2) #vector of total parallax
          px_projection_paral=val_px1*vect_paral[0]+val_px2*vect_paral[1]
          px_paral=(px_projection_paral*vect_paral[0],px_projection_paral*vect_paral[1]) #parallel part
          px_paral_norm=sqrt(px_paral[0]**2+px_paral[1]**2)
          tab_ecartMoy_profil[i]+=abs(interpol(px_paral_norm,col,lig)-mes_tendanceCentrale[i])*interpol(coef_correl,col,lig)**int(puiss_poids)
      if type_output==4:
        if not (isnan(interpol(data_px1,col,lig)) or isnan(interpol(data_px2,col,lig))):
          val_px1=interpol(data_px1,col,lig)
          val_px2=interpol(data_px2,col,lig)
          total=(val_px1, val_px2) #vector of total parallax
          px_projection_perp=val_px1*vect_perp[0]+val_px2*vect_perp[1]
          px_perp=(px_projection_perp*vect_perp[0],px_projection_perp*vect_perp[1]) #perp part
          px_perp_norm=sqrt(px_perp[0]**2+px_perp[1]**2)
          tab_ecartMoy_profil[i]+=abs(interpol(px_perp_norm,col,lig)-mes_tendanceCentrale[i])*interpol(coef_correl,col,lig)**int(puiss_poids)
      diviseur+=interpol(coef_correl,col,lig)**int(puiss_poids)
    tab_ecartMoy_profil[i]/=diviseur
  return tab_ecartMoy_profil


