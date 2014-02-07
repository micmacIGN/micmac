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
  v=(vectAB[0]/normeAB,vectAB[1]/normeAB) #vecteur unitaire de (AB)
  u=(v[1],-v[0]) #vect unitaire perp
  pt_deb=(ptP[0]+longueur*u[0],ptP[1]+longueur*u[1])
  pt_fin=(ptP[0]-longueur*u[0],ptP[1]-longueur*u[1])
  return pt_deb,pt_fin

#create a perpendicular every "pas_droit" et the perpendiculars are trimmed every "pas_perp"
#a perpendicular = (x,y) array
#tab_toutes_perp[0] #all the points of the 1st perp
#tab_toutes_perp[0][0] #1st point of the 1st perp
#tab_toutes_perp[0][0][0] #x of the 1st point of the 1st perp
def create_all_perp(tab_pts,pas_droit,longueur_profil,pas_perp):
  tab_pts_intermed=calc_pts_intermed_tab(tab_pts,pas_droit)
  #dessine_polylig(tab_pts_intermed,0.5) # draw polyline

  #all the perpendiculars (1 per intermidiary point of the polyline)
  tab_toutes_perp=[0]*(len(tab_pts_intermed))

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
def calc_profil_coef_correl(data,coef_correl,larg_moy,tab_toutes_perp,num_profil,pas_perp,interpol, puiss_poids):
  ordonnees_profil=[0]*(len(tab_toutes_perp[num_profil])) #ordinate: parallax values
  abscisses_profil=[0]*(len(tab_toutes_perp[num_profil])) #abscissa: distance along profile
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    #computing the mean value using the "before" and "after" profiles
    diviseur=0
    abscisses_profil[i]=i*pas_perp #abscissa of the i point
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1):#for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      if not isnan(interpol(data,col,lig)) :
        ordonnees_profil[i]+=interpol(data,col,lig)*interpol(coef_correl,col,lig)**int(puiss_poids)
        diviseur+=interpol(coef_correl,col,lig)**int(puiss_poids)
    ordonnees_profil[i]/=diviseur
  return abscisses_profil,ordonnees_profil


#compute sigmas of a stack of perpendicular profiles when using weighted mean (mean on 2*larg_moy+1 profiles; weights=correlation coefficients)
def calc_sigma_profil_coef_correl(data,coef_correl,larg_moy,tab_toutes_perp,num_profil,interpol,moyenne_profil_perp, puiss_poids) :
  tab_sigma_profil=[0]*(len(tab_toutes_perp[num_profil]))
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    diviseur=0
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1): #for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      tab_sigma_profil[i]+=((interpol(data,col,lig)-moyenne_profil_perp[i])**2)*interpol(coef_correl,col,lig)**int(puiss_poids)
      diviseur+=interpol(coef_correl,col,lig)**int(puiss_poids)
    tab_sigma_profil[i]/=diviseur
    tab_sigma_profil[i]=sqrt(tab_sigma_profil[i])
  return tab_sigma_profil


#compute weighted median value
def calc_mediane_pond(val,poids):
  if len(val)!=len(poids):
    print 'Erreur calc mediane pond !'
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
  return mediane_pond


#compute stack of perpendicular profiles when using the weighted median method (median on 2*larg_moy+1 profiles; weights=correlation coefficients)
def calc_profil_mediane_pond(data,coef_correl,larg_moy,tab_toutes_perp,num_profil,pas_perp,interpol, puiss_poids):
  ordonnees_profil=[0]*(len(tab_toutes_perp[num_profil])) #ordinate: parallax values
  abscisses_profil=[0]*(len(tab_toutes_perp[num_profil])) #abscissa: distance along profile
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    valeurs_px=[] #all the px on j
    poids_px=[] #all the weights on j
    abscisses_profil[i]=i*pas_perp #abscissa of the i point
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1): #for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      if not isnan(interpol(data,col,lig)) :
        valeurs_px.append(interpol(data,col,lig))
        poids_px.append(interpol(coef_correl,col,lig)**int(puiss_poids) )
    ordonnees_profil[i]=calc_mediane_pond(valeurs_px,poids_px)
  return abscisses_profil,ordonnees_profil


#compute sigmas of a stack of perpendicular profiles when using weighted median
def calc_ecartMoy_profil_mediane_pond(data,coef_correl,larg_moy,tab_toutes_perp,num_profil,interpol,med_profil_perp, puiss_poids) :
  tab_ecartMoy_profil=[0]*(len(tab_toutes_perp[num_profil]))
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    diviseur=0
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1): #for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      tab_ecartMoy_profil[i]+=abs(interpol(data,col,lig)-med_profil_perp[i])*interpol(coef_correl,col,lig)**int(puiss_poids)
      diviseur+=interpol(coef_correl,col,lig)**int(puiss_poids)
    tab_ecartMoy_profil[i]/=diviseur
  return tab_ecartMoy_profil


#generalized function for absolute deviation (measurement of the central tendency: weighted median or weighted mean)
def calc_ecartMoy_profil(data,coef_correl,larg_moy,tab_toutes_perp,num_profil,interpol,mes_tendanceCentrale, puiss_poids):
  tab_ecartMoy_profil=[0]*(len(tab_toutes_perp[num_profil]))
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    diviseur=0
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1): #for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      tab_ecartMoy_profil[i]+=abs(interpol(data,col,lig)-mes_tendanceCentrale[i])*interpol(coef_correl,col,lig)**int(puiss_poids)
      diviseur+=interpol(coef_correl,col,lig)**int(puiss_poids)
    tab_ecartMoy_profil[i]/=diviseur
  return tab_ecartMoy_profil


#compute a single perpendicular profile
def calc_profil_perp(data,tab_toutes_perp,num_profil,pas_perp,interpol,larg_moy=0):
  ordonnees_profil=[0]*(len(tab_toutes_perp[num_profil])) #ordinate: parallax values (=hauteur dans le profil)
  abscisses_profil=[0]*(len(tab_toutes_perp[num_profil])) #abscissa: distance along profile
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    abscisses_profil[i]=i*pas_perp #abscissa of the i point
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1): #for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      if not isnan(interpol(data,col,lig)):
        ordonnees_profil[i]+=interpol(data,col,lig)
  return abscisses_profil,ordonnees_profil


#compute stack of perpendicular profiles using the weighted mean method (mean on 2*larg_moy+1 profiles; weights=correlation coefficients)
#for the abscissa, we take into account the results resolution (resol_im*sousEch_res, where resol_im: initial resolution of the images and sousEch_res: results' resampling factor)
def calc_profil_coef_correl_m(data,coef_correl,larg_moy,tab_toutes_perp,num_profil,pas_perp,interpol, puiss_poids, resol_im, sousEch_res):
  ordonnees_profil=[0]*(len(tab_toutes_perp[num_profil])) #ordinate: parallax values
  abscisses_profil=[0]*(len(tab_toutes_perp[num_profil])) #abscissa: distance along profile
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    #computing the mean value using the "before" and "after" profiles
    diviseur=0
    abscisses_profil[i]=(i-len(tab_toutes_perp[num_profil])/2)*pas_perp*resol_im*sousEch_res #abscissa of the i point
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1): #for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      if not isnan(interpol(data,col,lig)) :
        ordonnees_profil[i]+=interpol(data,col,lig)*interpol(coef_correl,col,lig)**int(puiss_poids)
        diviseur+=interpol(coef_correl,col,lig)**int(puiss_poids)
    ordonnees_profil[i]/=diviseur
  return abscisses_profil,ordonnees_profil


#compute stack of perpendicular profiles using the weighted median method (median on 2*larg_moy+1 profiles; weights=correlation coefficients)
#for the abscissa, we take into account the results resolution (resol_im*sousEch_res, where resol_im: initial resolution of the images and sousEch_res: results' resampling factor)
def calc_profil_mediane_pond_m(data,coef_correl,larg_moy,tab_toutes_perp,num_profil,pas_perp,interpol, puiss_poids, resol_im, sousEch_res):
  ordonnees_profil=[0]*(len(tab_toutes_perp[num_profil])) #ordinate: parallax values
  abscisses_profil=[0]*(len(tab_toutes_perp[num_profil])) #abscissa: distance along profile
  for i in range(len(tab_toutes_perp[num_profil])): #for every point of the central profile
    valeurs_px=[] #parallax values on j
    poids_px=[] #weights values on j
    abscisses_profil[i]=(i-len(tab_toutes_perp[num_profil])/2)*pas_perp*resol_im*sousEch_res #abscissa of the i point
    for j in range(num_profil-larg_moy, num_profil+larg_moy+1): #for every profile we search for the i point
      (col,lig)=tab_toutes_perp[j][i]
      if not isnan(interpol(data,col,lig)) :
        valeurs_px.append(interpol(data,col,lig))
        poids_px.append( interpol(coef_correl,col,lig)**int(puiss_poids) )
    ordonnees_profil[i]=calc_mediane_pond(valeurs_px,poids_px)
  return abscisses_profil,ordonnees_profil

