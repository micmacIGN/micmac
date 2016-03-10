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

import re
import os
import sys


from scipy import *
from pylab import *
from osgeo import gdal
from fonctions_perp import *
from ConstructCG import *


# function for doing the stacks of perpendicular profiles
#  params:
#   filename_Px: filename of the disparity image
#   filename_poids: filename of weigths (correlation coefficients image)
#   puiss_poids: power(exponent) of weights when calculating the weighted median
#   longueur_profil: the total length of a profile is 2*longueur_profil+1
#   largeur_moy: the total width of a stack is 2*largeur_moy+1
#   ecart_profils: number of pixels between central profiles of stacks
#   stack_calcMethod: defines the function to use for computing the stacks (calc_profil_mediane_pond for weighted median or calc_profil_coef_correl for weighted mean)
#   filename_polyline: filename of the polyline which describes the fault
#   filename_info_out: filename out
#   type_output: offsets direction
#   information needed for plotting:
#     label_fig: label of the plot
#     color_fig: color of the plot
#     xlabel_fig, ylabel_fig: labels of plot axes
#     title_fig: title of the figure
#     racine_nom_fig: commun part of the name when saving figure
def stackPerp(filename_Px1, filename_Px2, filename_poids,im_resolution, puiss_poids, longueur_profil, largeur_moy, ecart_profils, stack_calcMethod,filename_polyline, filename_info_out, type_output,label_fig, color_fig, xlabel_fig, ylabel_fig, title_fig, racine_nom_fig, showErrBar, showFig=False):
  print "-------------------"
  print "Save offsets to file: ",filename_info_out
  pas_perp=1
  pas_droit=1
  longueur_profil=(longueur_profil-1)/2+0.0001
  largeur_moy=(largeur_moy-1)/2
  if (os.path.exists(filename_info_out)):
    os.remove(filename_info_out)
  if (os.path.exists(filename_polyline)):
    tab_pts=TraceFromFile(filename_polyline)
    #print "Trace points: ",tab_pts
    tab_toutes_perp=create_all_perp(tab_pts,pas_droit,longueur_profil,pas_perp)
    #print "Nb of perpendicular profiles: ",len(tab_toutes_perp)
  if (os.path.exists(filename_Px1)):
    ds_Px1=gdal.Open(filename_Px1, gdal.GA_ReadOnly)
    nb_col_Px1=ds_Px1.RasterXSize #number of columns
    nb_lig_Px1=ds_Px1.RasterYSize #number of lines
    #nb_b_Px1=ds_Px1.RasterCount #number of bands
    #print 'nb_col: ',nb_col_Px1,' nb_lig: ',nb_lig_Px1,' nb_b:',nb_b_Px1
    data_Px1=ds_Px1.ReadAsArray()#!! attention: gdal data's structure is data[lig][col]
  else:
    print 'Error! ',str(filename_Px1), 'does not exist!'
  if (os.path.exists(filename_Px2)):
    ds_Px2=gdal.Open(filename_Px2, gdal.GA_ReadOnly)
    nb_col_Px2=ds_Px2.RasterXSize #number of columns
    nb_lig_Px2=ds_Px2.RasterYSize #number of lines
    #nb_b_Px2=ds_Px2.RasterCount #number of bands
    #print 'nb_col: ',nb_col_Px2,' nb_lig: ',nb_lig_Px2,' nb_b:',nb_b_Px2
    data_Px2=ds_Px2.ReadAsArray()#!! attention: gdal data's structure is data[lig][col]
  else:
    print 'Error! ',str(filename_Px2), 'does not exist!'
  if (len((filename_poids))!=0):
    if (os.path.exists(filename_poids)):
      ds_poids=gdal.Open(filename_poids, gdal.GA_ReadOnly)
      nb_col_poids=ds_poids.RasterXSize #number of columns
      nb_lig_poids=ds_poids.RasterYSize #number of lines
      #nb_b_poids=ds_poids.RasterCount #number of bands
      #print 'nb_col_poids: ',nb_col_poids,' nb_lig_poids: ',nb_lig_poids
      data_poids=ds_poids.ReadAsArray()#!! attention: with gdal, data's structure is data[lig][col]
      if nb_col_poids!=nb_col_Px1 and nb_lig_poids!=nb_lig_Px1:
        print "!!!Different image size for weights ("+str(filename_poids)+") and parallax ("+str(filename_Px)+")!!!"
    else:
      print 'Error! ',str(filename_poids), 'does not exist!'
  else:
    if (os.path.exists(filename_Px1)): #!!!!!!!!!
      data_poids=ones((nb_lig_Px1,nb_col_Px1))
      #print len(data_poids), len(data_poids[0])
  compteur_profilMoy=0
  for num_profil_central in range(largeur_moy,len(tab_toutes_perp)-largeur_moy,ecart_profils):
    #coord of central point on the central profile of the stack:
    col_cen_num_profil_central=tab_toutes_perp[num_profil_central][int(longueur_profil+1)][0]
    lig_cen_num_profil_central=tab_toutes_perp[num_profil_central][int(longueur_profil+1)][1]
    print "Central profile number:", num_profil_central
    #compute orientation vector of the fault for this profile
    firstProfilePoint=tab_toutes_perp[num_profil_central][0]
    lastProfilePoint=tab_toutes_perp[num_profil_central][-1]
    vectAB=(lastProfilePoint[0]-firstProfilePoint[0],lastProfilePoint[1]-firstProfilePoint[1])
    normeAB=sqrt(vectAB[0]**2+vectAB[1]**2)
    u=(vectAB[0]/normeAB,vectAB[1]/normeAB) #unit vector parallel to fault
    v=(u[1],-u[0]) #unit vector perp to fault
    #~ print "Profile ",num_profil_central,": parall ",u,"; perp ",v
    fig=figure(1)
    #~ profil_absc,profil_ordo,tab_ecart=stack_calcMethod(data_Px1,data_Px2, data_poids,largeur_moy,tab_toutes_perp,num_profil_central,pas_perp,interpol_bilin,puiss_poids, u, v, type_output)
    profil_absc,profil_ordo,tab_ecart=stack_calcMethod(data_Px1*im_resolution,data_Px2*im_resolution, data_poids,largeur_moy,tab_toutes_perp,num_profil_central,pas_perp,interpol_bilin,puiss_poids, u, v, type_output)
    compteur_profilMoy+=1
    ax = fig.add_subplot(111)
    #~ ax = plt.subplot2grid((1,2),(0, 0))
    #~ print "**** type_output: ", type_output
    constr_cg=ConstructCG(filename_Px1,filename_Px2,filename_poids,im_resolution, puiss_poids, longueur_profil, largeur_moy, ecart_profils, stack_calcMethod,fig,ax,profil_absc,profil_ordo,tab_ecart,label_fig,color_fig, xlabel_fig, ylabel_fig,title_fig, racine_nom_fig, compteur_profilMoy, num_profil_central, col_cen_num_profil_central,lig_cen_num_profil_central, filename_polyline, filename_info_out,type_output,showFig,showErrBar)

    #~ g=figure(2)
    #~ ds = gdal.Open(filename_Px1, gdal.GA_ReadOnly)
    #~ data_im=ds.ReadAsArray()
    #~ imshow(data_im,cmap=cm.Greys_r, interpolation=None)

    if showFig:
      show()

#function for retrieving the coordinates of the fault (polyline) from a file
def TraceFromFile(filepath_trace):
  coords_trace=[]
  reading_status=0 #0: begin , 1: getting polyline points, 2: polyline finished
  for line in open(filepath_trace,'r').readlines():
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
  nrProf_offsets=[]
  coords_fault=[]
  reading_status=0 #0: begin , 1: getting info on offsets, 2: start getting info on fault polyline, 3: end polyline, 4: getting resolution value
  if (os.path.exists(filepath_infoOffsets)):
    for line in open(filepath_infoOffsets,'r').readlines():
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
      #~ print 'list_y', list_y
      #~ ym=[]
      #~ for y in list_y:
        #~ ym.append(y*resol)
      #~ ax.plot(list_x,ym,color='k', marker='o')
      ax.plot(list_x,list_y,color='k', marker='o')
    xlabel('distance along the fault (px)')
    ylabel('offset(m)')
    title('Slipe-curve')
    savefig(nom_fig)
    plt.show()
