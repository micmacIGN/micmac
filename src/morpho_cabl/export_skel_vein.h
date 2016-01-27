/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/



/*

Copyright (C) 1998 Marc PIERROT DESEILLIGNY

   Skeletonization by veinerization. 

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28


   Detail of the algoprithm in Deseilligny-Stamon-Suen
   "Veinerization : a New Shape Descriptor for Flexible
    Skeletonization" in IEEE-PAMI Vol 20 Number 5, pp 505-521

    It also give the signification of main parameters.

    See some commemts at end of this file.
*/

/*
     Code in export_skel_vein.cpp is  C++ code.
	 However, for technical reasons the interface is
	 limitide to C.


     Version of 15/01/1999 :

         - OK on g++ 2.7.2, Visual 6.0, Borland 5.0
         - works only with distance 3-2; 
*/

using namespace std;






typedef struct ResultVeinSkel
{
      unsigned short *  x;
      unsigned short *  y;
      int               nb;
} ResultVeinSkel;

void freeResultVeinSkel(ResultVeinSkel *);

ResultVeinSkel VeinerizationSkeleton
(
     unsigned char **    out,
     unsigned char **    in,
     int                 tx,
     int                 ty,
     int                 surf_threshlod,
     double              angular_threshlod,
     bool                skel_of_disk,
     bool                prolgt_extre,
     bool                with_result,
     unsigned short **   tmp
);

const unsigned char * NbBitsOfFlag();


/*
                  EXPLANATION OF PARAMETERS :


   Detail of the algoprithm in [PAMI-MPD] : Deseilligny-Stamon-Suen
   "Veinerization : a New Shape Descriptor for Flexible
    Skeletonization" in IEEE-PAMI Vol 20 Number 5, pp 505-521 


       out  :  result image  adressable by "out[y][x]"  for   0<= x < tx and 0<= y < ty
       ###      resulting image is a graph  of pixel.  If you are only
               interested by a set of pixel, you should interpret it like that :
     
                   out[y][x] = 0  => (x,y) not in the skeleton
                   out[y][x] != 0  => (x,y)  in the skeleton

                    
               If you are interested in the graph structure  you mut interpret it like that :

                   - decompose out[y][x] a tab of eight bits ;
                   - for each bit k :
                           * if k= 0 :  nothing
                           * if k = 1 : there is a link between the pixel and it Kth neighboor
                             according to Freeman number;

               Lets take an example, if the value is 150;  the binary representation
               of 150  is "10010110" (as 150=2+4+16+128); so the pixel is to be limked
               with neighboor 1,2,4 and 7 (see bellow).


                            V3     V2     V1
                                   |    /
                                   |  /
                            V4---- P      V0
                                    \
                                      \
                            V5     V6     V7

                  This graph is always symetric (this mean for example, that bits 3 of V7 
                  will value 1, because if P is linked to V7 then V7 is linked to P).


       in   :  input image, same format as out
       ##      pixels value 0 are considered out of the shape,
                all other value are considered inside the shape.
              ``input'' image can be modified by the algorithm.

       tx,ty : size of image
       #####

      surf_threshlod, angular_threshlod : two threshold who signification is detailled
      ################################### in  [PAMI-MPD]. If you do not want to read 
      the paper, begin by surf_threshlod = 8 and angular_threshlod = 3.14, then adapt 
      then iteratively to you application


      skel_of_disk : true if  you want a skeleton for shape that are almost like a circle
      ############                    (recommanded : set it to false for begining)

      prolgt_extre : true if you want extremity to be prolongated until the frontier of input shape
      ############                    (recommanded : set it to false for begining)


     with_result: true if you want to have something interesting as a result of the function.
     ###########   
                 The struct result will then contain the center of shape that are not in the result graph "out" ,
            that is :
                  *   isolated pixels
                  * center of shape that are almost like a circle if you have set skel_of_disk to false.
             
     tmp  : an optional memory auxiliary zone that can acceralate the computation.
     ###    You can pass :

                * or the null pointer if you want to economize memory.
                * or a valid pointer adressable with the same x,y  as  in and out.


     The only memory not deallocated by the algorithm is the result. You can free it by
     ``freeResultVeinSkel'' (It is safe to use it, wether or not with_result was set to true).
*/








/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
