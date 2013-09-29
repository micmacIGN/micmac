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
//    INITIAL COPRYRIGHT OF S. Roy/NEC

/***
Sofware: stereomf
Author : Sebastien Roy (sebastien@research.nj.nec.com)

               Copyright (c) 1999, NEC Research Institute Inc.
                            All Rights Reserved.

Permission to use, copy, modify, and distribute this software and its
associated documentation for non-commercial purposes is hereby granted,
provided that the above copyright notice appears in all copies, derivative
works or modified versions of the software and any portions thereof, and
that both the copyright notice and this permission notice appear in the
documentation..  NEC Research Institute Inc. shall be given a copy of any
such derivative work or modified version of the software and NEC Research
Institute Inc. and its affiliated companies (collectively referred to as
NECI) shall be granted permission to use, copy, modify and distribute the
software for internal use and research.  The name of NEC Research Institute
Inc. and its affiliated companies shall not be used in advertising or
publicity related to the distribution of the software, without the prior
written consent of NECI.  All copies, derivative works or modified versions
of the software shall be exported or reexported in accordance with
applicable laws and regulations relating to export control.  This software
is experimental.  NECI does not make any representations regarding the
suitability of this software for any purpose and NECI will not support the
software.
THE SOFTWARE IS PROVIDED AS IS.  NECI DOES NOT MAKE ANY
WARRANTIES EITHER EXPRESS OR IMPLIED WITH REGARD TO THE SOFTWARE.  NECI
ALSO DISCLAIMS ANY WARRANTY THAT THE SOFTWARE IS FREE OF INFRINGEMENT OF
ANY INTELLECTUAL PROPERTY RIGHTS OF OTHERS.  NO OTHER LICENSE EXPRESS OR
IMPLIED IS HEREBY GRANTED. NECI SHALL NOT BE LIABLE FOR ANY DAMAGES,
INCLUDING GENERAL, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, ARISING
OUT OF THE USE OR INABILITY TO USE THE SOFTWARE.
***/


/**

               Copyright (c) 2002, Insitut Geographique National
                            All Rights Reserved.

Modification : Marc PIERROT DESEILLIGNY
               marc.pierrot-deseilligny@ign.fr
               11/02/01


Par rapport a l'implementation  initiale, on a les modifs suivantes :

      - possibilite d'utiliser des nappes afin de restreindre memoire et CPU; c'est
        la modification majeure;

      - possibilite d'utiliser aussi bien le 4 que le 8 voisinage;

      - diminution de la taille memoire prise par chaque cellule; (de 36 a 16 octets en 4 Voisinage);

      - reecriture complete : implementation en C++ avec un style resolument objet,
        utilisation de la STL pour toutes les structures de donnees, pas de variable
        globale, separation "interface/implementation";

      - utilisation de structure de points 3D plutot que d'utiliser des entiers
        pour indexer un tableau 1D; ce qui a l'effet (inattendu) de faire gagner
        un facteur 2;

      - correction d'un "bug" dans la structure de donnee file de priorite,
        dans la version initiale; dans  certaine configuration  apparaissant d'autant
        plus facilement que la regularisation est forte, la file devient tres creuse
        avec des entrees variant dans un tres grand intervalle (typiquement 5 objets dans la file,
        4 entree mais avec des valeurs en 100 et 30 000); dans la version initiale
        un temps enorme est perdu pour satisfaire la requete, trouver l'index non vide le plus
        grand en dessous d'un index donne; la degenerescence pouvait multiplier jusqu'a 500
        le temps de calcul; le probleme a ete fixe en rajoutant a la file un set<int> contenant
        l'ensemble des indexes occupes, dans les cas non degenere il y a un leger surcout (20%)
        du a la gestion du set<int>, mais les temps deviennent deterministes;


Dans "cox_roy.cpp" on trouvera , en debut  de fichier , la fonction  "ExempleUtilisationCoxRoy"
qui devrait pouvoir faire office de documentation minimale sur l'utilisation de cette implementation.

**/
#pragma once

class cInterfaceCoxRoyAlgo
{
      public :

	   inline int ZMin(int anX,int anY)const {return mZMin[anY][anX];}
	   inline int ZMax(int anX,int anY)const {return mZMax[anY][anX];}
           inline int X0() const {return mX0Glob;}
           inline int X1() const {return mX1Glob;}
           inline int Y0() const {return mY0Glob;}
           inline int Y1() const {return mY1Glob;}

           inline bool IsInside(int anX,int anY) const
           {
                 return     (anX >= mX0Glob)
                         && (anX <  mX1Glob)
                         && (anY >= mY0Glob)
                         && (anY <  mY1Glob);
           }

           inline bool IsInside(int anX,int anY,int aZ) const
           {
                 return     IsInside(anX,anY)
                         && (aZ  >= mZMin[anY][anX])
                         && (aZ  <  mZMax[anY][anX]);
           }


          // Fixe le cout d'un arc particulier, a priori pas tres utile 
          // (SetCostVert + SetStdCostRegul devraient  suffire en general)

           virtual void SetCost(int anX,int anY,int aZ,int mNumEdge,int aCost) = 0;

          // Fixe le cout de p1 vers  p2=(p1.x,p1.y,p1.z+1) , et  de p2 ver p1

           virtual void SetCostVert(int anX,int anY,int aZ,int aCost) = 0;

          // Suppose les couts verticaux fixes, par exemple par SetCostVert
          // fixe les couts horizontaux, a partir des cout verticaux C1,C2  selon la formule
          //       ((C1+C2)/2 * aCoeff + aCste ) * DIV
          //
          //     ou DIV vaut    : 
          //          1  si grape de 4 connexite
          //          1 / (1+ sqrt(2)) si arc de 4 connexite dans  une graphe de 8 conexite
          //          1 / (2+ sqrt(2)) si arc de 8 connexite dans  une graphe de 8 conexite
          //
          //  Compte tenu des arrondis, cost peut valoir assez souvent 0, notamment en 8 connexite,
          //  si on souhaite eviter cela a tout prix, passer aVmin a 1  (par exemple)
	   //
	   //  Dans tous les cas, les couts sont tronques sur l'intervalle  [0 100]

           virtual void SetStdCostRegul(double aCoeff,double aCste,int aVMin =0) =0;

           
           virtual ~cInterfaceCoxRoyAlgo();
           static cInterfaceCoxRoyAlgo * NewOne
                                         (
                                               int xsz,int ysz,
                                               signed short ** aDataZmin,
                                               signed short ** aDataZmax,
                                               bool  Cx8,
                 //   !! Nouveau param, si vaut true les valeur seront
                 //   stockee sur un octet, sinon sur deux (utile quand
                 //   le coeff de regul est tres faible)
                                               bool  OnUChar
                                         );

           virtual int TopMaxFlowStd(short **Sol) =0;

           int NbVois() const {return mNbVois;}
           int XOfNumVois(int aNum) const;
           int YOfNumVois(int aNum) const;
           int ZOfNumVois(int aNum) const;


      protected :

           cInterfaceCoxRoyAlgo
           (
                   int xsz,int ysz,
                   signed short ** aDataZmin,
                   signed short ** aDataZmax,
                   int             aNbVois
           );

          int mX0Glob,mX1Glob,mY0Glob,mY1Glob;  // Limit of rectangle
          signed short ** mZMin;
          signed short ** mZMax;
          int              mNbVois;
};






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
