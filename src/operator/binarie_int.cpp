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




#include "StdAfx.h"


/****************************************************************/
/*                                                              */
/*     OperatorAndBB                                            */
/*                                                              */
/****************************************************************/

class OperatorAndBB  : public OperBinInt
{
    public :
          static OperatorAndBB The_only_one;

          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { 
               for(int i=0 ; i<nb ; i++)
                  t0[i] = t1[i] & t2[i];
          }
};

OperatorAndBB OperatorAndBB::The_only_one;
const OperBinInt & OpAndBB =  OperatorAndBB::The_only_one;


/****************************************************************/
/*                                                              */
/*     OperatorAnd                                              */
/*                                                              */
/****************************************************************/

class OperatorAnd  : public OperBinInt
{
    public :
          static OperatorAnd The_only_one;

          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { 
               for(int i=0 ; i<nb ; i++)
                  t0[i] = t1[i] && t2[i];
          }
};

OperatorAnd OperatorAnd::The_only_one;
const OperBinInt & OperAnd =  OperatorAnd::The_only_one;

/****************************************************************/
/*                                                              */
/*     OperatorOrBB                                             */
/*                                                              */
/****************************************************************/

class OperatorOrBB  : public OperBinInt
{
    public :
          static OperatorOrBB The_only_one;

          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { 
               for(int i=0 ; i<nb ; i++)
                  t0[i] = t1[i] | t2[i];
          }
};

OperatorOrBB OperatorOrBB::The_only_one;
const OperBinInt & OpOrBB =  OperatorOrBB::The_only_one;


/****************************************************************/
/*                                                              */
/*     OperatorOr                                               */
/*                                                              */
/****************************************************************/

class OperatorOr   : public OperBinInt
{
    public :
          static OperatorOr The_only_one;

          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { 
               for(int i=0 ; i<nb ; i++)
                  t0[i] = t1[i] || t2[i];
          }
};

OperatorOr OperatorOr::The_only_one;
const OperBinInt & OperOr =  OperatorOr::The_only_one;


/****************************************************************/
/*                                                              */
/*     OperatorXorBB                                            */
/*                                                              */
/****************************************************************/

class OperatorXorBB   : public OperBinInt
{
    public :
          static OperatorXorBB The_only_one;

          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { 
               for(int i=0 ; i<nb ; i++)
                  t0[i] = t1[i] ^ t2[i];
          }
};

OperatorXorBB OperatorXorBB::The_only_one;
const OperBinInt & OperXorBB =  OperatorXorBB::The_only_one;


/****************************************************************/
/*                                                              */
/*     OperatorXor                                              */
/*                                                              */
/****************************************************************/

class OperatorXor   : public OperBinInt
{
    public :
          static OperatorXor The_only_one;

          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { 
               for(int i=0 ; i<nb ; i++)
                  t0[i] = (t1[i] && t2[i]) ? 0 : (t1[i] || t2[i]) ;
          }
};

OperatorXor OperatorXor::The_only_one;
const OperBinInt & OperXor =  OperatorXor::The_only_one;

/****************************************************************/
/*                                                              */
/*     OperatorLeftShift                                        */
/*                                                              */
/****************************************************************/

class OperatorLeftShift   : public OperBinInt
{
    public :
          static OperatorLeftShift The_only_one;

          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { 
               for(int i=0 ; i<nb ; i++)
                  t0[i] = t1[i] << t2[i] ;
          }
};

OperatorLeftShift OperatorLeftShift::The_only_one;
const OperBinInt & OperLeftShift =  OperatorLeftShift::The_only_one;

/****************************************************************/
/*                                                              */
/*     OperatoRightShift                                        */
/*                                                              */
/****************************************************************/

class OperatorRightShift   : public OperBinInt
{
    public :
          static OperatorRightShift The_only_one;

          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { 
               for(int i=0 ; i<nb ; i++)
                  t0[i] = t1[i] >> t2[i] ;
          }
};

OperatorRightShift OperatorRightShift::The_only_one;
const OperBinInt & OperRightShift =  OperatorRightShift::The_only_one;





/****************************************************************/
/*                                                              */
/*     OperatorStdMod                                           */
/*                                                              */
/****************************************************************/

class OperatorStdMod   : public OperBinInt
{
    public :
          static OperatorStdMod The_only_one;

          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { 
                ASSERT_USER
                (
                    index_values_null(t2,nb) == INDEX_NOT_FOUND,
                    " a % b with b = 0"
                );

               for(int i=0 ; i<nb ; i++)
                  t0[i] = t1[i] % t2[i] ;
          }
};

OperatorStdMod OperatorStdMod::The_only_one;
const OperBinInt & OperStdMod =  OperatorStdMod::The_only_one;


/****************************************************************/
/*                                                              */
/*     OperatorMod                                              */
/*                                                              */
/****************************************************************/

class OperatorMod   : public OperBinInt
{
    public :
          static OperatorMod The_only_one;

          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { 
                ASSERT_USER
                (
                    index_values_null(t2,nb) == INDEX_NOT_FOUND,
                    " a % b with b = 0"
                );

               for(int i=0 ; i<nb ; i++)
                  t0[i] = mod_gen(t1[i],t2[i]);
          }
};

OperatorMod OperatorMod::The_only_one;
const OperBinInt & OperMod =  OperatorMod::The_only_one;




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
