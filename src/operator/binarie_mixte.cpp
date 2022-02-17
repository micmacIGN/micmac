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
/*     OperatorDiv                                              */
/*                                                              */
/****************************************************************/

template <class T0,class T1,class T2> inline void div_t0_eg_t1_op_t2
         (T0 * t0, const T1 * t1,const T2 * t2,INT nb)
{
    ASSERT_USER
    (
        index_values_null(t2,nb) == INDEX_NOT_FOUND,
        "division by 0"
    );

    for (int i=0; i<nb ; i++)
        t0[i] = t1[i] / t2[i];
}

class OperatorDiv : public OperBinMixte
{
    public :
          static OperatorDiv The_only_one;

 
      //--------------
      //   t0 = t1 + t2 
      //--------------

          void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const REAL *t2,INT nb) const
          { div_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const INT  *t2,INT nb) const
          { div_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          void t0_eg_t1_op_t2(REAL * t0,const INT  * t1,const REAL *t2,INT nb) const
          { div_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          
          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { div_t0_eg_t1_op_t2(t0,t1,t2,nb); }

};

OperatorDiv OperatorDiv::The_only_one;
const OperBinMixte & OpDiv =  OperatorDiv::The_only_one;

/****************************************************************/
/*                                                              */
/*     OperatorMinus2                                           */
/*                                                              */
/****************************************************************/

template <class T0,class T1,class T2> inline void F1OrF2IfBadNum_t0_eg_t1_op_t2
         (T0 * t0, const T1 * t1,const T2 * t2,INT nb)
{

    for (int i=0; i<nb ; i++)
        t0[i] = IsBadNum(t1[i]) ? t2[i] : t1[i];
}

class OperatorF1OrF2IfBadNum : public OperBinMixte
{
    public :
          static OperatorF1OrF2IfBadNum The_only_one;

 
      //--------------
      //   t0 = t1 + t2 
      //--------------

          void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const REAL *t2,INT nb) const
          { F1OrF2IfBadNum_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const INT  *t2,INT nb) const
          { F1OrF2IfBadNum_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          void t0_eg_t1_op_t2(REAL * t0,const INT  * t1,const REAL *t2,INT nb) const
          { F1OrF2IfBadNum_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          
          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { F1OrF2IfBadNum_t0_eg_t1_op_t2(t0,t1,t2,nb); }

};

OperatorF1OrF2IfBadNum OperatorF1OrF2IfBadNum::The_only_one;
const OperBinMixte & OpF1OrF2IfBadNum =  OperatorF1OrF2IfBadNum::The_only_one;


/****************************************************************/
/*                                                              */
/*     OperatorMinus2                                           */
/*                                                              */
/****************************************************************/

template <class T0,class T1,class T2> inline void minus2_t0_eg_t1_op_t2
         (T0 * t0, const T1 * t1,const T2 * t2,INT nb)
{

    for (int i=0; i<nb ; i++)
        t0[i] = t1[i] - t2[i];
}

class OperatorMinus2 : public OperBinMixte
{
    public :
          static OperatorMinus2 The_only_one;

 
      //--------------
      //   t0 = t1 + t2 
      //--------------

          void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const REAL *t2,INT nb) const
          { minus2_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const INT  *t2,INT nb) const
          { minus2_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          void t0_eg_t1_op_t2(REAL * t0,const INT  * t1,const REAL *t2,INT nb) const
          { minus2_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          
          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { minus2_t0_eg_t1_op_t2(t0,t1,t2,nb); }

};

OperatorMinus2 OperatorMinus2::The_only_one;
const OperBinMixte & OpMinus2 =  OperatorMinus2::The_only_one;


/****************************************************************/
/*                                                              */
/*     OperatorPow2                                             */
/*                                                              */
/****************************************************************/

/*
    A Optimiser pour tenir compte des exposants entiers
*/

template <class T0,class T1,class T2> inline void pow2_t0_eg_t1_op_t2
         (T0 * t0, const T1 * t1,const T2 * t2,INT nb)
{
    if (El_User_Dyn.active())
    {
        for (int i=0; i<nb ; i++)
        {
             if ((t1[i]<0) && (t2[i] != (int) (t2[i])))
                 El_User_Dyn.error
                 (
                     EEM0 << "Bad value in pow : pow (" 
                          << t1[i] << "," << t2[i] <<")"
                 );
        }
    }

    for (int i=0; i<nb ; i++)
        t0[i] = (T0) pow((double)t1[i],(double)t2[i]);
}

class OperatorPow2 : public OperBinMixte
{
    public :
          static OperatorPow2 The_only_one;

 
      //--------------
      //   t0 = t1 + t2 
      //--------------

          void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const REAL *t2,INT nb) const
          { pow2_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const INT  *t2,INT nb) const
          { pow2_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          void t0_eg_t1_op_t2(REAL * t0,const INT  * t1,const REAL *t2,INT nb) const
          { pow2_t0_eg_t1_op_t2(t0,t1,t2,nb); }

          
          void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const
          { pow2_t0_eg_t1_op_t2(t0,t1,t2,nb); }

};

OperatorPow2 OperatorPow2::The_only_one;
const OperBinMixte & OpPow2 =  OperatorPow2::The_only_one;






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
