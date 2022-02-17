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
#include "hough_include.h"


/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

class ElHoughBasic : public ElHoughImplem
{
    public :
       ElHoughBasic(Pt2di Sz,REAL StepRho,REAL StepTeta,bool Adapt,bool SubPix,REAL RabRho,REAL RabTeta);
	   void PostInit();
    private :
       virtual void clean();
       virtual void ElemPixel(tLCel &,Pt2di);         

       REAL CoeffStepOfTeta(INT iTeta)
       {
          if (!mAdapt) return 1.0;
          REAL aTeta = G2S_Teta(iTeta);
          return ElMax(ElAbs(cos(aTeta)),ElAbs(sin(aTeta)));
       }
       bool mAdapt;
       bool mSubPix;
};


ElHoughBasic::ElHoughBasic
(
     Pt2di Sz,
     REAL StepRho,
     REAL StepTeta,
     bool adapt,
     bool subpix,
     REAL RabRho,
     REAL RabTeta
)   :
    ElHoughImplem(Sz,StepRho,StepTeta,RabRho,RabTeta),
    mAdapt       (adapt),
    mSubPix      (subpix)
{
}

void ElHoughBasic::PostInit()
{
	ElHoughImplem::PostInit();
   if (mAdapt)
      for (INT iTeta=0; iTeta<NbTetaTot() ; iTeta++)
      {
          SetStepRho(iTeta, StepRhoInit()*CoeffStepOfTeta(iTeta));
      }
}

void ElHoughBasic::clean()
{
}

void ElHoughBasic::ElemPixel(tLCel & aV,Pt2di aPix)
{
    Pt2dr aPt =  G2S_XY(Pt2dr(aPix));

    REAL aRh0 = euclid(aPt);
    REAL aTeta0 =  (aRh0 < 1e-5) ? 0.0 : angle(aPt);

    for (INT iTeta=0 ; iTeta<NbTeta() ; iTeta++)
    {
       REAL aTeta = G2S_Teta(iTeta);
       REAL aRho = aRh0*cos(aTeta-aTeta0);
       REAL gRho =  CentS2G_Rho(aRho,iTeta);
       if (mSubPix)
       {
          INT iRho0  =  round_down(gRho);
          INT iRho1  =  iRho0+1;
          REAL pds1 = gRho-iRho0;
          REAL pds0 = 1-pds1;
          aV.push_back(tCel(Pt2di(iRho0,iTeta),pds0/StepRho(iTeta)));
          aV.push_back(tCel(Pt2di(iRho1,iTeta),pds1/StepRho(iTeta)));
       }
       else
       {
          INT iRho  =  round_ni(gRho);
          aV.push_back(tCel(Pt2di(iRho,iTeta),1.0/StepRho(iTeta)));
       }
    }
}

ElHoughImplem * ElHoughImplem::Basic
                (
                   Pt2di SzXY,
                   REAL  StepRho,
                   REAL  StepTeta,
                   bool  adapt,
                   bool  subpix,
                   REAL  RabRho,
                   REAL  RabTeta
                )
{
   ElHoughBasic * aRes = new ElHoughBasic(SzXY,StepRho,StepTeta,adapt,subpix,RabRho,RabTeta);
   aRes->PostInit();
   return aRes;
}


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
