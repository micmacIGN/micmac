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

/*
   La distorsion d'une camera Bilineaire est codee par la valeur du deplacement sur les N Noeuds.

   Le modele bilineaire a ete prefere au modele triangule car :

      * plus simple a gerer au niveau des indexe
      * naturellement continu en extrapolation
      * ni pire ni meilleur  que la triangulation en continuite
      

*/



class cDistorBilin :   public ElDistortion22_Gen 
{
     public :
          friend void Test_DBL();

          cDistorBilin(Pt2dr aSz,Pt2dr aP0,Pt2di aNb);
          Pt2dr Direct(Pt2dr) const ;

          Pt2dr & Dist(const Pt2di aP) {return mVDist[aP.y][aP.x];}
          const Pt2dr & Dist(const Pt2di aP) const {return mVDist[aP.y][aP.x];}


          virtual cCalibDistortion ToXmlStruct(const ElCamera *) const;
          cCalibrationInterneGridDef ToXmlGridStruct() const;


          bool  AcceptScaling() const;
          bool  AcceptTranslate() const;
          void V_SetScalingTranslate(const double &,const Pt2dr &);


     private  :
        //  ==== Tests ============
          Box2dr BoxRab(double aMulStep) const;
          void Randomize();

        //  =============
          Pt2dr ToCoordGrid(const Pt2dr &) const;
          Pt2dr FromCoordGrid(const Pt2dr &) const;
          // Renvoie le meilleur interval [X0, X0+1[ contenat aCoordGr, valide qqsoit aCoordGr
          void GetDebInterval(int & aX0,const int & aSzGrd,const double & aCoordGr) const;
          //  tel que aCoordGr soit le barry de (aX0,aX0+1) avec (aPdsX0,1-aPdsX0)  et 0<= aX0 < aSzGr, aX0 entier
          void GetDebIntervalAndPds(int & aX0,double & aPdsX0,const int & aSzGrd,const double & aCoordGr) const;
          //  A partir d'un points en coordonnees grille retourne le coin bas-gauche et le poids 
          void GetParamCorner(Pt2di & aCornerBG,Pt2dr & aPdsBG,const Pt2dr & aCoorGr) const;
          void InitEtatFromCorner(const Pt2dr & aCoorGr) const; 

          Pt2dr                               mP0;
          Pt2dr                               mP1;
          Pt2dr                               mStep;
          Pt2di                               mNb;
          std::vector<std::vector<Pt2dr > >   mVDist;

          mutable Pt2di                               mCurCorner;
          mutable double                              mP00;
          mutable double                              mP10;
          mutable double                              mP01;
          mutable double                              mP11;
};

/**************************************************************/
/*                                                            */
/*                 cDistorBilin                               */
/*                                                            */
/**************************************************************/

cDistorBilin::cDistorBilin(Pt2dr aP0,Pt2dr aP1,Pt2di aNb) :
   mP0     (aP0),
   mP1     (aP1),
   mStep   ((aP1-aP0).dcbyc(Pt2dr(aNb))),
   mNb     (aNb)
{
    std::vector<Pt2dr > aV0;
    for (int aKX=0 ; aKX<= mNb.x ; aKX++)
        aV0.push_back(Pt2dr(0,0));

    for (int aKY=0 ; aKY<= mNb.y ; aKY++)
        mVDist.push_back(aV0);
}

Pt2dr cDistorBilin::ToCoordGrid(const Pt2dr & aP) const   { return (aP-mP0).dcbyc(mStep); } 
Pt2dr cDistorBilin::FromCoordGrid(const Pt2dr & aP) const { return  mP0+aP.mcbyc(mStep); } 


void  cDistorBilin::GetDebInterval(int & aX0,const int & aSzGrd,const double & aCoordGr) const
{
   aX0 =  ElMax(0,ElMin(aSzGrd-1,round_down(aCoordGr)));
}


void cDistorBilin::GetDebIntervalAndPds(int & aX0,double & aPdsX0,const int & aSzGrd,const double & aCoordGr) const
{
    GetDebInterval(aX0,aSzGrd,aCoordGr);
    aPdsX0 = 1.0 - (aCoordGr-aX0);
}

void  cDistorBilin::GetParamCorner(Pt2di & aCornerBG,Pt2dr & aPdsBG,const Pt2dr & aCoorGr) const
{
     GetDebIntervalAndPds(aCornerBG.x,aPdsBG.x,mNb.x,aCoorGr.x);
     GetDebIntervalAndPds(aCornerBG.y,aPdsBG.x,mNb.y,aCoorGr.y);
}

void cDistorBilin::InitEtatFromCorner(const Pt2dr & aCoorGr) const
{
   Pt2dr aPds;
   GetParamCorner(mCurCorner,aPds,aCoorGr);
   mP00 = aPds.x * aPds.y;
   mP10 = (1-aPds.x) * aPds.y;
   mP01 = aPds.x * (1-aPds.y);
   mP11 = (1-aPds.x) * (1-aPds.y);
    
}
Pt2dr cDistorBilin::Direct(Pt2dr aP) const
{
    InitEtatFromCorner(ToCoordGrid(aP));

    return   aP 
           + Dist(mCurCorner             ) * mP00
           + Dist(mCurCorner + Pt2di(1,0)) * mP10
           + Dist(mCurCorner + Pt2di(0,1)) * mP01
           + Dist(mCurCorner + Pt2di(1,1)) * mP11;
}

Box2dr cDistorBilin::BoxRab(double aMulStep) const
{
    Pt2dr aRab= mStep * aMulStep;
    return Box2dr (mP0-aRab,mP1+aRab);
}
  
cCalibrationInterneGridDef  cDistorBilin::ToXmlGridStruct() const
{
   cCalibrationInterneGridDef aRes;
   aRes.P0() = mP0;
   aRes.P1() = mP1;
   aRes.Nb() = mNb;

   for (int aKY=0 ; aKY<= mNb.y ; aKY++)
   {
       for (int aKX=0 ; aKX<= mNb.x ; aKX++)
       {
           aRes.PGr().push_back(Dist(Pt2di(aKX,aKY)));
       }
   }

   return aRes;
}

void cDistorBilin::Randomize()
{
   for (int aKY=0 ; aKY<= mNb.y ; aKY++)
   {
       for (int aKX=0 ; aKX<= mNb.x ; aKX++)
       {
             Dist(Pt2di(aKX,aKY)) = Pt2dr(NRrandC(),NRrandC()).mcbyc(mStep) * 0.1;
       }
   }
}

extern cCalibDistortion GlobXmlDistNoVal();


cCalibDistortion FromCIGD(const cCalibrationInterneGridDef & aCIGD)
{
    cCalibDistortion  aRes = GlobXmlDistNoVal();
    aRes.ModGridDef().SetVal(aCIGD);

    return aRes;
}


cCalibDistortion cDistorBilin::ToXmlStruct(const ElCamera *) const
{
   return FromCIGD(ToXmlGridStruct());
}

bool  cDistorBilin::AcceptScaling() const {return true; }
bool  cDistorBilin::AcceptTranslate() const {return true; }

/*

   Extrait de photogram.h :
     Soit H (X) == PP + X * F   se transforme en H-1 D H

     Pt2dr cDistorBilin::ToCoordGrid(const Pt2dr & aP) const   { return (aP-mP0).dcbyc(mStep); } 
    ( PP + X * F -mP0) / S  = (X-P')/S'
    (PP -mP0)/S = -P' *F/S

    P' = (mP0 -PP) /F   ; S' = S/F     

*/

void cDistorBilin::V_SetScalingTranslate(const double & aF,const Pt2dr & aPP)
{
   for (int aKY=0 ; aKY<= mNb.y ; aKY++)
   {
       for (int aKX=0 ; aKX<= mNb.x ; aKX++)
       {
           Dist(Pt2di(aKX,aKY)) = ( Dist(Pt2di(aKX,aKY))- aPP) / aF;
       }
   }
   mP0 = (mP0-aPP) / aF;
   mP1 = (mP1-aPP) / aF;
   mStep = mStep / aF;
}

void Test_DBL()
{
    Pt2dr aP0(-10,-20);
    Pt2dr aP1(1500,2000);
    Pt2di aNb(10,15);

    

    cDistorBilin aDBL1(aP0,aP1,aNb);
    aDBL1.Randomize();

    cDistorBilin aDBL2 = aDBL1;

    Box2dr aBoxRab1 = aDBL1.BoxRab(0.3);

   // Test copy
    for (int aK=0 ; aK<100000 ; aK++)
    {
         Pt2dr aP0 = aBoxRab1.RandomlyGenereInside();
         Pt2dr aP1 = aDBL1.Direct(aP0);
         Pt2dr aP2 = aDBL2.Direct(aP0);
         double aDist = euclid(aP1,aP2);
         ELISE_ASSERT(aDist==0,"Test_DBL dist");
    }
    
    for (int aTime=0 ; aTime<100000 ; aTime++)
    {
        double aF = pow(2.0,NRrandC()*8);
        Pt2dr aPP = Pt2dr(NRrandC(),NRrandC()) * aF;
        aDBL2 = aDBL1;
        aDBL2.V_SetScalingTranslate(aF,aPP);
        Box2dr aBoxRab2 = aDBL2.BoxRab(0.3);
        for (int aK=0 ; aK<10; aK++)
        {
            Pt2dr aP0 = aBoxRab2.RandomlyGenereInside();
            Pt2dr aP2 = aDBL2.Direct(aP0);

            Pt2dr aP1 = (aDBL1.Direct(aPP+aP0*aF)-aPP) /aF;

            std::cout << aP1 << " " << aP2 << "\n";
     // Soit H (X) == PP + X * F   se transforme en H-1 D H
             
        }
        // aDBL2 = 
    }


    std::cout << "DONE Test cDistorBilin\n";

}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
