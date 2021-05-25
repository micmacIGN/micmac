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



/********************************************************************/
/********************************************************************/
/********************************************************************/

void ElHoughImplem::VerifIsImageXY(Im2D_U_INT1 anIm)
{
   ELISE_ASSERT
   (
     (NbX()==anIm.tx()) &&  (NbY()==anIm.ty()),
      "Bad image size in Hough"
   );
}

void ElHoughImplem::SetImageXYCur(Im2D_U_INT1 anIm)
{
   VerifIsImageXY(anIm);
   mImageXY = anIm;
   mDataImXY = mImageXY.data();
}
 
void ElHoughImplem::SetImageRTCur(Im2D_INT4 anIm)
{
   ELISE_ASSERT
   (
     (mNbTetaTot==anIm.tx()) &&  (NbRho()==anIm.ty()),
      "Bad image size in Hough"
   );
   mImageRT = anIm;
   mDataImRT = mImageRT.data();
}



/********************************************************************/
/********************************************************************/

void ElHoughImplem::Transform_Ang
                    (
                        Im2D_INT4   aHoughTr,
                        Im2D_U_INT1 ImMod,
                        Im2D_U_INT1 ImAng,
                        REAL        RealIncAng,
                        bool        AngIsGrad  // Si vrai += 90 degre
                    )
{
   SetImageRTCur(aHoughTr);
   SetImageXYCur(ImMod);
   VerifIsImageXY(ImAng);
   aHoughTr.raz();
   INT  IincTeta = ElMin(round_ni(ElAbs(RealIncAng)/mStepTeta),NbTeta()/2-1);
   ElSetMax(IincTeta,1);
 
   U_INT1 ** dataAng = ImAng.data();

   for (INT iTeta=0;iTeta<=(NbTeta()/2); iTeta++)
   {
        mDataPdsEcTeta[iTeta] 
      = mDataPdsEcTeta[NbTeta()-1-iTeta] 
      = ElMax(0,(255*(IincTeta-iTeta))/IincTeta);
   }
 
   for (INT y=0; y<NbY() ; y++)
   {
       for (INT x=0; x<NbX() ; x++)
       {
           INT val = mDataImXY[y][x];
           if (val)
           {
              INT iAng = dataAng[y][x];
              if (! AngIsGrad)          // pour passer de gradient a ligne de niveau
                 iAng += 64;
              iAng = iAng %128;       // pour faire un angle de droite
              iAng =(iAng*NbTeta())/128; // dynamic locale
              INT iAng1 = mod(iAng-IincTeta,NbTeta());
              INT iAng2 = mod(iAng+IincTeta,NbTeta());

              INT AdrDeb = mDataAdE[y][x];
              INT Nb =  mDataNbE[y][x];
              tElIndex * TabIT = mDataITeta+AdrDeb;
              tElIndex * TabIR = mDataIRho+AdrDeb;
              U_INT1   * TabP = mDataPds+AdrDeb;

              INT AdrAng1 = GetIndTeta(iAng1,TabIT,Nb);
              INT AdrAng2 = GetIndTeta(iAng2,TabIT,Nb);
              if (AdrAng2<AdrAng1) 
                 AdrAng2+=Nb;

              INT teta,adr;
              for (INT ADR=AdrAng1; ADR<AdrAng2 ; ADR++)
              {
                  adr = ADR%Nb;
                  teta = TabIT[adr];
                  mDataImRT[TabIR[adr]][teta] 
                          += TabP[adr] * val * mDataPdsEcTeta[ElAbs(teta-iAng)];
              }
           }
       }
   }

   MakeCirc(aHoughTr);
}




void ElHoughImplem::Transform(Im2D_INT4 aHoughTr,Im2D_U_INT1 anIm)
{
   SetImageRTCur(aHoughTr);
   SetImageXYCur(anIm);
   aHoughTr.raz();
 
 
   for (INT y=0; y<NbY() ; y++)
   {
       for (INT x=0; x<NbX() ; x++)
       {
           INT val = mDataImXY[y][x];
           if (val)
           {
              INT AdrDeb = mDataAdE[y][x];
              INT AdrFin = AdrDeb + mDataNbE[y][x];
              for (INT adr=AdrDeb; adr<AdrFin ; adr++)
                  mDataImRT[mDataIRho[adr]][mDataITeta[adr]] 
                                       += mDataPds[adr] * val;
           }
       }
   }
 
   MakeCirc(aHoughTr);
}                                                               

void ElHoughImplem::MakeCirc(Im2D_INT4 aHoughTr)
{
   INT NbTetaAcc = aHoughTr.tx();
   INT NbRhoAcc = aHoughTr.ty();
   INT4 ** dataHT = aHoughTr.data();

   for (INT iTeta0 = NbTeta(); iTeta0<NbTetaAcc ; iTeta0 +=  NbTeta())
   {
        INT nbTetaBande = ElMin(NbTeta(),NbTetaAcc-iTeta0);
        for (INT iRho=0; iRho<NbRhoAcc; iRho++)
        {
             INT iRhoSym = IndRhoSym(iRho);
             if ((iRhoSym>=0) && (iRhoSym<NbRhoAcc))
                convert
                (
                   dataHT[iRho]+iTeta0,
                   dataHT[iRhoSym]+iTeta0-NbTeta(),
                   nbTetaBande
                );
        }
   }
}

/********************************************************************/
/********************************************************************/

REAL ElHoughImplem::DynamicModeValues() 
{

    if (! mDMV_IsCalc)
    {
        Im2D_U_INT1 anIm(NbX(),NbY(),(U_INT1)1);
        Im2D_INT4 aPdsOri = mHouhAccul;

        mHouhAccul = Im2D_INT4(mHouhAccul.tx(),mHouhAccul.ty());
        Im2D_INT4 aImPds = Pds(anIm);

        INT4 ** P = aImPds.data();
        REAL SumP = 0.0;
        REAL SumD = 0.0;

        for (INT iR = 0 ; iR<NbRho() ; iR++)
            for (INT iT = 0 ; iT<NbTeta() ; iT++)
                if (P[iR][iT])
                {
                    SumP += P[iR][iT];
                    Seg2d s = Grid_Hough2Euclid(Pt2dr(iT,iR));
                     if ( ! s.empty())
                        SumD += euclid(s.p0(),s.p1());
                }

        mDMV_Val = SumP / SumD;
        mDMV_IsCalc = true;
         
        mHouhAccul = aImPds;
    }

    return  mDMV_Val;
}


REAL ElHoughImplem::DynamicModeGradient()
{
    return   255 * DynamicModeValues();
}

REAL ElHough::Dynamic(ElHough::tModeAccum  aMode)
{
    switch (aMode)
    {
           case eModeValues : return DynamicModeValues();
           case eModeGradient : return DynamicModeGradient();
    }

    return 0;
}
   
/********************************************************************/
/********************************************************************/

void ElHoughImplem::CalcMaxLoc
     (
         Im2D_INT4    Im,
         ElSTDNS vector<Pt2di> & Pts,
         REAL VoisRho,
         REAL VoisTeta,
         REAL VMin
      ) 
{

// std::cout << "STEPS " << mStepTeta << " "<< mStepRhoInit << "\n";


    INT iVoisTeta = ElMin(mNbRabTeta,round_ni(VoisTeta/mStepTeta));
    INT iVoisRho  = ElMin(mNbRabRho,round_ni(VoisRho/mStepRhoInit));
    INT iVmin = round_ni(VMin);


    mCML.AllMaxLoc
    (
        Pts,
        Im,
        Pt2di(iVoisTeta,iVoisRho),
        P0Vois(),P1Vois(),
        iVmin
    );

}


bool ElHoughImplem::BandeConnectedVsup
     (
           Pt2di       p1,
           Pt2di       p2,
           Im2D_INT4   Im,
           INT         VInf,
           REAL        Tol
     )
{
    return mCML.BandeConnectedVsup
           (
               p1,p2,
               Im,VInf,
               Tol,
               mMarqBCVS
           );
}
                                  
void ElHoughImplem::FiltrMaxLoc_BCVS
     (
        ElSTDNS vector<Pt2di> & Pts,
        Im2D_INT4       Im,
        REAL            FactInf,
        REAL            TolGeom,
        REAL            VoisRho,
        REAL            VoisTeta
     )
{
   Pt2di IVois
         (
             round_ni(VoisTeta/mStepTeta),
             round_ni(VoisRho/mStepRhoInit)
         );
   
    mCML.FiltrMaxLoc_BCVS
    (
           Pts,
           Im,
           FactInf,
           TolGeom,
           IVois,
           mMarqBCVS
    );
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
