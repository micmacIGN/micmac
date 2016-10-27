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





    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/
    /*                                                                           */
    /*                         EHFS_ScoreIm                                      */
    /*                                                                           */
    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/
 
EHFS_ScoreIm::EHFS_ScoreIm 
(
    REAL         Step,
    REAL         Width,
    REAL         LentghMax,
    Im2D_U_INT1  ImGlob,
    REAL         CostChg,
    REAL         VminRadiom
)   :
    ElHoughFiltSeg(Step,Width,LentghMax,ImGlob.sz()),
    mImLoc     (SzMax().x,SzMax().y),
    mGainIfSeg (mImLoc.data()[mNbYMax]),
    mImGlob    (ImGlob),
    mCostChg   (CostChg),
    mVminRadiom (VminRadiom)
{
}

REAL EHFS_ScoreIm::AverageCostChange() 
{
   return mCostChg;
}

REAL EHFS_ScoreIm::CostState(bool IsSeg,INT Abcisse)
{
    REAL GainIfSeg = mGainIfSeg[Abcisse] / mVminRadiom;
    GainIfSeg = ElMin(GainIfSeg,1.0);

    if (! IsSeg)  
       GainIfSeg = 1.0-GainIfSeg;
    return GainIfSeg;
}

Im2D_U_INT1  EHFS_ScoreIm::ImLoc(){return mImLoc;}


void  EHFS_ScoreIm::UpdateSeg()
{
   MakeIm(mImLoc,mImGlob,0);
}

void EHFS_ScoreIm::ExtendImage_proj(INT Delta)
{
    ElHoughFiltSeg::ExtendImage_proj(mImLoc,Delta);
}

    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/
    /*                                                                           */
    /*                         EHFS_ScoreGrad                                    */
    /*                                                                           */
    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/

REAL EHFS_ScoreGrad::AverageCostChange()
{
   return mCostChgt;
}

REAL EHFS_ScoreGrad::CostState(bool  IsSeg,INT  x)
{
   REAL GainIfSeg = mDataGainIfSeg[x];
   return IsSeg ? GainIfSeg : (1-GainIfSeg);
}

EHFS_ScoreGrad::EHFS_ScoreGrad 
(
    REAL         Step,
    REAL         Width,
    REAL         LentghMax,
    Im2D_INT1    ImGlobGX,
    Im2D_INT1    ImGlobGY,
    REAL         CostChg ,
    REAL         EcarTeta,
    REAL         EcarMaxLoc,
    REAL         SeuilGrad
) :
  ElHoughFiltSeg (Step,Width,LentghMax,Inf(ImGlobGX.sz(),ImGlobGY.sz())),
  mImLocGX       (SzMax().x,SzMax().y), 
  mImLocGY       (SzMax().x,SzMax().y), 
  mImLocGRho     (SzMax().x,SzMax().y), 
  mDataLocRho    (mImLocGRho.data()),
  mImLocGTeta    (SzMax().x,SzMax().y), 
  mDataLocTeta   (mImLocGTeta.data()), 
  mImYMaxLoc     (SzMax().x),
  mDataYMaxLoc   (mImYMaxLoc.data()),
  mImGlobGX      (ImGlobGX),
  mImGlobGY      (ImGlobGY),
  mCostChgt      (CostChg),  
  //mEcarTeta      (EcarTeta),
  mPdsTeta       (256),
  mDataPdsTeta   (mPdsTeta.data()),
  //mEcarMaxLoc    (EcarMaxLoc),
  mPdsMaxLoc     (SzMax().y),
  mDataPdsMaxLoc (mPdsMaxLoc.data()),
  //mSeuilGrad     (SeuilGrad),
  mPdsRho        (256),
  mDataPdsRho    (mPdsRho.data()),
  mGainIfSeg     (SzMax().x),
  mDataGainIfSeg (mGainIfSeg.data())
{
     for (INT iTeta=0; iTeta<256 ; iTeta++)
     {
         REAL DeltaTeta = (iTeta%128) *PI /128.0 - PI/2;
         DeltaTeta /= EcarTeta;
         mDataPdsTeta[iTeta] =  pow(0.5,ElSquare(DeltaTeta));
     }

     for (INT y=0 ; y<SzMax().y ; y++)
     {
         REAL dY = (y-mNbYMax) * mStep;
         dY /= EcarMaxLoc;
         mDataPdsMaxLoc[y] = pow(0.5,ElSquare(dY));
     }

     for (INT iRho=0 ; iRho<256 ; iRho++)
     {
         mDataPdsRho[iRho] = ElMin(1.0,iRho/SeuilGrad);
     }
}

Im2D_U_INT1 EHFS_ScoreGrad::ImLocGRho()  { return mImLocGRho;}
Im2D_U_INT1 EHFS_ScoreGrad::ImLocGTeta() { return mImLocGTeta;}
Im1D_U_INT1 EHFS_ScoreGrad::ImMaxLoc()   { return mImYMaxLoc;}




void  EHFS_ScoreGrad::MakeGainIfSeg()
{
    for (INT x=0 ; x<=mNbX ; x++)
    {
       mDataGainIfSeg[x] =
              mDataPdsTeta[mDataLocTeta[mNbYMax][x]]
            * mDataPdsMaxLoc[mDataYMaxLoc[x]]
            * mDataPdsRho[mDataLocRho[mNbYMax][x]];
    }
}


void EHFS_ScoreGrad::TestGain(INT x)
{
   cout << "Glob " <<  mDataGainIfSeg[x]
        << " teta " << mDataPdsTeta[mDataLocTeta[mNbYMax][x]]
        << " MaxLoc " << mDataPdsMaxLoc[mDataYMaxLoc[x]] 
        << " Norm " << mDataPdsRho[mDataLocRho[mNbYMax][x]] << "\n";
}




void  EHFS_ScoreGrad::MakeImGradXY
     ( 
            Im2D_INT1 OutGx   , Im2D_INT1 OutGy,
            Im2D_INT1 InPutGx , Im2D_INT1 InPutGy,
            INT1 def
     )
{
    MakeIm(OutGx,InPutGx,def);
    MakeIm(OutGy,InPutGy,def);

    INT1 ** dGx = OutGx.data();
    INT1 ** dGy = OutGy.data();

    for (INT y=0; y<= 2* mNbYMax ; y++)
        for (INT x=0; x<=mNbX ; x++)
        {
            Pt2dr grad = Pt2dr(dGx[y][x],dGy[y][x]) * mInvTgtSeg;
            dGx[y][x] = ElMax(-128,ElMin(127,round_ni(grad.x)));
            dGy[y][x] = ElMax(-128,ElMin(127,round_ni(grad.y)));
        }
}

void  EHFS_ScoreGrad::MakeImGradRhoTeta
      ( 
           Im2D_U_INT1 OutRho, 
           Im2D_U_INT1 OutTeta,
           Im2D_INT1   InPutGx, 
           Im2D_INT1   InPutGy,
           INT1 def
      )
{
    VerifSize(OutRho);
    VerifSize(OutTeta);
    MakeImGradXY(mImLocGX,mImLocGY,InPutGx,InPutGy,def);

    U_INT1 ** dRho  = OutRho.data();
    U_INT1 ** dTeta = OutTeta.data();
    INT1 **   dGx   = mImLocGX.data();
    INT1 **   dGy   = mImLocGY.data();

    for (INT y=0; y<= 2* mNbYMax ; y++)
        for (INT x=0; x<=mNbX ; x++)
        {
            Pt2dr rt = Pt2dr::polar(Pt2dr(dGx[y][x],dGy[y][x]),PI); 
            dRho[y][x] = ElMin(255,round_ni(rt.x));
            dTeta[y][x] =  mod(round_ni(rt.y*128.0/PI),256);
        }
}


void   EHFS_ScoreGrad::MakeImMaxLoc(Im1D_U_INT1 ImMaxLoc,Im2D_U_INT1 InRho)
{
    VerifSize(ImMaxLoc);
    VerifSize(InRho);

    U_INT1 *  dMax = ImMaxLoc.data();
    U_INT1 ** dRho = InRho.data();

    for (INT x=0; x<=mNbX ; x++)
    {
         INT yMaxL1 = mNbYMax;
         while ((yMaxL1<2*mNbYMax) && (dRho[yMaxL1][x]<=dRho[yMaxL1+1][x]))
               yMaxL1++;

         INT yMaxL2 = mNbYMax;
         while ((yMaxL2>0) && (dRho[yMaxL2][x]<=dRho[yMaxL2-1][x]))
               yMaxL2--;

        dMax[x] = (ElAbs(yMaxL1-mNbYMax)>ElAbs(yMaxL2-mNbYMax)) ? yMaxL1  : yMaxL2;
    }
}



void EHFS_ScoreGrad::UpdateSeg()
{
    MakeImGradRhoTeta
    (
         mImLocGRho,
         mImLocGTeta,
         mImGlobGX,
         mImGlobGY,
         0
    );
    MakeImMaxLoc(mImYMaxLoc,mImLocGRho);

    MakeGainIfSeg();
}


/*
void  ElHoughFiltSeg::AddPi(Im2D_INT1 OutTeta)
{
    VerifSize(OutTeta);
    INT1 ** dRho  = OutTeta.data();

    for (INT y=0; y<= 2* mNbYMax ; y++)
        for (INT x=0; x<=mNbX ; x++)
        {
             if (dRho[y][x]>=0) 
                 dRho[y][x] -= 128;
              else
                 dRho[y][x] += 128;
        }
}
                                                  




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
