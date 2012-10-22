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
               
/******************************************/
/*                                        */
/*       TabuledCollecPt2di               */
/*                                        */
/******************************************/


TabuledCollecPt2di::TabuledCollecPt2di(Pt2di aP0,Pt2di aP1,REAL aRatio) :
    mP0    (Inf(aP0,aP1)),
    mP1    (Sup(aP0,aP1)),
    mSz    (mP1-mP0),
    mIm    (mSz.x,mSz.y,0),
    mCpt   (mIm.data()),
    mVPts  (round_up(aRatio*mSz.x*mSz.y))
{
}

bool TabuledCollecPt2di::InDom(Pt2di aP) const
{
   return aP.in_box(mP0,mP1);
}

TabuledCollecPt2di::tElem & TabuledCollecPt2di::Val(Pt2di aP)
{
   return mCpt[aP.y-mP0.y][aP.x-mP0.x];
}
const TabuledCollecPt2di::tElem & TabuledCollecPt2di::Val(Pt2di aP) const
{
   return mCpt[aP.y-mP0.y][aP.x-mP0.x];
}

INT  TabuledCollecPt2di::NbPres (Pt2di aP,INT aDef) const
{
    return InDom(aP) ? Val(aP) : aDef;
}

void TabuledCollecPt2di::Add(Pt2di aP)
{
   if (!InDom(aP))
      return;

   if (Val(aP)==0)
      mVPts.push_back(aP);
   Val(aP)++;
}

void TabuledCollecPt2di::clear()
{
   for (tIterPts anIt = mVPts.begin(); anIt != mVPts.end() ; anIt++)
       Val(*anIt) = 0;

   mVPts.clear();
}

/******************************************/
/*                                        */
/*          EliseDecCor2D                 */
/*                                        */
/******************************************/

EliseDecCor2D::EliseDecCor2D(Pt2di aSz) :
   mDecX    (aSz.x,aSz.y,-1e5),
   mDataX   (mDecX.data()),
   mDecY    (aSz.x,aSz.y,-1e5),
   mDataY   (mDecY.data())
{
}

Pt2di EliseDecCor2D::anIDec(Pt2di aP)  const
{
   return Pt2di(Pt2dr(mDataX[aP.y][aP.x],mDataY[aP.y][aP.x]));
}

Pt2dr EliseDecCor2D::RDec(Pt2di aP)  const
{
   return Pt2dr(mDataX[aP.y][aP.x],mDataY[aP.y][aP.x]);
}

void  EliseDecCor2D::SetDec(Pt2di aP,Pt2dr aVal) 
{
   mDataX[aP.y][aP.x] = (EliseDecCor2D::tElem)aVal.x;
   mDataY[aP.y][aP.x] = (EliseDecCor2D::tElem)aVal.y;
}



Fonc_Num EliseDecCor2D::In()
{
    return Virgule(mDecX.in(0),mDecY.in(0));
}

Fonc_Num EliseDecCor2D::In(Pt2dr aDef)
{
    return Virgule(mDecX.in(aDef.x),mDecY.in(aDef.y));
}

Output EliseDecCor2D::Out()
{
    return Virgule(mDecX.out(),mDecY.out());
}



/******************************************/
/*                                        */
/*          EliseCorrel2D                 */
/*                                        */
/******************************************/

bool EliseCorrel2D::InImage(Pt2di aP)
{
   return    
             (aP.x >= mSzV )
          && (aP.y >= mSzV )
          && (aP.x < mSzIm.x - mSzV)
          && (aP.y < mSzIm.y - mSzV) ;
}

bool EliseCorrel2D::OkIm1(Pt2di aP)
{
    return mUseVOut                             ?
            (mIsImOk1.get_def(aP.x,aP.y,0)!=0)  :
            InImage(aP)                         ;
}
bool EliseCorrel2D::OkIm2(Pt2di aP)
{
    return mUseVOut                             ?
            (mIsImOk2.get_def(aP.x,aP.y,0)!=0)  :
            InImage(aP)                         ;
}


EliseCorrel2D::EliseCorrel2D
(
        Fonc_Num f1,
        Fonc_Num f2,
        Pt2di aSzIm,
        INT   aSzVgn,
        bool  aUseVOut,
        INT   aVOut,
        bool  WithPreCompute,
        bool  WithDec
)  :
   mSzIm           (aSzIm),
   mWithPreCompute (WithPreCompute),
   mWithDec        (WithDec),
   mSzPrec         (WithPreCompute ? mSzIm : Pt2di(1,1)),
   mP0Correl       (aSzVgn,aSzVgn),
   mP1Correl       (aSzIm.x-aSzVgn,aSzIm.y-aSzVgn),
   mDec            (WithDec ? mSzIm : Pt2di(1,1)),
   mCorrelMax      (mSzPrec.x,mSzPrec.y,-1e5),
   mCorrel         (mSzPrec.x,mSzPrec.y,-1e5),
   mSzI12          (aSzIm),
   mI1             (mSzIm.x,mSzIm.y,tElInit(0)),
   mDataI1         (mI1.data()),
   mSzImOk         ((WithPreCompute && aUseVOut) ? mSzIm :  Pt2di(1,1)),
   mIsImOk1        (mSzImOk.x,mSzImOk.y),
   mI2             (mSzIm.x,mSzIm.y,tElInit(0)),
   mDataI2         (mI2.data()),
   mIsImOk2        (mSzImOk.x,mSzImOk.y),
   mS1             (mSzPrec.x,mSzPrec.y,-1e5),
   mDataS1         (mS1.data()),
   mS2             (mSzPrec.x,mSzPrec.y,-1e5),
   mDataS2         (mS2.data()),
   mS11            (mSzPrec.x,mSzPrec.y,-1e5),
   mDataS11        (mS11.data()),
   mS22            (mSzPrec.x,mSzPrec.y,-1e5),
   mDataS22        (mS22.data()),
   mI1GeomRadiomI2 (1,1),
   mD1GRI2         (0),
   mSzV            (-1000),
   mUseVOut        (aUseVOut),
   mVOut           (aVOut)
{
    SetFoncs(f1,f2);
    if (aSzVgn > 0) 
       SetSzVign(aSzVgn);
}

INT EliseCorrel2D::SzV() const
{
   return mSzV;
}

bool EliseCorrel2D::OKPtCorrel(Pt2di aPt) const
{
   return 
               (aPt.x >= mP0Correl.x) 
           &&  (aPt.y >= mP0Correl.y) 
           &&  (aPt.x <  mP1Correl.x) 
           &&  (aPt.y <  mP1Correl.y)  ;
}

REAL EliseCorrel2D::CorrelBrute(Pt2di aPIm1,Pt2di aPIm2) const
{
    tBaseElInit  aRes = 0;
    for (INT y=-mSzV ; y<=mSzV ; y++)
    {
         tElInit * aL1 =  mDataI1[y+aPIm1.y]+aPIm1.x;
         tElInit * aL2 =  mDataI2[y+aPIm2.y]+aPIm2.x;
         for (INT x=-mSzV ; x<=mSzV ; x++)
             aRes += aL1[x]*aL2[x];
    }
    return aRes;
}

REAL EliseCorrel2D::Correl(Pt2di aPIm1,Pt2di aPIm2) const
{
   if (! (OKPtCorrel(aPIm1) && OKPtCorrel(aPIm2)))
      return -1;

   return  (
                CorrelBrute(aPIm1,aPIm2) /mNbVois
              - mDataS1[aPIm1.y][aPIm1.x] *  mDataS2[aPIm2.y][aPIm2.x]
           )
           /
           sqrt(mDataS11[aPIm1.y][aPIm1.x] *  mDataS22[aPIm2.y][aPIm2.x]) ;
}

REAL EliseCorrel2D::CorrelStdIDec(Pt2di aPIm2) const
{
   return Correl(aPIm2+mDec.anIDec(aPIm2),aPIm2);
}


Fonc_Num EliseCorrel2D::CorrelMax()
{
   return mCorrelMax.in(0);
}

void EliseCorrel2D::SetFoncs(Fonc_Num f1,Fonc_Num f2)
{
   ELISE_COPY(mI1.all_pts(),f1,mI1.out());
   ELISE_COPY(mI2.all_pts(),f2,mI2.out());
}

Fonc_Num EliseCorrel2D::Moy(Fonc_Num aFonc)
{
    return rect_som(Rconv(aFonc),mSzV) / ElSquare(2*mSzV+1.0);
}

#define Epsilon 1e-2

void EliseCorrel2D::SetSzVign(INT aSzV)
{
    if (mSzV == aSzV) 
       return;

    mSzV = aSzV;
    if (! mWithPreCompute)
       return;

    mNbVois = ElSquare(2*mSzV+1);
    Symb_FNum aF1 (mI1.in_proj());
    Symb_FNum aF2 (mI2.in_proj());


    Symb_FNum aMoy(  Virgule
                     (
                        Moy(Virgule(aF1,Square(aF1))),
                        Moy(Virgule(aF2,Square(aF2)))
                     )
                  );

    ELISE_COPY
    (
         mS1.all_pts(),
         Virgule
         (
             aMoy.v0(),
             Max(Epsilon,aMoy.v1()-Square(aMoy.v0())),
             aMoy.v2(),
             Max(Epsilon,aMoy.kth_proj(3)-Square(aMoy.v2()))
         ),
         Virgule
         (
              mS1.out(),mS11.out(),
              mS2.out(),mS22.out()
         )
    );
    InitImOk(mIsImOk1,mI1);
    InitImOk(mIsImOk2,mI2);
}

void EliseCorrel2D::InitImOk(Im2D_Bits<1> aIOk,Im2D<tElInit,tBaseElInit> aIVals)
{
    ELISE_COPY
    (
        aIOk.all_pts(),
        erod_d8(aIVals.in(mVOut) != mVOut,mSzV+3),
        aIOk.out()
    );
}

Box2di EliseCorrel2D::BoxOk(Pt2di aDec,INT anIncert)
{
   INT aRab = anIncert + mSzV;
   Pt2di aPtRab(aRab,aRab);

   return Box2di
          (
               Sup(Pt2di(0,0),-aDec) + aPtRab,
               Inf(mSzIm,mSzIm-aDec) - aPtRab -Pt2di(1,1)
          );
}


void EliseCorrel2D::ComputeCorrel(Pt2di aDec)
{
     Box2di aBox = BoxOk(aDec,0);

     ELISE_COPY(mCorrel.all_pts(),-1e6,mCorrel.out());


     Symb_FNum aFonc =  (
                           (
                              Moy(trans(mI1.in_proj(),aDec)*mI2.in_proj())
                              -trans(mS1.in(0),aDec)*mS2.in(0)
                           )
                           / sqrt(trans(mS11.in(Epsilon),aDec)*mS22.in(Epsilon))
                        );

     REAL aMin,aMax;
     ELISE_COPY
     (
         aBox.Flux(),
         aFonc,
         mCorrel.out() | VMax(aMax) | VMin(aMin)
     );
}


/*
void EliseCorrel2D::ComputeCorrelMax(Pt2di aDec0,INT anIncert)
{
    ELISE_COPY(mCorrelMax.all_pts(),-1e7,mCorrelMax.out());

    for (INT x=-anIncert ; x<=anIncert ; x++)
        for (INT y=-anIncert ; y<=anIncert ; y++)
        {

              Pt2di aDec = aDec0 + Pt2di(x,y);
              ComputeCorrel(aDec);

              ELISE_COPY
              (
                  select(mCorrelMax.all_pts(),mCorrel.in()>mCorrelMax.in()),
                  Virgule(aDec.x,aDec.y),
                     mDec.Out()
                  | (mCorrelMax.out() << mCorrel.in())
              );
        }
}
*/

Fonc_Num EliseCorrel2D::DecIn()
{
   return mDec.In();
}

Output EliseCorrel2D::DecOut()
{
   return mDec.Out();
}

Pt2di EliseCorrel2D::TrInitFromScratch
       (
	    INT   aZoomOverRes,
	    REAL  aRatioBord,
	    REAL  FactLissage
       )
{
   Im2D<tElInit,tBaseElInit>  anI1 = mI1.gray_im_red(aZoomOverRes);
   Im2D<tElInit,tBaseElInit>  anI2 = mI2.gray_im_red(aZoomOverRes);

   Pt2di aP = EliseCorrelation::RechTransMaxCorrel
              (
                  anI1.in(mVOut),
                  anI2.in(mVOut),
		  anI1.sz(),
		  aRatioBord,
		  FactLissage/aZoomOverRes,
                  1,
                  1e-5,
                  mUseVOut,
                  mVOut
              );

   return aP * aZoomOverRes;
}

void EliseCorrel2D::InitFromSousResol(EliseCorrel2D & aC2D_Red,INT aRatio)
{
     ELISE_COPY
     (
         rectangle(Pt2di(0,0),mSzIm),
         aC2D_Red.mDec.In()[Virgule(FX,FY)/aRatio]*aRatio,
         mDec.Out()
     );
     ELISE_COPY
     (
         rectangle(Pt2di(0,0),mSzIm),
         aC2D_Red.mCorrelMax.in(-1)[Virgule(FX,FY)/aRatio],
         mCorrelMax.out()
     );
}

void EliseCorrel2D::RaffineFromVois
     (
         INT   anIncert,
         Pt2di aDecGlob,
         EliseDecCor2D &       aDecOut,
         const EliseDecCor2D & aDecIn,
         INT aNbVoisRech,
         INT aSzRech
     )
{
   Pt2di aPIncert(anIncert,anIncert);
   TabuledCollecPt2di aSet(-aPIncert,aPIncert+Pt2di(1,1));


   Box2di aBox = BoxOk(aDecGlob,aNbVoisRech);

   INT cptMv =0;
   INT cptStat =0;
   for (INT y=aBox.P0().y ; y<aBox.P1().y ; y++)
   {
       for (INT x=aBox.P0().x ; x<aBox.P1().x ; x++)
       {

            Pt2di aPIm2(x,y);
            REAL CorMax = -1e9;
            Pt2di aDecMax = aDecIn.anIDec(aPIm2);
            Pt2di aD0 = aDecMax;
            for (INT dx = -aNbVoisRech ; dx<= aNbVoisRech ; dx++)
            {
               for (INT dy = -aNbVoisRech ; dy<= aNbVoisRech ; dy++)
               {
                    Pt2di aDec = aDecIn.anIDec(aPIm2 + Pt2di(dx,dy));
                    for (INT rx = -aSzRech; rx <=aSzRech; rx++)
                        for (INT ry = -aSzRech; ry <=aSzRech; ry++)
                        {
                             Pt2di aNewDec(aDec.x+rx,aDec.y+ry);
                             Pt2di aDecLoc = aNewDec-aDecGlob;
                             if (aSet.NbPres(aDecLoc,1) == 0)
                             {
                                 aSet.Add(aDecLoc);
                                 REAL aCor = Correl(aPIm2+aNewDec,aPIm2);
                                 if (aCor > CorMax)
                                 {
                                     CorMax = aCor;
                                     aDecMax = aNewDec;
                                 }
                             }
                        }
               }
            }
            if (aD0 == aDecMax)
               cptStat++;
            else
               cptMv++;
            aDecOut.SetDec(aPIm2,Pt2dr(aDecMax));
            aSet.clear();
       }
   }
}


void EliseCorrel2D::RaffineFromVois
     (
         INT   anIncert,
         Pt2di aDecGlob,
         INT aNbVoisRech,
         INT aSzRech,
         bool enParal
     )
{
    if (enParal)
    {
       EliseDecCor2D aRes(mSzIm);
       RaffineFromVois(anIncert,aDecGlob,aRes,mDec,aNbVoisRech,aSzRech);
       mDec = aRes;
    }
    else
       RaffineFromVois(anIncert,aDecGlob,mDec,mDec,aNbVoisRech,aSzRech);
}



Im2D_REAL4 EliseCorrel2D::SubPixelaire
     (
             Pt2di aDecGlob,
             REAL aStepLim,
             INT  aSzV
     ) 
{ 
   Im2D_REAL4 aRes(mSzIm.x,mSzIm.y,-2.0);

   OptimTranslationCorrelation<tElInit> anOpt
                                        (
                                             aStepLim,
                                             1.0,
                                             1,
                                             mI2,
                                             mI1,
                                             aSzV,
                                             1.0
                                        );
   Box2di aBox = BoxOk(aDecGlob,aSzV);

   Pt2dr PtEcart (0,0);
   INT   NbEcart = 0;

   for (INT y=aBox.P0().y ; y<aBox.P1().y ; y++)
   {
       for (INT x=aBox.P0().x ; x<aBox.P1().x ; x++)
       {
            Pt2di aP2(x,y);
            aRes.data()[y][x] = -2;

            if (OkIm2(aP2))
            {
               anOpt.SetP0Im1(Pt2dr(aP2));

               Pt2dr aPInit = mDec.RDec(aP2);
               anOpt.optim(aPInit);
               Pt2dr aPSol = anOpt.param();

                if (OkIm1 (aP2+Pt2di(aPSol)))
                {
                    mDec.SetDec(aP2,aPSol);
                    PtEcart += (aPSol-aPInit);
                    NbEcart++;
                    aRes.data()[y][x] = (float) anOpt.ScOpt();
                }
            }
       }
    }
    return aRes;
}


#include "im_tpl/correl_imget.h"

Im2D_REAL4  EliseCorrel2D::DiffSubPixelaire
     (
             Pt2di aDecGlob,
             INT  aSzV
     ) 
{ 
  ELISE_ASSERT(false,"in EliseCorrel2D::DiffSubPixelair : ADD okIM1, okIM2");
   Pt3dr aDefOut(-10,-10,-10);
   Im2D_REAL4 aRes(mSzIm.x,mSzIm.y,-2.0);

   OptCorrSubPix_Diff<tElInit>     anOpt (mI2,mI1,aSzV,1.0,aDefOut);
   Box2di aBox = BoxOk(aDecGlob,aSzV);

   Pt2dr PtEcart (0,0);
   INT   NbEcart = 0;

   for (INT y=aBox.P0().y ; y<aBox.P1().y ; y++)
   {
       for (INT x=aBox.P0().x ; x<aBox.P1().x ; x++)
       {
            Pt2di aP(x,y);
            for (INT aStep =0; aStep < 3 ; aStep++)
            {
                Pt2dr aPInit = Pt2dr(aP)+mDec.RDec(aP);
 
                Pt3dr aPopt = anOpt.Optim(Pt2dr(aP),aPInit);
                Pt2dr aPSol(aPopt.x,aPopt.y);
                mDec.SetDec(aP,aPSol-Pt2dr(aP));

                 if (aStep==0)
                 {
                     PtEcart += (aPSol-aPInit);
                     NbEcart++;
                 }
                 aRes.data()[y][x] = (float) aPopt.z;
             }
       }
    }
    return  aRes;
}

void EliseCorrel2D:: SetSzI12(Pt2di aSz)
{
   mSzI12 = aSz;
   I1().Resize(aSz);
   I2().Resize(aSz);
}

void EliseCorrel2D::SetI1GeomRadiomI2()
{
     if (!mD1GRI2)
     {
         mI1GeomRadiomI2 =  Im2D<tElem,REAL> (mSzIm.x,mSzIm.y,0.0);
         mD1GRI2 = mI1GeomRadiomI2.data();
     }
}

Fonc_Num EliseCorrel2D::Fonc1() { return mI1.in(0); }
Fonc_Num EliseCorrel2D::Fonc2() { return mI2.in(0); }

Pt2dr EliseCorrel2D::Homol(Pt2di aP) const
{
     Pt2dr aRes =  Pt2dr(aP) + mDec.RDec(aP);
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
