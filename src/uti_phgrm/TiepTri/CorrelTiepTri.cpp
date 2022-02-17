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

#include "TiepTri.h"

template <class Type> bool inside_window(const Type & Im1, const Pt2di & aP1,const int &   aSzW)
{
   return Im1.inside(aP1-Pt2di(aSzW,aSzW)) && Im1.inside(aP1+Pt2di(aSzW,aSzW));
}

Pt2dr  DoubleSol(const  RMat_Inertie & aMatr)
{
   return Pt2dr
          (
              aMatr.correlation(),
              LSQMoyResiduDroite(aMatr)
          );
}

bool  USE_SCOR_CORREL = true;

double STD_SCORE(const Pt2dr & a2Sol)
{
    return  USE_SCOR_CORREL ? a2Sol.x : (-a2Sol.y);
}

double DefScoreOpt() {return USE_SCOR_CORREL ? TT_DefCorrel : -1e20;}

/********************************************************************************/
/*                                                                              */
/*                  Auto-Correlation                                            */
/*                                                                              */
/********************************************************************************/

// Classe pour calculer de l'autocorrelation rapide
/*
template <class TypeIm> class cCutAutoCorrelDir : public cAutoCorrelDir<TypeIm>
{
    public :
         cCutAutoCorrelDir(TypeIm anIm,const Pt2di & aP0,double aRho,int aSzW ) :
             cAutoCorrelDir<TypeIm> (anIm,aP0,aRho,aSzW),
             mNbPts                 (SortedAngleFlux2StdCont(mVPt,circle(Pt2dr(0,0),aRho)).size())
         {
         }

         bool  AutoCorrel(double aRejetInt,double aRejetReel,double aSeuilAccept)
         {
               double aCorrMax = -2;
               int    aKMax = -1;
               for (int aK=0 ; aK<mNbPts ; aK++)
               {
                    double aCor = ICorrelOneOffset(this->mP0,mVPt[aK],this->mSzW); 
                    if (aCor > aSeuilAccept) return true;
                    if (aCor > aCorrMax)
                    {
                        aCorrMax = aCor;
                        aKMax = aK;
                    }
               }
               ELISE_ASSERT(aKMax!=1,"AutoCorrel no K");
               if (aCorrMax < aRejetInt) return false;

               Pt2dr aRhoTeta = Pt2dr::polar(Pt2dr(mVPt[aKMax]),0.0);

               double aStep0 = 1/this->mRho;
               Pt2dr aRes1 =  this->DoItOneStep(aRhoTeta.y,aStep0*0.5,2);

               if (aRes1.y>aSeuilAccept)   return true;
               if (aRes1.y<aRejetReel)     return false;

               Pt2dr aRes2 =  this->DoItOneStep(aRes1.x,aStep0*0.2,2);

               return aRes2.y > aCorrMax;
         }

    private :
         int mNbPts;
         std::vector<Pt2di> mVPt;
};
*/


void UUUU()
{
    TIm2D<double,double> anIm(Pt2di(1,1));

    cCutAutoCorrelDir<TIm2D<double,double> > aCACD(anIm,Pt2di(1,1),3.0,3);
}




/********************************************************************************/
/*                                                                              */
/*                  Correlation sub-pixel, interpol bilin basique               */
/*                                                                              */
/********************************************************************************/


Pt2dr TT_CorrelBilin
                             (
                                const tTImTiepTri & Im1,
                                const Pt2di & aP1,
                                const tTImTiepTri & Im2,
                                const Pt2dr & aP2,
                                const int   aSzW
                             )
{
 
     if (! (inside_window(Im1,aP1,aSzW) && inside_window(Im2,round_ni(aP2),aSzW+1))) return Pt2dr(TT_DefCorrel,1e20);

     Pt2di aVois;
     RMat_Inertie aMatr;

     for  (aVois.x = -aSzW ; aVois.x<=aSzW  ; aVois.x++)
     {
          for  (aVois.y = -aSzW ; aVois.y<=aSzW  ; aVois.y++)
          {
               aMatr.add_pt_en_place(Im1.get(aP1+aVois),Im2.getr(aP2+Pt2dr(aVois)));
          }
     }
     
     return  DoubleSol(aMatr);
}

/********************************************************************************/
/*                                                                              */
/*                  Correlation entiere                                         */
/*                                                                              */
/********************************************************************************/

Pt2dr TT_CorrelBasique
                             (
                                const tTImTiepTri & Im1,
                                const Pt2di & aP1,
                                const tTImTiepTri & Im2,
                                const Pt2di & aP2,
                                const int   aSzW,
                                const int   aStep
                             )
{
 
     if (! (inside_window(Im1,aP1,aSzW*aStep) && inside_window(Im2,aP2,aSzW*aStep))) return Pt2dr(TT_DefCorrel,1e20);

     Pt2di aVois;
     RMat_Inertie aMatr;

     for  (aVois.x = -aSzW ; aVois.x<=aSzW  ; aVois.x++)
     {
          for  (aVois.y = -aSzW ; aVois.y<=aSzW  ; aVois.y++)
          {
               aMatr.add_pt_en_place(Im1.get(aP1+aVois*aStep),Im2.get(aP2+aVois*aStep));
          }
     }
     
     return DoubleSol(aMatr);
}




cResulRechCorrel   TT_RechMaxCorrelBasique
                      (
                             const tTImTiepTri & Im1,
                             const Pt2di & aP1,
                             const tTImTiepTri & Im2,
                             const Pt2di & aP2,
                             const int   aSzW,
                             const int   aStep,
                             const int   aSzRech
                      )
{
    double aScoreMax = -1e30;
    Pt2di  aDecMax;
    Pt2dr  a2SolMax;
    Pt2di  aP;
    for (aP.x=-aSzRech ; aP.x<= aSzRech ; aP.x++)
    {
        for (aP.y=-aSzRech ; aP.y<= aSzRech ; aP.y++)
        {
             Pt2dr a2Sol  = TT_CorrelBasique(Im1,aP1,Im2,aP2+aP,aSzW,aStep);
       
             if (STD_SCORE(a2Sol) > aScoreMax)
             {
                 aScoreMax = STD_SCORE(a2Sol);
                 a2SolMax = a2Sol;
                 aDecMax = aP;
             }
        }
     }

     return cResulRechCorrel(Pt2dr(aP2+aDecMax),a2SolMax.x);

}

/********************************************************************************/
/*                                                                              */
/*                  Optimisation                                                */
/*                                                                              */
/********************************************************************************/
typedef enum eTypeModeCorrel
{
    eTMCInt = 0,
    eTMCBilinStep1 = 1,
    eTMCOther = 2
}  eTypeModeCorrel;


class c2Sol
{
     public :
         c2Sol () :
            mScoreMax (-1e20)
         {
         }

         void Update(const Pt2dr& a2Sol)
         {
             mLastSol = a2Sol;
             double aSc = STD_SCORE(a2Sol);
             if (aSc>mScoreMax)
             {
                 mScoreMax = aSc;
                 mSolMax = a2Sol;
             }
         }
         double ScoreFinal() {return  mSolMax.x;}
         double Score4Opt()  {return  STD_SCORE(mLastSol);}

     private :

         Pt2dr   mLastSol;
         Pt2dr   mSolMax;
         double  mScoreMax;
};

 
class cTT_MaxLocCorrelBasique : public Optim2DParam
{
    public :
         REAL Op2DParam_ComputeScore(REAL aDx,REAL aDy) 
         {
            if (mMode == eTMCInt)
            {
               m2Sol.Update(TT_CorrelBasique(mIm1,Pt2di(mP1),mIm2,Pt2di(mP2)+Pt2di(round_ni(aDx),round_ni(aDy)),mSzW,mStep));
               return m2Sol.Score4Opt();
            }
            if (mMode == eTMCBilinStep1)
            {
               m2Sol.Update(TT_CorrelBilin(mIm1,Pt2di(mP1),mIm2,mP2+Pt2dr(aDx,aDy),mSzW));
               return m2Sol.Score4Opt();
            }

            return 0;
         }
 
         cTT_MaxLocCorrelBasique 
         ( 
              eTypeModeCorrel     aMode,
              const tTImTiepTri & aIm1,
              Pt2dr               aP1,
              const tTImTiepTri & aIm2,
              Pt2dr               aP2,
              const int           aSzW,
              const int           aStep,
              double              aStepRechCorrel
         )  :
            Optim2DParam ( aStepRechCorrel, DefScoreOpt() ,1e-5, true),
            mMode (aMode),
            mIm1 (aIm1),
            mP1  (aP1),
            mIm2 (aIm2),
            mP2  (aP2),
            mSzW (aSzW),
            mStep (aStep)
         {
         }
            
         eTypeModeCorrel     mMode;
         const tTImTiepTri & mIm1;
         Pt2dr               mP1;
         const tTImTiepTri & mIm2;
         Pt2dr               mP2;
         const int           mSzW;
         const int           mStep;
         c2Sol               m2Sol;
};

cResulRechCorrel      TT_RechMaxCorrelLocale
                      (
                             const tTImTiepTri & aIm1,
                             const Pt2di & aP1,
                             const tTImTiepTri & aIm2,
                             const Pt2di & aP2,
                             const int   aSzW,
                             const int   aStep,
                             const int   aSzRechMax
                      )

{
   cTT_MaxLocCorrelBasique  anOpt(eTMCInt,aIm1,Pt2dr(aP1),aIm2,Pt2dr(aP2),aSzW,aStep,0.9);
   anOpt.optim_step_fixed(Pt2dr(0,0),aSzRechMax);

   return cResulRechCorrel(Pt2dr(aP2+round_ni(anOpt.param())),anOpt.m2Sol.ScoreFinal());
}

cResulRechCorrel TT_RechMaxCorrelMultiScaleBilin
                      (
                             const tTImTiepTri & aIm1,
                             const Pt2dr & aP1,
                             const tTImTiepTri & aIm2,
                             const Pt2dr & aP2,
                             const int   aSzW,
                             double aStepFinal
                      )

{
   cTT_MaxLocCorrelBasique  anOpt(eTMCBilinStep1,aIm1,aP1,aIm2,aP2,aSzW,1,aStepFinal);
   anOpt.optim();

   return cResulRechCorrel(aP2+anOpt.param(),anOpt.m2Sol.ScoreFinal());
}


/********************************************************************/
/*                                                                  */
/*                                                                  */
/*           Correlation avec  :                                    */
/*                        *   interpol SinC                         */
/*                        *   fenetre dense                         */
/*                        *   1 seul reech                          */
/*                                                                  */
/*                                                                  */
/********************************************************************/

class cTT_MaxLocCorrelDS1R : public Optim2DParam
{
     private :

         REAL Op2DParam_ComputeScore(REAL aDx,REAL aDy) ;
         double               mStep0;
         std::vector<double>  mVals1; // Les valeurs interpolees de l'image 1 sont stockees une fois pour toute
         const tTImTiepTri & mIm1;
         
         std::vector<Pt2dr> mVois2; // Les voisin de l'images 2 sont stockes une fois pour toute
         const tTImTiepTri & mIm2;
         tElTiepTri **       mData2;
         Pt2dr               mDecInit;
         tInterpolTiepTri *  mInterpol;
         double              mSzInterp;
         Pt2dr               mPInf2;
         Pt2dr               mPSup2;
         Pt2di               mSzIm1;
         Pt2di               mSzIm2;
         bool                mOkIm1;

     public :
         c2Sol               m2Sol;

         bool   OkIm1() const {return mOkIm1;}
         cTT_MaxLocCorrelDS1R 
         ( 
              tInterpolTiepTri *  anInterpol,
              cElMap2D *          aMap,
              const tTImTiepTri & aIm1,
              Pt2dr               aPC1,
              const tTImTiepTri & aIm2,
              Pt2dr               aPC2,
              const int           aSzW,
              const int           aNbByPix,
              double              aStep0,
              double              aStepEnd
         )  :
            Optim2DParam ( aStepEnd/aStep0 , DefScoreOpt() ,1e-5, true),
            mStep0    (aStep0),
            mIm1      (aIm1),
            mIm2      (aIm2),
            mData2    (aIm2._the_im.data()),
            mInterpol (anInterpol),
            mSzInterp (anInterpol->SzKernel()+2),
            mPInf2    (1e30,1e30),
            mPSup2    (-1e30,-1e30),
            mSzIm1    (aIm1.sz()),
            mSzIm2    (aIm2.sz()),
            mOkIm1    (true)
         {
            tElTiepTri ** aData1 = aIm1._the_im.data();
            int aNbVTot = aSzW * aNbByPix;
            mDecInit = aPC2 - (*aMap)(aPC1);
            for (int aKx = -aNbVTot ; aKx<=aNbVTot ; aKx++)
            {
               for (int aKy = -aNbVTot ; aKy<=aNbVTot ; aKy++)
               {
                   Pt2dr aVois(aKx/double(aNbByPix),aKy/double(aNbByPix));
                   Pt2dr aP1 = aPC1 + aVois;
                   if (
                            (aP1.x <= mSzInterp)
                         || (aP1.y <= mSzInterp)
                         || (aP1.x >= mSzIm1.x+mSzInterp)
                         || (aP1.y >= mSzIm1.y+mSzInterp)
                      )
                   {
                       mOkIm1 = false;
                   }
                   else
                   {
                       mVals1.push_back(anInterpol->GetVal(aData1,aP1));
                   }
                   Pt2dr aP2 = (*aMap)(aP1) + mDecInit;
                   mVois2.push_back(aP2);
                   mPInf2 = Inf(mPInf2,aP2);
                   mPSup2 = Sup(mPSup2,aP2);
               }
            }
            mPInf2 = mPInf2 - Pt2dr(mSzInterp,mSzInterp);
            mPSup2 = mPSup2 + Pt2dr(mSzInterp,mSzInterp);
         }
};

REAL cTT_MaxLocCorrelDS1R::Op2DParam_ComputeScore(REAL aDx,REAL aDy) 
{
    ELISE_ASSERT(mOkIm1,"cTT_MaxLocCorrelDS1R::Op2DParam_ComputeScore not ok Im1");
    aDx *= mStep0;
    aDy *= mStep0;

    if (
             (mPInf2.x + aDx <=0) 
          || (mPInf2.y + aDy <=0) 
          || (mPSup2.x + aDx >=mSzIm2.x) 
          || (mPSup2.y + aDy >=mSzIm2.y) 
        )
     {
// std::cout << "RRRRr OOUTTTtttttttttttt\n";
        return DefScoreOpt();
     }
// std::cout << "RRRRr IIIIIiiinnnnnnnnnnn\n";

     Pt2dr aDec(aDx,aDy);
     RMat_Inertie aMatr;
     for (int aKV=0 ; aKV<int(mVals1.size()) ; aKV++)
     {
         aMatr.add_pt_en_place
         (
              mVals1[aKV],
              mInterpol->GetVal(mData2,mVois2[aKV] + aDec)
         );
     }
     // double aRes =  aMatr.correlation();
     m2Sol.Update(DoubleSol(aMatr));
     return  m2Sol.Score4Opt();
}

cResulRechCorrel TT_MaxLocCorrelDS1R 
                         ( 
                              tInterpolTiepTri *  anInterpol,
                              cElMap2D *          aMap,
                              const tTImTiepTri & aIm1,
                              Pt2dr               aPC1,
                              const tTImTiepTri & aIm2,
                              Pt2dr               aPC2,
                              const int           aSzW,
                              const int           aNbByPix,
                              double              aStep0,
                              double              aStepEnd
                         )
{

   cTT_MaxLocCorrelDS1R anOptim(anInterpol,aMap,aIm1,aPC1,aIm2,aPC2,aSzW,aNbByPix,aStep0,aStepEnd);

   cResulRechCorrel aResult;
   if (!anOptim.OkIm1()) return aResult;

   anOptim.optim();

   return cResulRechCorrel(aPC2+anOptim.param() * aStep0, anOptim.m2Sol.ScoreFinal());
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
aooter-MicMac-eLiSe-25/06/2007*/
