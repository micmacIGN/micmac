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



#ifndef _ELISE_IM_OPBUF_TPL_
#define _ELISE_IM_OPBUF_TPL_

template <class tArg> 
class  cTplOpbBufImage
{
       public :
     typedef typename tArg::tElem  tElem;
     typedef tElem *               tElPtr;
     typedef typename tArg::tCumul tCumul;
    // Conventions aBoxIm  : exclut P1
    //             aBoxWin : inclut P1
             cTplOpbBufImage 
             (
                  tArg &  anArg,
                  Box2di  aBoxIm,
                  Box2di  aBoxWin
             )  :
                mArg         (anArg),
                mBoxIm       (aBoxIm),
                mBoxWin      (aBoxWin),

                mYCurOut     (aBoxIm._p0.y),
                mY1Out       (aBoxIm._p1.y),

                mY0CurIn     (aBoxWin._p0.y+aBoxIm._p0.y),
                mY1CurIn     (aBoxWin._p1.y+aBoxIm._p0.y),
                mNbYIn       (mY1CurIn-mY0CurIn+1),

                mX0CurOut    (aBoxIm._p0.x),
                mX1CurOut    (aBoxIm._p1.x),
                mNbXOut      (mX1CurOut-mX0CurOut),

                mX0CurIn     (aBoxWin._p0.x+aBoxIm._p0.x),
                mX1CurIn     (aBoxWin._p1.x+aBoxIm._p1.x),
                mNbXIn       (mX1CurIn-mX0CurIn),
                mNbWX        (aBoxWin._p1.x-aBoxWin._p0.x),

                mMatElem     ((new tElPtr [mNbYIn])-mY0CurIn),
                mVCumul      ((new tCumul [mNbXOut])-mX0CurOut)
             {
                 for (int aYIn=mY0CurIn ; aYIn<= mY1CurIn ; aYIn++)
                 {
                     mMatElem[aYIn] = (new tElem [mNbXIn]) -mX0CurIn;
                     if (aYIn != mY1CurIn)
                        AddNewLine(aYIn);
                 }
             }
             int  XOutDebLigne() const {return mX0CurOut;}
             int  XOutFinLigne() const {return mX1CurOut;}
             ~cTplOpbBufImage()
             {
                 for (int aYIn=mY0CurIn; aYIn<=mY1CurIn ; aYIn++)
                     delete [] (mMatElem[aYIn]+mX0CurIn);
                 delete [] (mMatElem+mY0CurIn);
                 delete [] (mVCumul+mX0CurOut);
             }

             void DoIt()
             {
                  for (;mYCurOut<mY1Out ; mYCurOut++)
                  {
                      AddNewLine(mY1CurIn);

                      mArg.OnNewLine(mYCurOut);
                      for (int aXOut=mX0CurOut ; aXOut<mX1CurOut ; aXOut++)
                          mArg.UseAggreg(Pt2di(aXOut,mYCurOut),mVCumul[aXOut]);

                      CumulLine(mY0CurIn,-1);
                      // Permuation circulaire du "Buffer de lignes"
                      tElPtr aL0 = mMatElem[mY0CurIn];
                      for (INT aYIn = mY0CurIn  ; aYIn<mY1CurIn ; aYIn++)
                          mMatElem[aYIn] = mMatElem[aYIn+1];
                       mMatElem[mY1CurIn] = aL0;

                      mY0CurIn++;
                      mY1CurIn++;
                      mMatElem--;
                  }
             }

       private :

             cTplOpbBufImage(const cTplOpbBufImage &); // Non Implemente
             void AddNewLine(int aYIn)
             {
                  tElem * aL = mMatElem[aYIn];
                  
                  for (Pt2di aPIn(mX0CurIn,aYIn) ; aPIn.x<mX1CurIn ; aPIn.x++)
                     mArg.Init(aPIn,aL[aPIn.x]);
                  CumulLine(aYIn,1);
             }

             void CumulLine(int aYIn,int aSigne)
             {
                 tElem * aL = mMatElem[aYIn];
                 tCumul  anAccum;
                 for (int aXIn = mX0CurIn ; aXIn < mX0CurIn+mNbWX; aXIn++)
                     anAccum.AddElem(1,aL[aXIn]);

                 int aXArIn = mX0CurIn;
                 int aXAvIn = mX0CurIn+mNbWX;

                 for (int aXOut=mX0CurOut ; aXOut<mX1CurOut ; aXOut++)
                 {
                     anAccum.AddElem(1,aL[aXAvIn]);
                     mVCumul[aXOut].AddCumul(aSigne,anAccum);
                     anAccum.AddElem(-1,aL[aXArIn]);
                     aXArIn++;
                     aXAvIn++;
                 }
             }



            tArg &    mArg;
            Box2di    mBoxIm;
            Box2di    mBoxWin;

            int       mYCurOut;
            int       mY1Out;

            int       mY0CurIn;
            int       mY1CurIn;
            int       mNbYIn;

            int       mX0CurOut;
            int       mX1CurOut;
            int       mNbXOut;

            int       mX0CurIn;
            int       mX1CurIn;
            int       mNbXIn;
            int       mNbWX;
            INT       mCurYIn;

            tElem **  mMatElem;
            tCumul *  mVCumul;
};



//  Classe pour utilisation  dans de la correlation par moindres
// carres .

struct cElemOBICorrelMoindreCarres
{
      public :

         bool    mIsOk;
         double  mIm1;
         Pt2dr   mP1;
         Pt2dr   mP2;
         Pt3dr   mGrI2;
};

struct cCumulOBICorrelMoindreCarres
{
    public :
        typedef double tCum;

        cCumulOBICorrelMoindreCarres()
        {
             mP1 = Pt2dr(0,0);
             mP2 = Pt2dr(0,0);
             mMom.Init();
        }

       void AddCumul(int aS,const cCumulOBICorrelMoindreCarres & aCum)
       {
           mP1 +=   aCum.mP1 * aS;
           mP2 +=   aCum.mP2 * aS;
           mMom.Cumul(aS,aCum.mMom);
       }

        void AddElem(int aS,const cElemOBICorrelMoindreCarres & anEl)
        {
            if (anEl.mIsOk)
            {
                mP1 +=   anEl.mP1 * aS;
                mP2 +=   anEl.mP2 * aS;
                mMom.Add(aS,anEl.mIm1,anEl.mGrI2);
            }
        }

        Pt2dr            mP1;
        Pt2dr            mP2;
        cMomentCor2DLSQ  mMom;

};

template <class tElemIm> 
struct cArgOBICorrelMoindreCarres
{
     public :
         typedef cCumulOBICorrelMoindreCarres tCumul;
         typedef cElemOBICorrelMoindreCarres  tElem;

          void OnNewLine(int anY)
          {
              cout << anY << "\n";
              getchar();
          }

         void UseAggreg(Pt2di aP1,const tCumul & aCum)
         {
 // Pt3dr aR1 = aCum.mMom.GetSol(mBufSol);
         }

         void Init(Pt2di aP1,cElemOBICorrelMoindreCarres & anEl)
         {
             if (IsOkP1(aP1))
             {
                 Pt2dr aP2 = P1toP2(aP1);
                 if (IsOkP2(aP2))
                 {
                     anEl.mIsOk=true;
                     anEl.mIm1 = Im1(aP1);
                     anEl.mP1 = aP1;
                     anEl.mP2 = aP2;
                     anEl.mGrI2 = Im2AndG2(aP2);
                 }
                 else
                 {
                     anEl.mIsOk=false;
                 }
             }
             else
             {
                 anEl.mIsOk=false;
             }
         }

       // Fin des pre-requis 


         typedef typename El_CTypeTraits<tElemIm>::tBase  tBaseIm; 
         typedef Im2D<tElemIm,tBaseIm>                    tImage;
         cArgOBICorrelMoindreCarres
         (
              const ElDistortion22_Gen & aDist1to2,
              REAL            aVBiCub,
              tImage          aIm1,
              Im2D_Bits<1>    aMasq1,
              tImage          aIm2,
              Im2D_Bits<1>    aMasq2,
              INT             aSzV
         )  :
            mDist1to2 (aDist1to2),
            mBCKernel (aVBiCub),
            mIm1      (aIm1.data()),
            mMasq1    (aMasq1),
            mIm2      (aIm2.data()),
            mMasq2    (aMasq2),
            mVerif    (aIm1,aIm2,aSzV,1.0,Pt3dr(-1.0,-1.0,-1.0)),
            mPtOut    (-1,-1,-1),
            mBufSol   (mPtOut)
         {
              mVerif.SetModeBicub(true);
         }
    private :

         Pt2dr   P1toP2(Pt2di aP1)   {return mDist1to2.Direct(aP1);}
         bool    IsOkP1(Pt2di aP1)   {return mMasq1.get(aP1,0);}
         bool    IsOkP2(Pt2dr aP2)   {return mMasq2.get(round_ni(aP2),0);}
         REAL    Im1(Pt2di aP1)      {return mIm1[aP1.y][aP1.x];}
         Pt3dr   Im2AndG2(Pt2dr aP2) {return BicubicInterpol(mBCKernel,mIm2,aP2);}


         cArgOBICorrelMoindreCarres(const cArgOBICorrelMoindreCarres&); // N.I.

         const ElDistortion22_Gen & mDist1to2;
         cCubicInterpKernel         mBCKernel;
         tElemIm **                 mIm1;
         TIm2DBits<1>               mMasq1;                 
         tElemIm **                 mIm2;
         TIm2DBits<1>               mMasq2;                 
         cTplDiffCorrelSubPix<tElemIm>  mVerif;
         Pt3dr                         mPtOut;
    protected :
         cBufResMomentCor2DLSQ         mBufSol;
        
};



#endif //  _ELISE_IM_OPBUF_TPL_


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
