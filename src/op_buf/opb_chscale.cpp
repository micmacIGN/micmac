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

    Fonc_Num_OP_Buf  :  pour memoriser+bufferiser la fonction entrante

        Arg_FNOPB :  Fonc_Num entrante + Box de la FoncNum entrante
        Arg_Fonc_Num_Comp : le flux compile

*/

/************************************************************/
/*                                                          */
/*             cFiltreInterpol                              */
/*                                                          */
/************************************************************/

class cFiltreInterpol
{
     public :
         double Val(double aV) const
         {
              return  Val_0_1((aV-mV0)/mSc);
         }
         cFiltreInterpol(double aSupport) :
             mSupport (aSupport),
             mV0(0),
             mSc (1.0)
         {
         }
         virtual ~cFiltreInterpol() {}

         void SetVO(double aV0) {mV0=aV0;}
         void SetSc(double aSc) {mSc=aSc;}
         double Support() const {return mSupport*mSc;}
     private :
         double  mSupport;
         double  mV0;
         double  mSc;

         virtual double  Val_0_1(double) const = 0;
};

/************************************************************/
/*                                                          */
/*             cFiltreInterpol_BiCub                        */
/*                                                          */
/************************************************************/

class cFiltreInterpol_BiCub : public cFiltreInterpol
{
     public :
         cFiltreInterpol_BiCub(double aValB) :
             cFiltreInterpol (2.0),
             mKernel         (aValB)
         {
         }

         static cFiltreInterpol_BiCub * StdFromScale(double aSc)
         {
             if (aSc< 1.0)
               return new cFiltreInterpol_BiCub(-0.5);
             if (aSc< 1.5)
               return new cFiltreInterpol_BiCub(aSc-1.5);
             return new cFiltreInterpol_BiCub(0.0);
         }

         static cFiltreInterpol_BiCub * StdFromScaleNonNeg(double aSc)
         {
             return new cFiltreInterpol_BiCub(0.0);
         }

     private :

         double  Val_0_1(double aV) const 
         {
            return mKernel.Value(aV);
         }
         cCubicInterpKernel mKernel;
};

class cFiltreInterpol_Bilin : public cFiltreInterpol
{
     public :
         cFiltreInterpol_Bilin(double aValB) :
             cFiltreInterpol (1.0)
         {
         }


     private :

         double  Val_0_1(double aV) const 
         {
            if (aV>0)
            {
                return (aV<1) ? (1-aV) : 0 ;
            }
            return  (aV>-1) ? (1+aV) : 0;
         }
};














/************************************************************/
/*                                                          */
/*                 cHomTr1D                                 */
/*                                                          */
/************************************************************/

class cHomTr1D
{
      public :

          cHomTr1D(double aTr,double aSc) :
             mTr (aTr),
             mSc (aSc)
          {
          }

          double Direct(double aV)   const { return  aV*mSc + mTr; }
          double Inverse(double aV)  const { return  (aV-mTr)/mSc ; }

          double Sc() const {return mSc;}
      private :
          double mTr;
          double mSc;
};


/************************************************************/
/*                                                          */
/*             cPixelSortie_Convol                          */
/*                                                          */
/************************************************************/



class cPixelSortie_Convol
{
    public :
       cPixelSortie_Convol
       (
            cFiltreInterpol &  aFiltre,
            const cHomTr1D &         aHom,
            double                   aPixOut,
            double                   aDilate
       );

       cPixelSortie_Convol(const cPixelSortie_Convol & aMPS):
			mVPds(aMPS.mVPds),
			mVPdsInit(aMPS.mVPdsInit)
       {
            mPixIn0 =   aMPS.mPixIn0;
            mPixIn1 =   aMPS.mPixIn1;
            mPixOut =   aMPS.mPixOut;
            mPdsInit  = &(mVPdsInit[0]) - mPixIn0;
            mPds      = &(    mVPds[0]) - mPixIn0;
       }
       int PixIn0() const {return mPixIn0;}
       int PixIn1() const {return mPixIn1;}
 
 
       double SomPond(double * aData) const;
       double Pds(int aPix) const {return mPds[aPix];}

    private :
       int mPixIn0;
       int mPixIn1;

       double  mPixOut;


       double * mPdsInit;
       double * mPds;

       std::vector<double> mVPds;
       std::vector<double> mVPdsInit;
};

double cPixelSortie_Convol::SomPond(double * aData) const
{
   double aRes = 0;
   for (int aPix=mPixIn0 ; aPix<mPixIn1 ; aPix++)
       aRes += aData[aPix] * mPds[aPix];
   return aRes;
}

cPixelSortie_Convol::cPixelSortie_Convol
(
            cFiltreInterpol &  aFiltre,
            const cHomTr1D &         aHom,
            double                   aPixOut,
            double                   aDilate
)   :
    mPixOut  (aPixOut)
{
    double aPixIn = aHom.Direct(mPixOut);
    aFiltre.SetVO(aPixIn);
    aFiltre.SetSc(ElMax(1.0,aHom.Sc()*aDilate));

    mPixIn0 = round_up(aPixIn -aFiltre.Support());
    mPixIn1 = 1+ round_down(aPixIn +aFiltre.Support());
  
    double aSom=0.0;
    for (int aPix = mPixIn0; aPix<mPixIn1 ; aPix++)
    {
         mVPdsInit.push_back(aFiltre.Val(aPix));
         aSom+=mVPdsInit.back();
    }    
    mPdsInit = &(mVPdsInit[0]) - mPixIn0;
    for (int aPix = mPixIn0; aPix<mPixIn1 ; aPix++)
    {
        mPdsInit[aPix] /= aSom;
        mVPds.push_back(mPdsInit[aPix]);
    }
    mPds =  &(mVPds[0]) - mPixIn0;
}

/************************************************************/
/*                                                          */
/*             cLineSortie_Convol                           */
/*                                                          */
/************************************************************/

class cLineSortie_Convol
{
     public :
        cLineSortie_Convol
        (
            cFiltreInterpol &  aFiltre,
            const cHomTr1D &   aHom,
            int                aPixOut0,
            int                aPixOut1,
            double             aDilateF
        );
        const cPixelSortie_Convol & Pix(int aPix) const;
        int LargMax() const {return mLargMax;}
        int PixIn0()  const {return mPixIn0;}
        int PixIn1()  const {return mPixIn1;}

     private :
        cLineSortie_Convol(const cLineSortie_Convol &);

        int mPixOut0;
        int mPixOut1;
        int mPixIn0;
        int mPixIn1;
        int mLargMax;
        std::vector<cPixelSortie_Convol> mVPSC;
};


const cPixelSortie_Convol & cLineSortie_Convol::Pix(int aPix) const
{
   ELISE_ASSERT
   (
       (aPix>=mPixOut0) && (aPix<mPixOut1),
       "Bad Access to cLineSortie_Convol::Pix"
   );
   return mVPSC[aPix-mPixOut0];
}

cLineSortie_Convol::cLineSortie_Convol
(
    cFiltreInterpol &  aFiltre,
    const cHomTr1D &   aHom,
    int                aPixOut0,
    int                aPixOut1,
    double             aDilateF
) :
   mPixOut0 (aPixOut0),
   mPixOut1 (aPixOut1)
{
   mVPSC.reserve(mPixOut1-mPixOut0);
   mLargMax =0;
   for (int aPixOut=mPixOut0 ; aPixOut<mPixOut1 ; aPixOut++)
   {
       mVPSC.push_back(cPixelSortie_Convol(aFiltre,aHom,aPixOut,aDilateF));
       cPixelSortie_Convol & mPCur = mVPSC[mVPSC.size()-1];
       ElSetMax(mLargMax,mPCur.PixIn1()-mPCur.PixIn0());
       if (aPixOut==mPixOut0)
       {
       }
       else
       {
           cPixelSortie_Convol & mPPrec = mVPSC[mVPSC.size()-2];
           ELISE_ASSERT
           (
                    mPCur.PixIn0()>=mPPrec.PixIn0()
               &&  (mPCur.PixIn1()>=mPPrec.PixIn1()),
               "Incoherence in cPixelSortie_Convol"
           );
       }
   }
   mPixIn0 = mVPSC[0].PixIn0();
   mPixIn1 = mVPSC.back().PixIn1();
}

/************************************************************/
/*                                                          */
/*             cLineSortie_Convol                           */
/*                                                          */
/************************************************************/

class cOpChc_BufCannaux
{
      public :
          cOpChc_BufCannaux(int aNbCanal,int aInOut0,int aInOut1) :
              mP0     (aInOut0,0),
              mP1     (aInOut1,aNbCanal),
              mBufOut (NEW_MATRICE(mP0,mP1,double))
          {
          }
          ~cOpChc_BufCannaux()
          {
              DELETE_MATRICE(mBufOut,mP0,mP1);
          }
          double * BufOut(int aDim)
          {
              return mBufOut[aDim];
          }
      private :
          Pt2di     mP0;
          Pt2di     mP1;
          double ** mBufOut;
};


class cOpBuf_Chc : public Simple_OPBuf1<double,double>
{
     public :
         Simple_OPBuf1<double,double> * dup_comp();
         ~cOpBuf_Chc();
         void  calc_buf (double ** output,double *** input);

         cOpBuf_Chc
         (
            Fonc_Num           aFonc,
            Pt2dr              aTr,
            Pt2dr              aSc,
            Pt2dr              aDilate,
            cFiltreInterpol *  aFiltreX,
            cFiltreInterpol *  aFiltreY,
            bool               DeleteFiltre
          );
           
     private :
         
         cOpChc_BufCannaux & BufCan(int aY)
         {
             return *(mCanaux[mod(aY,mLineY->LargMax())]);
         }
        
          
         Fonc_Num             mFonc;
         Pt2dr                mTr;
         Pt2dr                mSc;
         Pt2dr                mDilate;
         cFiltreInterpol    * mFiltreX;
         cFiltreInterpol    * mFiltreY;
         bool                 mDeleteFiltre;

         cLineSortie_Convol * mLineX;
         cLineSortie_Convol * mLineY;
         Flux_Pts_Computed *  mFluxRectIn;
         Fonc_Num_OP_Buf *    mFBuf;
         int                  mYInCur;
         std::vector<cOpChc_BufCannaux *>  mCanaux;

};

cOpBuf_Chc::cOpBuf_Chc
(
    Fonc_Num           aFonc,
    Pt2dr              aTr,
    Pt2dr              aSc,
    Pt2dr              aDilate,
    cFiltreInterpol *  aFiltreX,
    cFiltreInterpol *  aFiltreY,
    bool               aDeleteFiltre
) :
  mFonc         (aFonc),
  mTr           (aTr),
  mSc           (aSc),
  mDilate       (aDilate),
  mFiltreX      (aFiltreX),
  mFiltreY      (aFiltreY),
  mDeleteFiltre (aDeleteFiltre),
  mLineX        (0),
  mLineY        (0),
  mFluxRectIn   (0),
  mFBuf         (0),
  mYInCur       (123456789)
{
}

cOpBuf_Chc::~cOpBuf_Chc()
{
   if (mDeleteFiltre)
   {
      delete mFiltreX;
      delete mFiltreY;
   }

   delete mLineX;
   delete mLineY;
   delete mFluxRectIn;
   delete mFBuf;
   DeleteAndClear(mCanaux);
}



Simple_OPBuf1<double,double> * cOpBuf_Chc::dup_comp()
{
    cOpBuf_Chc * aRes = new cOpBuf_Chc(mFonc,mTr,mSc,mDilate,mFiltreX,mFiltreY,false);
    aRes->mLineX = new cLineSortie_Convol(*mFiltreX,cHomTr1D(mTr.x,mSc.x),x0(),x1(),mDilate.x);
    aRes->mLineY = new cLineSortie_Convol(*mFiltreY,cHomTr1D(mTr.y,mSc.y),y0(),y1(),mDilate.y);

    aRes->mFluxRectIn = RLE_Flux_Pts_Computed::rect_2d_interface
                        (
                            Pt2di(aRes->mLineX->PixIn0(),aRes->mLineY->PixIn0()),
                            Pt2di(aRes->mLineX->PixIn1(),aRes->mLineY->PixIn1()),
                            500
                        );


    aRes->mFBuf = new Fonc_Num_OP_Buf
                 (
                      Arg_Fonc_Num_Comp(aRes->mFluxRectIn),
                      Arg_FNOPB(aRes->mFonc,Box2di(Pt2di(0,0),Pt2di(0,0)),GenIm::real8),
                      Arg_FNOPB::def,
                      Arg_FNOPB::def,
                      false
                 );

    aRes->mYInCur = aRes->mLineY->PixIn0();

    for (int aK=0 ; aK<aRes->mLineY->LargMax() ; aK++)
    {
        aRes->mCanaux.push_back
        (
             new cOpChc_BufCannaux (dim_out(),x0(),x1())
        ); 
    }

    return aRes;
    
}





void  cOpBuf_Chc::calc_buf(double ** output,double *** )
{
    const cPixelSortie_Convol &  aYPixC = mLineY->Pix(ycur());

    for (   ;mYInCur<aYPixC.PixIn1()    ; mYInCur++)
    {
// std::cout << "Y = "  << mYInCur << "\n";
// if (mYInCur<=100) getchar();
         mFBuf->maj_buf_values(mYInCur);
         cOpChc_BufCannaux & aCan = BufCan(mYInCur);
         for (int aD=0 ; aD<dim_out() ; aD++)
         {
              double * aBufOut = aCan.BufOut(aD);
              double * aBufIn = mFBuf->kth_buf((double *)0,0)[aD][0];

              for (int anX=x0 () ; anX<x1() ; anX++)
                  aBufOut[anX] = mLineX->Pix(anX).SomPond(aBufIn);
         }
    }

    for (int aD=0 ; aD<dim_out() ; aD++)
    {
         double * OutL =  output[aD];

         for (int anX=x0 () ; anX<x1() ; anX++)
             OutL[anX] = 0;

         for (int anY=aYPixC.PixIn0(); anY<aYPixC.PixIn1() ; anY++)
         {
              double * aBufOut = BufCan(anY).BufOut(aD);
              double aPds = aYPixC.Pds(anY);
              for (int anX=x0 () ; anX<x1() ; anX++)
                  OutL[anX] +=  aPds * aBufOut[anX];
         }
    }
}



Fonc_Num  StdFoncChScale_Gen
          (
             Fonc_Num aFonc,Pt2dr aTr,Pt2dr aSc,Pt2dr aDilate,
             cFiltreInterpol * aIntX,
             cFiltreInterpol * aIntY
          )
{
   
  Pt2di aITr = round_ni(aTr);
  if ( (aSc==Pt2dr(1,1)) && (aTr == Pt2dr(aITr)))
  {
      return trans(aFonc,aITr);
  }


   aFonc = Rconv(aFonc);
   return create_op_buf_simple_tpl
          (
                (Simple_OPBuf1<INT,INT> *)0,
                new cOpBuf_Chc
                (
                     aFonc,aTr,aSc,aDilate,
                     aIntX,
                     aIntY,
                     true
                ),
                Fonc_Num(0),
                aFonc.dimf_out(),
                Box2di(Pt2di(0,0),Pt2di(0,0))
          );
}

Fonc_Num  StdFoncChScale(Fonc_Num aFonc,Pt2dr aTr,Pt2dr aSc,Pt2dr aDilate)
{
    return StdFoncChScale_Gen
           (
                 aFonc,aTr,aSc,aDilate,
                 cFiltreInterpol_BiCub::StdFromScale(aSc.x),
                 cFiltreInterpol_BiCub::StdFromScale(aSc.y)
           );
}

Fonc_Num  StdFoncChScale_BicubNonNeg(Fonc_Num aFonc,Pt2dr aTr,Pt2dr aSc,Pt2dr aDilate)
{
    return StdFoncChScale_Gen
           (
                 aFonc,aTr,aSc,aDilate,
                 cFiltreInterpol_BiCub::StdFromScaleNonNeg(aSc.x),
                 cFiltreInterpol_BiCub::StdFromScaleNonNeg(aSc.y)
           );
}


Fonc_Num  StdFoncChScale_Bilin(Fonc_Num aFonc,Pt2dr aTr,Pt2dr aSc,Pt2dr aDilate)
{
    return StdFoncChScale_Gen
           (
                 aFonc,aTr,aSc,aDilate,
                 new cFiltreInterpol_Bilin(aSc.x),
                 new cFiltreInterpol_Bilin(aSc.y)
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
