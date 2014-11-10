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
#include "im_special/hough.h"
#include "api/vecto.h"



/*
Pt2di Std2Elise(const ComplexI & aComp)
{
    return Pt2di(aComp.real(),aComp.imag());
}
Pt2dr Std2Elise(const ComplexR & aComp)
{
    return Pt2dr(aComp.real(),aComp.imag());
}



ComplexI  Elise2Std(const Pt2d<int> & aPt)
{
    return ComplexI(aPt.x,aPt.y);
}
ComplexR  Elise2Std(const Pt2d<double> & aPt)
{
    return ComplexR(aPt.x,aPt.y);
}
*/


template <class Type> class Hough_Mappped_Comp : public  Simple_OPBuf1<INT,Type>     
{
    public :

          void  calc_buf (INT ** output,Type *** input);
          Simple_OPBuf1<INT,Type> *         dup_comp();     


          Hough_Mappped_Comp
          (
             HoughMapedParam &,
             HoughMapedInteractor &
          );

          virtual ~Hough_Mappped_Comp();

    private :

         
         REAL   LengthBufOptim();
         Seg2d  OptimSeg(Seg2d aSeg,REAL aPrec);
         void MakeOneHough();
         void TraiteAHoughPts(Pt2di);
         Im2D_INT4 CalcPds();
         Im2D<Type,INT> ImOptim();
         // INT  NbImage();

         ElHoughFiltSeg * CompScore(Hough_Mappped_Comp<Type> & ares);


         // Seuils 




         ElHough &  mHough;
         HoughMapedParam  & mParam;
         HoughMapedInteractor &  mCurHMI;
         std::vector<Seg2d>      mSegs;
         ElHough::tModeAccum mModeH;
         INT                 mDimHough;

         ElHoughFiltSeg *  mScore;    // L'un ou l'autre suivant les cas

         std::vector<Im2D<Type,INT> >   mImGlobTmp;
         std::vector<Im2D<U_INT1,INT> >  mImHough;


         INT     mX0LocHough;
         INT     mDimIn;

         Pt2di   mSzHoughUti;
         Pt2di   mP0HoughUti;
         Pt2di   mP1HoughUti;
         Pt2di   mRabBoxUti;
         Box2di  mBoxUtiElargie;
         Pt2di   mRabBoxOptim;
         Box2di  mBoxOptim;

         Pt2di   mP0HInImGlob;
         Pt2di   mP0TmpInGlob;

          
         std::vector<Pt2di> mPtsHough;
         std::vector<Seg2d> mSegPrg;

};


/*****************************************************/
/*                                                   */
/*           HoughMapedParam                         */
/*                                                   */
/*****************************************************/

ElHough & HoughMapedParam::HoughOfString(const std::string & aName)
{
    if (!theStdH)
    {
        if (ELISE_fp:: exist_file(aName.c_str()))
        {
           theStdH =  ElHough::NewOne(aName);
        }
        else 
        {
           theStdH =  ElHough::NewOne
                   (
                       Pt2di(200,200),
                       1.0,
                       1.0,
                       ElHough::ModeStepAdapt,
                       20.0,
                       0.5
                   );
           theStdH->write_to_file(aName);
        }
    }

    return * theStdH;
}

ElHough *  HoughMapedParam::theStdH = 0;




HoughMapedParam::HoughMapedParam
(
     const std::string  & aNameHough,
     REAL aLengthMinSeg,   
     REAL aLengthMaxSeg,
     REAL aVminRadiom,
     bool aModeSar
)  :
   mNameHough  (aNameHough),
   mLengthMinSeg (aLengthMinSeg),
   mLengthMaxSeg (aLengthMaxSeg),
   mVminRadiom   (aVminRadiom),
   mModeH        (aModeSar ? ElHough::eModeValues : ElHough::eModeGradient),


   mFactCaniche           (DefFactCaniche()),
   mGradUseFiltreMaxLocDir  (DefGradUseFiltreMaxLocDir()),
   mGradUseSeuilPreFiltr    (DefGradUseSeuilPreFiltr()),
   mGradSeuilPreFiltr       (aVminRadiom),

   mSAR_NbVoisMoyCFAR       (10),
   mSAR_NbStepMoyCFAR       (4),
   mSAR_NbVoisMedian        (1),
   mSAR_FactExpFilter       (0.4),
   mSAR_BinariseSeuilBas    (180),
   mSAR_BinariseSeuilHaut   (230),
   mSAR_SzConnectedCompMin  (round_ni(2*aLengthMinSeg)),

   mVoisRhoMaxLocInit  (DefVoisRhoMaxLocInit()),
   mVoisTetaMaxLocInit (DefVoisTetaMaxLocInit()),
   mSeuilAbsHough      (aLengthMinSeg*aVminRadiom*RatioDefSeuilAbsHough()),
   
   mUseBCVS               (mModeH ==  ElHough::eModeGradient),
   mFiltrBCVSFactInferior (DefBCVSFactInf()),
   mVoisRhoMaxLocBCVS     (mVoisRhoMaxLocInit * DefRatioBCVS()),
   mVoisTetaMaxLocBCVS    (mVoisTetaMaxLocInit * DefRatioBCVS()),

   mDistMinPrgDyn         (aLengthMinSeg),
   mFiltrSzHoleToFill     (aLengthMinSeg),
   mFiltrSeuilRadiom      (mVminRadiom),

   mHoughIncTeta          (DefHoughIncTeta()),
   mFiltrIncTeta          (mHoughIncTeta*DefRatioFiltrIncTeta()),
   mFiltrEcMaxLoc         (DefEcMaxLoc()),

   mDoPreOptimSeg         (false),
   mPrecisionPreOptimSeg  (0.1),
   mDoPostOptimSeg        (true),
   mPrecisionPostOptimSeg (0.01),

   mLongFiltrRedond       (aLengthMinSeg),
   mLargFiltrRedond       (aLengthMinSeg),
   mRatioRecouvrt         (DefRatioRecouvrt()),

   mDeltaMinExtension     (1.0),
   mFiltrBCVSFactTolGeom  (2.0),
   mStepInterpPrgDyn      (1.0),
   mWidthPrgDyn           (3.0)
{
}



Box2di HoughMapedParam::BoxRab()
{
    INT dM = ElMax3(Hough().NbX(),Hough().NbY(),round_up(mLengthMaxSeg));
    Pt2di  RabDmax(dM,dM);

    return Box2di(-RabDmax-P0HoughUti(),RabDmax+SzHoughUti()-P0HoughUti());
}

Pt2di HoughMapedParam::SzHoughUti()
{
    return Pt2di (Pt2dr(Hough().SzXY()) * (1.0 - mRatioRecouvrt));
}

Pt2di HoughMapedParam::P0HoughUti()
{
   return (Hough().SzXY() -SzHoughUti()) /2;
}

Pt2di HoughMapedParam::P1HoughUti()
{
   return P0HoughUti()+SzHoughUti() ;
}


ElHough &  HoughMapedParam::Hough()
{
   return HoughOfString(mNameHough);
}


Fonc_Num HoughMapedParam::Filtr_CFAR(Fonc_Num fIm,Fonc_Num fInside)
{
    if ( mSAR_NbStepMoyCFAR <=0)
    {
        return fIm;
    }


    Symb_FNum pds = fInside;
    Symb_FNum I   = fIm;

    Fonc_Num FSta = Rconv(Virgule(pds,I,Square(I)));
    for (INT k=0; k<mSAR_NbStepMoyCFAR; k++)
    {
        Symb_FNum  SFS = rect_som(Virgule(FSta,fIm),mSAR_NbVoisMoyCFAR,true);
        FSta =    Virgule(SFS.v0(),SFS.v1(),SFS.v2())
                 / ElSquare(2*mSAR_NbVoisMoyCFAR+1);
        fIm = SFS.kth_proj(7);
    }
    Symb_FNum Stats = FSta;
 
 
    Symb_FNum  s0 = Max(Stats.v0(),1e-3);
    Symb_FNum  s1 = Stats.v1()/s0;
    Symb_FNum  s2 = Max(1e-3,Stats.v2()/s0-Square(s1));
 
 
    Symb_FNum   Erf = erfcc((fIm-s1)/sqrt(s2));
 
    Fonc_Num res =  255.0 * Max(0,Min(1,Erf));              
    return res;
}


Fonc_Num HoughMapedParam::FiltrMedian(Fonc_Num f)
{
    if (mSAR_NbVoisMedian <=0)
       return f;

    return rect_median(f,mSAR_NbVoisMedian,256);
}

Fonc_Num HoughMapedParam::FiltrCanExp(Fonc_Num f,Fonc_Num FInside)
{
    if (mSAR_FactExpFilter <=0)
       return f;

    return Iconv
           (
                canny_exp_filt(f,mSAR_FactExpFilter,mSAR_FactExpFilter) 
              / Max(1e-3,canny_exp_filt(FInside,mSAR_FactExpFilter,mSAR_FactExpFilter))
           ); 
}

Fonc_Num HoughMapedParam::FiltrBinarise(Fonc_Num f)
{
    Im1D_U_INT1 aLut (256);
    ELISE_COPY
    (
          aLut.all_pts(),
          Min(255,Max(0,((FX-mSAR_BinariseSeuilBas)*255)/(mSAR_BinariseSeuilHaut-mSAR_BinariseSeuilBas))),
          aLut.out()
    );

    return aLut.in()[Max(0,Min(255,f))];
}

Fonc_Num  HoughMapedParam::FiltrConc(Fonc_Num fIm)
{
    Pt2di pConc(mSAR_SzConnectedCompMin,mSAR_SzConnectedCompMin);

    Symb_FNum SIm (fIm);
    Symb_FNum  SB  =  BoxedConc(Virgule(SIm!=0,SIm),pConc,true,true) ;

    return  (SB.v0() == ParamConcOpb::DefColBig()  ) * SB.v2();
}


Fonc_Num HoughMapedParam::FiltreSAR(Fonc_Num fIm,Fonc_Num fInside)
{
   Fonc_Num f = Filtr_CFAR(fIm,fInside).v0();

   f = FiltrMedian(f);
   f = FiltrCanExp(f,fInside);
   f = FiltrBinarise(f);
   f = FiltrConc(f);

   return Iconv(f);
}

/*****************************************************/
/*                                                   */
/*             HoughMapedInteractor                  */
/*                                                   */
/*****************************************************/


HoughMapedInteractor::~HoughMapedInteractor() {}
void HoughMapedInteractor::OnNewSeg(const ComplexR &,const ComplexR &) {} 
void HoughMapedInteractor::OnNewCase(const ComplexI &,const ComplexI &) {} 


/*****************************************************/
/*                                                   */
/*             Hough_Mappped_Comp<Type>              */
/*                                                   */
/*****************************************************/


template <> ElHoughFiltSeg * Hough_Mappped_Comp<INT1>::CompScore(Hough_Mappped_Comp<INT1> & aRes)
{
     ELISE_ASSERT(mModeH== ElHough::eModeGradient,"Inc in Hough_Mappped_Comp<INT1>::CompScore");
     return new EHFS_ScoreGrad
                (
                    mParam.mStepInterpPrgDyn,
                    mParam.mWidthPrgDyn,
                    LengthBufOptim(),
                    aRes.mImGlobTmp[0],
                    aRes.mImGlobTmp[1],
                    mParam.mFiltrSzHoleToFill / 2.0,
                    mParam.mFiltrIncTeta,
                    mParam.mFiltrEcMaxLoc,
                    mParam.mFiltrSeuilRadiom
                );
}


template <> ElHoughFiltSeg * Hough_Mappped_Comp<U_INT1>::CompScore(Hough_Mappped_Comp<U_INT1> & aRes)
{
     ELISE_ASSERT(mModeH== ElHough::eModeValues,"Inc in Hough_Mappped_Comp<INT1>::CompScore");
     return new EHFS_ScoreIm
                (
                    mParam.mStepInterpPrgDyn,
                    mParam.mWidthPrgDyn,
                    LengthBufOptim(),
                    aRes.mImGlobTmp[0],
                    mParam.mFiltrSzHoleToFill / 2.0,
                    mParam.mFiltrSeuilRadiom
                );
}


template <class Type> REAL Hough_Mappped_Comp<Type>::LengthBufOptim()
{
    return euclid(mHough.SzXY()) + mParam.mLengthMaxSeg;
}

template <class Type> Hough_Mappped_Comp<Type>::~Hough_Mappped_Comp()
{
   ElSegMerge
   (
       mSegs,
       mParam.mLongFiltrRedond,
       mParam.mLargFiltrRedond
   );

   for 
   (
        std::vector<Seg2d>::iterator anItS=mSegs.begin();
        anItS!=mSegs.end();
        anItS++
   )
   {
          mCurHMI.Extr0().push_back(Elise2Std(anItS->p0()));
          mCurHMI.Extr1().push_back(Elise2Std(anItS->p1()));
   }
   delete mScore;
}


template <class Type> Simple_OPBuf1<INT,Type> * Hough_Mappped_Comp<Type>::dup_comp()
{
   Hough_Mappped_Comp<Type> * res = new Hough_Mappped_Comp<Type>(mParam,mCurHMI);

   for (INT k=0; k<this->dim_in() ; k++)
   {
       res->mImGlobTmp.push_back(this->AllocImageBufIn());
   }


   {
   for (INT k=0; k<mDimHough ; k++)
   {
       res->mImHough.push_back(Im2D_U_INT1(mHough.NbX(),mHough.NbY()));
   }
   }

   res->mScore = CompScore(*res);
   return res;
}



template <class Type> Hough_Mappped_Comp<Type>::Hough_Mappped_Comp
                      (

                             HoughMapedParam & aParam,
                             HoughMapedInteractor &  anInteractor
                      ) :

      mHough         (aParam.Hough()),
      mParam         (aParam),
      mCurHMI        (anInteractor),
      mSegs          (),
      mModeH         (ElHough::tModeAccum(aParam.mModeH)),
      mDimHough      (mModeH==ElHough::eModeValues ? 1 : 2),
      mScore         (0),
      mSzHoughUti    (aParam.SzHoughUti()),  //  (aHough.SzXY() * (1.0 - aParam.mRatioRecouvrt)),
      mP0HoughUti    (aParam.P0HoughUti()), // ((aHough.SzXY()-mSzHoughUti) / 2),
      mP1HoughUti    (aParam.P1HoughUti()), //(mP0HoughUti+mSzHoughUti),
      mRabBoxUti     (1,1),
      mBoxUtiElargie (mP0HoughUti-mRabBoxUti,mP1HoughUti+mRabBoxUti),
      mRabBoxOptim   (4,4),
      mBoxOptim      (mRabBoxOptim,mHough.SzXY()-mRabBoxOptim)
{
}

template <class Type> Im2D_INT4  Hough_Mappped_Comp<Type>::CalcPds()
{
     if (mModeH == ElHough::eModeValues)
     {
         ELISE_COPY
         (
             mImHough[0].all_pts(),
             trans(mImGlobTmp[0].in(0),mP0HInImGlob),
             mImHough[0].out()
         );
         return mHough.Pds(mImHough[0]);
     }
     else
     {
         Symb_FNum FInit
                   (
                       trans
                       (
                           Virgule(mImGlobTmp[0].in(),mImGlobTmp[1].in()),
                           mP0HInImGlob
                       )
                   );

         Fonc_Num RT = (Polar_Def_Opun::polar(FInit,0));
         if (mParam.mGradUseFiltreMaxLocDir)
         {
            Symb_FNum RT2 = RMaxLocDir
                            (
                                 RT,
                                 0.0,
                                 MaxLocDir_Def_OrientedMaxLoc,
                                 MaxLocDir_Def_RhoCalc,
                                 true
                            );

             RT = (Virgule(RT2.v1(),RT2.v2()) * (RT2.v0()));
         }

         if (mParam.mGradUseSeuilPreFiltr)
         {
              Symb_FNum RT2 (RT);
              RT = RT2 * (RT2.v0()>mParam.mGradSeuilPreFiltr);
         }



         Symb_FNum RhoTeta  (RT);


         Symb_FNum Rho  (Min(RhoTeta.v0(),255));
         Symb_FNum Teta (mod(Iconv((RhoTeta.v1() * (255.0/(2*PI)))),256));      

         ELISE_COPY
         (
             mImHough[0].all_pts(),
             Virgule(Rho,Teta),
             Virgule(mImHough[0].out(),mImHough[1].out())
         );
         return mHough.PdsAng(mImHough[0],mImHough[1],mParam.mHoughIncTeta);
     }
}


template <class Type> Im2D<Type,INT> Hough_Mappped_Comp<Type>::ImOptim()
{
     INT ind = (mModeH == ElHough::eModeGradient ? 2 : 0);
     return mImGlobTmp[ind];
}


template <class Type> Seg2d  Hough_Mappped_Comp<Type>::OptimSeg(Seg2d aSeg,REAL aPrec)
{
   REAL aScore;
   return ImOptim().OptimizeSegTournantSomIm
          (
                 aScore,
                 aSeg,
                 ElMax(1,round_ni(euclid(aSeg.p0(),aSeg.p1()))),
                 1.0,
                 aPrec
          );
}

template <class Type> void Hough_Mappped_Comp<Type>::TraiteAHoughPts(Pt2di aHPt)
{
     Seg2d  aSeg = mHough.Grid_Hough2Euclid(Pt2dr(aHPt));

     if (aSeg.clip(mBoxUtiElargie).empty())
        return;


     aSeg = aSeg.trans (Pt2dr(mP0HInImGlob));
     if (mParam.mDoPreOptimSeg)
     {
         aSeg = OptimSeg(aSeg,mParam.mPrecisionPreOptimSeg);
     }
     mScore->SetSeg(aSeg);
     mScore->GenPrgDynGet(mSegPrg,mParam.mDistMinPrgDyn);


     for 
     ( 
          std::vector<Seg2d>::iterator itSeg  = mSegPrg.begin();
          itSeg != mSegPrg.end() ;
          itSeg++
     )
     {

          Im2D<Type,INT> anIm = ImOptim();
          Seg2d aSOpt = mScore->ExtendSeg(*itSeg,mParam.mDeltaMinExtension,anIm);
          if (mParam.mDoPostOptimSeg)
             aSOpt = OptimSeg(aSOpt,mParam.mPrecisionPostOptimSeg);
          aSOpt = aSOpt.trans(Pt2dr(mP0TmpInGlob));
          mSegs.push_back(aSOpt);
          mCurHMI.OnNewSeg(Elise2Std(aSOpt.p0()),Elise2Std(aSOpt.p1()));
     }
}




template <class Type> void Hough_Mappped_Comp<Type>::MakeOneHough()
{

   mP0HInImGlob = -Pt2di(this->dx0(),this->dy0())+Pt2di(mX0LocHough,0) -mP0HoughUti;

   Im2D_INT4 anImPds = CalcPds();


   Pt2di P0 (this->x0()+mX0LocHough,this->ycur());
   Pt2di P1 = P0+mSzHoughUti;
   mCurHMI.OnNewCase(Elise2Std(P0),Elise2Std(P1));


    mHough.CalcMaxLoc 
    (
       anImPds,
       mPtsHough,
       mParam.mVoisRhoMaxLocInit,
       mParam.mVoisTetaMaxLocInit,
       mParam.mSeuilAbsHough *  mHough.Dynamic(mModeH)
    );


    if (mParam.mUseBCVS)
    {
       mHough.FiltrMaxLoc_BCVS
       (
          mPtsHough,
          anImPds,
          mParam.mFiltrBCVSFactInferior,
          mParam.mFiltrBCVSFactTolGeom,
          mParam.mVoisRhoMaxLocBCVS,
          mParam.mVoisTetaMaxLocBCVS
       );                      
    }

    for 
    (
		 std::vector<Pt2di>::iterator anIt = mPtsHough.begin();
         anIt != mPtsHough.end();
         anIt++
    )
          TraiteAHoughPts(*anIt);

} 



template <class Type> void Hough_Mappped_Comp<Type>::calc_buf (INT ** output,Type *** input)
{
   mP0TmpInGlob  = Pt2di(this->x0()+this->dx0(),this->ycur()+this->dy0());

   if (! this->first_line_in_pack())
       return;


    for (mDimIn = 0; mDimIn<this->dim_in() ; mDimIn++)
    {
        this->Simple_OPBuf1<INT,Type>::SetImageOnBufEntry(mImGlobTmp[mDimIn],input[mDimIn]);
    }


    for ( mX0LocHough= 0 ; mX0LocHough<this->x1()-this->x0() ; mX0LocHough+=mSzHoughUti.x)
    {
        MakeOneHough();
    }
}

template class Hough_Mappped_Comp<U_INT1>;
template class Hough_Mappped_Comp<INT1>;



Output Hough_Mapped_Sar
         (
             Fonc_Num                    aFonc,
             Fonc_Num                    aFoncPond,
             HoughMapedParam &           aParam,
             HoughMapedInteractor &      anHMI 
         )
{

   return Output::onul() <<
          create_op_buf_simple_tpl
          (
              new Hough_Mappped_Comp<U_INT1>(aParam,anHMI),
              aParam.FiltreSAR(aFonc,aFoncPond),
              1,
              aParam.BoxRab(),
              aParam.SzHoughUti().y,
              false
          );
}    


Output Hough_Mapped_Grad
         (
             Fonc_Num                    aFonc,
             HoughMapedParam &           aParam,
             HoughMapedInteractor &      anHMI 
         )
{


   Symb_FNum aGrad = deriche(aFonc,aParam.mFactCaniche,20);
   Symb_FNum aRho = Polar_Def_Opun::polar(aGrad,0).v0();

   return Output::onul() <<
          create_op_buf_simple_tpl
          (
              new Hough_Mappped_Comp<INT1>(aParam,anHMI),
              Max(-128,Min(127,Virgule(aGrad,aRho-128))),
              1,
              aParam.BoxRab(),
              aParam.SzHoughUti().y,
              false
          );
}    



Output   Hough_Mapped
         (
             Fonc_Num                    aFonc,
             Fonc_Num                    aFoncPond,
             HoughMapedParam &           aParam,
             HoughMapedInteractor &      anHMI 
         )
{
   return (aParam.mModeH == ElHough::eModeValues)  ?
          Hough_Mapped_Sar (aFonc,aFoncPond,aParam,anHMI) : 
          Hough_Mapped_Grad(aFonc,aParam,anHMI) ;
}

void HoughMapFromFoncNum
     (
          Fonc_Num aFoncH,
          Fonc_Num aFoncInside,
          const ComplexI & p0, const ComplexI & p1 ,
          HoughMapedParam & aParam,    
          HoughMapedInteractor &   aHMI
     )
{



    ELISE_COPY
    (
        rectangle(Std2Elise(p0),Std2Elise(p1)),
        1,
        Hough_Mapped(aFoncH,aFoncInside,aParam,aHMI)
    );
}


void HoughMapFromImage
     (
          unsigned char ** im ,    int Tx, int Ty,  // PARAMETRE IMAGE
          const ComplexI & p0, const ComplexI & p1 ,     // rectangle image
          HoughMapedParam & aParam,      // Parametre de detection de segment
          HoughMapedInteractor & aHMI // Interacteur 
     )
{
    Im2D_NoDataLin aINDL;
    Im2D_U_INT1 anIm(aINDL,Tx,Ty);

    for (INT anY=0 ; anY<Ty ; anY++)
        anIm.data()[anY] = im[anY];


    HoughMapFromFoncNum
    (
         anIm.in(0), anIm.inside(),
         p0,p1,
         aParam,aHMI
    );
}




void HoughMapFromFile
     (
          const std::string & aName,    // NOM DU FICHIER
          const ComplexI & p0, const ComplexI & p1 ,     // rectangle image
          HoughMapedParam & aParam,      // Parametre de detection de segment
          HoughMapedInteractor & aHMI  // Interacteur 
     )
{
   Tiff_Im aTifFile =  Tiff_Im::BasicConvStd(aName);

   HoughMapFromFoncNum
   (
      aTifFile.in(0),aTifFile.inside(),
      p0,p1,
      aParam,aHMI 
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
