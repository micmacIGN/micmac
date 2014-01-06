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
#include "general/all.h"
#include "private/all.h"
#include <map>
#include "im_tpl/correl_imget.h"


const INT TheZoomInit = 16;
const INT TheNbPtsInit = 400;

const REAL   TheCorrel_PrecisionFinale = 0.1;
const REAL   TheCorrel_IncertitudeInit0 =  60.0;
const REAL   TheCorrel_IncertitudeStd =  1.0;
const INT    TheCorrel_NbVoisInit0    =  3;

REAL TheCorrel_SzVignette = 10.0;
REAL TheCorrel_StepCalcul = 1;

const REAL   TheSeuilRecipr_0   = 1.0;
const REAL   TheSeuilRecipr_Std = 1.0;


template <class Type> class PyrImages;
template <class Type> class ImagesOfPyr;


template <class Type> class ImagesOfPyr
{
     private :
        typedef El_CTypeTraits<Type>::tBase  tBase;
        typedef Im2D<Type,tBase>             tIm;

        friend PyrImages<Type>;



        tIm         mIm;
        INT         mZ;
        Video_Win * mPW;


         
     public :


        Pt2dr Sz();
        void   ShowGray();
        Output OGray();
        void draw_circle(const Pt2dr &,INT aCoul,REAL aDiam);
        void draw_seg(const Pt2dr &,const Pt2dr &,INT aCoul);

        tIm Im()   {return mIm;}
        INT Zoom() {return mZ;}
        ImagesOfPyr(tIm anIm,INT aZoom,Video_Win *);

};


template <class Type,class Container> void Pyr_draw_circles
                                      (
                                          ImagesOfPyr<Type> & ,
                                          const Container &,
                                          INT aCoul,
                                          REAL aDiam
                                      );



template <class Type> class PyrImages
{
        typedef El_CTypeTraits<Type>::tBase  tBase;
        typedef Im2D<Type,tBase>             tElIm;
        typedef ImagesOfPyr<Type>            tIm;

     public :

           PyrImages
           (
                 tElIm anIm1,
                 INT aZoomMax,
                 INT aRatio = 2,
                 const std::list<Video_Win> & = std::list<Video_Win>()
           );
           std::vector<tIm> & VIm() {return mPyr;}
           INT NbImages() const {return mPyr.size();}
           tIm &  ImageOfZoom(INT aZoom);

     private :

        PyrImages(const PyrImages<Type> &);
        void post_init();

        tElIm                   mIm1;
        std::vector<INT>        mVZoomRel; // mVZoomRel[0] => Bidon
        std::vector<tIm>        mPyr;
        std::list<Video_Win>  mWinsInit;
        std::list<Video_Win>  mWinsChc;
};




/*******************************************************************/
/*                                                                 */
/*                      ImagesOfPyr                                */
/*                                                                 */
/*******************************************************************/

template <class Type> Pt2dr ImagesOfPyr<Type>::Sz()
{
   return mIm.sz();
}

template <class Type> Output ImagesOfPyr<Type>::OGray()
{
    return mPW ? mPW->ogray() : Output::onul();
}

template <class Type> void ImagesOfPyr<Type>::ShowGray()
{
    if (mPW)
    {
       mPW->clear();
       ELISE_COPY
       (
          mIm.all_pts(),
          mIm.in(),
          OGray()
       );
    }
}

template <class Type> 
         ImagesOfPyr<Type>::ImagesOfPyr
         (
              tIm anIm,
              INT aZoom,
              Video_Win * aPW
         )  :
            mIm (anIm),
            mZ  (aZoom),
            mPW (aPW)
{
    ShowGray();
}



template <class Type> void ImagesOfPyr<Type>::draw_circle(const Pt2dr & aPt,INT aCoul,REAL aDiam)
{
     if (mPW)
        mPW->draw_circle_abs(aPt,aDiam,mPW->pdisc()(aCoul));
}

template <class Type> void ImagesOfPyr<Type>::draw_seg(const Pt2dr & aP1,const Pt2dr & aP2,INT aCoul)
{
     if (mPW)
        mPW->draw_seg(aP1,aP2,mPW->pdisc()(aCoul));
}

template <class Type,class Container> void Pyr_draw_circles
                                      (
                                          ImagesOfPyr<Type> & anIZ,
                                          const Container & aContPt,
                                          INT aCoul,
                                          REAL aDiam 
                                      )
{
       for 
       (
            typename Container::const_iterator anItP = aContPt.begin();
            anItP != aContPt.end();
            anItP ++
       )
           anIZ.draw_circle(*anItP,aCoul,aDiam);
}



/*******************************************************************/
/*                                                                 */
/*                      PyrImages                                  */
/*                                                                 */
/*******************************************************************/

template <class Type> 
         void PyrImages<Type>:: post_init()
{
     mVZoomRel[0] = 1;

     INT aZoomAbs = 1;

     std::list<Video_Win>::iterator itW = mWinsInit.begin();

     for (INT iZ=0 ; iZ<(INT)mVZoomRel.size() ; iZ++)
     {
         INT aZR = mVZoomRel[iZ];
         aZoomAbs *= aZR;

         tElIm anIm = (iZ ==0) ? mIm1 : (mPyr[iZ-1].mIm.gray_im_red(aZR));
         Video_Win  * aPW = 0;

         if (itW != mWinsInit.end())
         {
              Video_Win aW = *itW;
              REAL aRatio = aW.sz().RatioMin(anIm.sz());
              aW = aW.chc(Pt2dr(0,0),Pt2dr(aRatio,aRatio));
              mWinsChc.push_back(aW);
              aPW = &mWinsChc.back();

              {
                 std::list<Video_Win>::iterator itW2 = itW;
                 itW2++;
                 if (itW2 != mWinsInit.end())
                    itW = itW2;
              }
         }
         
         mPyr.push_back(tIm(anIm,aZoomAbs,aPW));
     }
}



template <class Type> 
         PyrImages<Type>::PyrImages
         (
                 tElIm anIm1,
                 INT aZoomMax,
                 INT aRatio ,
                 const std::list<Video_Win> &  aVWins
         ) :
             mIm1      (anIm1),
             mWinsInit (aVWins)
{
     for (INT aZ=1 ; aZ<=aZoomMax ; aZ *= aRatio)
          mVZoomRel.push_back(aRatio);

     post_init();
}




template <class Type>  
         PyrImages<Type>::tIm &  PyrImages<Type>::ImageOfZoom(INT aZoom)
{
    for (INT k=0 ; k<NbImages() ; k++)
        if (mPyr[k].Zoom() == aZoom)
          return mPyr[k];

    ELISE_ASSERT(false,"Cannot Find Zoom in PyrImages<Type>::ImageOfZoom");
    return mPyr[0];
}

template class PyrImages<U_INT1>;

/*******************************************************/
/*                                                     */
/*                    RecalPyr2                        */
/*                                                     */
/*******************************************************/

template <class Type> class ResulRecal
{
     public :
          ResulRecal
          (
               ElDistortionPolynomiale  aDist,
               ImagesOfPyr<Type> &      aImP1,
               ImagesOfPyr<Type> &      aImP2,
               ElDistortionPolynomiale  aDistPrec
          ) ;
 
          ImagesOfPyr<Type>        & ImP1()  {return mImP1;}
          ImagesOfPyr<Type>        & ImP2()  {return mImP2;}
          ElDistortionPolynomiale  & Dist()  {return mDist;}

          ElDistortionPolynomiale  DistChc(ImagesOfPyr<Type> & anImP)  ;

     private :
          ElDistortionPolynomiale  mDist;
          ImagesOfPyr<Type> & mImP1;
          ImagesOfPyr<Type> & mImP2;
          Pt2di               mSz;
};


template <class Type> ElDistortionPolynomiale  ResulRecal<Type>::DistChc(ImagesOfPyr<Type> & anImP) 
{
     REAL aRatio =   (REAL)mImP1.Zoom() /  (REAL)anImP.Zoom();

     return mDist.MapingChScale(aRatio);
}

template <class Type> ResulRecal<Type>::ResulRecal
                      (
                                     ElDistortionPolynomiale  aDist,
                                     ImagesOfPyr<Type> &      aImP1,
                                     ImagesOfPyr<Type> &      aImP2,
                                     ElDistortionPolynomiale  aDistPrec
                      )  :
   mDist  (aDist),
   mImP1  (aImP1),
   mImP2  (aImP2),
   mSz    (aImP1.Sz())
{
    INT Nb = 10;

    mImP1.ShowGray();

    REAL MaxEc = 0;
    REAL SomEc = 0;

    for (INT x =0 ; x < Nb ; x++)
        for (INT y =0 ; y < Nb ; y++)
        {
             Pt2dr aP = Pt2dr(mSz.x *x,mSz.y*y) / (REAL) Nb;
             Pt2dr aPPrec = aDistPrec.Direct(aP);
             Pt2dr aPNew = aDist.Direct(aP);

             mImP1.draw_circle(aP,P8COL::green,3.0);
             mImP1.draw_seg(aP,aP+(aPNew-aP)*20,P8COL::blue);
             mImP1.draw_seg(aP,aP+(aPPrec-aP)*20,P8COL::red);

             const REAL aDist = euclid(aPPrec,aPNew);
             SomEc += aDist;
             ElSetMax(MaxEc,aDist);
        }


    SomEc /= Nb * Nb;
    MaxEc *= aImP1.Zoom();
    SomEc *= aImP1.Zoom();

}

typedef CalcPtsInteret::tContainerPtsInt tLPInt;


template <class Type>  class RecalPyr2
{
     private :
          typedef PyrImages<Type>      tPyr;

          tPyr &                       mPyr1;
          tPyr &                       mPyr2;
          std::map<INT,ResulRecal<Type> *>  mRecals;

          ResulRecal<Type> & MakeOneRecal 
                             (
                                 INT aZoom,
                                 INT aDegre,
                                 INT aNbPtsInit,
                                 ResulRecal<Type> * Prec
                             );

     public :

         void MakeCamNum();

         RecalPyr2(tPyr &,tPyr &);
         ResulRecal<Type> & GetRecal(INT aZoom);
};


template <class Type>  
ResulRecal<Type> & RecalPyr2<Type>::MakeOneRecal
                 (
                        INT                        aDegre,
                        INT                        aZoom,
                        INT                        aNbPts,
                        ResulRecal<Type> *         aRecPrec
                 )
{

     ImagesOfPyr<Type> &        anIZ1 = mPyr1.ImageOfZoom(aZoom);
     ImagesOfPyr<Type> &        anIZ2 = mPyr2.ImageOfZoom(aZoom);

     anIZ1.ShowGray();
     Im2D<Type,INT> anI1 = anIZ1.Im();
     Im2D<Type,INT> anI2 = anIZ2.Im();

     tLPInt PtsI =   CalcPtsInteret::GetEnsPtsInteret_Nb(anI1,aNbPts);
     Pyr_draw_circles(anIZ1,PtsI,P8COL::red,3.0);

     bool FirstRecal = (aRecPrec==0);
     ElDistortionPolynomiale aDistInit = (aRecPrec==0) ?
                                         ElDistortionPolynomiale(euclid(anI1.sz())) :
                                         aRecPrec->DistChc(anIZ1)                   ;

     REAL aPrecInit = FirstRecal ?
                     (TheCorrel_IncertitudeInit0/(aZoom*TheCorrel_NbVoisInit0)) :
                     TheCorrel_IncertitudeStd                                   ;

     INT aNbVois = FirstRecal ? TheCorrel_NbVoisInit0 : 1;
     REAL aSeuilRec = FirstRecal ? TheSeuilRecipr_0 : TheSeuilRecipr_Std;



     OptimTranslationCorrelation<Type> anOpt12
     (
         TheCorrel_PrecisionFinale,
         aPrecInit,
         aNbVois,
         anI1,
         anI2,
         TheCorrel_SzVignette,
         TheCorrel_StepCalcul
     );



     OptimTranslationCorrelation<Type> anOpt21
     (
         TheCorrel_PrecisionFinale,
         aPrecInit,
         aNbVois,
         anI2,
         anI1,
         TheCorrel_SzVignette,
         TheCorrel_StepCalcul
     );

    OptCorrSubPix_Diff<Type>  aDifOpt12
                         (
                             anI1,
                             anI2,
                             TheCorrel_SzVignette,
                             TheCorrel_StepCalcul,
                             Pt3dr(-100,-100,-100)
                         );


      static const REAL Exag = 20;

     INT aNbPtsOk =0;
     ElPackHomologue  aLPH;

REAL aTDif = 0;
REAL aTComb = 0;
ElTimer aChrono;

     for (tLPInt::const_iterator itP=PtsI.begin(); itP!=PtsI.end() ; itP++)
     {
          Pt2dr aP1 = *itP; //  +Pt2dr(0.1256,0.08762);
          Pt2dr aP2 = aDistInit.Direct(aP1);
          Pt2dr aTrInit = aP2-aP1;
          
          anOpt12.SetP0Im1(aP1) ;
          aChrono.reinit();
          anOpt12.optim(aTrInit);
          aTComb +=  aChrono.uval();
 
          if (anOpt12.FreelyOptimized())
          {
               Pt2dr aTrOpt  =  anOpt12.param();


//    TEST de coherence avec  diff
if (! FirstRecal)
{
    Pt3dr aDifPt (aP2.x,aP2.y,0);

for (INT aNbIterDif=0 ; aNbIterDif<9 ; aNbIterDif++)
{
    aChrono.reinit();
        aDifPt = aDifOpt12.Optim(aP1,Pt2dr(aDifPt.x,aDifPt.y));
    aTDif +=  aChrono.uval();

/*
    Pt2dr aP2Comb = aP1 + aTrOpt;
    Pt2dr aP2Dif(aDifPt.x,aDifPt.y);
*/


}
}


               anIZ1.draw_circle(aP1,P8COL::green,3.0);

               anOpt21.SetP0Im1(aP1+aTrOpt);
               anOpt21.optim(-aTrOpt);
               Pt2dr aTrNull = aTrOpt + anOpt21.param();

               if (euclid(aTrNull) < aSeuilRec)  
               {
                   aNbPtsOk++;
                   anIZ1.draw_seg(aP1,aP1+aTrOpt*Exag,P8COL::blue);
                   anIZ1.draw_seg(aP1+aTrNull*Exag,aP1+aTrOpt*Exag,P8COL::cyan);
                   // cout << (euclid(aTrNull) * aZoom) << " " << (euclid(aTrOpt) *aZoom) << "\n";
                   aLPH.add(ElCplePtsHomologues(aP1,aP1+aTrOpt));
               }
                 
          }
          
     }

     ElDistortionPolynomiale aNewDist =  aLPH.FitDistPolynomiale(aDegre,euclid(anI1.sz()));

     for (tLPInt::const_iterator itP=PtsI.begin(); itP!=PtsI.end() ; itP++)
     {
          Pt2dr aP1 = *itP; //  +Pt2dr(0.1256,0.08762);
          Pt2dr aP2 = aNewDist.Direct(aP1);
          Pt2dr aTrPol = aP2-aP1;
          anIZ1.draw_seg(aP1,aP1+aTrPol*Exag,P8COL::red);

          Pt2dr aP3 = aDistInit.Direct(aP1);
          Pt2dr aTrInit = aP3-aP1;
          anIZ1.draw_seg(aP1,aP1+aTrInit*Exag,P8COL::yellow);
     }


     mRecals[aZoom] = new  ResulRecal<Type>(aNewDist,anIZ1,anIZ2,aDistInit);
          
     return  GetRecal(aZoom);
     
}





template <class Type> ResulRecal<Type> &RecalPyr2<Type>::GetRecal(INT aZoom)
{
    ResulRecal<Type> * aRR = mRecals[aZoom];
    ELISE_ASSERT(aRR!=0,"Cannot Get RecalPyr2<Type>::GetRecal");
    return *aRR;
}

template <class Type>  
         RecalPyr2<Type>::RecalPyr2
         (
             tPyr & aPyr1,
             tPyr & aPyr2
          ) :
            mPyr1 (aPyr1),
            mPyr2 (aPyr2)
{

    ResulRecal<Type> & aRecal16 = MakeOneRecal
    (
        1,
        TheZoomInit,
        TheNbPtsInit,
        (ResulRecal<Type> *)NULL
    );


    ResulRecal<Type> & aRecal8 = MakeOneRecal
    (
        2,
        TheZoomInit/2,
        800,
        &aRecal16
    );

    ResulRecal<Type> & aRecal4 = MakeOneRecal
    (
        3,
        TheZoomInit/4,
        1600,
        &aRecal8
    );

    ResulRecal<Type> & aRecal2 =  MakeOneRecal
    (
        5,
        TheZoomInit/8,
        3200,
        &aRecal4
    );


    /* ResulRecal<Type> & aRecal4 = */ MakeOneRecal
    (
        7,
        TheZoomInit/16,
        6400,
        &aRecal2
    );


}

template class RecalPyr2<U_INT1>;



/*******************************************************/
/*                                                     */
/*                    Recal_23sur1                     */
/*                                                     */
/*******************************************************/

template <class Type>  class Resul_Recal_23sur1
{
     public :

          Resul_Recal_23sur1
          (
                const ElDistortionPolynomiale & mD12 ,
                const ElDistortionPolynomiale & mD13 
          )  :
             mMap2to1 (mD12.FNum()),
             mMap3to1 (mD13.FNum())
          {
          }

          Fonc_Num Map2to1() const {return mMap2to1;}
          Fonc_Num Map3to1() const {return mMap3to1;}


     private  :

         Fonc_Num mMap2to1;
         Fonc_Num mMap3to1;
};








template <class Type>  class Recal_23sur1
{
     private :
          typedef El_CTypeTraits<Type>::tBase  tBase;
          typedef Im2D<Type,tBase>             tElIm;
          typedef PyrImages<Type>              tPyr;
          typedef RecalPyr2<Type>              tRec2;


          tPyr   mPyr1;
          tPyr   mPyr2;
          tPyr   mPyr3;

          tRec2    mRec12;
          tRec2    mRec13;
          tRec2 *  mRec23;

     public :

         void MakeCamNum();
         void MakeBouclage(INT aZoom);

         Resul_Recal_23sur1<Type>  Resul(INT aZoom);
         Recal_23sur1
         (
                 bool  aBoucl,
                 tElIm anIm1,
                 tElIm anIm2,
                 tElIm anIm3,
                 INT aZoomMax,
                 INT aRatio = 2,
                 const std::list<Video_Win> & = std::list<Video_Win>()
         );
};



template <class Type> Resul_Recal_23sur1<Type> Recal_23sur1<Type>::Resul(INT aZoom)
{
    return Resul_Recal_23sur1<Type>
           (
               mRec12.GetRecal(aZoom).Dist(),
               mRec13.GetRecal(aZoom).Dist()
           );
}

/*
template <class Type>  void Recal_23sur1<Type>::MakeBouclage(INT aZoom)
{
    ELISE_ASSERT(mRec23!=0,"No Rec23 in MakeBouclage");
    
    ResulRecal<Type> & aRes12 = mRec12.GetRecal(aZoom);
    ImagesOfPyr<Type> & aImP1 = aRes12.ImP1();
};
*/





template <class Type>  void Recal_23sur1<Type>::MakeBouclage(INT aZoom)
{
    ELISE_ASSERT(mRec23!=0,"No Rec23 in MakeBouclage");
    
    ResulRecal<Type> & aRes12 = mRec12.GetRecal(aZoom);
    ImagesOfPyr<Type> & aImP1 = aRes12.ImP1();
    Pt2di aSz = aImP1.Im().sz();
    
    ElDistortionPolynomiale & aDist12 = mRec12.GetRecal(aZoom).Dist();
    ElDistortionPolynomiale & aDist13 = mRec13.GetRecal(aZoom).Dist();
    ElDistortionPolynomiale & aDist23 = mRec23->GetRecal(aZoom).Dist();


    aImP1.ShowGray();
    INT aNb = 20;
    REAL exag = 30.0;

    REAL EcMax = 0.0;
    REAL SomEc = 0.0;

    REAL EcMaxComp = 0.0;
    REAL SomEcComp = 0.0;

    for (INT x=0 ; x<aNb ; x++)
    {
        for (INT y=0 ; y<aNb ; y++)
        {
             Pt2dr aP = Pt2dr(aSz.x *x,aSz.y*y) / (REAL) aNb;
            
             Pt2dr aP1 = aP;
             Pt2dr aP2 = aDist12.Direct(aP1);
             Pt2dr aP3 = aDist13.Direct(aP1);
             Pt2dr aQ3 = aDist23.Direct(aP2);



             Pt2dr aV12 =  aDist12.Direct(aP)-aP;
             Pt2dr aV13 =  aDist13.Direct(aP)-aP;
             Pt2dr aV23 =  aDist23.Direct(aP)-aP;

             aImP1.draw_seg(aP,aP+aV12*exag,P8COL::blue);
             aImP1.draw_seg(aP+aV12*exag,aP+(aV12+aV23)*exag,P8COL::red);
             aImP1.draw_seg(aP+(aV12+aV23)*exag,aP+(aV12+aV23-aV13)*exag,P8COL::green);

             const REAL aDist = euclid(aV12+aV23-aV13);

             ElSetMax(EcMax,aDist);
             SomEc += aDist;


             const REAL aDistComp = euclid(aQ3,aP3);

             ElSetMax(EcMaxComp,aDistComp);
             SomEcComp += aDistComp;

        }
   }

   SomEc /= aNb * aNb ;
   cout << "Som Ecart " << SomEc << " ; EcMax " << EcMax << "\n";

   SomEcComp /= aNb * aNb ;
   cout << "Som EcartComp " << SomEcComp << " ; EcMaxComp " << EcMaxComp << "\n";
}


template <class Type>  
         Recal_23sur1<Type>::Recal_23sur1
         (
                 bool  aBoucl,
                 tElIm anIm1,
                 tElIm anIm2,
                 tElIm anIm3,
                 INT aZoomMax,
                 INT aRatio = 2,
                 const std::list<Video_Win> & aVWins
         ) :
         mPyr1 (anIm1,aZoomMax,aRatio,aVWins),
         mPyr2 (anIm2,aZoomMax,aRatio,aVWins),
         mPyr3 (anIm3,aZoomMax,aRatio,aVWins),
         mRec12 (mPyr1,mPyr2),
         mRec13 (mPyr1,mPyr3),
         mRec23 (aBoucl ? (new tRec2(mPyr2,mPyr3)) : 0)
{
   if (aBoucl)
   {
      MakeBouclage(4);
      MakeBouclage(2);
      MakeBouclage(1);
   }
}

template class Recal_23sur1<U_INT1>;


/*
template <class Type> ImagesOfPyr<Type>::ImagesOfPyr
                      (
                            tIm          anIm,
                            INT          aZoom,
                            VideoWin *
                      )
{
}
*/

void MakeGrad(Im2D_U_INT1 anIm,REAL aFact)
{
    cout << "Begin Grad \n";
    ELISE_COPY
    (
         anIm.all_pts(), 
         Max(0,Min(255,polar(deriche(anIm.in_proj(),aFact,20),0.0).v0())),
         anIm.out()
    );
    cout << "End Grad \n";
}


std::string Substitute
            (
                 const std::string & aStr,
                 const std::string & aOld,
                 const std::string & aNew
             )
{
   std::string aRes = aStr;
   std::string::size_type aPosition = aRes.rfind(aOld);
   ELISE_ASSERT(aPosition != std::string::npos,"Dont find SubString in Substitute");
   aRes.replace(aPosition,aNew.length(),aNew);
   return aRes;
}



int main (int argc,char ** argv)
{
    std::string aNameIn;
    INT aDegre = 5;
    INT CanalFixe = 2;
    INT GenTest = 0;
    INT WithBouclage = 1;
 
    REAL aGradFact = -1;

    INT NbVois = 10;
    REAL Step = 1.0;


    std::vector<INT> ZoomsOut;
    ZoomsOut.push_back(1);

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAM(aNameIn),
          LArgMain()  << EAM(aDegre,"Degre",true)
                      << EAM(ZoomsOut,"ZoomsOut",true)
                      << EAM(CanalFixe,"CanalFixe",true)
                      << EAM(GenTest,"GenTest",true)
                      << EAM(WithBouclage,"Bouclage",true)
                      << EAM(aGradFact,"aGradFact",true)
                      << EAM(NbVois,"NbVois",true)
                      << EAM(Step,"Step",true)
    );
    TheCorrel_SzVignette  = NbVois * Step;
    TheCorrel_StepCalcul  = Step;

     Tiff_Im aTifIn(aNameIn.c_str());
     Pt2di aSzW = aTifIn.sz();

     Fonc_Num  aFoncIn = aTifIn.in().shift_coord(-CanalFixe);

     INT Sep2RGB = (aTifIn.nb_chan() != 3);


     if (Sep2RGB)
     {
        std::string r("_r");
        std::string v("_v");
        std::string b("_b");

        std::string Namev = Substitute(aNameIn,r,v);
        std::string Nameb = Substitute(aNameIn,r,b);

        cout << "Namev = " << Namev.c_str() << "\n";
        cout << "Nameb = " << Nameb.c_str() << "\n";

        Tiff_Im  Tiffv = Tiff_Im::StdConv(Namev.c_str());
        Tiff_Im  Tiffb = Tiff_Im::StdConv(Nameb.c_str());

        aFoncIn = Virgule(aTifIn.in() ,Tiffv.in(),Tiffb.in()).shift_coord(-CanalFixe);
     }






     //  aSzW  = Pt2di(1000,1000);

     Im2D_U_INT1 aI1(aSzW.x,aSzW.y);
     Im2D_U_INT1 aI2(aSzW.x,aSzW.y);
     Im2D_U_INT1 aI3(aSzW.x,aSzW.y);

     Video_Win aW = Video_Win::WStd(aSzW,0.15);

     ELISE_COPY
     (
         aI1.all_pts(),
         aFoncIn,
           Virgule(aI1.out(),aI2.out(),aI3.out())
          | aW.orgb()
     );

     if (aGradFact > 0)
     {
          MakeGrad(aI1,aGradFact);
          MakeGrad(aI2,aGradFact);
          MakeGrad(aI3,aGradFact);
     }






     std::list<Video_Win> aVWins;
     aVWins.push_back(Video_Win::WStd(Pt2di(800,800),1.0));
     // aVWins.push_back(Video_Win::WStd(Pt2di(800,800),1.0));
     // aVWins.push_back(Video_Win::WStd(Pt2di(800,800),1.0));

     Recal_23sur1<U_INT1>  aPyr1((bool)WithBouclage,aI1,aI2,aI3,TheZoomInit,2,aVWins);



     Resul_Recal_23sur1<U_INT1>  aRes = aPyr1.Resul(1);


     if (aGradFact > 0)
     {
         ELISE_COPY
         (
             aI1.all_pts(),
             aFoncIn,
               Virgule(aI1.out(),aI2.out(),aI3.out())
              | aW.orgb()
         );
     }

     for (INT aKZ=0 ; aKZ<(INT)ZoomsOut.size() ; aKZ++)
     {
          INT aZ = ZoomsOut[aKZ];

          std::string StrZ = "" ;
          if (aZ !=1)
          {
             char CChZ[100];
             sprintf(CChZ,"%d",aZ);
             StrZ = CChZ;
          }

          std::string aNameOut =    StdPrefix(aNameIn) 
                                 + std::string(StrZ) 
                                 + std::string(GenTest ? "_test.tif"  : "_recal.tif");
          Tiff_Im aTifOut
             (
                  aNameOut.c_str(),
                  aSzW * aZ, 
                  GenIm::u_int1,
                  Tiff_Im::No_Compr,
                  Tiff_Im::RGB
             );



          if (GenTest)
          {
               ELISE_ASSERT(false,"GenTest Not Implemented \n");
          }
          else
          {
              Pt2di aP0(0,0);
              Pt2di aP1 = aTifOut.sz();

              Fonc_Num FIm1 =  aI1.in(0);
              Fonc_Num aF21 = aRes.Map2to1();
              Fonc_Num aF31 = aRes.Map3to1();
              if (aZ != 1)
              {
                 Fonc_Num xZ = FX/(REAL)aZ;
                 Fonc_Num yZ = FY/(REAL)aZ;
                 FIm1 = aI1.ImGridReech(xZ,yZ,64,aP0,aP1,128);
                 aF21 = aF21[Virgule(xZ,yZ)];
                 aF31 = aF31[Virgule(xZ,yZ)];
              }
              aW.clear();
              aW = aW.chc(Pt2dr(0,0),Pt2dr(1/(REAL)aZ,1/(REAL)aZ));
              ELISE_COPY
              (
                  aTifOut.all_pts(),
                  Virgule
                  (
                      FIm1,
                      aI2.ImGridReech(aF21.v0(),aF21.v1(),16,aP0,aP1,128),
                      aI3.ImGridReech(aF31.v0(),aF31.v1(),16,aP0,aP1,128)
                  ).shift_coord(CanalFixe),
                  aTifOut.out() | aW.orgb()
              );
          }
     }


    return 1;
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
