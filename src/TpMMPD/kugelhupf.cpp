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
#include "kugelhupf.h"
/**
 * Kugelhupf: Automatic fiducial point determination
 * Klics Ubuesques Grandement Evites, Lent, Hasardeux mais Utilisable pour Points Fiduciaux
 * Inputs:
 *  - list of images
 *  - fiducial points on one image
 * 
 * Output:
 *  - fiducial points on all images
 * 
 * Call example:
 *   mm3d Kugelhupf ".*.tif" Ori-InterneScan/MeasuresIm-1987_FR4074_0202.tif.xml SearchIncertitude=10 TargetHalfSize=32
 * 
 * TODO: support 16 bit images?
 *  2 search levels, or dezoom ?
 * */

/***************************************************************/
/*                                                             */
/*             cOneHypMarkFid                                  */
/*                                                             */
/***************************************************************/

cOneHypMarkFid::cOneHypMarkFid (const cOneMesureAF1I & aMes) :
   mMes (aMes)
{
}

// "PAquet" d'image correspondant a une resolution

double cQuickCorrelPackIm::Correl(const Pt2dr & aDec,eModeInterpolation aMode)
{
    Pt2di aDecI = round_ni(aDec);
    RMat_Inertie aMat;
    double aV1=0,aV2=0,aPds=0;
    int aRab = 3;
    Pt2di aPRab(aRab,aRab);

    Pt2di aP0 = Sup(Pt2di(0,0), aPRab-aDecI );
    Pt2di aP1 = Inf(mTRef.sz(), mTIm.sz() - aDecI - aPRab);

    Pt2di aP;
    for (aP.x =aP0.x ; aP.x<aP1.x ; aP.x++)
    {
        for (aP.y =aP0.y ; aP.y<aP1.y ; aP.y++)
        {
            aPds = mTMasq.get(aP);
            if (aPds)
            {
                aV1 = mTRef.get(aP);
                if (aMode==eInterpolPPV)
                {
                    aV2 = mTIm.get(aP+aDecI);
                }
                else // if (aMode==eInterpolBiLin)
                {
                    aV2 = mTIm.getr(Pt2dr(aP)+aDec);
                }

                aMat.add_pt_en_place(aV1,aV2,aPds);
            }
        }
    }
    if (aMat.s()==0) return -1;
    return aMat.correlation(1e-5);
}

Pt2di cQuickCorrelPackIm::OptimizeInt(const Pt2di aP0,int aSzW)
{
    double aMaxCor = -1e9;
    Pt2di aPMax(0,0);

    for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)
    {
       for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)
       {
           Pt2di aP  = Pt2di(aDx,aDy) + aP0;
           double aCor =  Correl(Pt2dr(aP),eInterpolPPV);
           if (aCor>aMaxCor)
           {
               aMaxCor = aCor;
               aPMax = aP;
           }
       }
    }
    return aPMax;
}



Pt2dr cQuickCorrelPackIm::OptimizeReel(const Pt2dr aP0,double aStep,int aSzW,double & aMaxCor)
{
    aMaxCor = -1e9;
    Pt2dr aPMax(0,0);

    for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)
    {
       for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)
       {
           Pt2dr aP  =  aP0+ Pt2dr(aDx,aDy) * aStep;
           double aCor =  Correl(Pt2dr(aP),eInterpolBiLin);
           if (aCor>aMaxCor)
           {
               aMaxCor = aCor;
               aPMax = aP;
           }
       }
    }
    return aPMax;
}




void ReducePackIm( TIm2D<REAL4,REAL8>  ImOut, TIm2D<REAL4,REAL8> ImIn,double aSsRes,double aDil)
{
   ELISE_COPY
   (
         ImOut.all_pts(),
         StdFoncChScale_BicubNonNeg(ImIn.in(0),Pt2dr(0,0),Pt2dr(1/aSsRes,1/aSsRes),Pt2dr(aDil,aDil)),
         ImOut.out()
   );

}

void cQuickCorrelPackIm::InitByReduce(const cQuickCorrelPackIm & aPack,double aDil)
{
    double aSsRes = mResol / aPack.mResol;

    ReducePackIm(mTIm  , aPack.mTIm  , aSsRes, aDil);
    ReducePackIm(mTRef , aPack.mTRef , aSsRes, aDil);
    ReducePackIm(mTMasq, aPack.mTMasq, aSsRes, aDil);
}

void cQuickCorrelPackIm::FinishLoad()
{
   ELISE_COPY(mTMasq.all_pts(),mTMasq.in(),sigma(mSomPds));
}

cQuickCorrelPackIm::cQuickCorrelPackIm(Pt2di aSzBuf,Pt2di aSzMarq,double aResol,bool Debug) :
    mSzIm    (round_ni(Pt2dr(aSzBuf)*aResol)),
    mSzMarq  (round_ni(Pt2dr(aSzMarq)*aResol)),
    mResol   (aResol),
    mTIm     (mSzIm),
    mTRef    (mSzMarq),
    mTMasq   (mSzMarq),
    mW       (0),
    mDebug   (Debug)
{
    ELISE_COPY(mTMasq._the_im.all_pts(),1.0,mTMasq._the_im.out());
}


std::list<Pt3dr>  cQuickCorrelPackIm::DecFFT()
{
    bool VisuFFT=mDebug;
    if (VisuFFT)
    {
        if (mW==0)
           mW = Video_Win::PtrWStd(Pt2di(512,512));
    }
    if (false)  // mettre 1 si on veut visualiser les FFT
    {

        ELISE_COPY(mW->all_pts(),Min(255,Max(0,mTIm._the_im.in(0))),mW->ogray());
        double aVMin,aVMax;
        ELISE_COPY
        (
            mTIm._the_im.all_pts(),
            mTIm._the_im.in(),
            VMax(aVMax) | VMin(aVMin)
        );
        std::cout << "MAX MIN " << aVMin << " " << aVMax << "\n";
        mW->clik_in();

        ELISE_COPY(mW->all_pts(),Min(255,Max(0,mTRef._the_im.in(0))),mW->ogray());
        ELISE_COPY
        (
            mTRef._the_im.all_pts(),
            mTRef._the_im.in(),
            VMax(aVMax) | VMin(aVMin)
        );
        std::cout << "MAX MIN " << aVMin << " " << aVMax << "\n";
        mW->clik_in();



        ELISE_COPY(mW->all_pts(),Min(255,Max(0,mTMasq._the_im.in(0))),mW->ogray());
        ELISE_COPY
        (
            mTMasq._the_im.all_pts(),
            mTMasq._the_im.in(),
            VMax(aVMax) | VMin(aVMin)
        );
        std::cout << "MAX MIN " << aVMin << " " << aVMax << "\n";

        ELISE_COPY(mW->all_pts(),mTMasq._the_im.in(0),mW->odisc());
        mW->clik_in();
    }

// if (mDebug) std::cout << "RUNNNN DecFFT\n";
    ElTimer aChrono;
    Im2D_REAL8 aRes = ElFFTPonderedCorrelNCPadded
                      (
                           mTIm.in(0),
                           mTRef.in(0),
                           mTIm.sz(),
                           1.0,
                           mTMasq.in(0),
                           1e-5,
                           mSomPds * 0.99
                      );
    Im2D_REAL8 aVisuRes = aRes.dup();

    std::list<Pt3dr> aLRes;
    bool Cont = true;
    double aBestCorr = -1;
    int aCpt = 0;
    while (Cont)
    {
        TIm2D<REAL8,REAL8> aTRes(aRes);
        Pt2di aPMax;
        ELISE_COPY(aRes.all_pts(),aRes.in(),aPMax.WhichMax());
        double aCorr = aTRes.get(aPMax);
        aBestCorr = ElMax(aBestCorr,aCorr);
        Cont = (aCpt<2) || ((aCorr > (aBestCorr - 0.3)) && (aCpt<5));
        aCpt++;
        if (Cont)
        {
            aLRes.push_back(Pt3dr(aPMax.x,aPMax.y,aCorr));
            int aDist=10;
            Pt2di aPDist(aDist,aDist);
            ELISE_COPY
            (
                 rectangle(aPMax-aPDist,aPMax+aPDist).chc(Virgule(mod(FX,aRes.sz().x),mod(FY,aRes.sz().y))),
                 0,
                 aRes.out()
            );
        }
    }

/*
    Pt2di aSz = aRes.sz();
    Pt2di aP;
    for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
        {
             if (std_isnan(aTRes.get(aP))) std::cout << "NNAAAAn\n";
        }
    }
*/

    if (VisuFFT &&(mW!=0))
    {
         Fonc_Num aF = (1.0+aVisuRes.in(0))*128;
         // ELISE_COPY(mW->all_pts(),mTIm.in(0) ,mW->ogray());
         ELISE_COPY(mW->all_pts(),Max(0.0,Min(255.0,aF)) ,mW->ogray());
         bool First= true;
         for (std::list<Pt3dr>::iterator itP=aLRes.begin() ; itP!=aLRes.end() ; itP++)
         {
            // std::cout << "Correl=" << itP->z << "\n";
            mW->draw_circle_loc(Pt2dr(itP->x,itP->y),5.0,mW->pdisc()(First ? P8COL::red : P8COL::green));

            Pt2di aP(round_ni(itP->x),round_ni(itP->y));
            aP =  DecFFT2DecIm(aRes,aP);
            itP->x = aP.x;
            itP->y = aP.y;
            First = false;
         }
         mW->clik_in();
    }

    return aLRes;
}





cQuickCorrelOneFidMark::cQuickCorrelOneFidMark
(
              Fonc_Num            aFoncIm,
              Fonc_Num            aFoncRef,
              Fonc_Num            aFoncMasq,
              Box2di              aBoxRef,
              Pt2di               aIncLoc,
              int                 aSzFFT,
              bool                Debug
) :
    mFoncFileIm     (aFoncIm),
    mFoncRef        (aFoncRef),
    mFoncMasqRef    (aFoncMasq),
    mNoMasq         (mFoncMasqRef.is1()),
    mBoxRef         (aBoxRef),
    mSzRef          (mBoxRef.sz()),
    mIncLoc         (aIncLoc),
    mSzBuf          (mSzRef + mIncLoc*2),
    mSzSsResFFT     (aSzFFT),
    mSsRes          (double(mSzSsResFFT) / dist8(mSzBuf)),
    mNbNiv          (3),
    mSsResByNiv     (pow(mSsRes,1.0/(mNbNiv-1))),
    mDebug          (Debug)
{
    // Pour l'instant on met en dur une pyram a trois niveau
    
    for (int aK=0 ; aK<mNbNiv ; aK++)
    {
         mPyram.push_back(cQuickCorrelPackIm(mSzBuf,mSzRef,pow(mSsResByNiv,aK),Debug));
    }
}


cOneSol_QuickCor  cQuickCorrelOneFidMark::TestCorrel(const Pt2dr & aP0)
{
    // Calcul de l'offset de chargement
    mCurDecRef = round_ni(aP0) + mBoxRef._p0;
    mCurDecIm = mCurDecRef -mIncLoc;


    LoadIm();
    std::list<Pt3dr> aLDecCor  = mPyram[mNbNiv-1].DecFFT();

    double aBestCor=-1;
    Pt2dr  aBestP(0,0);

    for (std::list<Pt3dr>::iterator itP=aLDecCor.begin() ; itP!=aLDecCor.end() ; itP++)
    {
        Pt3dr & aDecCor0 = *itP;
    
        Pt2di aPDec(aDecCor0.x,aDecCor0.y);

        // std::cout << "DECC " << aPDec << "\n";
        for (int aK=mNbNiv-2 ;  aK>=0 ; aK--)
        {
              aPDec = round_ni(Pt2dr(aPDec) /mSsResByNiv);
              Pt2di aPOpt =  mPyram[aK].OptimizeInt(aPDec,1+round_up(2/mSsResByNiv));
 
              aPDec = aPOpt;
        }

        Pt2dr aRPDec = Pt2dr(aPDec);
        double aNewCor;
        for (double aStep=0.5 ; aStep>0.01 ; aStep/=2)
        {
           aRPDec =  mPyram[0].OptimizeReel(aRPDec,aStep,2,aNewCor);
        }
        // std::cout << "Coorel " << aDecCor0.z << " => " << aNewCor << "\n";

        aRPDec = aRPDec + aP0 +  Pt2dr(mCurDecIm-mCurDecRef);

        itP->x = aRPDec.x;
        itP->y = aRPDec.y;
        itP->z = aNewCor;

        if (aNewCor > aBestCor)
        {
            aBestP = aRPDec;
            aBestCor = aNewCor;
        }
    }
    //   std::cout << "==#=#=#=#=#=#=#*******************ppppp\n";

    cOneSol_QuickCor aRes;
    // aPInc   = aP0 - mCurDecRef + mCurDecIm + aPDec
    // 

    aRes.mPOut = aBestP;
    aRes.mLSols =  aLDecCor;

    // aRes.mPOut =  aP0 + aBestP + Pt2dr(mCurDecIm-mCurDecRef) ;
    return aRes;
}

void cQuickCorrelOneFidMark::LoadIm()
{
    // On charge les fichiers images dans les buffer a pleine resol
    ELISE_COPY(mPyram[0].mTIm.all_pts() ,trans(mFoncFileIm,mCurDecIm),mPyram[0].mTIm.out());
    ELISE_COPY(mPyram[0].mTRef.all_pts(),trans(mFoncRef,mCurDecRef),mPyram[0].mTRef.out());
    ELISE_COPY(mPyram[0].mTMasq.all_pts(),trans(mFoncMasqRef,mCurDecRef),mPyram[0].mTMasq.out());



    // calcule des images reduite
   for (int aK=1 ; aK<mNbNiv ; aK++)
   {
        mPyram[aK].InitByReduce(mPyram[aK-1],pow(2.0,1.0/(mNbNiv-1)));
   }

   for (int aK=0 ; aK<mNbNiv ; aK++)
      mPyram[aK].FinishLoad();

   //     Tiff_Im::Create8BFromFonc("TestKU-Im0.fif",mPyram[0].mTIm.sz(),mPyram[0].mTIm.in());
   //     Tiff_Im::Create8BFromFonc("TestKU-Im1.fif",mPyram[1].mTIm.sz(),mPyram[1].mTIm.in());
   //     Tiff_Im::Create8BFromFonc("TestKU-Im2.fif",mPyram[2].mTIm.sz(),mPyram[2].mTIm.in());
}



/*****************************************************************************/


const std::string cAppli_FFTKugelhupf_main::TheKeyOI = "Key-Assoc-STD-Orientation-Interne";

cAppli_FFTKugelhupf_main::cAppli_FFTKugelhupf_main(int argc,char ** argv) :
  cAppliWithSetImage      (argc-1,argv+1,TheFlagNoOri),
  mTargetHalfSzPx         (150,150),
  mSearchIncertitudePx    (500),
  mExtMasq                ("NONE"),
  mDebug                  (false),
  mValSim                 (false),
  mPdsCorr                (10.0)
{
    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mFullPattern, "Pattern of scanned images",  eSAM_IsPatFile)
                     << EAMC(mFiducPtsFileName, "2d fiducial points of an image", eSAM_IsExistFile),
         LArgMain()  << EAM(mTargetHalfSzPx,"TargetHalfSize",true,"Target half size in pixels (Def= computed from mask or 150)")
                     << EAM(mExtMasq,"Masq",true,"Masq extension for ref image, Def=NONE (means unused), USE OF MASK IS STRONGLY RECOMMANDED !!!!")
                     << EAM(mSearchIncertitudePx,"SearchIncertitude",true,"Def 3*Targ HalfSz")
                     << EAM(mSzFFT,"SzFFT",true,"Sz of initial fft research, power of recomanded, Def=256 or 128 depending other")
                     << EAM(mDebug,"Debug",true,"Debugin/Tuning option")
                     << EAM(mValSim,"ValSim",true,"Validation by similitude/affinity (Def size depent, affinity if NbPoint>=6)")
/*
                     << EAM(aSearchIncertitudePx,"SearchIncertitude",true,"Search incertitude in pixels (Def=5)")
                     << EAM(aSearchStepPx,"SearchStep",true,"Search step in pixels (Def=0.5)")
                     << EAM(aThreshold,"Threshold",true,"Limit to accept a correlation (Def=0.90)")
*/
    );


     const cInterfChantierNameManipulateur::tSet * aSetIm = mEASF.SetIm();

     mEASF.mICNM->Assoc2To1(TheKeyOI,"XXX",true);


     if (aSetIm->size() ==0)
     {
          ELISE_ASSERT(false,"No image in FFTKugelhupf");
     }
     else if (aSetIm->size() > 1)
     {
         std::cout << "cAppli_FFTKugelhupf_main::cAppli_FFTKugelhupf_main \n";
         ExpandCommand(2,"",true);
         return;
     }




     mDico = StdGetFromPCP(mFiducPtsFileName,MesureAppuiFlottant1Im);

     if (! EAMIsInit(&mValSim))
     {
           mValSim = (mDico.OneMesureAF1I().size() <6);
     }

     mNameIm2Parse = (*aSetIm)[0]; 
     mNameImRef    = mDico.NameIm();

     if (mExtMasq != "NONE")
     {
        CorrecNameMasq(mEASF.mDir,mNameImRef,mExtMasq);
        mWithMasq = true;
        mNameFileMasq = StdPrefix(mNameImRef) + mExtMasq + ".tif";
     }
     else
     {
        mWithMasq = false;
     }

     // Initialisation des resultats, calcul eventuel de la taille des vignettes
    Pt2di  aCompTHSP(0,0);
    for 
    (
        std::list<cOneMesureAF1I>::const_iterator itM=mDico.OneMesureAF1I().begin() ;
        itM!=mDico.OneMesureAF1I().end() ;
        itM++
    )
    {
          Pt2di aTHSP =  mTargetHalfSzPx;
          if (mWithMasq)
          {
               Tiff_Im aTifM(mNameFileMasq.c_str());

               Pt2di aC = Pt2di(itM->PtIm());
               Pt2di aP0 = aC - aTHSP;
               Pt2di aP1 = aC + aTHSP;
               Pt2di aSzIm = aP1 - aP0;
               Pt2di aGerm = aSzIm / 2;

               Im2D_Bits<1> anIm(aSzIm.x,aSzIm.y,0);
               TIm2DBits<1>  aTM(anIm);
               ELISE_COPY(anIm.all_pts(),trans(aTifM.in(0),aP0),anIm.out());

               // Debordement des composantes connexes si taches autour des marques trop grandes
               ELISE_COPY(anIm.border(1),0,anIm.out());

               int aValC = aTM.get(aGerm);
               if (aValC!=1)
               {
                   std::cout << "For pt= " << itM->NamePt() << " image=" <<  mNameFileMasq << "\n";
                   ELISE_ASSERT(false,"Centre of target not in masq ");
               }
               Pt2di pmax,pmin;
               ELISE_COPY
               (
                    conc(aGerm,anIm.neigh_test_and_set(Neighbourhood::v4(),1,0,2)),
                    Virgule(FX,FY),
                    pmax.VMax() | pmin.VMin()
               );

               aTHSP = Sup(pmax-aGerm,aGerm-pmin) + Pt2di(2,2);

               aCompTHSP  = Sup(aCompTHSP,aTHSP);

          }
          cOneHypMarkFid aHMF(*itM);
          aHMF.mTargetSize = aTHSP;
          mListHMF.push_back(aHMF);
    }
    if ( mWithMasq && (! EAMIsInit(&mTargetHalfSzPx)))
    {
       mTargetHalfSzPx = Inf(aCompTHSP,mTargetHalfSzPx);
    }

    if (! EAMIsInit(&mSearchIncertitudePx))
    {
         mSearchIncertitudePx = ElMax(30, 3 * dist8(mTargetHalfSzPx));
    }
    if (mDebug)
        std::cout << "Computed Target 1/2 size= " << mTargetHalfSzPx << "\n";
   
    if (! EAMIsInit(&mSzFFT))
    {
          double aNb = 2*sqrt(double((mTargetHalfSzPx.x+mSearchIncertitudePx)*(mTargetHalfSzPx.y+mSearchIncertitudePx)));
          if (aNb < 200)
             mSzFFT = 64;
          else if (aNb < 1200)
             mSzFFT = 128;
          else if (aNb < 2400)
             mSzFFT = 256;
          else 
             mSzFFT = 512;

        if (mDebug) 
           std::cout << "SzFFT= " << mSzFFT << "\n";
    }

     mQCor = new cQuickCorrelOneFidMark
                 (
                      Tiff_Im::StdConvGen(mNameIm2Parse,1,true).in(0),
                      Tiff_Im::StdConvGen(mNameImRef,1,true).in(0),
                      mWithMasq ?  Tiff_Im::StdConvGen(mNameFileMasq,1,true).in(0) : 1,
                      Box2di(-mTargetHalfSzPx,mTargetHalfSzPx),
                      Pt2di(mSearchIncertitudePx,mSearchIncertitudePx),
                      mSzFFT,
                      mDebug
                 );

    DoResearch();
    // std::cout << "EXTMASQ = " << mExtMasq  << " " << mNameFileMasq<< "\n";
     // mQCor = new  
}

void cAppli_FFTKugelhupf_main::DoResearch()
{
    cMesureAppuiFlottant1Im aRes;
    aRes.NameIm() = mNameIm2Parse;
    
    // Recherche pour chaque point d'une ou plusieurs solutions
    for 
    (
        std::list<cOneMesureAF1I>::const_iterator itM=mDico.OneMesureAF1I().begin() ;
        itM!=mDico.OneMesureAF1I().end() ;
        itM++
    )
    {
        cOneMesureAF1I aMes;
        aMes.NamePt() = itM->NamePt();
        cOneSol_FFTKu aSol = Research1(*itM);
        aMes.PtIm() = aSol.mOut.mPOut;
        aRes.OneMesureAF1I().push_back(aMes);

        if (mDebug)
        {
             std::cout << "FftKgh : " << aMes.NamePt() << " "<< aMes.PtIm() -itM->PtIm()  << "\n";
        }
    }


    mBestCostComb = 1e20;
    // Recherche combinatoire avec approche de type Ransac
    for (int aKS1=0 ; aKS1<int(mVSols.size()) ; aKS1++)
    {
         for (int aKS2=aKS1+1 ; aKS2<int(mVSols.size()) ; aKS2++)
         {
              if (mValSim)
              {
                  TestOneSolCombine(aKS1,aKS2,-1);
              }
              else
              {
                 for (int aKS3=aKS2+1 ; aKS3<int(mVSols.size()) ; aKS3++)
                 {
                     Pt2dr aV12 = mVSols[aKS1].mOut.mPOut-mVSols[aKS2].mOut.mPOut;
                     Pt2dr aV13 = mVSols[aKS1].mOut.mPOut-mVSols[aKS3].mOut.mPOut;

                     double aDet = (vunit(aV12) ^ vunit(aV13));
                     if (ElAbs(aDet) > 0.1)
                     {
                         TestOneSolCombine(aKS1,aKS2,aKS3);
                     }
                 }
              }
         }
    }

    int aKComb=0;
    ElPackHomologue aNewPack;
    for 
    (
        std::list<cOneMesureAF1I>::iterator itM=aRes.OneMesureAF1I().begin() ;
        itM!=aRes.OneMesureAF1I().end() ;
        itM++
    )
    {
        aNewPack.Cple_Add(ElCplePtsHomologues(mBestSolComb[aKComb],mVSols[aKComb].mIn.PtIm()));
        itM->PtIm() = mBestSolComb[aKComb];
        aKComb++;
    }
    ELISE_ASSERT(aKComb==int(mBestSolComb.size()),"Incohe in combine");
   
    double     aResidu;
    ElAffin2D  anAff = ElAffin2D::L2Fit(aNewPack,&aResidu);
    aRes.PrecPointeByIm().SetVal(aResidu);

    MakeFileXML(aRes,mEASF.mICNM->Assoc2To1(TheKeyOI,mNameIm2Parse,true).second);
    std::cout << "RESIDU = " << aResidu << " for " << mNameIm2Parse << "\n";
}


void  cAppli_FFTKugelhupf_main::TestOneSolCombine(int aKS1,int aKS2,int aKS3)
{
   ElPackHomologue aPack;
   aPack.Cple_Add(ElCplePtsHomologues(mVSols[aKS1].mOut.mPOut,mVSols[aKS1].mIn.PtIm()));
   aPack.Cple_Add(ElCplePtsHomologues(mVSols[aKS2].mOut.mPOut,mVSols[aKS2].mIn.PtIm()));
   if (aKS3 >= 0)
   {
       aPack.Cple_Add(ElCplePtsHomologues(mVSols[aKS3].mOut.mPOut,mVSols[aKS3].mIn.PtIm()));
   }
   cElHomographie aHom (aPack,true);

   double aCostGlob = 0;
   std::vector<Pt2dr> aVPtsOut;
   for (int aKS=0 ; aKS<int(mVSols.size()) ; aKS++)
   {
       double aCostMin = 1e20;
       Pt2dr aPOutMin(-1,-1);
       Pt2dr aPIn= mVSols[aKS].mIn.PtIm();
       const std::list<Pt3dr> & aLP = mVSols[aKS].mOut.mLSols;
       for (std::list<Pt3dr>::const_iterator itP=aLP.begin(); itP!=aLP.end() ; itP++)
       {
            Pt2dr aPOut (itP->x,itP->y);
// std::cout << "HHHHH  " << aPOut << " " << mVSols[aKS].mOut.mPOut << "\n";
            double aCost = euclid(aHom.Direct(aPOut)-aPIn) + (1-ElMin(1.0,itP->z)) * mPdsCorr;
            if (aCost < aCostMin)
            {
               aCostMin = aCost;
               aPOutMin = aPOut;
            }
       }
       aVPtsOut.push_back(aPOutMin);
       aCostGlob += aCostMin;
       // std::cout << "COST MIN " << aCostMin  << "\n";
   }

   if (aCostGlob<mBestCostComb)
   {
       mBestCostComb = aCostGlob;
       mBestSolComb  = aVPtsOut;
   }
    
/*
    for (int aKS1=0 ; aKS1<int(mVSols.size()) ; aKS1++)
*/
}


cOneSol_FFTKu cAppli_FFTKugelhupf_main::Research1(const cOneMesureAF1I & aMes)
{
    cOneSol_FFTKu aSol;
    aSol.mOut = mQCor->TestCorrel(aMes.PtIm());
    aSol.mIn = aMes;

    mVSols.push_back(aSol);
    
    mPackH.Cple_Add(ElCplePtsHomologues(aSol.mOut.mPOut,aSol.mIn.PtIm()));
    return aSol;
}


int FFTKugelhupf_main(int argc,char ** argv) 
{
   cAppli_FFTKugelhupf_main anAppli(argc,argv);


   return EXIT_SUCCESS;
}



//----------------------------------------------------------------------------



int cCorrelImage::mSzW=8;
void cCorrelImage::setSzW(int aSzW)
{
  mSzW=aSzW;
}

int cCorrelImage::getSzW()
{
  return this->mSzW;
}

Pt2di cCorrelImage::getmSz()
{
  return this->mSz;
}

cCorrelImage::cCorrelImage():
  mSz    (Pt2di(mSzW*2+1,mSzW*2+1)),
  mTIm    (mSz),
  mIm     (mTIm._the_im),
  mTImS1  (mSz),
  mImS1   (mTImS1._the_im),
  mTImS2  (mSz),
  mImS2   (mTImS2._the_im)
{}


void cCorrelImage::getFromIm(Im2D<U_INT1,INT4> * anIm,double aCenterX,double aCenterY)
{
  ELISE_COPY
    (
     mIm.all_pts(),
     anIm->in(0)[Virgule(FX+aCenterX-mSzW,FY+aCenterY-mSzW)], //put in (x,y) on destination pic what is in (x+400,y+400) in source pic
     mIm.out()
    );

  //to write to a file:
  //Tiff_Im(
  //          "toto.tif",
  //          Pt2di(400,400),
  //          GenIm::u_int1,
  //          Tiff_Im::No_Compr,
  //          Tiff_Im::BlackIsZero,
  //          Tiff_Im::Empty_ARG ).out()
  prepare();
}

void cCorrelImage::getWholeIm(Im2D<U_INT1,INT4> * anIm)
{
  ELISE_COPY
    (
     mIm.all_pts(),
     anIm->in()[Virgule(FX,FY)],
     mIm.out()
    );

  //to write to a file:
  //Tiff_Im(
  //          "toto.tif",
  //          Pt2di(400,400),
  //          GenIm::u_int1,
  //          Tiff_Im::No_Compr,
  //          Tiff_Im::BlackIsZero,
  //          Tiff_Im::Empty_ARG ).out()
  prepare();
}

void cCorrelImage::prepare()
{
  ELISE_COPY
    (
     mIm.all_pts(),
     rect_som(mIm.in_proj(),mSzW) / ElSquare(1.0+2*mSzW),
     mImS1.out()
    );

  ELISE_COPY
    (
     mIm.all_pts(),
     rect_som(Square(mIm.in_proj()),mSzW) / ElSquare(1.0+2*mSzW),
     mImS2.out()
    );

}


double cCorrelImage::CrossCorrelation( const cCorrelImage & aIm2 )
{
  //if (! InsideW(aPIm1,mSzW)) return TheDefCorrel;

  Pt2di aPIm1(mSzW,mSzW);
  Pt2di aPIm2 = aPIm1;
  //if (! aIm2.InsideW(aPIm2,mSzW)) return TheDefCorrel;

  double aS1 = mTImS1.get(aPIm1);
  double aS2 = aIm2.mTImS1.get(aPIm2);
  //std::cout<<"aS1 "<<aS1<<"   aS2 "<<aS2<<std::endl;


  double aCov = Covariance(aIm2)  -aS1*aS2;
  //std::cout<<"aCov "<<aCov<<std::endl;

  double aVar11 = mTImS2.get(aPIm1) - ElSquare(aS1);
  double aVar22 = aIm2.mTImS2.get(aPIm2) - ElSquare(aS2);
  //std::cout<<"aVar11 "<<aVar11<<"   aVar22 "<<aVar22<<std::endl;

  return aCov / sqrt(ElMax(1e-5,aVar11*aVar22));
}

double cCorrelImage::Covariance( const cCorrelImage & aIm2 )
{
  Pt2di aPIm1(mSzW,mSzW);
  if (1) // A test to check the low level access to data - pixel access
  {
    unsigned char ** aRaw2 = mIm.data();
    unsigned char *  aRaw1 = mIm.data_lin();
    ELISE_ASSERT(mTIm.get(aPIm1)==aRaw2[aPIm1.y][aPIm1.x],"iiiii");
    ELISE_ASSERT((aRaw1+aPIm1.y*mSz.x) ==aRaw2[aPIm1.y],"iiiii");
  }
  double aSom =0;
  Pt2di aPIm2 = aPIm1;

  Pt2di aVois;
  /*for (aVois.x=0; aVois.x<=mSzW*2 ; aVois.x++)
    {
    for (aVois.y=0; aVois.y<=mSzW*2 ; aVois.y++)
    {
    aSom +=  mTIm.get(aPIm1+aVois) * aIm2.mTIm.get(aPIm2+aVois);
  //std::cout<<"aPIm1+aVois "<<aPIm1+aVois<<"    mTIm.get(aPIm1+aVois) "<<mTIm.get(aPIm1+aVois)<<"    aIm2.mTIm.get(aPIm2+aVois) "<<aIm2.mTIm.get(aPIm2+aVois)<<std::endl;
  } 
  }*/
  for (aVois.x=-mSzW; aVois.x<=mSzW ; aVois.x++)
  {
    for (aVois.y=-mSzW; aVois.y<=mSzW ; aVois.y++)
    {
      aSom +=  mTIm.get(aPIm1+aVois) * aIm2.mTIm.get(aPIm2+aVois);
    } 
  }

  //std::cout<<"aSom /ElSquare(1+2*mSzW) "<<aSom<<"/"<<ElSquare(1+2*mSzW)<<std::endl;
  return aSom /ElSquare(1+2*mSzW);
}


//----------------------------------------------------------------------------


// ScannedImage class


cScannedImage::cScannedImage
( std::string aNameScannedImage,
  cInterfChantierNameManipulateur * aICNM,
  std::string aXmlDir  ):
  mName             (aNameScannedImage),
  mNameImageTif     (NameFileStd(mName,1,false,true,true,true)),
  mXmlFileName      (aXmlDir+"MeasuresIm-"+mName+".xml"),
  mTiffIm           (mNameImageTif.c_str()),
  mImgSz            (mTiffIm.sz()),
  mImT              (mImgSz),
  mIm               (mImT._the_im),
  mIsLoaded         (false)
{
  //std::cout<<"ScannedImageName: "<<mName<<std::endl;
  
}

void cScannedImage::load()
{
    ELISE_COPY(mIm.all_pts(),mTiffIm.in(),mIm.out());
    mIsLoaded=true;
}

//----------------------------------------------------------------------------

int Kugelhupf_main(int argc,char ** argv)
{
  std::string aFullPattern;//pattern of all scanned images
  std::string aFiducPtsFileName;//2d fiducial points of 1 image
  int aTargetHalfSzPx=64;//target size in pixel
  int aSearchIncertitudePx=5;//Search incertitude
  double aSearchStepPx=0.5;//Search step
  double aThreshold=0.9;//limit to accept a correlation
 
  bool verbose=false;

  std::cout<<"Kugelhupf (Klics Ubuesques Grandement Evites, Lent, Hasardeux mais Utilisable pour Points Fiduciaux): Automatic fiducial point determination"<<std::endl;
  

  ElInitArgMain
    (
     argc,argv,
     //mandatory arguments
     LArgMain()  << EAMC(aFullPattern, "Pattern of scanned images",  eSAM_IsPatFile)
     << EAMC(aFiducPtsFileName, "2d fiducial points of an image", eSAM_IsExistFile),
     //optional arguments
     LArgMain()  << EAM(aTargetHalfSzPx,"TargetHalfSize",true,"Target half size in pixels (Def=64)")
     << EAM(aSearchIncertitudePx,"SearchIncertitude",true,"Search incertitude in pixels (Def=5)")
     << EAM(aSearchStepPx,"SearchStep",true,"Search step in pixels (Def=0.5)")
     << EAM(aThreshold,"Threshold",true,"Limit to accept a correlation (Def=0.90)")
    );
    
  if (MMVisualMode) return EXIT_SUCCESS;

  std::cout<<"aFiducPtsFileName: "<<aFiducPtsFileName<<std::endl;

  // Initialize name manipulator & files
  std::string aDirXML,aDirImages,aPatIm;
  std::string aFiducPtsFileTmpName;
  SplitDirAndFile(aDirXML,aFiducPtsFileTmpName,aFiducPtsFileName);
  SplitDirAndFile(aDirImages,aPatIm,aFullPattern);
  std::cout<<"Working dir: "<<aDirImages<<std::endl;
  std::cout<<"Images pattern: "<<aPatIm<<std::endl;


  cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
  const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));


  //read xml file
  //see MesureAppuiFlottant1Im definition in include/XML_GEN/ParamChantierPhotogram.xml
  cMesureAppuiFlottant1Im aDico = StdGetFromPCP(aFiducPtsFileName,MesureAppuiFlottant1Im);
  std::list< cOneMesureAF1I > aOneMesureAF1IList= aDico.OneMesureAF1I();
  std::string aMainPictureName=aDico.NameIm();
  cScannedImage aMainImg(aMainPictureName,aICNM,aDirXML);
  std::cout<<"On "<<aMainPictureName<<", found 2d points:\n";

  for (std::list<cOneMesureAF1I>::iterator itP=aOneMesureAF1IList.begin(); itP != aOneMesureAF1IList.end(); itP ++)
  {
    std::cout<<" - "<<itP->NamePt()<<" "<<itP->PtIm()<<"\n";
    aMainImg.getAllFP().OneMesureAF1I().push_back((*itP));
  }


  std::cout<<"Found pictures:\n";
  for (unsigned int i=0;i<aSetIm.size();i++)
  {
    std::cout<<" - "<<aSetIm[i]<<"\n";
  }



  Pt2di aTargetImSz(aTargetHalfSzPx*2+1,aTargetHalfSzPx*2+1);
  //Pt2di aSearchImSz(aTargetHalfSzPx*2+2*aSearchIncertitudePx+1,aTargetHalfSzPx*2+2*aSearchIncertitudePx+1);

  std::cout<<"Create sub pictures..."<<std::endl;

  cCorrelImage::setSzW(aTargetHalfSzPx);
  cCorrelImage aTargetIm;
  cCorrelImage aTargetImSearch;


  std::vector<cScannedImage*> aImgList;
  for (unsigned int i=0;i<aSetIm.size();i++)
  {
    if (aSetIm[i]==aMainPictureName)
      continue;

    std::cout<<"Working on image "<<aSetIm[i]<<std::endl;
    cScannedImage * aImg=new cScannedImage(aSetIm[i],aICNM,aDirXML);
    if (aImg->isExistingXmlFile())
    {
      std::cout<<"  Already has an xml file."<<std::endl;
      delete aImg;
      continue;
    }

    aImgList.push_back(aImg);
    aImg->getAllFP().NameIm()=aImg->getName();

    for (std::list<cOneMesureAF1I>::iterator itP=aMainImg.getAllFP().OneMesureAF1I().begin(); itP != aMainImg.getAllFP().OneMesureAF1I().end(); itP ++)
    {
      if (verbose)
        std::cout<<"  Target "<<itP->NamePt()<<"  "<<itP->PtIm()<<"\n";
      else
        std::cout<<"."<<std::flush;

      aTargetIm.getFromIm(aMainImg.getIm(),itP->PtIm().x,itP->PtIm().y);

      double aCoefCorrelMax=-1.0;
      double aTmpCoefCorrel;
      double aStepApprox=aTargetHalfSzPx/20.0;
      cOneMesureAF1I aBestPtApprox;
      cOneMesureAF1I aBestPt;
      aBestPt.NamePt()=itP->NamePt();
      aBestPtApprox.NamePt()=itP->NamePt();

      for (double x=-aSearchIncertitudePx;x<=aSearchIncertitudePx;x+=aStepApprox)
      {
        for (double y=-aSearchIncertitudePx;y<=aSearchIncertitudePx;y+=aStepApprox)
        {
          aTargetImSearch.getFromIm(aImg->getIm(),itP->PtIm().x+x,itP->PtIm().y+y);
          if (verbose)
            std::cout<<aTargetIm.CrossCorrelation(aTargetImSearch)<<"     ";
          aTmpCoefCorrel=aTargetIm.CrossCorrelation(aTargetImSearch);
          if (aTmpCoefCorrel>aCoefCorrelMax)
          {
            if (verbose)
              std::cout<<"   new best\n";
            aCoefCorrelMax=aTmpCoefCorrel;
            aBestPtApprox.PtIm()=Pt2dr(itP->PtIm().x+x,itP->PtIm().y+y);
          }
        }
        if (verbose)
          std::cout<<std::endl;
      }
      aCoefCorrelMax=-1.0;
 
      for (double x=-aStepApprox*2;x<=aStepApprox*2;x+=aSearchStepPx)
      {
        for (double y=-aStepApprox*2;y<=aStepApprox*2;y+=aSearchStepPx)
        {
          aTargetImSearch.getFromIm(aImg->getIm(),aBestPtApprox.PtIm().x+x,aBestPtApprox.PtIm().y+y);
          if (verbose)
            std::cout<<aTargetIm.CrossCorrelation(aTargetImSearch)<<"     ";
          aTmpCoefCorrel=aTargetIm.CrossCorrelation(aTargetImSearch);
          if (aTmpCoefCorrel>aCoefCorrelMax)
          {
            aCoefCorrelMax=aTmpCoefCorrel;
            aBestPt.PtIm()=Pt2dr(aBestPtApprox.PtIm().x+x,aBestPtApprox.PtIm().y+y);
          }
        }
        if (verbose)
          std::cout<<std::endl;
      }
      if (verbose)
        std::cout<<"Best: "<<aBestPt.PtIm()<<" ("<<aCoefCorrelMax<<")\n";
      if (aCoefCorrelMax>aThreshold)
      {
        aImg->getAllFP().OneMesureAF1I().push_back(aBestPt);
      }else{
        std::cout<<"Bad match on "<<itP->NamePt()<<": "<<aCoefCorrelMax<<"/"<<aThreshold<<std::endl;
        break;
      }
    }
    std::cout<<"\n";
    //write xml file only if all points found:
    if (aImg->getAllFP().OneMesureAF1I().size()==aMainImg.getAllFP().OneMesureAF1I().size())
    {
      std::cout<<"  Save xml file."<<std::endl;
      MakeFileXML(aImg->getAllFP(),aImg->getXmlFileName());
    }
    delete aImg;
  }

  return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

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
   Footer-MicMac-eLiSe-25/06/2007/*/
