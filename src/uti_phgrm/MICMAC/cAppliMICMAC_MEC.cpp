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
#include "../src/uti_phgrm/MICMAC/MICMAC.h"

int cAppliMICMAC::MemSizePixelImage() const
{
    // Nombre d'octet par pixel et par image,
    // compte tenu d'un rab de recouvrement
    return   12;
}



double cAppliMICMAC::MemSizeProcess(int aTxy) const
{

   double aRes=     ElSquare(aTxy+2.0*mSzRec)
	   *  (
	         mMemPart.NbMaxImageOn1Point().Val() * MemSizePixelImage() 
               + mCurEtape->MemSizePixelAlgo()
	      );

  return aRes;
}

int cAppliMICMAC::GetTXY() const
{
    int aSz=mSzDMin;
    int aStep = 20;
    while
    ( 
          (aSz <mSzDMax)
       && (MemSizeProcess(aSz)<(AvalaibleMemory().Val()*1e6))
    )
	   aSz += aStep;
    return aSz-aStep;
}

void cAppliMICMAC::DoAllMEC()
{


#if CUDA_ENABLED

    CGpGpuContext<cudaContext> gpgpuContext;
    gpgpuContext.createContext();

#endif

     for 
     (
		tContEMC::const_iterator itE = mEtapesMecComp.begin();
		itE != mEtapesMecComp.end();
		itE++
     )
     {          
        OneEtapeSetCur(**itE);
        if (mDoTheMEC  && (!DoNothingBut().IsInit()))
        {
           DoOneEtapeMEC(**itE);
           //std::cout<<" NUM ETAPE ======= >>>> "<<(*itE)->Num()<< "  "<<FirstEtapeMEC().Val()<<std::endl;
        }

        if (
                 ( (*itE)->Num()>=FirstEtapeMEC().Val())
           &&    ( (*itE)->Num()<=LastEtapeMEC().Val())
           )
        {
            if (! CalledByProcess().Val())
               MakeResultOfEtape(**itE);


            if (
                      (mDoTheMEC  && (!DoNothingBut().IsInit()))
                 ||   (DoNothingBut().IsInit() && (ButDoOrtho().Val()||ButDoPartiesCachees().Val()))
                 ||   ( Paral_Pc_NbProcess().IsInit())
               )
            {
               MakePartiesCachees();
            }


            if (     (! CalledByProcess().Val())
                 &&  (
                            (mDoTheMEC  && (!DoNothingBut().IsInit()))
                        ||  (DoNothingBut().IsInit() &&  ButDoRedrLocAnam().Val())
                     )
               )
            {
               MakeRedrLocAnamSA();
            }
        }
     }

#if CUDA_ENABLED
    if (mCorrelAdHoc && mCorrelAdHoc->GPU_CorrelBasik().IsInit())    
        gpgpuContext.deleteContext();
#endif

}

/*
 Formule ancienne, egalise les ecart moyen avec une somme discrete et
 approximation
*/

double Old_FromSzW2FactExp(double aSzW,double mCurNbIterFenSpec)
{
    aSzW  /= sqrt(mCurNbIterFenSpec);
   return (2*aSzW+1)/(2*aSzW+5);
}

/* Formule nouvelle, egalise les ecart type avec une somme integrale,
 * sans approximation
 */

/*
double FromSzW2FactExp(double aSzW,double mCurNbIterFenSpec)
{
   double aRes = exp(- (sqrt(6*mCurNbIterFenSpec))/(1+aSzW));
   // std::cout << "ANCIEN " << Old_FromSzW2FactExp(aSzW,mCurNbIterFenSpec) << "\n";
   // std::cout << "FromSzW2FactExp : " << aRes << "\n"; getchar();
   return aRes;
}
*/


void cAppliMICMAC::OneEtapeSetCur(cEtapeMecComp & anEtape)
{
     mPrecEtape = mCurEtape;
     mCurEtape = & anEtape;
     mZoomChanged=false;


     if (anEtape.EtapeMEC().GenCubeCorrel().ValWithDef(false))
     {
        ELISE_fp::MkDirSvp(DirCube());
     }
     mEBI = mCurEtape->EBI();
     const cEtapeMEC & anEM = mCurEtape->EtapeMEC();
     mCorrelAdHoc = anEM.CorrelAdHoc().PtrVal();

     mSzWR.x = anEM.SzW().Val();
     mSzWR.y = anEM.SzWy().ValWithDef(mSzWR.x);
     mCurTypeWC = anEM.TypeWCorr().ValWithDef(eWInCorrelFixe);
     mCurFenSpec = (mCurTypeWC != eWInCorrelFixe);

     if (mCurFenSpec)
     {
         ELISE_ASSERT(mMapEquiv==0,"Incompatibilite Fentre-Speciale/Equivalenc");
     }
     mCurNbIterFenSpec =  anEM.NbIterFenSpec().ValWithDef(1);

     mCurWSpecUseMasqGlob =   mCurFenSpec
                           && anEM.WSpecUseMasqGlob().ValWithDef(mNbPDV==2);

     mFactFenetreExp.x = FromSzW2FactExp(mSzWR.x,mCurNbIterFenSpec);
     mFactFenetreExp.y = FromSzW2FactExp(mSzWR.y,mCurNbIterFenSpec);

     mCurForceCorrelPontcuelle = false;
     mCurForceCorrelByRect = false;

     if (InterditCorrelRapide().Val())
     {
         mCurForceCorrelPontcuelle = true;
     }


     if (anEM.AggregCorr().Val()==eAggregInfoMut)
     {
         mCurForceCorrelPontcuelle =true;
     }
     if (ForceCorrelationByRect().Val())
     {
	 mCurForceCorrelByRect = true;
     }
     
     if (mCurFenSpec)
     {
         mSzWFixe = round_ni(mSzWR.x);
	 if (mCurTypeWC == eWInCorrelRectSpec)
	    mPtSzWMarge = round_up(mSzWR)+Pt2di(1,1); // Un peu lache
	 else
            mPtSzWMarge = round_ni(mSzWR*3.0);

 	 mCurForceCorrelByRect = true;
	 ELISE_ASSERT
	 (
	     !mCurEtape->EtapeMEC().SzWInt().IsInit(),
	     "En mode FenSpec pas de SzWInt"
	 );
	 ELISE_ASSERT
	 (
	     !mCurEtape->EtapeMEC().SurEchWCor().IsInit(),
	     "En mode FenSpec pas de SurEchWCor"
	 );
     }
     else
     {
         mSzWFixe = round_ni(mSzWR.x);
         mPtSzWMarge = round_ni(mSzWR);
         ELISE_ASSERT
	 (
	      mSzWFixe==mSzWR.x,
	      "Fenetre reelle en mode fenetre fixe"
	 );
         if (! mCorrelAdHoc)
         {
             ELISE_ASSERT
	     (
	          mSzWR.x==mSzWR.y,
	          "tx!=ty en mode fenetre fixe"
	     );
         }
     }
     mPtSzWFixe = Pt2di(mSzWFixe,round_ni(mSzWR.y));


     mModeIm1Maitre = IsModeIm1Maitre(mCurEtape->EtapeMEC().AggregCorr().Val());
     mCurSurEchWCor =  mCurEtape->EtapeMEC().SurEchWCor().ValWithDef(1);
     mCurSzWInt =  mCurEtape->EtapeMEC().SzWInt().ValWithDef(mSzWFixe*mCurSurEchWCor);

     if (mCurSurEchWCor != 1)
     {
        mCurForceCorrelPontcuelle = true;
     }

     if (mModeIm1Maitre)
     {
         ELISE_ASSERT
         (
              mGIm1IsInPax,
              "Aggreg with Im1 Master requires Geom Im1 master"
         );
     }


     mCMS_ModeEparse = false;
     if (mCorrelAdHoc)
     {
         // C'est pour court circuiter les algo cd ChCorrel, c'est indépendant de ce qui est fait dans le code specifique
         mCurForceCorrelPontcuelle = true;
         if (mCorrelAdHoc->Correl2DLeastSquare().IsInit())
         {
              ELISE_ASSERT(anEtape.AlgoRegul()==eAlgoLeastSQ,"Least Sq require eAlgoLeastSQ");
         }
         else if (mCorrelAdHoc->TypeCAH().ScoreLearnedMMVII().IsInit())
         {
             if (mCorrelAdHoc->TypeCAH().ScoreLearnedMMVII().Val().FileModeleCost()=="MVCNNCorrel2D")
                 {
                       ELISE_ASSERT(mDimPx==2,"Pax should be 2D for calling DeepSimNets in Mode Epip");
                 }
         }
         else
         {
             ELISE_ASSERT(mDimPx==1,"Multiple Px in GPU");
         }
         ELISE_ASSERT(mCurSurEchWCor==1,"Sur ech in GPU");

        mCMS = mCorrelAdHoc->CorrelMultiScale().PtrVal();
        mCMS_ModeEparse = (mCMS!=0) && (! mCMS->ModeDense().ValWithDef( mModeIm1Maitre));
/*
         ELISE_ASSERT(mCurEtape->EtapeMEC().AggregCorr().Val()==eAggregSymetrique,"Aggreg non sym in GPU");
         ELISE_ASSERT(mCurEtape->EtapeMEC().ModeInterpolation().Val()==eInterpolMPD,"Interp non MPD in GPU");
*/


     }
     else
     {
         mCMS = 0;
         mCC = 0;
     }


     mIsOptDiffer = anEtape.IsOptDiffer();
     mIsOptDequant = anEtape.IsOptDequant();
     mIsOptIdentite = anEtape.IsOptIdentite();
     mIsOptimCont = anEtape.IsOptimCont();

     mCurCarDZ =  GetCaracOfDZ(mCurEtape->DeZoomTer());
     mSom0 = ElSquare(1+2*mSzWFixe * mCurSurEchWCor);
     if (
             (! CalledByProcess().Val())
          ||(anEtape.Num() <LastEtapeMEC().Val())
        )
     {
        mCurMAI = mCurEtape->ModeleAnImported();
        if (mCurMAI)
        {
           mCurMAI->LoadMA();
           for 
           (
               tCsteIterPDV itP = PdvBegin();
	       itP != PdvEnd();
	       itP++
           )
           {
                (*itP)->Geom().CorrectModeleAnalytique(mCurMAI);
           }
        }
     }
     ELISE_ASSERT
     (
        ! (mCurForceCorrelByRect && mCurForceCorrelPontcuelle),
	"Incohe mCurForceCorrelByRect/mCurForceCorrelPontcuelle"
     );
}

const std::string & mm_getstrpid();


std::string cAppliMICMAC::PrefixGenerikRecalEtapeMicmMac(cEtapeMecComp & anEtape)
{
    int aNumEt = anEtape.Num();
    //mNameExe + std::string(" ")
    //  Modif MPD, reordonne pour mettre Arg d'etape a la fin
    //  Modif Greg: probleme de '"' pour condor
    //std::string aNameProcess = std::string("\"")+mNameXML+std::string("\"")
   std::string aNameProcess = mNameXML 
                               + std::string(" CalledByProcess=1 ")
                               + std::string(" ByProcess=0 ")
                               + std::string(" IdMasterProcess="+ mm_getstrpid() + " ");

    // MODIF MPD mise entre " des parametre pour etre completement reentrant
    for (int aKArg=0; aKArg<mNbArgAux ; aKArg++)
        aNameProcess =   aNameProcess
                       + std::string(" \"")
                       + std::string(mArgAux[aKArg]) 
                       +  std::string("\"");

    aNameProcess = aNameProcess
                   + std::string(" FirstEtapeMEC=") + ToString(aNumEt)
                   + std::string(" LastEtapeMEC=") + ToString(aNumEt+1);

    return aNameProcess;
/*
                   + std::string(" FirstBoiteMEC=") + ToString(mKBox)
                   + std::string(" NbBoitesMEC=1") ;
*/

}



void cAppliMICMAC::DoOneEtapeMEC(cEtapeMecComp & anEtape)
{
     if (
            (anEtape.Num() <FirstEtapeMEC().Val())
          ||(anEtape.Num() >=LastEtapeMEC().Val())
          ||(mNbBoitesToDo <=0)
        )
        return;
    std::list<std::string> aLStrProcess;
    if (mShowMes)
    {
        mCout << "-------- BEGIN STEP,  "
           << ", Num = " << anEtape.Num()
           << ", DeZoomTer = " << anEtape.DeZoomTer()
           << ", DeZoomIm = " << anEtape.DeZoomIm()
           << "\n";
    }


     mNbPtsWFixe = (1+2*mPtSzWFixe.x*mCurSurEchWCor)*(1+2*mPtSzWFixe.y*mCurSurEchWCor);
     int aDZIm = anEtape.DeZoomIm();
     
     mVecPtsW = std::vector<Pt2dr>(mNbPtsWFixe,Pt2dr(0,0));
     mTabPtsW = &mVecPtsW[0];

     *mGeomDFPx = mCurEtape->GeomTer();
     for 
     (
         tCsteIterPDV itP = PdvBegin();
	 itP != PdvEnd();
	 itP++
     )
     {
         (*itP)->Geom().SetDeZoomIm(aDZIm);
     }

     Box2dr aPropClip = ProportionClipMEC().ValWithDef
                     (Box2dr(Pt2dr(0,0),Pt2dr(1,1)));
     Pt2dr aSzDz = Pt2dr(mGeomDFPx->SzDz());
     Box2di aBoxClip 
            (
                   aPropClip._p0.mcbyc(aSzDz),
                   aPropClip._p1.mcbyc(aSzDz)
            );
      if ((!  ClipMecIsProp().Val()) && ProportionClipMEC().IsInit())
      {
         Box2dr  aB = ProportionClipMEC().Val();
         double aZ =ZoomClipMEC().Val();
         double aF = aZ / anEtape.DeZoomTer();

         aBoxClip = Box2di(aB._p0 * aF,aB._p1 * aF);
      }

/*
std::cout << "CCMMM = " << aBoxClip._p0 << " " << aBoxClip._p1 << "\n"; getchar();
*/

      const cEtapeMEC & anEM = anEtape.EtapeMEC();
      if (mPtrIV)
      {
          Pt2di aCTer = mPtrIV->CentreVisuTerrain();
          int aSzWV   = mPtrIV->SzWTerr().Val()/2;
          aBoxClip._p0.SetSup(aCTer - Pt2di(aSzWV,aSzWV));
          aBoxClip._p1.SetInf(aCTer + Pt2di(aSzWV,aSzWV));
      }
      mSzRec =  anEM.SzRecouvrtDalles().ValWithDef(SzRecouvrtDalles().Val());
      mSzDMin=  anEM.SzDalleMin().ValWithDef(SzDalleMin().Val());
      mSzDMax=  anEM.SzDalleMax().ValWithDef(SzDalleMax().Val());


      int aTXY = GetTXY();
      if (mPtrIV)
          aTXY = 100000;
      Pt2di aPRec(mSzRec,mSzRec);

      cDecoupageInterv2D aDecInterv
                         (
                              aBoxClip,
                              Pt2di(aTXY,aTXY),
                              Box2di(-aPRec,aPRec)
                         );
     
     for (mKBox=0 ; mKBox<aDecInterv.NbInterv() ; mKBox++)
     {
          if (
                (   (mKBox>=FirstBoiteMEC().Val())
                 || (anEtape.Num() >FirstEtapeMEC().Val())
                )
                && (mNbBoitesToDo >0)
             )
          {
               if ( mShowMes)
               {
                  mCout << "   -- BEGIN BLOC  "
                        << "  Bloc= " << mKBox+1 
                        << ", Out of " << aDecInterv.NbInterv()  
                        << aDecInterv.KthIntervOut(mKBox)._p0
                        << aDecInterv.KthIntervOut(mKBox)._p1
                        << "\n";
               }
               if (ByProcess().Val()==0)
               {
                  DoOneBloc
                  (
                      aDecInterv.KthIntervOut(mKBox),
                      aDecInterv.KthIntervIn(mKBox),
                      0,
                      aBoxClip
                  );
               }
               else
               {
                   // at first bloc run homography or epipolar warping for MVS deep learning pipeline
                   if ((mKBox==0) && (ByProcess().Val()!=0))
                     {
                       std::cout<<"MKBBBBBOOXXXXX  "<<mKBox<<"  aDecInterv.NbInterv() "<<aDecInterv.NbInterv()<<" PROCCC: "<<
                               ByProcess().Val()<<std::endl;
                         if (mCorrelAdHoc)
                           {
                             mZoomChanged=true;
                             // Homography or Epipolar warping for multi view Deep similarity matching
                             const cTypeCAH & aTC  = mCorrelAdHoc->TypeCAH();
                             if (aTC.MutiCorrelOrthoExt().IsInit())
                               {

                                 const cMutiCorrelOrthoExt aMCOE = aTC.MutiCorrelOrthoExt().Val();
                                 //aMCOE.UseEpip().Val()
                                 if (aMCOE.UseEpip().IsInit())
                                   {
                                       if (aMCOE.UseEpip().Val())
                                         {
                                           DoEstimWarpersPDVs();
                                         }
                                        else
                                         {
                                            DoEstimHomWarpers();
                                         }
                                   }
                               }
                           }
                     }
/*
                   int aNumEt = anEtape.Num();
                   //mNameExe + std::string(" ")
		   //  Modif MPD, reordonne pour mettre Arg d'etape a la fin
                   //  Modif Greg: probleme de '"' pour condor
                   //std::string aNameProcess = std::string("\"")+mNameXML+std::string("\"")
		   std::string aNameProcess = mNameXML 
                               + std::string(" CalledByProcess=1 ")
                               + std::string(" ByProcess=0 ");

                   for (int aKArg=0; aKArg<mNbArgAux ; aKArg++)
                       aNameProcess =   aNameProcess
                                      + std::string(" ")
                                      + std::string(mArgAux[aKArg]);

                   aNameProcess = aNameProcess
                               + std::string(" FirstEtapeMEC=") + ToString(aNumEt)
                               + std::string(" LastEtapeMEC=") + ToString(aNumEt+1)
*/
		   std::string aNameProcess = 
                                 PrefixGenerikRecalEtapeMicmMac(anEtape)
                               + std::string(" FirstBoiteMEC=") + ToString(mKBox)
                               + std::string(" NbBoitesMEC=1") ;

                   std::cout<<"PROCESS  =====>>>>  "<<aNameProcess<<std::endl;
                   aLStrProcess.push_back(aNameProcess);
               }
          }
     }
     if (ByProcess().Val()!=0)
        ExeProcessParallelisable(true,aLStrProcess);
	 

   if (anEM.DoImageBSurH().IsInit())
   {
      DoImagesBSurH(anEM.DoImageBSurH().Val());
   }
}


void cAppliMICMAC::SauvFileChantier(Fonc_Num aF,Tiff_Im aFile) const
{
      ELISE_COPY
      (
          rectangle(mBoxOut._p0,mBoxOut._p1),
          trans(aF,-mBoxIn._p0),
          aFile.out()
      );

}


int cAppliMICMAC::NbApproxVueActive()
{
  if (mNbApproxVueActive<0)
  {
      mNbApproxVueActive = 0;
      for (tCsteIterPDV itFI=PdvBegin(); itFI!=PdvEnd(); itFI++)
      {
          // bool Loaded = false;
          if (
                   (mCurEtape->SelectImage(*itFI))
               &&  (*itFI)->LoadImageMM(true,*mLTer,Pt2di(0,0), itFI==PdvBegin())
             )
          {
             mNbApproxVueActive++;
          }
       }
  }
  return mNbApproxVueActive;
}

const Box2di & cAppliMICMAC::BoxIn()  const {return mBoxIn;}
const Box2di & cAppliMICMAC::BoxOut() const {return mBoxOut;}

std::string  cAppliMICMAC::DirCube() const
{
   return  FullDirMEC() + "Cube" + ToString(mCurEtape->Num());
}
std::string cAppliMICMAC::NameFileCurCube(const std::string & aName) const
{
   Pt2di aP0 = mBoxOut._p0;
   return DirCube() + "/Data_" + ToString(aP0.x) + "_" + ToString(aP0.y)  + "_" +aName;
}
  



void cAppliMICMAC::DoOneBloc
     (
          const Box2di & aBoxOut,
          const Box2di & aBoxIn,
          int aNiv,
          const Box2di & aBoxGlob
     )
{

    std::cout << "DO ONE BLOC " << aBoxOut._p0 << " " << aBoxOut._p1 << " " << aBoxIn._p0  << " MATP " << mCurEtape->MATP() << "\n";
   //  mStatN =0;
   mStatGlob =0;
   mLTer = 0;
   mSurfOpt = 0;
   mNbApproxVueActive = -1;

#if CUDA_ENABLED
   if (mCorrelAdHoc && mCorrelAdHoc->GPU_CorrelBasik().IsInit())
       IMmGg.SetTexturesAreLoaded(false);
#endif


   mBoxIn = aBoxIn;
   mBoxOut = aBoxOut;
   //std::cout<<" $$$$$$$$ DIMENSION PARALLAX   "<<mDimPx<<std::endl;
   mLTer = new cLoadTer(mDimPx,aBoxIn.sz(),*mCurEtape);

   double aNbCel = mCurEtape->LoadNappesAndSetGeom(*mLTer,aBoxIn);


   int aSzCel = mCurEtape->MultiplierNbSizeCellule();
   //std::cout << "SzzEcccellLLLLLLLLLLLLLLLLLLLL " <<  aSzCel*aNbCel <<"\n";

   int aLMin = ElMin(aBoxOut._p1.x-aBoxOut._p0.x,aBoxOut._p1.y-aBoxOut._p0.y);

/*
   for (int aK=0; aK<=aNiv; aK++)  std::cout <<"---|";
   std::cout << "OUT " << aBoxOut._p0 << aBoxOut._p1 << " IN " << aBoxIn._p0 << aBoxIn._p1
             << "  NbCel " << aNbCel << "\n"; 
*/


   if (
                 (!mCurEtape->MATP() )
              && (aNbCel*aSzCel) >NbCelluleMax().Val()
          // && (aBoxOut._p1.x-aBoxOut._p0.x) > (3*mSzRec+5)  // Evite recursion trop profonde, voir infinie
          // && (aBoxOut._p1.y-aBoxOut._p0.y) > (3*mSzRec+5)
      )
   {
       cLoadTer * aLTInit = mLTer;
       int aSzRecInit = mSzRec;

       Box2di aSplitOut[4];
       aBoxOut.QSplit(aSplitOut);

       mSzRec = ElMax(0,ElMin(mSzRec,aLMin/4));

       for (int aK=0 ; aK<4 ; aK++)
       {
/*
           std::cout  << "SPLIT " << aBoxOut._p0 << aBoxOut._p1  
                      << "  ; O = " << aBoxIn._p0 << aBoxIn._p1  
                      <<  "  => OUT " << aSplitOut[aK]._p0 << aSplitOut[aK]._p1 
                      <<  " REC " << mSzRec
                      <<  "  => IN " << aSplitIn[aK]._p0 << aSplitIn[aK]._p1 << "\n";
*/
           // getchar();
           DoOneBloc
           (
                aSplitOut[aK],
                Inf(aBoxGlob,aSplitOut[aK].dilate(mSzRec)),
                aNiv+1,
                aBoxGlob
          );
       }

       mSzRec = aSzRecInit;
       mLTer = aLTInit;
       return;
   }



   *mGeomDFPx = mCurEtape->GeomTer();  // actualisation en fonction du LoadNappes

   cEquiv1D anEqX;
   anEqX.InitByClipAndTr(mCurCarDZ->EqX(),mBoxIn._p0.x,0,mBoxIn.sz().x);
   cEquiv1D anEqY;
   anEqY.InitByClipAndTr(mCurCarDZ->EqY(),mBoxIn._p0.y,0,mBoxIn.sz().y);

  // Les structures de ChoixCalcCorrByQT sont assez consommatrices de
  // memoires, le surcout a spliter en different rectangle etant assez
  // faible, on a tout interet a le faire
   
/*
  int aRatio = 20;
  // Qd on utilise mCurFenSpec, bcp + sensible a la decoupe
  if (mCurFenSpec) 
      aRatio = 60;
*/
  int aSzMaxChCorr= round_ni(100 * SzMinDecomposCalc().Val());
  if  (    GetCurCaracOfDZ()->HasMasqPtsInt() 
        || (mCurSurEchWCor != 1)
        || mIsOptimCont 
        )
      aSzMaxChCorr = 10000;
  cDecoupageInterv2D aDecInterv =
	                cDecoupageInterv2D::SimpleDec(aBoxIn.sz(),aSzMaxChCorr,0);
  Pt2di aSzMaxDec = aDecInterv.SzMaxOut();


  mCurEtUseWAdapt =  mCurEtape->UseWAdapt();
  if (mCurEtUseWAdapt)
  {
       mImSzWCor.Resize(aBoxIn.sz());
       mTImSzWCor = TIm2D<U_INT1,INT>(mImSzWCor);
       ELISE_ASSERT(mCurEtape->DeZoomTer()==mCurEtape->DeZoomIm(),"ZoomTer!=ZoomIm with UseWAdapt");
       std::string aNameSzW = NameFileSzW(mCurEtape->DeZoomTer());
       Tiff_Im aTF(aNameSzW.c_str());
       ELISE_COPY
       (
            mImSzWCor.all_pts(),
            trans(aTF.in_proj(),aBoxIn._p0),
            mImSzWCor.out()
       );
  }
   
   // Chargement des images

   mPDVBoxGlobAct.clear();
   bool isFirstImLoaded = true;
   for (tCsteIterPDV itFI=PdvBegin(); itFI!=PdvEnd(); itFI++)
   {
       // bool Loaded = false;
       if (
                (mCurEtape->SelectImage(*itFI))
            &&  (*itFI)->LoadImageMM(false,*mLTer,aSzMaxDec,isFirstImLoaded)
          )
       {
         // Loaded = true;
          isFirstImLoaded = false;
          if ((mKBox == 0) && ShowLoadedImage().Val())
          {
               mCout << "== " << (*itFI)->Name() << "\n";
          }
          mPDVBoxGlobAct.push_back(*itFI);

          if (mCurEtape->UsePC() || (*itFI)->Geom().UseMasqTerAnamSA())
          {
              (*itFI)->LoadedIm().MakePC
                       (
                           **itFI,
                           mCurEtape,
                           mCurEtape->PredPC(),
                           aBoxIn,
                           mCurEtape->UsePC(),
                           (*itFI)->Geom().UseMasqTerAnamSA()
                       );
          }
       }
   }
   mNbImChCalc = (int) mPDVBoxGlobAct.size();

   if (mShowMes)
   {
	   
      mCout << "      " << mNbImChCalc << " Images Loaded\n";
   }

   // Initialisation de mTabV1 et aVPtInEc
   //  mTabV1 : zone memoire pour stocker les valeurs de Im1 (acceleration "speciale")
   // aVPtInEc : au cas ou fenetre de correl != fenetre de normalisation
   std::vector<int> aVPtInEc;
   std::vector<int> aVPtIndOK;
   std::vector<Pt2di> aVPtOK;


   mVecV1.clear();
   int aNbTot = mSzWFixe *mCurSurEchWCor;
   for (int anY=-aNbTot ; anY<=aNbTot ;anY++)
   {
       for (int anX=-aNbTot ; anX<=aNbTot ;anX++)
       {
            aVPtInEc.push_back(dist8(Pt2di(anX,anY))<=mCurSzWInt);
            if (aVPtInEc.back())
            {
               aVPtIndOK.push_back((int)(aVPtInEc.size()-1));
               aVPtOK.push_back(Pt2di(anX,anY));
            }
            mVecV1.push_back(0);
       }
   }
   mTabV1 = &(mVecV1[0]);


   if (mPtrIV)
   {
      Visualisation(aBoxIn);
      exit(0);
   }
   // --------------------------------------
   double aTimeCorrel = 0;
   ElTimer aChrono;

   if (mIsOptimCont)
   {
      OptimisationContinue();
   }
   else
   {

       // mStatN = new cStatOneClassEquiv(*this,(int) mPDVBoxGlobAct.size(),aVPtInEc);

       mStatGlob = new cStatGlob(*this,aVPtInEc,aVPtIndOK,aVPtOK);

       for (int aKPdv=0;aKPdv<int( mPDVBoxGlobAct.size()); aKPdv++)
       {
           mStatGlob->AddVue(*(mPDVBoxGlobAct[aKPdv]));
       }

       mStatGlob->InitSOCE();
   
       mDefCost =  mStatGlob->CorrelToCout(mDefCorr);
       mSurfOpt = cSurfaceOptimiseur::Alloc(*this,*mLTer,anEqX,anEqY);

       InitCostCalcCorrel();

       if (mCorrelAdHoc)
       {
          GlobDoCorrelAdHoc(aBoxOut,aBoxIn);
       }
       else
       {
          for (int aKBox=0 ; aKBox<aDecInterv.NbInterv() ; aKBox++)
          {
              ChoixCalcCorrByQT(aDecInterv.KthIntervOut(aKBox));
          }  
       }

   
        aTimeCorrel = aChrono.ValAndInit();
        if (mShowMes)
        {
			if((mCorrelAdHoc != 0 && mCorrelAdHoc->TypeCAH().GPU_CorrelBasik().IsInit())||
			   (mCMS!=0 && mCMS->UseGpGpu().Val()))

				mCout << "       Cuda Correlation Finished, Begin Cuda Optimisation\n";
            else
                mCout << "       Correl Calc, Begin Opt\n";
        }

        mSurfOpt->SolveOpt();

#if CUDA_ENABLED
        if (mCorrelAdHoc && mCorrelAdHoc->GPU_CorrelBasik().IsInit())
        {
            IMmGg.Data().DeallocDeviceData();
            IMmGg.Data().DeallocHostData();
        }
#endif
    }



// for (int aK=0; aK<=aNiv; aK++)  std::cout <<"---|";
// std::cout << "AVANT SN " << "\n";

    mCurEtape->SauvNappes(*mLTer,aBoxOut,aBoxIn);

// for (int aK=0; aK<=aNiv; aK++)  std::cout <<"---|";
// std::cout << "APRES SN  " << "\n";


    double aTimeOptim = aChrono.ValAndInit();

    mTimeTotCorrel += aTimeCorrel;
    mTimeTotOptim += aTimeOptim;

    if (mShowMes)
    {
       double aNbPtsTot= mNbPointsByRect2 + mNbPointsByRectN + mNbPointsIsole + mNbPointByRectGen;
       mCout 
             << "       TCor " << mTimeTotCorrel 
             << " CTimeC " << (mTimeTotCorrel * 1e6)/ aNbPtsTot
             <<       " TOpt " << mTimeTotOptim
             << " Pts , R2 " <<  (mNbPointsByRect2/aNbPtsTot)*100.0
             << ", RN " <<  (mNbPointsByRectN/aNbPtsTot)*100.0
             << " Pts , R-GEN " <<  (mNbPointByRectGen/aNbPtsTot)*100.0
             << ", Isol " << (mNbPointsIsole/aNbPtsTot)*100.0
             << "  PT  "   <<  aNbPtsTot

             <<"\n";
    }
  
	

    //  delete mStatN;
    delete mStatGlob;
    delete mLTer;
    delete mSurfOpt;
    mSurfOpt = 0;
    mNbBoitesToDo--;
}

template <class aTVect> double EcartType(const aTVect & aV,const  std::vector<double> * aVPds)
{
   typename aTVect::value_type aSomT;
   typename aTVect::value_type aSomT2;
   double aSP = 0;
   double aSP2 = 0;

   int aK=0;
   for (typename aTVect::const_iterator itV=aV.begin() ; itV!=aV.end() ; itV++)
   {
          double aPds = aVPds ? (*aVPds)[aK] : 1.0 ;
          typename aTVect::value_type aT = *itV;
          aSomT = aSomT + aT*aPds;
          aSomT2 = aSomT2 + Pcoord2(aT) *aPds;
          aSP += aPds;
          aSP2 += ElSquare(aPds);
          aK++;
   }
   aSomT = aSomT / aSP;
   aSomT2 = aSomT2 / aSP;
   aSomT2 = aSomT2 - Pcoord2(aSomT);

      // Ce debiaisement est necessaire, par exemple si tous les poids sauf 1 sont
      // presque nuls
   double aDebias = 1 - aSP2/ElSquare(aSP);
   ELISE_ASSERT(aDebias>0,"Singularity in cManipPt3TerInc::CalcPTerInterFaisceauCams ");
   aSomT2 =  aSomT2/ aDebias;

   // double anEc2 = aSomT2.x+aSomT2.y+aSomT2.z;
   double anEc2 = SomCoord(aSomT2);
   if (anEc2 <= -1e-7)
   {
       std::cout << "EC2 =" << anEc2 << "\n";
       ELISE_ASSERT(false,"Singularity in BSurH_SetOfDir ");
   }
   anEc2 = sqrt(ElMax(0.0,anEc2));
    return anEc2;
}

double BSurH_SetOfDir(const std::vector<Pt3dr> & aV,const  std::vector<double> * aVPds)
{
      // Adaptation purement heuristique
   return     1.35 * EcartType(aV,aVPds);
}

        // virtual ElSeg3D FaisceauPersp(const Pt2dr & )  const;
void cAppliMICMAC::DoImagesBSurH(const cDoImageBSurH& aParBsH)
{
     ElTimer aChrono;
     double aDownScale = aParBsH.ScaleNuage();

     cXML_ParamNuage3DMaille aXmlN =  mCurEtape->DoRemplitXML_MTD_Nuage();
     cElNuage3DMaille *  aFullNuage = cElNuage3DMaille::FromParam(PDV1()->Name(),aXmlN,FullDirMEC());

     cElNuage3DMaille *  aNuage = aFullNuage->ReScaleAndClip(aDownScale);


     Pt2di aSz = aNuage->SzUnique();
     Im2D_U_INT1 aRes(aSz.x,aSz.y);
     TIm2D<U_INT1,INT> aTRes(aRes);

     Im2D_Bits<1>  aNewMasq(aSz.x,aSz.y,0);
     TIm2DBits<1>  aTNMasq(aNewMasq);
     double aSeuilBsH = aParBsH.SeuilMasqExport().ValWithDef(0);

     Pt2di aPIndex;
     // double aPax[theDimPxMax] ={0,0};

     for (aPIndex.x=0 ; aPIndex.x<aSz.x ; aPIndex.x++)
     {
         for (aPIndex.y=0 ; aPIndex.y<aSz.y ; aPIndex.y++)
         {
             int aVal = 255;
             if (aNuage->IndexHasContenu(aPIndex))
             {
                Pt3dr aPEucl = aNuage->PtOfIndex(aPIndex);
                // Pt3dr aQE = aNuage->Euclid2ProfAndIndex(aPEucl);
                // Pt2dr aPE2(aPEucl.x,aPEucl.y);
                // aPax[0] = aPEucl.z;
                std::vector<Pt3dr> aVN;
                for (int aKPdv=0 ; aKPdv<int(mPrisesDeVue.size()) ; aKPdv++)
                {
                    cPriseDeVue & aPDV = *(mPrisesDeVue[aKPdv]);
                    cGeomImage & aGeom = aPDV.Geom();
                    CamStenope *  aCS = aGeom.GetOriNN() ;

                    Pt2dr  aPIm = aCS->R3toF2(aPEucl);


                    //  Pt2dr  aPIm = aGeom.Objet2ImageInit_Euclid(aPE2,aPax);
                    Pt2di  aSz = aPDV.SzIm() ;
                    if ((aPIm.x>0) && (aPIm.y>0) && (aPIm.x<aSz.x) && (aPIm.y<aSz.y))
                    {
                          Pt3dr aN = vunit(aPEucl-aCS->PseudoOpticalCenter());
                          aVN.push_back(aN);
                    }
                }
                if (aVN.size() > 1)
                {
                   double aRVal = BSurH_SetOfDir(aVN,0);
                   aTNMasq.oset(aPIndex,aRVal > aSeuilBsH);
                   aRVal = (aRVal+aParBsH.Offset().Val()) * aParBsH.Dyn().Val();
                   aVal = ElMax(0,ElMin(253,round_ni(aRVal)));
                }
                else
                {
                   aVal = 254;
                }
             }
             aTRes.oset(aPIndex,aVal);
         }
     }
     Tiff_Im::Create8BFromFonc
     (
         FullDirMEC()+aParBsH.Name(),
         aRes.sz(),
         aRes.in()
     );
     Im2D_Bits<1>  anOldMasq = aNuage->ImDef();
     ELISE_COPY
     (
          anOldMasq.all_pts(),
          anOldMasq.in() && close_32(aNewMasq.in(0),6),
          anOldMasq.out()
     );
     aNuage->Save(aParBsH.NameNuage());
     std::cout << "Time-BSH " << aChrono.uval();
}



void cAppliMICMAC::InitBlocInterne( const Box2di & aBox)
{
   mBoxInterne = aBox;
   int aPxMin[theDimPxMax],aPxMax[theDimPxMax];   
   mLTer->CalculBornesPax(aBox,aPxMin,aPxMax);

   // On calcule les images qui "voient" au - une petite partie du terrain
   mPDVBoxInterneAct.clear();
   mNb_PDVBoxInterne=0;
   for (tCsteIterPDV itFI=mPDVBoxGlobAct.begin(); itFI!=mPDVBoxGlobAct.end(); itFI++)
   {
       if (  (*itFI)->Geom().BoxTerHasIntersection
             (
                 *mGeomDFPx,
                 aPxMin,aPxMax,
                 Box2dr(aBox._p0,aBox._p1)
             )
          )
       {
          mPDVBoxInterneAct.push_back(*itFI);
          mNb_PDVBoxInterne++;
       }
   }

}

void cAppliMICMAC::DoOneBlocInterne
     (
            const cPxSelector & aSelector,
            const Box2di & aBox
     )
{

   InitBlocInterne(aBox);

/*
   mBoxInterne = aBox;
   int aPxMin[theDimPxMax],aPxMax[theDimPxMax];   
   mLTer->CalculBornesPax(aBox,aPxMin,aPxMax);

   // On calcule les images qui "voient" au - une petite partie du terrain
   mPDVBoxInterneAct.clear();
   mNb_PDVBoxInterne=0;
   for (tCsteIterPDV itFI=mPDVBoxGlobAct.begin(); itFI!=mPDVBoxGlobAct.end(); itFI++)
   {
       if (  (*itFI)->Geom().BoxTerHasIntersection
             (
                 *mGeomDFPx,
                 aPxMin,aPxMax,
                 Box2dr(aBox._p0,aBox._p1)
             )
          )
       {
          mPDVBoxInterneAct.push_back(*itFI);
          mNb_PDVBoxInterne++;
       }
   }
*/


  // ii  const cEtapeMEC & anEM = mCurEtape->EtapeMEC();
   mCSAccelIm1 =    (mModeGeomMEC==eGeomMECIm1) 
                 &&  (
		          ( mCurEtape->EtapeMEC().AggregCorr().Val()==eAggregSymetrique)
		      ||  IsModeIm1Maitre(mCurEtape->EtapeMEC().AggregCorr().Val())
	             )
                 &&  (! mCurEtape->UsePC())
                 &&  (! mCurForceCorrelByRect)
                 && (mNb_PDVBoxInterne==2) 
                 && (mNbPDV==2) 
                 && (mPDV1 == *(mPDVBoxInterneAct.begin()))
                 && (mSzWFixe==mCurSzWInt)
                 && (mMapEquiv==0)
                 && (! InterditAccelerationCorrSpec().Val());


   if (mCorrelAdHoc)
   {
       DoCorrelAdHoc(aBox);
       return;
   }
   
   for (mCurPterI.x = aBox._p0.x ; mCurPterI.x <aBox._p1.x ; mCurPterI.x++)
   {
       for (mCurPterI.y = aBox._p0.y ; mCurPterI.y <aBox._p1.y ; mCurPterI.y++)
       {
           int aPxMax[theDimPxMax] ={1,1};
           int aPxMin[theDimPxMax] ={0,0};
           mLTer->GetBornesPax(mCurPterI,aPxMin,aPxMax);
           if (mLTer->IsInMasq(mCurPterI))
           {
               mCurBoxW._p0 = mCurPterI-mPtSzWFixe;
               mCurBoxW._p1 = mCurPterI+mPtSzWFixe;
               // Pour traiter de maniere generique 1 ou 2 dimension
               // de paralaxe, on va toujours faire comme si il y en 
               // avait deux, on s'arrange pour que la dimension inutile ne
               // rajoute qu'un imbrication dans une boucle vide


   
               // IsFirst evite de faire des PrepareAccelIm1 quand ce ne sera utile
               // pour aucune paralaxe, ce qui est le cas avec des tres grande fenetre
               // ou tout a deja ete fait par algo rapide
               bool isFirst = true;
               for (mCurPxI[1] = aPxMin[1] ; mCurPxI[1]<aPxMax[1] ; mCurPxI[1]++)
               {
                   for (mCurPxI[0]=aPxMin[0] ; mCurPxI[0]<aPxMax[0] ; mCurPxI[0]++)
                   {
                       if (aSelector.SelectPx(mCurPxI))
                       {
		          ELISE_ASSERT
			  (
			      ! mCurForceCorrelByRect,
			      "MICMAC:Incoherence interne / mCurForceCorrelByRect"
			  );
                          if (isFirst && mCSAccelIm1)
                          {
                              PrepareAccelIm1(aPxMin);
                          } 
                          REAL aCost =  mCSAccelIm1 ? 
                                        CalculAccelereScore() :
                                        CalculScore();
                          mSurfOpt->SetCout(mCurPterI,mCurPxI,aCost);
                          isFirst = false;
                       }
                   }
               }
           }
           else
           {
               double aDefCost =  mStatGlob->CorrelToCout(mDefCorr);
               for (mCurPxI[1] = aPxMin[1] ; mCurPxI[1]<aPxMax[1] ; mCurPxI[1]++)
               {
                   for (mCurPxI[0]=aPxMin[0] ; mCurPxI[0]<aPxMax[0] ; mCurPxI[0]++)
                   {
                       if (aSelector.SelectPx(mCurPxI))
                       {
                          mSurfOpt->SetCout(mCurPterI,mCurPxI,aDefCost);
                       }
                   }
               }
           }
       }
   }
// std::cout << "END " << aChrono.uval() << "\n";
}


void cAppliMICMAC::PrepareAccelIm1(int * aPx)
{
     mLineProj.InitLP
     (
        mCurBoxW,
        aPx,
        mPDV1->Geom(),
        *mGeomDFPx,
        *mCurEtape,
        *mLTer,
        Pt2di(0,0),
        mCurSurEchWCor
     );

     mOkStatIm1 = mPDV1->LoadedIm().StatIm1(mLineProj,mSomI1,mSomI11,mTabV1);

     mSomI1 /= mSom0;
     mSomI11 /= mSom0;
     mSomI11 -= ElSquare(mSomI1);
}

REAL cAppliMICMAC::CalculAccelereScore()
{
    mNbPointsIsole++;
    if (! mOkStatIm1)
       return  mStatGlob->CorrelToCout(mDefCorr);
    mLineProj.InitLP
    (
        mCurBoxW,
        mCurPxI,
        mPDV2->Geom(),
        *mGeomDFPx,
        *mCurEtape,
        *mLTer,
        Pt2di(0,0),
        mCurSurEchWCor
    );
    bool aOkStatIm2 = mPDV2->LoadedIm().StatIm2(mLineProj,mTabV1,mSomI2,mSomI22,mSomI12);

    if (! aOkStatIm2)
       return  mStatGlob->CorrelToCout(mDefCorr);

     mSomI2 /= mSom0;
     mSomI22 /= mSom0;
     mSomI22 -= ElSquare(mSomI2);

     mSomI12 /= mSom0;
     mSomI12 -= mSomI1 * mSomI2;
     double aCor = mSomI12 /sqrt(ElMax(mEpsCorr,mSomI11*mSomI22));


    return mStatGlob->CorrelToCout(aCor);
}


REAL cAppliMICMAC::CalculScore()
{
   mNbPointsIsole++;
   mStatGlob->Clear();
   mStatGlob->SetSomsMade(false);
   for 
   (
       tCsteIterPDV itFI=mPDVBoxInterneAct.begin(); 
       itFI!=mPDVBoxInterneAct.end(); 
       itFI++
   )
   {
       const cLoadedImage & aLIm = (*itFI)->LoadedIm();
       if (aLIm.IsVisible(mCurPterI))
       {
           mLineProj.InitLP
           (
                mCurBoxW,
                mCurPxI,
                (*itFI)->Geom(),
                *mGeomDFPx,
                *mCurEtape,
                *mLTer,
                Pt2di(0,0),
                mCurSurEchWCor
           );
           bool IsOk = true;
           Pt2dr * mTopPts = mTabPtsW;
           while ( mLineProj.Continuer() && IsOk)
           {
                 Pt2dr aP = mLineProj.PCurIm();
                 if(aLIm.IsOk(aP))
                 {
                     *(mTopPts++) = aP;
                 }
                 else
                 {
                    IsOk =false;
                 }
                 mLineProj.NexStep();
           }
           if (IsOk)
           {
             cStat1Distrib *aDist = mStatGlob->NextDistrib(**itFI);
             aLIm.GetVals(mTabPtsW,aDist->Vals(),mNbPtsWFixe);
           }
       }
   }
   return mStatGlob->Cout();
}


void   cAppliMICMAC::CalcCorrelByRect(Box2di aBox,int * aPx)
{

  //std::cout<<"correl by rect)))))))))))))))))  "<<std::endl;
   mLTer->MakeImTerOfPx(aBox,aPx);
   mStatGlob->SetSomsMade(false);
   mPDVBoxInterneAct.clear();
   for 
   (
       tCsteIterPDV itFI=mPDVBoxGlobAct.begin(); 
       itFI!=mPDVBoxGlobAct.end(); 
       itFI++
   )
   {
       if (  (*itFI)->Geom().BoxTerHasIntersection
             (
                 *mGeomDFPx,
                 aPx,aPx,
                 Box2dr(aBox._p0,aBox._p1)
             )
          )
       {
             mPDVBoxInterneAct.push_back(*itFI);
       }
   }

   // On le fait en deux fois, car LoadImInGeomTerr depend
   // de la taille de mPDVBoxInterneAct
   bool IsFirstLIIGT = true;
   for 
   (
       tCsteIterPDV itFI=mPDVBoxInterneAct.begin(); 
       itFI!=mPDVBoxInterneAct.end(); 
       itFI++
   )
   {
        (*itFI)->LoadedIm().LoadImInGeomTerr(aBox,aPx,IsFirstLIIGT);
	IsFirstLIIGT = false;
   }
   for 
   (
       tCsteIterPDV itFI=mPDVBoxInterneAct.begin(); 
       itFI!=mPDVBoxInterneAct.end(); 
       itFI++
   )
   {
        (*itFI)->LoadedIm().PostLoadImInGeomTerr(aBox);
   }


   if (UseAlgoSpecifCorrelRect())
   {
       INT aNbIm = (int) mPDVBoxInterneAct.size();
       if (aNbIm<2)
       {
           TIm2D<REAL8,REAL8> aTIC(mLTer->ImCorrel());
           Pt2di aP;

           double aDef = DefCorrelation().Val();
           for (aP.y =aBox._p0.y; aP.y<aBox._p1.y ; aP.y++)
           {
               for (aP.x =aBox._p0.x; aP.x<aBox._p1.x ; aP.x++)
               {
                  if (mLTer->IsInMasqOfPx(aP))
                      aTIC.oset(aP,aDef);
               }
          }
       }
       else if (mCurFenSpec)
       {
// std::cout << "Enter " << aBox._p0 << aBox._p1 << "\n";
             mPDVBoxInterneAct[0]->LoadedIm().CalcFenSpec
               (
                    aBox,
                    mPDVBoxInterneAct
               );
             mNbPointsByRect2 += aBox.surf();
// std::cout << "----END \n";
          // Correl Spec Exp
       }
       else if (aNbIm == 2)
       {
          mPDVBoxInterneAct[0]->LoadedIm().CalcCorrelRapide
               (
                    aBox,
                    mPDVBoxInterneAct[1]->LoadedIm()
               );
           mNbPointsByRect2 += aBox.surf();
       }
       else if (aNbIm <= theNbImageMaxAlgoRapide)
       {
           mPDVBoxInterneAct[0]->LoadedIm().NCalcCorrelRapide
                 (
                       aBox,
                       mPDVBoxInterneAct
                 );
           mNbPointsByRectN += aBox.surf();
       }
       else
       {
          ELISE_ASSERT(false,"CalcCorrelByRect");
       }
       TIm2D<REAL8,REAL8> aTIC(mLTer->ImCorrel());
       Pt2di aP;


       for (aP.y =aBox._p0.y; aP.y<aBox._p1.y ; aP.y++)
       {
           for (aP.x =aBox._p0.x; aP.x<aBox._p1.x ; aP.x++)
           {
               if (mLTer->IsInMasqOfPx(aP))
               {
                  double aCost = mStatGlob->CorrelToCout(aTIC.get(aP));
                  mSurfOpt->SetCout(aP,aPx,aCost);
               }
           }
       }
   }
   else
   {
      Pt2di aP;
      mStatGlob->SetSomsMade(true);
      for (aP.y =aBox._p0.y; aP.y<aBox._p1.y ; aP.y++)
      {
          for (aP.x =aBox._p0.x; aP.x<aBox._p1.x ; aP.x++)
          {
              // if (mLTer->OkPx(aP,aPx))

              if (mLTer->IsInMasqOfPx(aP))
              {
                 mNbPointByRectGen++;
                 mStatGlob->Clear();
                 for 
                 (
                     tCsteIterPDV itFI=mPDVBoxInterneAct.begin(); 
                     itFI!=mPDVBoxInterneAct.end(); 
                     itFI++
                 )
                 {
                        (*itFI)->LoadedIm().AddToStat(*mStatGlob,aP);
                 }
                 REAL aCost = mStatGlob->Cout();

                 mSurfOpt->SetCout(aP,aPx,aCost);
              }
          }
      }
      mStatGlob->SetSomsMade(false);
   }
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
