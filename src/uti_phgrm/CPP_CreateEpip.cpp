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



cCpleEpip * StdCpleEpip
          (
             std::string  aDir,
             std::string  aNameOri,
             std::string  aNameIm1,
             std::string  aNameIm2
          )
{
    if (aNameIm1 > aNameIm2) ElSwap(aNameIm1,aNameIm2);
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    std::string aNameCam1 =  anICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aNameOri,aNameIm1,true);
    std::string aNameCam2 =  anICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aNameOri,aNameIm2,true);

    CamStenope * aCam1 = CamStenope::StdCamFromFile(true,aNameCam1,anICNM);
    CamStenope * aCam2 = CamStenope::StdCamFromFile(true,aNameCam2,anICNM);
    return new cCpleEpip (aDir,1,*aCam1,aNameIm1,*aCam2,aNameIm2);

}

class cApply_CreateEpip_main
{
   public :
      cApply_CreateEpip_main(int argc,char ** argv);

      cInterfChantierNameManipulateur * mICNM;
      void  IntervZAddIm(cBasicGeomCap3D *);


      int    mDegre;
      int    mNbZ ;
      int    mNbXY ;
      int    mNbZRand ;
      // int    mNbZCheck ;   // Ceux la sont aussi random
      double mLengthMin ;
      double mStepReech ;
      bool   mForceGen;
      int    mNumKer;
      bool   mDebug;
      std::string mPostMasq;
      std::string mPostIm;
      bool mExpTxt;

      cBasicGeomCap3D *  mGenI1;
      cBasicGeomCap3D *  mGenI2;
      std::string        mName1;
      std::string        mName2;
      std::string        mOri;
      bool               mWithOri;
      int                mNbBloc;
      int                mDegSupInv;
      double             mEpsCheckInv;
      Pt2dr              mDir1;
      Pt2dr              mDir2;
      Pt2di              mNbCalcAutoDir;
      bool               mMakeAppuis;
      bool               mIntervZIsDef;
      double             mZMin;
      double             mZMax;
      double             mIProf;
      Pt2dr              mIntZ;             
      std::vector<double> mParamEICE;

      void DoEpipGen(bool DoIm);

      Pt2dr DirEpipIm2(cBasicGeomCap3D * aG1,cBasicGeomCap3D * aG2,ElPackHomologue & aPack,bool ForCheck,bool AddToP1, std::list<Appar23> &   aL23);

      void Ressample(cBasicGeomCap3D * aG1,EpipolaireCoordinate & E1,double aStep);

      void MakeAppuis(bool Is1,const std::list<Appar23>&aLT1,EpipolaireCoordinate & anEpi,Pt2dr aP0Epi,Pt2dr aP1Epi);

      void   ExportImCurveEpip(EpipolaireCoordinate & e1,const Box2dr& aBoxIn1,const Box2dr&,const std::string & aName,EpipolaireCoordinate & e2,const Box2dr& aBoxIn2,double aX2mX1);
      double PhaseValue(const double & aV,const double & aSzY) const;

};

void  cApply_CreateEpip_main::IntervZAddIm(cBasicGeomCap3D * aGeom)
{
    if (aGeom->AltisSolMinMaxIsDef())
    {
        mIntervZIsDef = true;
        Pt2dr aPZ =aGeom->GetAltiSolMinMax();
        mZMin= ElMin(mZMin,aPZ.x);
        mZMax= ElMax(mZMax,aPZ.y);
    }
}

// Compute direction of epip in G2 and fill Pack, AddToP1 indicate which way it must be added
Pt2dr  cApply_CreateEpip_main::DirEpipIm2
       (
            cBasicGeomCap3D *      aG1,
            cBasicGeomCap3D *      aG2,
            ElPackHomologue &      aPack,
            bool                   ForCheck,
            bool                   AddToP1,
            std::list<Appar23> &   aL23
       )
{
   
    Pt2dr aSz =  Pt2dr(aG1->SzBasicCapt3D());
 

    Pt2dr aSomTens2(0,0);
    // On le met tres petit, ce qui a priori n'est pas genant et evite 
    // d'avoir des point hors zone
    //  GetVeryRoughInterProf est une proportion

    if (!EAMIsInit(&mIProf))
       mIProf = aG1->GetVeryRoughInterProf() / 100.0;

/*
if (MPD_MM())
{
    std::cout << "aIProfaIProf " << aIProf << "\n";
    aIProf = 1/ 10000.0;
}
*/


    // aEps avoid points to be too close to limits
    double aEps = 5e-4;

    // Comput size of grid that will give mNbbXY ^2 points
    double aLenghtSquare = ElMin(mLengthMin,sqrt((aSz.x*aSz.y) / (mNbXY*mNbXY)));


    // Assure it give a sufficient reduduncy
    int aNbX = ElMax(1+3*mDegre,round_up(aSz.x /aLenghtSquare));
    int aNbY = ElMax(1+3*mDegre,round_up(aSz.y /aLenghtSquare));


     

    for (int aKX=0 ; aKX<= aNbX ; aKX++)
    {
        // Barrycentrik weighting, 
        double aPdsX = ForCheck ? NRrandom3() :  (aKX /double(aNbX));
        aPdsX = ElMax(aEps,ElMin(1-aEps,aPdsX));
        for (int aKY=0 ; aKY<= aNbY ; aKY++)
        {
            // Barrycentrik weighting, 
            double aPdsY = ForCheck ? NRrandom3() : (aKY/double(aNbY));
            aPdsY = ElMax(aEps,ElMin(1-aEps,aPdsY));
            // Point in image 1 on regular gris
            Pt2dr aPIm1 = aSz.mcbyc(Pt2dr(aPdsX,aPdsY));
            if (aG1->CaptHasData(aPIm1))
            {
                Pt3dr aPT1;
                Pt3dr aC1;
                // Compute bundle with origin on pseudo center
                aG1->GetCenterAndPTerOnBundle(aC1,aPT1,aPIm1);

                // std::cout << "IPROF " << aIProf * euclid(aPT1-aC1)  << " " << aPT1  << "\n";

                std::vector<Pt2dr> aVPIm2;
                double aNbZSup = mNbZ + mNbZRand; 
                for (int aKZ = -mNbZ ; aKZ <= aNbZSup  ; aKZ++)
                {
                     bool isZRand =  (aKZ > mNbZ)  || ForCheck ;
                     // Compute P Ground on bundle
                     Pt3dr aPT2 ;
                     // Now if we have interval Z, we are probably in RPC and it imporatnt to "fill" all the space
                     // we do it by selecting one more randomly in the space 
                     if ( mIntervZIsDef)
                     {
                          double aPds = (aKZ+mNbZ) / double(2*mNbZ);  // Standard weithin, betwen 0 and 1
                          if (isZRand)  // if additionnal add a random
                              aPds = NRrandom3();
                          aPT2 = aG1->ImEtZ2Terrain(aPIm1,mZMin*aPds + mZMax * (1-aPds));
                     }
                     else
                     {
                          double aPds = aKZ / double(mNbZ);
                          if (isZRand)  // if additionnal add a random
                              aPds = NRrandC();
                          aPT2 = aC1 + (aPT1-aC1) * (1+mIProf*aPds);
                     }

                     if (aG1->PIsVisibleInImage(aPT2) && aG2->PIsVisibleInImage(aPT2))
                     {
                        // Add projection
                        Pt2dr aPIm2 = aG2->Ter2Capteur(aPT2);
                        if (aG2->CaptHasData(aPIm2))
                        {
                            aL23.push_back(Appar23(aPIm1,aPT2));

                            if (! isZRand)
                               aVPIm2.push_back(aPIm2);
                            ElCplePtsHomologues aCple(aPIm1,aPIm2,1.0);
                            if (! AddToP1)  // If Im1/Im2 were swapped
                               aCple.SelfSwap();
                            aPack.Cple_Add(aCple);
                        }
                     }
                }
                // If more than one point is Ok
                if (aVPIm2.size() >=2)
                {
                    Pt2dr aDir2 = vunit(aVPIm2.back()-aVPIm2[0]);
                    aSomTens2 = aSomTens2 + aDir2 * aDir2; // On double l'angle pour en faire un tenseur
                }
            }
        }
    }
    Pt2dr aRT = Pt2dr::polar(aSomTens2,0.0);
    return Pt2dr::FromPolar(1.0,aRT.y/2.0); // Divide angle as it was multiplied
}

class cTmpReechEpip
{
     public :
        cTmpReechEpip
        (
                bool aConsChan,
                const std::string &,
                Box2dr aBoxImIn,
                ElDistortion22_Gen * anEpi,
                Box2dr aBoxOut,
                double aStep,
                const std::string & aNameOut,
                const std::string & aPostMasq,
                int aNumKer,
                bool aDebug,
                int  aNbBloc,
                double aEpsChekInv = 1e-2 // Check accuracy of inverse
        );
     private :
        Box2dr                 mBoxImIn;
        ElDistortion22_Gen *   mEpi;
        double                 mStep;
        Pt2dr                  mP0Epi;
        Pt2di                  mSzEpi;
        Pt2di                  mSzRed;

        Pt2dr ToFullEpiCoord(const Pt2dr & aP)
        {
            return mP0Epi + aP * mStep;
        }
        Pt2dr ToFullEpiCoord(const Pt2di & aP) {return ToFullEpiCoord(Pt2dr(aP));}

        Im2D_Bits<1> mRedIMasq;
        TIm2DBits<1> mRedTMasq;

        Im2D_REAL4         mRedImX;
        TIm2D<REAL4,REAL8> mRedTImX;
        Im2D_REAL4         mRedImY;
        TIm2D<REAL4,REAL8> mRedTImY;
};


void ReechFichier
    (
        bool  ConsChan,
        const std::string & aNameImInit,
        Box2dr aBoxImIn,
        ElDistortion22_Gen * anEpi,
        Box2dr aBox,
        double aStep,
        const std::string & aNameOut,
        const std::string & aPostMasq,
        int   aNumKer,
        int   aNbBloc
)
{
    cTmpReechEpip aReech(ConsChan,aNameImInit,aBoxImIn,anEpi,aBox,aStep,aNameOut,aPostMasq,aNumKer,false,aNbBloc);
}




cTmpReechEpip::cTmpReechEpip
(
        bool aConsChan,
        const std::string & aNameImInit,
        Box2dr aBoxImIn,
        ElDistortion22_Gen * anEpi,
        Box2dr aBoxImOut,
        double aStep,
        const std::string & aNameOut,
        const std::string & aPostMasq,
        int aNumKer ,
        bool Debug,
        int  aNbBloc,
        double aEpsCheckInv
) :
    mBoxImIn(aBoxImIn),
    mEpi    (anEpi),
    mStep   (aStep),
    mP0Epi  (aBoxImOut._p0),
    mSzEpi  (aBoxImOut.sz()),
    mSzRed  (round_up (aBoxImOut.sz() / aStep) + Pt2di(1,1)),
    mRedIMasq  (mSzRed.x,mSzRed.y,0),
    mRedTMasq  (mRedIMasq),
    mRedImX    (mSzRed.x,mSzRed.y),
    mRedTImX   (mRedImX),
    mRedImY    (mSzRed.x,mSzRed.y),
    mRedTImY   (mRedImY)
{
    std::cout << "=== RESAMPLE EPIP " << aNameImInit 
              << " Ker=" << aNumKer 
              << " Step=" << mStep 
              << " SzRed=" << mSzRed 
              << "======\n";


    cInterpolateurIm2D<REAL4> * aPtrSCI = 0;


    if (aNumKer==0)
    {
        aPtrSCI = new cInterpolBilineaire<REAL4>;
    }
    else 
    {
      
       cKernelInterpol1D * aKer = 0;
if (0)
{
           aKer  =  cKernelInterpol1D::StdInterpCHC(1.5);

}
       if (aNumKer==1)
          aKer = new cCubicInterpKernel(-0.5);
       else
          aKer = new cSinCardApodInterpol1D(cSinCardApodInterpol1D::eTukeyApod,aNumKer,aNumKer/2,1e-4,false);

       aPtrSCI =  new  cTabIM2D_FromIm2D<REAL4>   (aKer,1000,false);

       // cTabIM2D_FromIm2D<REAL4>   aSSCI (&aKer,1000,false);
    }

    cInterpolateurIm2D<REAL4> & aSCI = *aPtrSCI;



    Pt2di aPInd;

    for (aPInd.x=0 ; aPInd.x<mSzRed.x ; aPInd.x++)
    {
       for (aPInd.y=0 ; aPInd.y<mSzRed.y ; aPInd.y++)
       {
          bool Ok= false;
          Pt2dr aPEpi = ToFullEpiCoord(aPInd);
          Pt2dr aPIm =  anEpi->Inverse(aPEpi);
          if ((aPIm.x>mBoxImIn._p0.x) && (aPIm.y>mBoxImIn._p0.y) && (aPIm.x<mBoxImIn._p1.x) && (aPIm.y<mBoxImIn._p1.y))
          {
               Pt2dr aPEpi2 = anEpi->Direct(aPIm);
               if (euclid(aPEpi-aPEpi2) < aEpsCheckInv)
               {
                    Ok= true;
                    mRedTMasq.oset(aPInd,Ok);
               }
          }
          mRedTImX.oset(aPInd,aPIm.x);
          mRedTImY.oset(aPInd,aPIm.y);
       }
    }
    ELISE_COPY(mRedIMasq.all_pts(),dilat_d8(mRedIMasq.in(0),4),mRedIMasq.out());


    Tiff_Im aTifOri = Tiff_Im::StdConvGen(aNameImInit.c_str(),aConsChan ? -1 :1 ,true);
    Tiff_Im aTifEpi  = Debug                       ?
                       Tiff_Im(aNameOut.c_str())     :
                       Tiff_Im
                       (
                           aNameOut.c_str(),
                           mSzEpi,
                           aTifOri.type_el(),
                           Tiff_Im::No_Compr,
                           aTifOri.phot_interp()
                       )                            ;

    Tiff_Im aTifMasq = aTifEpi;
    bool ExportMasq = (aPostMasq!="NONE");

// std::cout << "POSTMAS " << aPostMasq << "\n";

    if (ExportMasq)
    {
        std::string aNameMasq = StdPrefix(aNameOut)+ aPostMasq  +".tif";
        if (Debug)
        {
           Tiff_Im::Create8BFromFonc("Reduc-"+aNameMasq,mRedIMasq.sz(),mRedIMasq.in()*255);
        }
        aTifMasq =  Debug                         ?
                    Tiff_Im(aNameMasq.c_str())    :
                    Tiff_Im
                    (
                        aNameMasq.c_str(),
                        mSzEpi,
                        GenIm::bits1_msbf,
                        Tiff_Im::No_Compr,
                        Tiff_Im::BlackIsZero
                    )                             ;
    }





    int aBrd = aNumKer+10;
    Pt2di aSzBrd(aBrd,aBrd);

    int aX00 = 0;
    int aY00 = 0;

    for (int aX0=aX00 ; aX0<mSzEpi.x ; aX0+=aNbBloc)
    {
         int aX1 = ElMin(aX0+aNbBloc,mSzEpi.x);
         for (int aY0=aY00 ; aY0<mSzEpi.y ; aY0+=aNbBloc)
         {
// std::cout << "X0Y0 " << aX0 << " " << aY0 << "\n";

             int aY1 = ElMin(aY0+aNbBloc,mSzEpi.y);

             Pt2di aP0Epi(aX0,aY0);
             Pt2di aSzBloc(aX1-aX0,aY1-aY0);

             TIm2D<REAL4,REAL8> aTImX(aSzBloc);
             TIm2D<REAL4,REAL8> aTImY(aSzBloc);
             TIm2DBits<1>       aTImMasq(aSzBloc,0);

             Pt2dr aInfIm(1e20,1e20);
             Pt2dr aSupIm(-1e20,-1e20);
             bool  NonVide= false;

             for (int anX =aX0 ; anX<aX1  ; anX++)
             {
                 for (int anY =aY0 ; anY<aY1  ; anY++)
                 {
                     Pt2dr aIndEpi (anX/mStep , anY/mStep);
                     Pt2di aPIndLoc (anX-aX0,anY-aY0);
                     if (mRedTMasq.get(round_down(aIndEpi)))
                     {
                        double aXIm = mRedTImX.getr(aIndEpi,-1,true);
                        double aYIm = mRedTImY.getr(aIndEpi,-1,true);

                        if ((aXIm>0) && (aYIm>0))
                        {
                            // aTImMasq.oset(aPIndLoc,1);
                            aTImX.oset(aPIndLoc,aXIm);
                            aTImY.oset(aPIndLoc,aYIm);

                            aInfIm  = Inf(aInfIm,Pt2dr(aXIm,aYIm));
                            aSupIm  = Sup(aSupIm,Pt2dr(aXIm,aYIm));
                            NonVide= true;
                        }
                     }
                 }
             }
             Pt2di aP0BoxIm = Sup(Pt2di(0,0),Pt2di(round_down(aInfIm) - aSzBrd));
             Pt2di aP1BoxIm = Inf(aTifOri.sz(),Pt2di(round_down(aSupIm) + aSzBrd));
             Pt2di aSzIm = aP1BoxIm - aP0BoxIm;
             NonVide = NonVide && (aSzIm.x>0) && (aSzIm.y>0);
             if (NonVide)
             {

                 // std::vector<Im2D_REAL4>  aVIm;

                 std::vector<Im2D_REAL4>  aVIm= aTifOri.VecOfImFloat(aSzIm);

                 ELISE_COPY
                 (
                     rectangle(Pt2di(0,0),aSzIm),
                     trans(aTifOri.in(),aP0BoxIm),
                     StdOut(aVIm)
                 );

                 std::vector<Im2D_REAL4>  aVImEpi = aTifEpi.VecOfImFloat(aSzBloc);
                 ELISE_ASSERT(aVImEpi.size()==aVIm.size(),"Incohe in nb chan, cTmpReechEpip::cTmpReechEpip");

                 for (int aKIm=0 ; aKIm <int(aVImEpi.size()) ; aKIm++)
                 {
                      TIm2D<REAL4,REAL8> aImEpi(aVImEpi[aKIm]);
                      REAL4 ** aDataOri = aVIm[aKIm].data();
                      for (int anX =0 ; anX<aSzBloc.x ; anX++)
                      {
                           for (int anY =0 ; anY<aSzBloc.y ; anY++)
                           {

                               Pt2di aIndEpi(anX,anY);
                               aImEpi.oset(aIndEpi,0);
                               Pt2di anIndEpiGlob  = aIndEpi + aP0Epi;

                               Pt2dr aIndEpiRed (anIndEpiGlob.x/mStep , anIndEpiGlob.y/mStep);
                               if (mRedTMasq.get(round_down(aIndEpiRed),0))
                               {
                                   double aXIm = mRedTImX.getr(aIndEpiRed,-1,true);
                                   double aYIm = mRedTImY.getr(aIndEpiRed,-1,true);
                                   Pt2dr aPImLoc = Pt2dr(aXIm,aYIm) - Pt2dr(aP0BoxIm);
                                   double aV= 128;
                                   if ((aPImLoc.x>aNumKer+2) && (aPImLoc.y>aNumKer+2) && (aPImLoc.x<aSzIm.x-aNumKer-3) && (aPImLoc.y<aSzIm.y-aNumKer-3))
                                   {
                                       aTImMasq.oset(aIndEpi,1);
                                       aV = aSCI.GetVal(aDataOri,aPImLoc);
                                       // aV= 255;
                                   }
                                   aImEpi.oset(aIndEpi,aV);
                               }
                           }
                      }
                 }
                 ELISE_COPY
                 (
                     rectangle(aP0Epi,aP0Epi+aSzBloc),
                     Tronque(aTifEpi.type_el(),trans(StdInput(aVImEpi),-aP0Epi)),
                     aTifEpi.out()
                 );
             }
             if (ExportMasq)
             {
                ELISE_COPY
                (
                    rectangle(aP0Epi,aP0Epi+aSzBloc),
                    trans(aTImMasq._the_im.in(0),-aP0Epi),
                    aTifMasq.out()
                );
             }
             // std::cout << "ReechDONE " <<  aX0 << " "<< aY0 << "\n";

         }
    }
}


// void CalcDirEpip(const ElPackHomologue)

extern void SetExagEpip(double aVal,bool AcceptApprox);

double cApply_CreateEpip_main::PhaseValue(const double & aV,const double & aSzY) const
{
  double   aNbLine = mParamEICE.at(1);
  double   aPer =  aSzY / aNbLine;
  double   aLarg =  mParamEICE.at(2);
  double  aPhase = ElAbs(Centered_mod_real(aV,aPer)/aPer);
  aPhase = ElMin(aPhase/aLarg,1.0) *PI;

  double aRes =  (1+cos(aPhase)) /2.0;
  return aRes;
}

void cApply_CreateEpip_main::ExportImCurveEpip
     (
        EpipolaireCoordinate & anEpip1,const Box2dr& aBoxIn1,const Box2dr& aBoxOut1,
        const std::string & aName,
        EpipolaireCoordinate & anEpip2,const Box2dr& aBoxIn2,
        double aX2mX1
     )
{
    ELISE_ASSERT(mParamEICE.size()==5,"Bad Size for ExpCurve");
    bool ShowOut = mParamEICE.at(4)>0.5;
    double aLRed =  mParamEICE.at(0);
    SetExagEpip(mParamEICE.at(3),true);

    Pt2dr aSzIn = aBoxIn1.sz();
    double aScale = ElMax(aSzIn.x,aSzIn.y) / aLRed;
    Pt2di aSzRed = round_ni(aSzIn / aScale);

    Pt2di aP;
    Im2D_U_INT1 aIm(aSzRed.x,aSzRed.y);

    for (aP.x=0 ; aP.x<aSzRed.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<aSzRed.y ; aP.y++)
        {
            Pt2dr aPIm1  = aBoxIn1.P0() + Pt2dr(aP) * aScale;

            Pt2dr aPEpi1 = anEpip1.Direct(aPIm1);
            double aVy = PhaseValue(aPEpi1.y,aBoxOut1.sz().y);
            
            double aVal = 1-aVy;

            if (ShowOut)
            {
                Pt2dr aPEpi2 (aPEpi1.x+aX2mX1 ,aPEpi1.y);
                Pt2dr aPIm2 =  anEpip2.Inverse(aPEpi2);

                if (!aBoxIn2.inside(aPIm2))
                {
                    aVal = 1 -aVal;
                }
           }

            aIm.SetR(aP,255*aVal);
        }
    }
    SetExagEpip(1.0,false);
    Tiff_Im::CreateFromIm(aIm,aName);
}

void cApply_CreateEpip_main::DoEpipGen(bool DoIm)
{
      mLengthMin = 500.0;
      mStepReech = 10.0;
      //  mNbXY      = 100;
      ElPackHomologue aPack;
      ElPackHomologue aPackCheck;
      std::list<Appar23>  aLT1,aLT2;

      // Compute the direction  and the set of homologous points
      if (mWithOri)
      {
         // For checking
         {
             std::list<Appar23>  aLTCheck1,aLTCheck2;
             DirEpipIm2(mGenI1,mGenI2,aPackCheck,true,true  ,aLTCheck1);  // Dont Swap
             DirEpipIm2(mGenI2,mGenI1,aPackCheck,true,false ,aLTCheck2); // Swap Pt
         }


         mDir2 =  DirEpipIm2(mGenI1,mGenI2,aPack,false,true,aLT1);  // Dont Swap
         mDir1 =  DirEpipIm2(mGenI2,mGenI1,aPack,false,false,aLT2); // Swap Pt
      }
      else
      {
          aPack = mICNM->StdPackHomol("",mName1,mName2);
          if (EAMIsInit(&mNbCalcAutoDir))
          {
              aPack.DirEpipolaire(mDir1,mDir2,mNbCalcAutoDir.x,mNbCalcAutoDir.y,1);
          }
          else
          {
              ELISE_ASSERT
              (
                  EAMIsInit(&mDir1) &&  EAMIsInit(&mDir2),
                  "Dir1 and Dir2 must be initialized in mode no ori"
              );
          }
      }

      std::cout << "Compute Epip ; D1=" << mDir1 << " ,D2=" << mDir2 << "\n";
      CpleEpipolaireCoord * aCple = CpleEpipolaireCoord::PolynomialFromHomologue(false,aPack,mDegre,mDir1,mDir2,mDegSupInv);

      // Save the result of epipolar in (relatively) raw format, containg polynoms+ name of orent
      //  in case we want to do a Cap3D=> nuage from them
      if (1)
      {
          aCple->SaveOrientCpleEpip(mOri,mICNM,mName1,mName2);
      }


      EpipolaireCoordinate & e1 = aCple->EPI1();
      EpipolaireCoordinate & e2 = aCple->EPI2();

      Pt2dr aInf1(1e20,1e20),aSup1(-1e20,-1e20);
      Pt2dr aInf2(1e20,1e20),aSup2(-1e20,-1e20);
      double aX2mX1 ;

      for (int aKTime=0 ; aKTime<2 ; aKTime++)
      {
            aX2mX1 = 0.0;
            aInf1=Pt2dr(1e20,1e20);
            aSup1=Pt2dr(-1e20,-1e20);
            aInf2=Pt2dr(1e20,1e20);
            aSup2=Pt2dr(-1e20,-1e20);

            bool ForCheck = (aKTime==0);
            ElPackHomologue * aPackK = ForCheck ? &aPackCheck  : & aPack;

            double aBias = 0.0;
            double aErrMax = 0.0;
            double aErrMoy = 0.0;
            int    mNbP = 0;

            // Compute accuracy, bounding box 
            for (ElPackHomologue::const_iterator itC=aPackK->begin() ; itC!= aPackK->end() ; itC++)
            {
                 // Images of P1 and P2 by epipolar transforms
                 Pt2dr aP1 = e1.Direct(itC->P1());
                 Pt2dr aP2 = e2.Direct(itC->P2());
                 // Update bounding boxes
                 aInf1 = Inf(aInf1,aP1);
                 aSup1 = Sup(aSup1,aP1);
                 aInf2 = Inf(aInf2,aP2);
                 aSup2 = Sup(aSup2,aP2);

                 // Average off of X
                 aX2mX1 += aP2.x - aP1.x;

                 double aDifY = aP1.y-aP2.y; // Should be 0
                 double anErr = ElAbs(aDifY);
                 mNbP++;
                 aErrMax = ElMax(anErr,aErrMax);
                 aErrMoy += anErr;
                 aBias   += aDifY;
            }

            aX2mX1 /= mNbP;

            double aInfY = ElMax(aInf1.y,aInf2.y);
            double aSupY = ElMax(aSup1.y,aSup2.y);
            aInf1.y = aInf2.y = aInfY;
            aSup1.y = aSup2.y = aSupY;


            std::cout  << "======================= " << (ForCheck ? " CONTROL" : "LEARNING DATA") << " ========\n";
            std::cout << "Epip Rect Accuracy:" 
                      << " Bias " << aBias/mNbP 
                      << " ,Moy " <<  aErrMoy/mNbP 
                      << " ,Max " <<  aErrMax 
                      << "\n";

            if (! ForCheck)
            {
                std::cout << "DIR " << mDir1 << " " << mDir2 << " X2-X1 " << aX2mX1<< "\n";
                std::cout << "Epip NbPts= " << mNbP << " Redund=" << mNbP/double(ElSquare(mDegre)) << "\n";
            }
      
      }
      std::cout  << "BOX1 " << aInf1 << " " <<  aSup1 << "\n";
      std::cout  << "BOX2 " << aInf2 << " " <<  aSup2 << "\n";


      bool aConsChan = true;
      Pt2di aSzI1 = mWithOri ? 
                    mGenI1->SzBasicCapt3D() : 
                    Tiff_Im::StdConvGen(mName1.c_str(),aConsChan ? -1 :1 ,true).sz() ;
      Pt2di aSzI2 = mWithOri ? 
                    mGenI2->SzBasicCapt3D() : 
                    Tiff_Im::StdConvGen(mName2.c_str(),aConsChan ? -1 :1 ,true).sz() ;

      std::string aNI1 = mICNM->NameImEpip(mOri,mName1,mName2);
      std::string aNI2 = mICNM->NameImEpip(mOri,mName2,mName1);

      Box2dr aBIn1(Pt2dr(0,0),Pt2dr(aSzI1));
      Box2dr aBOut1(aInf1,aSup1);
      Box2dr aBIn2(Pt2dr(0,0),Pt2dr(aSzI2));
      Box2dr aBOut2(aInf2,aSup2);

      if (DoIm)
      {
         cTmpReechEpip aReech1(aConsChan,mName1,aBIn1,&e1,aBOut1,mStepReech,aNI1,mPostMasq,mNumKer,mDebug,mNbBloc,mEpsCheckInv);
         std::cout << "DONE IM1 \n";
         cTmpReechEpip aReech2(aConsChan,mName2,aBIn2,&e2,aBOut2,mStepReech,aNI2,mPostMasq,mNumKer,mDebug,mNbBloc,mEpsCheckInv);
         std::cout << "DONE IM2 \n";

         std::cout << "DONNE REECH TMP \n";
      }

//  ::cTmpReechEpip(cBasicGeomCap3D * aGeom,EpipolaireCoordinate * anEpi,Box2dr aBox,double aStep) :

      if (mMakeAppuis)
      {
           MakeAppuis(true ,aLT1,e1,aInf1,aSup1);
           MakeAppuis(false,aLT2,e2,aInf2,aSup2);
      }

      if (EAMIsInit(&mParamEICE))
      {
         ExportImCurveEpip(e1,aBIn1,aBOut1,"ImLineEpip1.tif",e2,aBIn2,aX2mX1);
         ExportImCurveEpip(e2,aBIn2,aBOut2,"ImLineEpip2.tif",e1,aBIn1,-aX2mX1);
      }
}

void cApply_CreateEpip_main::MakeAppuis
     (
          bool Is1,
          const std::list<Appar23>&aLT1,
          EpipolaireCoordinate & anEpi,
          Pt2dr aP0Epi,
          Pt2dr aP1Epi
      )
{
          //  Pt2dr ToFullEpiCoord(const Pt2dr & aP) { return mP0Epi + aP * mStep; }
          // Pt2dr aPEpi = ToFullEpiCoord(aPInd);
          // Pt2dr aPIm =  anEpi->Inverse(aPEpi);
          //  aPIm  = Inver(aPEpi+P0)
          //  PEpi = 
    std::list<Appar23> aLCor;
    Pt2dr aInfEpi( 1e20, 1e20);
    Pt2dr aSupEpi(-1e20,-1e20);
    Pt2dr aSzEpi = aP1Epi - aP0Epi;
    for (const auto & aP : aLT1)
    {
        Pt2dr aPEpi =  anEpi.Direct(aP.pim) - aP0Epi;
        aInfEpi = Inf(aPEpi,aInfEpi);
        aSupEpi = Sup(aPEpi,aSupEpi);
        // std::cout << aP.pim << " " << anEpi.Direct(aP.pim) - aP0Epi << " " << anEpi.Inverse(aP.pim) +aP0Epi << "\n";
        aLCor.push_back(Appar23(aPEpi,aP.pter));
    }
    std::string aN1 = Is1 ? mName1 : mName2 ;
    std::string aN2 = Is1 ? mName2 : mName1 ;

    std::string aNameAp = mICNM->NameAppuiEpip(mOri,aN1,aN2);


    cListeAppuis1Im  aLAp =  El2Xml(aLCor,aN1);
    MakeFileXML(aLAp,aNameAp);
    std::cout << "======  MakeAppuis ====== " << aInfEpi  << " :: " << aSupEpi -aSzEpi << "\n";//  getchar();
}



    // std::string  aNameEpi = Is1?"ImEpi1.tif":"ImEpi2.tif";



cApply_CreateEpip_main::cApply_CreateEpip_main(int argc,char ** argv) :
   mDegre     (-1),
   mNbZ       (1),
   mNbXY      (100),
   mNbZRand   (1),
   mForceGen  (false),
   mNumKer    (5),
   mDebug     (false),
   mPostMasq  (""),
   mGenI1     (0),
   mGenI2     (0),
   mWithOri   (true),
   mNbBloc    (2000),
   mDegSupInv (4),
   mEpsCheckInv (1e-1),
   mMakeAppuis  (false),
   mIntervZIsDef (false),
   mZMin         (1e20),
   mZMax         (-1e20)
{
    Tiff_Im::SetDefTileFile(50000);
    std::string aDir= ELISE_Current_DIR;
    std::string anOri;
    double  aScale=1.0;

    bool Gray = true;
    bool Cons16B = true;
    bool InParal = true;
    bool DoIm = true;
    std::string aNameHom;
    bool mExpTxt=false;


    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(mName1,"Name first image", eSAM_IsExistFile)
                << EAMC(mName2,"Name second image", eSAM_IsExistFile)
                << EAMC(anOri,"Name orientation", eSAM_IsExistDirOri),
    LArgMain()  << EAM(aScale,"Scale",true)
                << EAM(aDir,"Dir",true,"directory (Def=current)", eSAM_IsDir)
                    << EAM(Gray,"Gray",true,"One channel gray level image (Def=true)", eSAM_IsBool)
                    << EAM(Cons16B,"16B",true,"Maintain 16 Bits images if avalaible (Def=true)", eSAM_IsBool)
                    << EAM(InParal,"InParal",true,"Compute in parallel (Def=true)", eSAM_IsBool)
                    << EAM(DoIm,"DoIm",true,"Compute image (def=true !!)", eSAM_IsBool)
                    << EAM(aNameHom,"NameH",true,"Extension to compute Hom point in epi coord (def=none)", eSAM_NoInit)
                    << EAM(mDegre,"Degre",true,"Degre of polynom to correct epi (def=9)")
                    << EAM(mForceGen,"FG",true,"Force generik epip even with stenope cam")
                    << EAM(mNumKer,"Kern",true,"Kernel of interpol,0 Bilin, 1 Bicub, other SinC (fix size of apodisation window), Def=5")
                    << EAM(mPostMasq,"AttrMasq",true,"Atribut for masq toto-> toto_AttrMasq.tif, NONE if unused, Def=Ori")
                    << EAM(mPostIm,"PostIm",true,"Attribut for Im ")
		    << EAM(mExpTxt,"ExpTxt",false,"Use txt tie points (Def false, e.g. use dat format)")
		    << EAM(mDir1,"Dir1",false,"Direction of Epip one (when Ori=NONE)")
		    << EAM(mDir2,"Dir2",false,"Direction of Epip one (when Ori=NONE)")
		    << EAM(mNbBloc,"NbBloc",false,"Sz of Bloc (mostly tuning)")
		    << EAM(mDebug,"Debug",false,"Debuging")
		    << EAM(mDegSupInv,"SDI",false,"Supplementary degree for inverse")
		    << EAM(mEpsCheckInv,"ECI",false,"Espsilpn for check inverse accuracy")
		    << EAM(mNbZ,"NbZ",false,"Number of Z, def=1 (NbLayer=1+2*NbZ)")
		    << EAM(mNbZRand,"NbZRand",false,"Number of additional random Z in each bundle, Def=1")
		    << EAM(mIProf,"IProf",false,"Interval prof (for test in mode FG of frame cam, else unused)")
		    << EAM(mIntZ,"IntZ",false,"Z interval, for test or correct interval of RPC")
		    << EAM(mNbXY,"NbXY",false,"Number of point / line or col, def=100")
		    << EAM(mNbCalcAutoDir,"NbCalcDir",false,"Calc directions : Nbts / NbEchDir")
		    << EAM(mParamEICE,"ExpCurve",false,"0-SzIm ,1-Number of Line,2- Larg (in [0 1]),3-Exag deform,4-ShowOut")
    );


if (!MMVisualMode)
{
    if (mName1 > mName2) ElSwap(mName1,mName2);

    int aNbChan = Gray ? 1 : - 1;

    cTplValGesInit<std::string>  aTplFCND;
    mICNM = cInterfChantierNameManipulateur::StdAlloc
                                               (
                                                   argc,argv,
                                                   aDir,
                                                   aTplFCND
                                               );
     mWithOri = (anOri != "NONE");
     if (mWithOri)
     {
        mICNM->CorrecNameOrient(anOri);
     }
     else
     {
     }
     mOri = anOri;

     if (mPostMasq!="NONE") 
     {
          // if (!EAMIsInit(&mPostMasq)) mPostMasq =  anOri;
           mPostMasq = "_Masq";
     }
     if (!EAMIsInit(&mPostIm)) mPostIm =  "_"+ (mWithOri ? anOri : std::string("EpiHom"));


     if (mWithOri)
     {
        mGenI1 = mICNM->StdCamGenerikOfNames(anOri,mName1);
        mGenI2 = mICNM->StdCamGenerikOfNames(anOri,mName2);

        IntervZAddIm(mGenI1);
        IntervZAddIm(mGenI2);
     }
     if (EAMIsInit(&mIntZ))
     {
          mIntervZIsDef = true;
          mZMin         = mIntZ.x;
          mZMax         = mIntZ.y;
     }

     if ((!mWithOri) || (mGenI1->DownCastCS()==0) || (mGenI2->DownCastCS()==0) || mForceGen)
     {
         // In case Generik, by default we create Appuis
         if (!EAMIsInit(&mMakeAppuis))
            mMakeAppuis = true;

         if (! EAMIsInit(&mDegre))
         {
            mDegre = mWithOri ? 9 : 2;
         }
         std::cout << "DDDDDD " << mDegre << " " << mWithOri << "\n";
         DoEpipGen(DoIm);
         return;
     }

     std::string   aKey =  + "NKS-Assoc-Im2Orient@-" + anOri;

     std::string aNameOr1 = mICNM->Assoc1To1(aKey,mName1,true);
     std::string aNameOr2 = mICNM->Assoc1To1(aKey,mName2,true);

     // std::cout << "RREEEEEEEEEEEEEEead cam \n";
     CamStenope * aCam1 = CamStenope::StdCamFromFile(true,aNameOr1,mICNM);
     // std::cout << "EPISZPPPpp " << aCam1->SzPixel() << "\n";

     CamStenope * aCam2 = CamStenope::StdCamFromFile(true,aNameOr2,mICNM);

     Tiff_Im aTif1 = Tiff_Im::StdConvGen(aDir+mName1,aNbChan,Cons16B);
     Tiff_Im aTif2 = Tiff_Im::StdConvGen(aDir+mName2,aNbChan,Cons16B);



      // aCam1->SetSz(aTif1.sz(),true);
      // aCam2->SetSz(aTif2.sz(),true);


  //  Test commit


     cCpleEpip aCplE
               (
                    aDir,
                    aScale,
                    *aCam1,mName1,
                    *aCam2,mName2
               );

     const char * aCarHom = 0;
     if (EAMIsInit(&aNameHom))
        aCarHom = aNameHom.c_str();

     std::cout << "TimeEpi-0 \n";
     ElTimer aChrono;
     aCplE.ImEpip(aTif1,aNameOr1,true,InParal,DoIm,aCarHom,mDegre,mExpTxt);
     std::cout << "TimeEpi-1 " << aChrono.uval() << "\n";
     aCplE.ImEpip(aTif2,aNameOr2,false,InParal,DoIm,aCarHom,mDegre,mExpTxt);
     std::cout << "TimeEpi-2 " << aChrono.uval() << "\n";

     aCplE.SetNameLock("End");
     aCplE.LockMess("End cCpleEpip::ImEpip");



     return ;

}
else return ;
}


int CreateEpip_main(int argc,char ** argv)
{
     cApply_CreateEpip_main(argc,argv);

     return EXIT_SUCCESS;
}

/*************************************************************/
/*                                                           */
/*                 cAppliReechHomogr                         */                                         
/*                                                           */
/*************************************************************/


class cAppliReechHomogr    : public ElDistortion22_Gen
{
    public :
        cAppliReechHomogr(int argc,char ** argv);
        void DoReech();
    private :
        Pt2dr Direct(Pt2dr) const;
        bool OwnInverse(Pt2dr &) const ;

        cElemAppliSetFile mEASF;


        std::string mFullNameI1;
        std::string mFullNameI2;
        std::string mNameI1;
        std::string mNameI2;
        std::string mNameI2Redr;
        std::string mPostMasq;
        std::string mKeyHom;

        ElPackHomologue  mPack;
        cElHomographie  mH1To2;
        cElHomographie  mH2To1;
        double          mScaleReech;
};

Pt2dr cAppliReechHomogr::Direct(Pt2dr aP) const
{
    return mH2To1.Direct(aP);
}

bool cAppliReechHomogr::OwnInverse(Pt2dr & aP) const 
{
   aP = mH1To2.Direct(aP);
   return true;
}



cAppliReechHomogr::cAppliReechHomogr(int argc,char ** argv)  :
    mPostMasq ("Masq"),
    mH1To2    (cElHomographie::Id()),
    mH2To1    (cElHomographie::Id())
{
	bool aShow=false;
    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(mFullNameI1,"Name of \"Master\" Image", eSAM_IsExistFile)
                      << EAMC(mFullNameI2,"Name of \"Slave\" Image", eSAM_IsExistFile)
                      << EAMC(mNameI2Redr,"Name of resulting registered Image", eSAM_IsExistFile),
          LArgMain()  << EAM (mPostMasq,"PostMasq",true,"Name of Masq , Def = \"Masq\"")
                      << EAM (mScaleReech,"ScaleReech",true,"Scale Resampling, used for interpolator when downsizing")
					  << EAM (aShow,"Show",true,"Show the computed homographies")
    );

     mNameI1 = NameWithoutDir(mFullNameI1);
     mNameI2 = NameWithoutDir(mFullNameI2);
   
     mEASF.Init(mFullNameI1);

     mKeyHom = "NKS-Assoc-CplIm2Hom@@dat";

     std::string aNameH = mEASF.mDir + mEASF.mICNM->Assoc1To2(mKeyHom,mNameI1,mNameI2,true);
     ElPackHomologue aPack = ElPackHomologue::FromFile(aNameH);

     double anEcart,aQuality;
     bool Ok;
     mH1To2 = cElHomographie::RobustInit(anEcart,&aQuality,aPack,Ok,50,80.0,2000);
     mH2To1 = mH1To2.Inverse();
     std::cout << "Ecart " << anEcart << " ; Quality " << aQuality    << " \n";

	 if (aShow)
	 {
	 	std::cout << "H12=" << "\n";
		mH1To2.Show();
	 	std::cout << "H21=" << "\n";
		mH2To1.Show();
	 }

     cMetaDataPhoto aMTD1 = cMetaDataPhoto::CreateExiv2(mFullNameI1);
     cMetaDataPhoto aMTD2 = cMetaDataPhoto::CreateExiv2(mFullNameI2);


     ReechFichier
     (
          false,
          mFullNameI2,
          Box2dr(Pt2dr(0,0),Pt2dr(aMTD2.TifSzIm())),
          this,
          Box2dr(Pt2dr(0,0),Pt2dr(aMTD1.TifSzIm())),
          10.0,
          mNameI2Redr,
          mPostMasq,
          5,
          2000
     );

}

int OneReechHom_main(int argc,char ** argv)
{
     cAppliReechHomogr  anAppli(argc,argv);
     return EXIT_SUCCESS;
}

int AllReechHom_main(int argc,char ** argv)
{
    std::string aFullName1,aPat,aPref,aPostMasq = "Masq";
    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aFullName1,"Name of \"Master\" Image", eSAM_IsExistFile)
                      << EAMC(aPat,"Name of all \"Slaves\" Image", eSAM_IsExistFile)
                      << EAMC(aPref,"Name of Prefix for registered Images", eSAM_IsExistFile),
          LArgMain()   <<  EAM(aPostMasq,"PostMasq",true,"Name of Masq , Def = \"Masq\"")
    );

    cElemAppliSetFile anEASF(aPat);
    const cInterfChantierNameManipulateur::tSet *  aSet = anEASF.SetIm();

    std::string aName1 = NameWithoutDir(aFullName1);

    std::list<std::string>  aLCom;

    for (int aK=0 ; aK<int(aSet->size()) ; aK++)
    {
         std::string aName2 = (*aSet)[aK];
         if (aName1 != aName2)
         {
             std::string aNameRes = anEASF.mDir +  aPref +aName2 + ".tif";
             if (! ELISE_fp::exist_file(aNameRes))
             {
                 // std::cout << "RES = " << aNameRes << "\n";
                 std::string aCom =  MM3dBinFile_quotes("TestLib")
                                     + " OneReechHom " 
                                     +   aFullName1
                                     +  " " + anEASF.mDir + aName2 
                                     +  " " +  aNameRes
                                     +  " PostMasq=" + aPostMasq;

                 aLCom.push_back(aCom);
                 // std::cout << "COM= " << aCom << "\n";
             }
         }
    }
    cEl_GPAO::DoComInParal(aLCom);

   
     return EXIT_SUCCESS;
}


/*************************************************************/

/*************************************************************/
/*                                                           */
/*                 cAppliOneReechMarqFid                     */                                         
/*                                                           */
/*************************************************************/



class cAppliOneReechMarqFid : public ElDistortion22_Gen ,
                              public cAppliWithSetImage
{
    public :
        cAppliOneReechMarqFid(int argc,char ** argv);
        void DoReech();
    private :
        Pt2dr Direct(Pt2dr) const;
        bool OwnInverse(Pt2dr &) const ;
        Pt2dr       ChambreMm2ChambrePixel (const Pt2dr &) const;
        Pt2dr       ChambrePixel2ChambreMm (const Pt2dr &) const;
        Box2dr      mBoxChambreMm;
        Box2dr      mBoxChambrePix;
        ElPackHomologue  mPack;


        std::string mNamePat;
        std::string mNameIm;
        std::string mDir;
        cInterfChantierNameManipulateur * mICNM;
        double      mResol;
        double      mResidu;
        ElAffin2D   mAffPixIm2ChambreMm;
        ElAffin2D   mAffChambreMm2PixIm;
        Pt2di       mSzIm;
        bool        mBySingle;
        int         mNumKer;
        std::string mPostMasq;
		bool ExportAffine;
};

// mAffin (mm) = Pixel

bool cAppliOneReechMarqFid::OwnInverse(Pt2dr & aP) const 
{
    aP = mAffChambreMm2PixIm(ChambrePixel2ChambreMm(aP));
    return true;
}

Pt2dr  cAppliOneReechMarqFid::Direct(Pt2dr aP) const
{
    return  ChambreMm2ChambrePixel(mAffPixIm2ChambreMm(aP));
}



Pt2dr cAppliOneReechMarqFid::ChambreMm2ChambrePixel (const Pt2dr & aP) const
{
   return (aP-mBoxChambreMm._p0) / mResol;
}

Pt2dr cAppliOneReechMarqFid::ChambrePixel2ChambreMm (const Pt2dr & aP) const
{
   return mBoxChambreMm._p0 + aP * mResol;
}



cAppliOneReechMarqFid::cAppliOneReechMarqFid(int argc,char ** argv) :
     cAppliWithSetImage   (argc-1,argv+1,TheFlagNoOri),
     mAffPixIm2ChambreMm  (ElAffin2D::Id()),
     mBySingle            (true),
     mNumKer              (5),
     mPostMasq            ("NONE"),
	 ExportAffine 		  (true)
{
    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(mNamePat,"Pattern image", eSAM_IsExistFile)
                      << EAMC(mResol,"Resolution of scan, mm/pix"),
          LArgMain()   <<  EAM(mBoxChambreMm,"BoxCh",true,"Box in Chambre (generally in mm, [xmin,ymin,xmax,ymax])")
                       << EAM(mNumKer,"Kern",true,"Kernel of interpol,0 Bilin, 1 Bicub, other SinC (fix size of apodisation window), Def=5")
                       << EAM(mPostMasq,"AttrMasq",true,"Atribut for masq toto-> toto_AttrMasq.tif, NONE if unused, Def=NONE")
					   << EAM(ExportAffine,"ExpAff","true","Export the affine transformation")
    );

    if (mPostMasq!="NONE") 
       mPostMasq = "_"+mPostMasq+"Masq";

    const cInterfChantierNameManipulateur::tSet * aSetIm = mEASF.SetIm();
    if (aSetIm->size()>1)
    {
         mBySingle = false;
         ExpandCommand(2,"",true);
         return;
    }

    mNameIm =(*aSetIm)[0];
 
    mDir = DirOfFile(mNameIm);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);

    std::pair<std::string,std::string> aPair = mICNM->Assoc2To1("Key-Assoc-STD-Orientation-Interne",mNameIm,true);


    mSzIm = Tiff_Im::StdConvGen(mNameIm,-1,true).sz();

    cMesureAppuiFlottant1Im aMesCam = StdGetFromPCP(mDir+aPair.first,MesureAppuiFlottant1Im);
    cMesureAppuiFlottant1Im aMesIm  = StdGetFromPCP(mDir+aPair.second,MesureAppuiFlottant1Im);

    mPack = PackFromCplAPF(aMesCam,aMesIm);
    if (! EAMIsInit(&mBoxChambreMm))
    {
          mBoxChambreMm._p0 = Pt2dr(1e20,1e20);
          mBoxChambreMm._p1 = Pt2dr(-1e20,-1e20);
          for (ElPackHomologue::const_iterator itC=mPack.begin() ; itC!=mPack.end() ; itC++)
          {
               mBoxChambreMm._p0  = Inf(mBoxChambreMm._p0,itC->P1());
               mBoxChambreMm._p1  = Sup(mBoxChambreMm._p1,itC->P1());
          }
    }
    mBoxChambrePix._p0 = ChambreMm2ChambrePixel(mBoxChambreMm._p0);
    mBoxChambrePix._p1 = ChambreMm2ChambrePixel(mBoxChambreMm._p1);

/*
    mAffChambreMm2PixIm = ElAffin2D::L2Fit(mPack,&mResidu);
    mAffPixIm2ChambreMm  = mAffChambreMm2PixIm.inv();
    std::cout << "FOR " << mNameIm << " RESIDU " << mResidu  << " \n";
*/
}


void cAppliOneReechMarqFid::DoReech()
{
    if (! mBySingle) return;

    ElTimer aChrono;
   
    mAffChambreMm2PixIm = ElAffin2D::L2Fit(mPack,&mResidu);
    mAffPixIm2ChambreMm  = mAffChambreMm2PixIm.inv();

    ReechFichier
    (
          true,
          mNameIm,
          Box2dr(Pt2dr(0,0),Pt2dr(mSzIm)),
          this,
          mBoxChambrePix,
          10.0,
          "OIS-Reech_"+mNameIm,
          mPostMasq,
          mNumKer,
          2000
    );
    std::cout << "FOR " << mNameIm << " RESIDU " << mResidu   << " Time " << aChrono.uval() << " \n";

	if (ExportAffine)
	{
		std::string aAffFileDir = "Ori-InterneScan/";
		ELISE_fp::MkDirSvp(aAffFileDir);

		mAffChambreMm2PixIm.SaveInFile(aAffFileDir+"OIS-Reech_"+StdPrefix(mNameIm)+"_ChambreMm2Pix.xml");
		//MakeFileXML(mAffChambreMm2PixIm,aAffFileDir+"OIS-Reech_"+StdPrefix(mNameIm)+"_ChambreMm2Pix.xml");

	}
}
    


int OneReechFid_main(int argc,char ** argv)
{
     cAppliOneReechMarqFid anAppli(argc,argv);
     anAppli.DoReech();
     return EXIT_SUCCESS;
}

int AllReechFromAscii_main(int argc,char** argv)
{
	std::string aImNamePat;
    std::string aASCIINamePat;
    cElemAppliSetFile aEASF;
	bool aShow = false;

	ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aImNamePat,"Pattern of image names to crop/rotate/scale", eSAM_IsExistFile)
                      << EAMC(aASCIINamePat,"Pattern of the ascii files with 2D correspondences", eSAM_IsExistFile),
          LArgMain()  << EAM(aShow,"Show",true,"Show computed homographies")
    );

    #if (ELISE_windows)
      replace( aASCIINamePat.begin(), aASCIINamePat.end(), '\\', '/' );
      replace( aImNamePat.begin(), aImNamePat.end(), '\\', '/' );
    #endif

	aEASF.Init(aImNamePat);

	cElRegex  anAutom(aImNamePat.c_str(),10);

	std::list<std::string>  aLCom;
    for (size_t aKIm=0  ; aKIm< aEASF.SetIm()->size() ; aKIm++)
    {
        std::string aNameIm = (*aEASF.SetIm())[aKIm];

		std::string aNameCorresp  =  MatchAndReplace(anAutom,aNameIm,aASCIINamePat);

		std::string aCom = MM3dBinFile_quotes("TestLib")
                     	 + " OneReechFromAscii "
						 + aNameIm 
						 + " " + aNameCorresp
						 + " Out=" + StdPrefix(aNameCorresp) +".tif" 
						 + " Show=" + ToString(aShow);
		std::cout << aCom << "\n";
		aLCom.push_back(aCom);
	}

	cEl_GPAO::DoComInParal(aLCom);

	return EXIT_SUCCESS;
}

int OneReechFromAscii_main(int argc,char** argv)
{
	std::string aFullName;
	std::string aOutName;
	std::string aASCIIName;
	bool aShow = false;

	Pt2dr aP1, aP2;
	Pt2di aSzOut(0,0);
    ElPackHomologue aPack;
	cElemAppliSetFile aEASF;

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aFullName,"Name of image to crop/rotate/scale", eSAM_IsExistFile)
                      << EAMC(aASCIIName,"Name of the ascii file with 2D correspondences", eSAM_IsExistFile),
          LArgMain()  << EAM(aOutName,"Out",true,"Name of the output, Def=Image+ASCII_filename.tif")
		  			  << EAM(aShow,"Show",true,"Show computed homographies")
    );

	#if (ELISE_windows)
      replace( aASCIIName.begin(), aASCIIName.end(), '\\', '/' );
      replace( aFullName.begin(), aFullName.end(), '\\', '/' );
	#endif

	aEASF.Init(aFullName);

	if (!EAMIsInit(&aOutName))
		aOutName = StdPrefix(aFullName) + "_" + StdPrefix(aASCIIName) + ".tif";




	/*1- Save Homol */
    ELISE_fp aFIn(aASCIIName.c_str(),ELISE_fp::READ);
    char * aLine;

    while ((aLine = aFIn.std_fgets()))
    {
         int aNb=sscanf(aLine,"%lf  %lf %lf  %lf",&aP1.x , &aP1.y, &aP2.x , &aP2.y);
         ELISE_ASSERT(aNb==4,"Could not read 2 double values");

         //std::cout << aP1 << " " << aP2 << "\n";
         ElCplePtsHomologues aP(aP1,aP2);

         aPack.Cple_Add(aP);

		 if (aP1.x>aSzOut.x)
		 	aSzOut.x=aP1.x;
		 if (aP1.y>aSzOut.y)
		 	aSzOut.y=aP1.y;

    }

    ElSimilitude  aSim = L2EstimSimHom(aPack);

    //  cElHomographie  aHom = cElHomographie::RansacInitH(aPack,50,2000);

	std::string aKeyHom = "NKS-Assoc-CplIm2Hom@@dat";
    std::string aNameH = aEASF.mDir + aEASF.mICNM->Assoc1To2(aKeyHom,aOutName,aFullName,true);
	aPack.StdPutInFile(aNameH);


	/*2- Save empty "transformed" image */
	GenIm::type_el aTypeOut = GenIm::u_int1;
    Tiff_Im aTifEpi = Tiff_Im
                       (
                           aOutName.c_str(),
                           aSzOut,
                           aTypeOut,
                           Tiff_Im::No_Compr,
                           Tiff_Im::BlackIsZero
                       );

	/*3- Do resampling in the transformed image */
	std::string aCom = MM3dBinFile_quotes("TestLib") 
		             + " OneReechHom " 
				     + aOutName
				     + " " + aEASF.mDir + aFullName +
				     + " " + aOutName
				     + std::string(" PostMasq=NONE ")
				     + std::string(" ScaleReech=") + ToString(euclid(aSim.sc())) + std::string(" ")
                     + "Show=" + ToString(aShow);

	System(aCom);	 



	return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
