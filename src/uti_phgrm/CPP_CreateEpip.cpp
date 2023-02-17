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


class cMapAsDist22 : public ElDistortion22_Gen 
{
    public :
       cMapAsDist22(cElMap2D *,cElMap2D * aMapInv=nullptr);
       Pt2dr Direct(Pt2dr) const  override ;    //

    private :
        bool OwnInverse(Pt2dr &) const override ;    //  return false
        
        cElMap2D * mMapDir;
        cElMap2D * mMapInv;
};

cMapAsDist22::cMapAsDist22(cElMap2D * aMap,cElMap2D * aMapInv) :
    mMapDir (aMap),
    mMapInv ((aMapInv!=nullptr) ? aMapInv : mMapDir->Map2DInverse())
{
}

Pt2dr cMapAsDist22::Direct(Pt2dr aPt) const  
{
   return (*mMapDir)(aPt);
}

bool cMapAsDist22::OwnInverse(Pt2dr & aPt) const 
{
   aPt = (*mMapInv)(aPt);
   return true;
}

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
      Box2dr  CalcBoxIm(cBasicGeomCap3D *);



      int    mDegre;
      bool   mXFitHom;
      bool   mXFitModele;
      bool   mXFitL2;
      bool   mXFit;
      std::vector<int>  mVecIterDeg; 
      double  mPropIterDeg; 
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
      std::string mNameHom;
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
      bool               mGenereImageDirEpip;
      bool               mIntervZIsDef;
      double             mZMin;
      double             mZMax;
      double             mIProf;
      Pt2dr              mIntZ;             
      std::string        mNameOut;
      std::vector<double> mParamEICE;

      Box2dr              mBoxTerrain;  // A box in terrain to limit computation, done for micmac manager
      std::vector<Pt3dr>  mPtsLimTerr;  // Corner  in 3d
      /*
      Box2dr              mBoxIm1;      // Full image 1 or image of Boxterain in any
      Box2dr              mBoxIm2;      // Full image 2 or image of Boxterain in any
      */


      double SignedComputeEpipolarability(const Pt3dr & aP1) const;
      double AbsComputeEpipolarability(const Pt3dr & aP1) const;
      Pt3dr Tang(bool IsI1,const Pt3dr & aP1,double & aDist) const;

      void DoEpipGen(bool DoIm);
      void DoOh(Pt2dr aC1,const ElPackHomologue & aPackCheck,CpleEpipolaireCoord * aChekEpi);
      Pt2dr  OhNextPOnCurvE(const Pt2dr aP1,bool Im1,bool Forward);

      Pt2dr DirEpipIm2(cBasicGeomCap3D * aG1,cBasicGeomCap3D * aG2,ElPackHomologue & aPack,bool ForCheck,bool AddToP1, std::list<Appar23> &   aL23,bool OnlyZCentr=false);

      void Ressample(cBasicGeomCap3D * aG1,EpipolaireCoordinate & E1,double aStep);

      void MakeAppuis(bool Is1,const std::list<Appar23>&aLT1,EpipolaireCoordinate & anEpi,Pt2dr aP0Epi,Pt2dr aP1Epi);

      void   ExportImCurveEpip(ElDistortion22_Gen & e1,const Box2dr& aBoxIn1,const Box2dr&,const std::string & aName,ElDistortion22_Gen & e2,const Box2dr& aBoxIn2,double aX2mX1);
      double PhaseValue(const double & aV,const double & aSzY) const;


      std::vector<double>  mParamTestOh;


};


Box2dr    cApply_CreateEpip_main::CalcBoxIm(cBasicGeomCap3D * aG)
{
    Pt2dr aSz =  Pt2dr(aG->SzBasicCapt3D());

    Box2dr aBoxFull(Pt2dr(-1,-1),aSz+Pt2dr(1,1));

    if (EAMIsInit(&mBoxTerrain))
    {
         Pt2dr aPMin(1e20,1e20);
         Pt2dr aPMax(-1e20,-1e20);
	 for (const auto & aPTer : mPtsLimTerr)
	 {
              Pt2dr aPIm = aG->Ter2Capteur(aPTer);
              aPMin = Inf(aPMin,aPIm);
              aPMax = Sup(aPMax,aPIm);
	 }

	 aPMin = Sup(aPMin,aBoxFull.P0());
	 aPMax = Inf(aPMax,aBoxFull.P1());

         ELISE_ASSERT((aPMin.x<aPMax.x),"Proj of Box terrain dont intersect box image");
         ELISE_ASSERT((aPMin.y<aPMax.y),"Proj of Box terrain dont intersect box image");
	 
	 return Box2dr(aPMin,aPMax);
    }

    return aBoxFull;
}




static  const  double aDZ = 1; // "Small" value to compute derivative
Pt3dr  cApply_CreateEpip_main::Tang(bool IsI1,const Pt3dr & aPTer,double & aDist) const
{

    // Swap G1/G2
    cBasicGeomCap3D *  aG1 = IsI1 ?  mGenI1 : mGenI2; 
    cBasicGeomCap3D *  aG2 = IsI1 ?  mGenI2 : mGenI1;

    Pt2dr aPIm1 = aG1->Ter2Capteur(aPTer);  // Proj in I1
    Pt3dr aOTer = aG1->ImEtZ2Terrain(aPIm1,aPTer.z - aDZ);  // Point allong bundle
    Pt3dr aQTer = aG1->ImEtZ2Terrain(aPIm1,aPTer.z + aDZ);  // Point allong bundle

    Pt2dr aOIm2 = aG2->Ter2Capteur(aOTer); 
    Pt2dr aQIm2 = aG2->Ter2Capteur(aQTer);

    aDist = euclid (aQIm2-aOIm2)/(2*aDZ); // Compute sensibility to dist

// std::cout << "OooQQqq : " << aOTer << aQTer << (aQTer-aOTer) / (2*aDZ) << "\n";
    return (aQTer-aOTer) / (2*aDZ);
}

double  cApply_CreateEpip_main::SignedComputeEpipolarability(const Pt3dr & aPTer) const
{
    // double aDZ = 1e-2; // "Small" value to compute derivative
    double aD1, aD2,aD1P,aD2P;

    Pt3dr aT1 = Tang(true ,aPTer,aD1);
    Pt3dr aT2 = Tang(false,aPTer,aD2);

    Pt3dr aT1Plus = Tang(true ,aPTer+aT2*aDZ,aD1P);
    Pt3dr aT1Minus = Tang(true,aPTer-aT2*aDZ,aD1P);

    Pt3dr aT2Plus =  Tang(false,aPTer+aT1*aDZ,aD2P);
    Pt3dr aT2Minus = Tang(false,aPTer-aT1*aDZ,aD2P);

    // std::cout  << "DIFF " <<  (aD1-aD1P)/(aD1+aD1P)  << " " << (aD2-aD2P)/(aD2+aD2P) << " " << aD1 << " " << aD2<< "\n";

    Pt3dr  aDerT1 = (aT1Plus-aT1Minus) / (2*aDZ);
    Pt3dr  aDerT2 = (aT2Plus-aT2Minus) / (2*aDZ);
    // Pt3dr  aDifDer = (aDerT1-aDerT2) / (aD1*aD2);
    Pt3dr  aDifDer = (aDerT1-aDerT2) / (euclid(aDerT1)+euclid(aDerT2));


    Pt3dr  aVect = vunit(aT1^aT2);

    double aRes = scal(aVect,aDifDer);


// std::cout << "Ttt " << aT1 << aT2 << "\n";
// std::cout << aDerT1 << aDerT2 << aDifDer  << " RRR " << aRes << "\n"; // getchar();
    //  std::cout << "RRRRR " << aRes << "\n";
    return aRes;
}

double  cApply_CreateEpip_main::AbsComputeEpipolarability(const Pt3dr & aPTer) const
{
   return ElAbs(SignedComputeEpipolarability(aPTer));
}

/*
*/

void  cApply_CreateEpip_main::IntervZAddIm(cBasicGeomCap3D * aGeom)
{
    if (aGeom->AltisSolMinMaxIsDef())
    {
        Pt2dr aPZ =aGeom->GetAltiSolMinMax();
        if (mIntervZIsDef)
        {
           // mZMin= ElMin(mZMin,aPZ.x);
           // mZMax= ElMax(mZMax,aPZ.y);

           // Modif MPD : sinon interv Z par union depasse 
           mZMin= ElMax(mZMin,aPZ.x);
           mZMax= ElMin(mZMax,aPZ.y);
           ELISE_ASSERT(mZMin<mZMax,"Empty Z Intervall");
        }
        else
        {
           mZMin= aPZ.x;
           mZMax= aPZ.y;
        }
        mIntervZIsDef = true;
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
            std::list<Appar23> &   aL23,
	    bool                   ForXFitHom
       )
{
std::cout << "XXXX  " << aG1->AltisSolIsDef() << " " <<  ForXFitHom << "\n";
    Box2dr  aBoxIm1 =    CalcBoxIm(aG1);
    Box2dr  aBoxIm2 =    CalcBoxIm(aG2);

    bool DoCompEp = ForCheck;
    cElStatErreur aStatEp(10);
   
    Pt2dr aSz =  Pt2dr(aG1->SzBasicCapt3D());
 

    Pt2dr aSomDir(0,0);
    // On le met tres petit, ce qui a priori n'est pas genant et evite 
    // d'avoir des point hors zone
    //  GetVeryRoughInterProf est une proportion

    if (!EAMIsInit(&mIProf))
       mIProf = aG1->GetVeryRoughInterProf() / 100.0;


    // aEps avoid points to be too close to limits
    double aEps = 5e-4;

    // Comput size of grid that will give mNbbXY ^2 points
    double aLenghtSquare = ElMin(mLengthMin,sqrt((aSz.x*aSz.y) / (mNbXY*mNbXY)));


    // Assure it give a sufficient reduduncy
    int aNbX = ElMax(1+3*mDegre,round_up(aSz.x /aLenghtSquare));
    int aNbY = ElMax(1+3*mDegre,round_up(aSz.y /aLenghtSquare));


     
    Im2D<float,double> aImEpipAbs(aNbX+1,aNbY+1);
    Im2D<float,double> aImEpipSigned(aNbX+1,aNbY+1);

    int aNbTens=0;
    bool aPbDirNonStable = false;
    Im2D<float,double> aImAng(aNbX+1,aNbY+1);
    TIm2D<float,double> aTImAng(aImAng);
    for (int aKX=0 ; aKX<= aNbX ; aKX++)
    {
        // Barrycentrik weighting, 
        double aPdsX = ForCheck ? NRrandom3() :  (aKX /double(aNbX));
        aPdsX = ElMax(aEps,ElMin(1-aEps,aPdsX));
        for (int aKY=0 ; aKY<= aNbY ; aKY++)
        {
            aTImAng.oset(Pt2di(aKX,aKY),0.0);
            // Barrycentrik weighting, 
            double aPdsY = ForCheck ? NRrandom3() : (aKY/double(aNbY));
            aPdsY = ElMax(aEps,ElMin(1-aEps,aPdsY));
            // Point in image 1 on regular gris

            // Pt2dr aPIm1 = aSz.mcbyc(Pt2dr(aPdsX,aPdsY));
            Pt2dr aPIm1 = aBoxIm1.FromCoordLoc(Pt2dr(aPdsX,aPdsY));
            if (aG1->CaptHasData(aPIm1))
            {
                Pt3dr aPT1;
                Pt3dr aC1;
                // Compute bundle with origin on pseudo center
                aG1->GetCenterAndPTerOnBundle(aC1,aPT1,aPIm1);

                 //std::cout << "IPROF " << aIProf * euclid(aPT1-aC1)  << " " << aPT1  << "\n";

                std::vector<Pt2dr> aVPIm2;
                int aNbZSup = mNbZ + mNbZRand; 
                int aNbZInf = -mNbZ;
		if (ForXFitHom)
		{
                    aNbZSup =0 ;
                    aNbZInf =0 ;
		}
                for (int aKZ = aNbZInf ; aKZ <= aNbZSup  ; aKZ++)
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
                           // To avoid border of interval and pb with PIsVisibleInImage
                          double aEps = 1e-3;
                          aPds = ElMax(aEps,ElMin(1-aEps,aPds));
			  double aZ = mZMin*aPds + mZMax * (1-aPds);
                          aPT2 = aG1->ImEtZ2Terrain(aPIm1,aZ);
                     }
		     else if (aG1->AltisSolIsDef() && ForXFitHom)
		     {
                          aPT2 = aG1->ImEtZ2Terrain(aPIm1,aG1->GetAltiSol());
		     }
                     else
                     {
                          double aPds = aKZ / double(mNbZ);
                          if (isZRand)  // if additionnal add a random
                              aPds = NRrandC();
                          aPT2 = aC1 + (aPT1-aC1) * (1+mIProf*aPds);
                     }

                     if (aG1->PIsVisibleInImage(aPT2) && aG2->PIsVisibleInImage(aPT2) )
                     {
                        // Add projection
                        Pt2dr aPIm2 = aG2->Ter2Capteur(aPT2);
                        if (aG2->CaptHasData(aPIm2) &&  aBoxIm2.inside(aPIm2))
                        {
                            if (DoCompEp && (aKZ==0))
                            {
                                double aSignEpip = SignedComputeEpipolarability(aPT2);
                                double aAbsEpip = std::abs(aSignEpip);
                                aStatEp.AddErreur(aAbsEpip);
                                aImEpipAbs.SetR(Pt2di(aKX,aKY),aAbsEpip*1e4);
                                aImEpipSigned.SetR(Pt2di(aKX,aKY),aSignEpip*1e4);
                            }
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
                    aSomDir = aSomDir +  aDir2; 
                    //aSomTens2 = aSomTens2 + aDir2 * aDir2; // On double l'angle pour en faire un tenseur
		    if  (scal(aSomDir,aDir2)<0)
                        aPbDirNonStable = true;

		    Pt2dr aRhoTeta = Pt2dr::polar(aDir2,0.0);
		    aTImAng.oset(Pt2di(aKX,aKY),aRhoTeta.y);
		    // std::cout << "ooooDDDDD " << aDir2  << aRhoTeta << "\n";
                    aNbTens++;
                }
            }
        }
    }
    aPbDirNonStable = aPbDirNonStable && (!(ForCheck || ForXFitHom));
    if (aPbDirNonStable || (mGenereImageDirEpip && (!(ForCheck || ForXFitHom))))
    {
        std::string aNameImDir = std::string("Epip-ImageDirEpip-Im") + (AddToP1 ? "1" : "2") + std::string(".tif");
	Tiff_Im::CreateFromIm(aImAng,aNameImDir);
	/* Tiff_Im Create8BFromFonc ( its_to_rgb i       ); */
    }
    if (aPbDirNonStable)
    {
        ELISE_ASSERT(false,"Incoherent direction in dirs estimate");
    }
    if (!(ForCheck || ForXFitHom))
    {
       std::cout << "Number points for tensor " << aNbTens << "\n";
       ELISE_ASSERT(aNbTens!=0,"Cannot compute direction no valide point");
    }
    if (DoCompEp)
    {
       std::cout << "====================  Epipolarability ============================ \n";
       std::cout << " Avg=" << aStatEp.Avg() << " Med=" << aStatEp.Erreur(0.5) << " Ect=" << aStatEp.Ect() << "\n";
       std::cout << " Min=" << aStatEp.Erreur(0.0) << " Max=" << aStatEp.Erreur(1.0) << "\n";
       Tiff_Im::CreateFromIm(aImEpipAbs,"EpipolaribityAbs.tif");
    }
    // 4 check there was no point computed
    if (ForCheck || ForXFitHom) 
       return aSomDir;
    if (0)
    {
	    std::cout << "aSomDir BRUTE " << aSomDir <<  " P1:" << AddToP1 << "\n";
	    getchar();
    }
    return vunit(aSomDir) * (AddToP1 ? 1.0:-1.0);  // There is an inversion 
    /*
    Pt2dr aRT = Pt2dr::polar(aSomTens2,0.0);
    return Pt2dr::FromPolar(1.0,aRT.y/2.0); // Divide angle as it was multiplied
    */
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
                double aEpsChekInv = 1e-2, // Check accuracy of inverse,
		const ElPackHomologue * aPackCheck = nullptr,
                const ElDistortion22_Gen * aEpiCheck=nullptr
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
        double aEpsCheckInv,
        const ElPackHomologue * aPackCheck,
        const ElDistortion22_Gen * aEpiCheck
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


    // Calcul de l'interpolateur
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

    // Si aPackCheck, calcul des statistique de px sur X et Y apres correction
    if (aPackCheck && aPackCheck->size())
    {
        {
            EpipolaireCoordinate * anEE = static_cast<EpipolaireCoordinate *> (anEpi);
            std::cout << "HHHaass " << anEE->HasXFitHom() << "\n";
	    if (anEE->HasXFitHom())
               std::cout << "   XFITH  " << anEE->ParamFitHom() << "\n";
        }

	double aDXD=0.0;
	double aDYD=0.0;
	double aDI1=0.0;
	double aDI2=0.0;
	for (const auto & aCple : *aPackCheck)
	{
            Pt2dr aP1 = aCple.P1();
            Pt2dr aP2 = aCple.P2();
            Pt2dr aQ1 = anEpi->Direct(aP1);
            Pt2dr aQ2 = aEpiCheck->Direct(aP2);
	    aDXD += ElAbs(aQ1.x-aQ2.x);
	    aDYD += ElAbs(aQ1.y-aQ2.y);

            Pt2dr aI1 = anEpi->Inverse(aQ1);
            Pt2dr aI2 = aEpiCheck->Inverse(aQ2);
	    aDI1 += euclid(aP1-aI1);
	    aDI2 += euclid(aP2-aI2);


	}
	int aNb = aPackCheck->size();
        std::cout << "COH  x:" <<aDXD/aNb  << " y:" << aDYD/aNb << "\n";
        std::cout << "    i1:" <<aDI1/aNb  << " i2:" << aDI2/aNb << "\n";
    }


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
        ElDistortion22_Gen & anEpip1,const Box2dr& aBoxIn1,const Box2dr& aBoxOut1,
        const std::string & aName,
        ElDistortion22_Gen & anEpip2,const Box2dr& aBoxIn2,
        double aX2mX1
     )
{

    ElDistortion22_Gen &  aDist1 = anEpip1;
    ElDistortion22_Gen &  aDist2 = anEpip2;


    ELISE_ASSERT(mParamEICE.size()==5,"Bad Size for ExpCurve");
    bool ShowOut = mParamEICE.at(4)>0.5;
    double aLRed =  mParamEICE.at(0);
    SetExagEpip(mParamEICE.at(3),true);

    Pt2dr aSzIn = aBoxIn1.sz();
    double aScale = ElMax(aSzIn.x,aSzIn.y) / aLRed;
    Pt2di aSzRed = round_ni(aSzIn / aScale);

    Pt2di aP;
    Im2D_U_INT1 aIm(aSzRed.x,aSzRed.y);

    Pt2dr aSumGx(0,0),aSumGy(0,0);
    double aSumPG=0;
    for (aP.x=0 ; aP.x<aSzRed.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<aSzRed.y ; aP.y++)
        {
            Pt2dr aPIm1  = aBoxIn1.P0() + Pt2dr(aP) * aScale;

            Pt2dr aPEpi1 = aDist1.Direct(aPIm1);
            double aVy = PhaseValue(aPEpi1.y,aBoxOut1.sz().y);
            
            double aVal = 1-aVy;
            {
                 // Pt2dr aPIm00  = (aBoxIn1.P0() +aBoxIn1.P1())/2.0;
                 Pt2dr aPIm00  = aPIm1;
                 Pt2dr aPIm10  = aPIm00 + Pt2dr(1,0);
                 Pt2dr aPIm01  = aPIm00 + Pt2dr(0,1);

                 Pt2dr aGx = aDist1.Direct(aPIm10) -aDist1.Direct(aPIm00);
                 Pt2dr aGy = aDist1.Direct(aPIm01) -aDist1.Direct(aPIm00);
                 aSumGx = aSumGx + aGx;
                 aSumGy = aSumGy + aGy;
                 aSumPG++;
            }

            if (ShowOut)
            {
                Pt2dr aPEpi2 (aPEpi1.x+aX2mX1 ,aPEpi1.y);
                Pt2dr aPIm2 =  aDist2.Inverse(aPEpi2);

                if (!aBoxIn2.inside(aPIm2))
                {
                    aVal = 1 -aVal;
                }
           }

            aIm.SetR(aP,255*aVal);
        }
    }
    aSumGx = aSumGx / aSumPG;
    aSumGy = aSumGy / aSumPG;
    SetExagEpip(1.0,false);

    std::cout << "================== Create Im Curves =====================\n";
    std::cout << " Sc=" <<  aScale
              << " Grad " << aSumGx  << aSumGy  
              << " Nx:" << euclid(aSumGx) << " Ny:" << euclid(aSumGy)
              << " BOWY " << aBoxOut1.sz().y
              << "\n";
    Tiff_Im::CreateFromIm(aIm,aName);
}


Pt2dr  cApply_CreateEpip_main::OhNextPOnCurvE(const Pt2dr aPIm1,bool Im1,bool Forward)
{
  cBasicGeomCap3D *  aG1 =  mGenI1 ;
  cBasicGeomCap3D *  aG2 =  mGenI2 ;
  double     aZ1 = mZMin;
  double     aZ2 = mZMax;

  if (!Im1)
     ElSwap(aG1,aG2);

  if (Im1!=Forward)  // a priori pour que la variation soit ds le meme sens il faut intervertir pour Im2
     ElSwap(aZ1,aZ2);


  Pt3dr aPTer1 = aG1->ImEtZ2Terrain(aPIm1,aZ1);
  Pt2dr aPIm2  = aG2->Ter2Capteur(aPTer1);
  Pt3dr aPTer2 = aG2->ImEtZ2Terrain(aPIm2,aZ2);
  Pt2dr aQIm1  = aG1->Ter2Capteur(aPTer2);

  return aQIm1;
}

class cOhEpipPt
{
     public :
          cOhEpipPt(const Pt2dr & aP1,const Pt2dr & aP2,const Pt2dr & aPE) :
             mP1 (aP1) ,
             mP2 (aP2) ,
             mPE (aPE) 
           {
           }
 
           Pt2dr mP1;
           Pt2dr mP2;
           Pt2dr mPE;

     private :
};


void cApply_CreateEpip_main::DoOh(Pt2dr aC1,const ElPackHomologue & aPackCheck,CpleEpipolaireCoord * aChekEpi)
{
     Pt2dr aInfIn1(1e20,1e20),aSupIn1(-1e20,-1e20);
     Pt2dr aInfIn2(1e20,1e20),aSupIn2(-1e20,-1e20);
     Pt2dr aInfOut(1e20,1e20),aSupOut(-1e20,-1e20);

     // Calcul centre Image 2
     ELISE_ASSERT (mIntervZIsDef,"No IntervZ in Oh test");
     Box2dr aBox1 (Pt2dr(0,0),Pt2dr(mGenI1->SzBasicCapt3D()));
     Box2dr aBox2 (Pt2dr(0,0),Pt2dr(mGenI2->SzBasicCapt3D()));
     // Pt2dr aBox2 (Pt2dr(0,0),Pt2dr(mGenI2->SzBasicCapt3D()));


     Pt2dr   aPForw = OhNextPOnCurvE(aC1,true,true);
     Pt2dr   aPBckw = OhNextPOnCurvE(aC1,true,false);
     double aDist = euclid(aPForw,aPBckw)/2.0;

     Pt2dr  aDirEpi1 = vunit(aPForw-aPBckw) ;
     Pt2dr  aDirOrth = aDirEpi1  * Pt2dr(0,1) * aDist;

     // Calcul des point "centraux" debut des courves epip
     // parcourt la direction ortho a droit et a gauche
     std::vector<cOhEpipPt>  aVSeedOh; // seed of epipolar point
     for (int aSens=-1 ; aSens<=1 ; aSens+=2)
     {
           int aK0 = (aSens>=0) ? 0 : -1;
           Pt2dr aPIm1 = aC1 + aDirOrth*double(aK0);
           while (aBox1.inside(aPIm1))
           {
               Pt3dr aPTer = mGenI1->ImEtZ2Terrain(aPIm1,(mZMin+mZMax)/2.0);
               Pt2dr  aPIm2 =  mGenI2->Ter2Capteur(aPTer);
               if (aBox2.inside(aPIm2))
               {
                    aVSeedOh.push_back(cOhEpipPt(aPIm1,aPIm2,Pt2dr(0,aDist*aK0)));
               }
               aPIm1 = aPIm1 + aDirOrth*double(aSens);
               aK0 += aSens;
           }
     }

     // std::cout << "NbSeedOh=" <<  aVSeedOh.size() << "\n";
     ELISE_ASSERT(!aVSeedOh.empty(),"Empty Oh seed");

     // Calcul des courbes epipolaires et memo des corresp 
     double aMaxDif=0.0;  // Chekc that point are homolologues with MicMac epip
     double aSomDif=0.0;
     double aNbDif = 0;
     std::vector<cOhEpipPt>  aVPtsOh; // all epipolar point
     ElPackHomologue  aPackIm1;  // Corresp  Im1 -> Epi1
     ElPackHomologue  aPackIm2;
     for (const auto & aPOh : aVSeedOh)
     {
         for (int aSens=-1 ; aSens<=1 ; aSens+=2)
         {
             int aK = 0;
             Pt2dr aPIm1 = aPOh.mP1;
             Pt2dr aPIm2 = aPOh.mP2;
             Pt2dr aPE  = aPOh.mPE;
             bool goForW = (aSens>=0);
             while (aBox1.inside(aPIm1) && aBox2.inside(aPIm2))
             {
                  if ((aK!=0) || (goForW))
                  {
                       aVPtsOh.push_back(cOhEpipPt(aPIm1,aPIm2,aPE));
                       {
                           Pt2dr aPC1  = aChekEpi->EPI1().Direct(aPIm1);
                           Pt2dr aPC2  = aChekEpi->EPI2().Direct(aPIm2);
                           double aDif = ElAbs(aPC1.y-aPC2.y);
                           aMaxDif = ElMax(aDif,aMaxDif);
                           aSomDif += aDif;
                           aNbDif ++;
                       }
                       aInfIn1 = Inf(aInfIn1,aPIm1);
                       aInfIn2 = Inf(aInfIn2,aPIm2);
                       aInfOut = Inf(aInfOut,aPE);
                       aSupIn1 = Sup(aSupIn1,aPIm1);
                       aSupIn2 = Sup(aSupIn2,aPIm2);
                       aSupOut = Sup(aSupOut,aPE);

                       aPackIm1.Cple_Add(ElCplePtsHomologues (aPIm1,aPE));
                       aPackIm2.Cple_Add(ElCplePtsHomologues (aPIm2,aPE));
                  }

                  aPIm1  = OhNextPOnCurvE(aPIm1,true,goForW);
                  aPIm2  = OhNextPOnCurvE(aPIm2,false,goForW);
                  aPE.x += aSens * aDist;
                  aK    += aSens;
             }
         }
     }
     std::cout << "===============================Oh================  " << "\n";
     std::cout << "cohernce hom  "
               << " Avg=" << aSomDif/aNbDif 
               << " Max:" << aMaxDif 
               << " Zs: " << mZMin << " " << mZMax << "\n";

     int aDeg = round_ni(mParamTestOh.at(0));

     cElMap2D *  aMap1 = MapPolFromHom(aPackIm1,aBox1,aDeg,0);
     cElMap2D *  aMap2 = MapPolFromHom(aPackIm2,aBox2,aDeg,0);

     aMaxDif=0.0;  // Chekc that point are homolologues with MicMac epip
     aSomDif=0.0;
     aNbDif = 0;


     for (const auto & aCple :  aPackCheck)
     {
         Pt2dr aPE1 = (*aMap1)(aCple.P1());
         Pt2dr aPE2 = (*aMap2)(aCple.P2());
         double aDif = ElAbs(aPE1.y - aPE2.y);
         aMaxDif = ElMax(aDif,aMaxDif);
         aSomDif += aDif;
         aNbDif ++;
     }
     std::cout << "TestEpipOh : "
               << " Avg=" << aSomDif/aNbDif 
               << " Max:" << aMaxDif 
               << " \n";
     
   if (EAMIsInit(&mParamEICE))
   {
     Box2dr  aBIn1(aInfIn1,aSupIn1);
     Box2dr  aBIn2(aInfIn2,aSupIn2);
     Box2dr  aBOut(aInfOut,aSupOut);
     cMapAsDist22 aDist1(aMap1);
     cMapAsDist22 aDist2(aMap2);
     ExportImCurveEpip(aDist1,aBIn1,aBOut,"OhImLineEpip1.tif",aDist2,aBIn2,0);
     ExportImCurveEpip(aDist2,aBIn2,aBOut,"OhImLineEpip2.tif",aDist1,aBIn1,0);
   }
}

//cElMap2D *  MapPolFromHom(const ElPackHomologue & aPack,const Box2dr & aBox,int aDeg,int aRabDegInv);


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

	 // If we are alredy almost in epip, we dont want a 180 deg rotation
	 if ((mDir2.x+mDir1.x) <0)
         {
             mDir1 = -mDir1;
             mDir2 = -mDir2;
	 }
      }
      else
      {
          // MPD : c'est la qu'on intervient pour le calcul du modele epipolaiez
          aPack = mICNM->StdPackHomol(mNameHom,mName1,mName2);
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

      ElPackHomologue aPackXFitH;
      if (mXFitHom)
      {
	 aPackXFitH = mICNM->StdPackHomol(mNameHom,mName1,mName2);
      }
      else if (mXFitModele)
      {
          std::list<Appar23>  aLTCheck;
          DirEpipIm2(mGenI1,mGenI2,aPackXFitH,true,true,aLTCheck,true); 

	      /*std::cout << "mXFitModelemXFitModelemXFitModelemXFitModele \n"; 
	 aPackXFitH = mICNM->StdPackHomol(mNameHom,mName1,mName2);
	      getchar();
	      */
      }


      std::cout << "Compute Epip ; D1=" << mDir1 << " ,D2=" << mDir2 << "\n";
      CpleEpipolaireCoord * aCpleEpi = nullptr;
      EpipolaireCoordinate * aEpi1 = nullptr;
      EpipolaireCoordinate * aEpi2 = nullptr;

      Pt2dr aInf1(1e20,1e20),aSup1(-1e20,-1e20);
      Pt2dr aInf2(1e20,1e20),aSup2(-1e20,-1e20);
      double aX2mX1 =0.0 ;

      bool WithCheck = (aPackCheck.size()!=0);
      int aNbTimes = mVecIterDeg.size();
      if (WithCheck)
         aNbTimes ++;

      double aSigma=0.0;
      for (int aKTime=0 ; aKTime<aNbTimes ; aKTime++)
      {
             if ((aCpleEpi==nullptr) || (!mWithOri))
             {
                int aKDeg = std::min(aKTime,int(mVecIterDeg.size())-1);
                aCpleEpi = CpleEpipolaireCoord::PolynomialFromHomologue
                                       (false,aCpleEpi,aSigma,aPack,mVecIterDeg.at(aKDeg),mDir1,mDir2,mDegSupInv);
                aEpi1 = &(aCpleEpi->EPI1());
                aEpi2 = &(aCpleEpi->EPI2());
                if (mXFit)
                {
                   aEpi1->XFitHom(aPackXFitH,mXFitL2,aEpi2);
                }
             }
            // For check at first iter :
            //    * necessary because many value to export at last
            //    * possible as called only with geom model with no iteration, no need to refine
            bool ForCheck = (aKTime==0) && WithCheck ;
            aX2mX1 = 0.0;
            aInf1=Pt2dr(1e20,1e20);
            aSup1=Pt2dr(-1e20,-1e20);
            aInf2=Pt2dr(1e20,1e20);
            aSup2=Pt2dr(-1e20,-1e20);

            ElPackHomologue * aPackK = ForCheck ? &aPackCheck  : & aPack;

            double aBias = 0.0;
            double aErrMax = 0.0;
            double aErrMoy = 0.0;
            int    mNbP = 0;

            Pt2dr  aCIm1(0,0);

            // Compute accuracy, bounding box 
            std::vector<double> aVResid;
            for (ElPackHomologue::const_iterator itC=aPackK->begin() ; itC!= aPackK->end() ; itC++)
            {
                 aCIm1 = aCIm1 + itC->P1() ;
                 // Images of P1 and P2 by epipolar transforms
                 Pt2dr aP1 = aEpi1->Direct(itC->P1());
                 Pt2dr aP2 = aEpi2->Direct(itC->P2());
                 // Update bounding boxes
                 aInf1 = Inf(aInf1,aP1);
                 aSup1 = Sup(aSup1,aP1);
                 aInf2 = Inf(aInf2,aP2);
                 aSup2 = Sup(aSup2,aP2);

                 // Average off of X
                 aX2mX1 += aP2.x - aP1.x;

                 double aDifY = aP1.y-aP2.y; // Should be 0
                 double anErr = ElAbs(aDifY);
                 aVResid.push_back(anErr);
                 mNbP++;
                 aErrMax = ElMax(anErr,aErrMax);
                 aErrMoy += anErr;
                 aBias   += aDifY;
            }
            aSigma = KthValProp(aVResid,mPropIterDeg);

            aX2mX1 /= mNbP;
            aCIm1 = aCIm1/mNbP;

            double aInfY = ElMax(aInf1.y,aInf2.y);
            double aSupY = ElMax(aSup1.y,aSup2.y);
            aInf1.y = aInf2.y = aInfY;
            aSup1.y = aSup2.y = aSupY;
	    if (mXFit)
	    {
                double aInfX = ElMax(aInf1.x,aInf2.x);
                double aSupX = ElMax(aSup1.x,aSup2.x);
                aInf1.x = aInf2.x = aInfX;
                aSup1.x = aSup2.x = aSupX;
	    }


            std::cout  << "======================= " << (ForCheck ? " CONTROL" : "LEARNING DATA") << " ========\n";
            std::cout << "Epip Rect Accuracy:" 
                      << " Bias " << aBias/mNbP 
                      << " ,Moy " <<  aErrMoy/mNbP 
                      << " ,Max " <<  aErrMax 
                      << " Center:" << aCIm1 << " NbC:" << mNbP
                      << "\n";

            std::cout <<  "Prop/Err : ";
            for (const auto & aP : {0.5,0.75,0.85,0.95})
            {
                 std::cout <<  "["  << aP << " -> " <<  KthValProp(aVResid,aP)  << "]";
            }
            std::cout <<  "\n";

            if (! ForCheck)
            {
                std::cout << "DIR " << mDir1 << " " << mDir2 << " X2-X1 " << aX2mX1<< "\n";
                std::cout << "Epip NbPts= " << mNbP << " Redund=" << mNbP/double(ElSquare(mDegre)) << "\n";
            }
            if (EAMIsInit(&mParamTestOh) && (!ForCheck))
            {
                 DoOh(aCIm1,aPackCheck,aCpleEpi);
            }


      }
      // Save the result of epipolar in (relatively) raw format, containg polynoms+ name of orent
      //  in case we want to do a Cap3D=> nuage from them
      if (1)
      {
          aCpleEpi->SaveOrientCpleEpip(mOri,mICNM,mName1,mName2);
      }
      std::cout  << "===================================\n";
      std::cout  << "BOX1 " << aInf1 << " " <<  aSup1 <<  " SZ=" << aSup1-aInf1<< "\n";
      std::cout  << "BOX2 " << aInf2 << " " <<  aSup2 <<  " SZ=" << aSup2-aInf2<< "\n";


      bool aConsChan = true;
      Pt2di aSzI1 = mWithOri ? 
                    mGenI1->SzBasicCapt3D() : 
                    Tiff_Im::StdConvGen(mName1.c_str(),aConsChan ? -1 :1 ,true).sz() ;
      Pt2di aSzI2 = mWithOri ? 
                    mGenI2->SzBasicCapt3D() : 
                    Tiff_Im::StdConvGen(mName2.c_str(),aConsChan ? -1 :1 ,true).sz() ;

      std::string aNI1 = mICNM->NameImEpip(mOri,mName1,mName2);
      std::string aNI2 = mICNM->NameImEpip(mOri,mName2,mName1);
      if (EAMIsInit(&mNameOut))
      {
          aNI1 = mNameOut + "_1.tif";
          aNI2 = mNameOut + "_2.tif";
      }

      Box2dr aBIn1(Pt2dr(0,0),Pt2dr(aSzI1));
      Box2dr aBOut1(aInf1,aSup1);
      Box2dr aBIn2(Pt2dr(0,0),Pt2dr(aSzI2));
      Box2dr aBOut2(aInf2,aSup2);

      if (DoIm)
      {
{
	std::cout << "HAS XFIT H " << aEpi1->HasXFitHom() << " " << aEpi2->HasXFitHom() << "\n";
}
         cTmpReechEpip aReech1(aConsChan,mName1,aBIn1,aEpi1,aBOut1,mStepReech,aNI1,mPostMasq,mNumKer,mDebug,mNbBloc,mEpsCheckInv,&aPackXFitH,aEpi2);
         std::cout << "DONE IM1 \n";
         ElPackHomologue aPSym = aPackXFitH;
	 aPSym.SelfSwap();
         cTmpReechEpip aReech2(aConsChan,mName2,aBIn2,aEpi2,aBOut2,mStepReech,aNI2,mPostMasq,mNumKer,mDebug,mNbBloc,mEpsCheckInv,&aPSym,aEpi1);
         std::cout << "DONE IM2 \n";

         std::cout << "DONNE REECH TMP \n";
      }

//  ::cTmpReechEpip(cBasicGeomCap3D * aGeom,EpipolaireCoordinate * anEpi,Box2dr aBox,double aStep) :

      if (mMakeAppuis)
      {
           MakeAppuis(true ,aLT1,*aEpi1,aInf1,aSup1);
           MakeAppuis(false,aLT2,*aEpi2,aInf2,aSup2);
      }

      if (EAMIsInit(&mParamEICE))
      {
         ExportImCurveEpip(*aEpi1,aBIn1,aBOut1,"ImLineEpip1.tif",*aEpi2,aBIn2,aX2mX1);
         ExportImCurveEpip(*aEpi2,aBIn2,aBOut2,"ImLineEpip2.tif",*aEpi1,aBIn1,-aX2mX1);
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
    if (EAMIsInit(&mNameOut))
       aNameAp =  "Appuis_" + mNameOut + (Is1 ? "_1.xml": "_2.xml" );


    cListeAppuis1Im  aLAp =  El2Xml(aLCor,aN1);
    MakeFileXML(aLAp,aNameAp);
    std::cout << "======  MakeAppuis ====== " << aInfEpi  << " :: " << aSupEpi -aSzEpi << "\n";//  getchar();
}



    // std::string  aNameEpi = Is1?"ImEpi1.tif":"ImEpi2.tif";



cApply_CreateEpip_main::cApply_CreateEpip_main(int argc,char ** argv) :
   mDegre     (-1),
   mXFitHom   (false),
   mXFitModele  (false),
   mXFitL2      (false),
   mPropIterDeg  (0.85),
   mNbZ       (2),  // One more precaution ...
   mNbXY      (100),
   mNbZRand   (1),
   mForceGen  (false),
   mNumKer    (5),
   mDebug     (false),
   mPostMasq  (""),
   mNameHom   (""),
   mGenI1     (0),
   mGenI2     (0),
   mWithOri   (true),
   mNbBloc    (2000),
   mDegSupInv (4),
   mEpsCheckInv (1e-1),
   mMakeAppuis  (false),
   mGenereImageDirEpip (false),
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
    //std::string aNameHom;
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
                    << EAM(mNameHom,"NameH",true,"Extension to compute Hom point in epi coord (def=none)", eSAM_NoInit)
                    << EAM(mDegre,"Degre",true,"Degre of polynom to correct epi (def=9)")
                    << EAM(mVecIterDeg,"VecIterDeg",true,"Vector of degree in case of iterative approach")
                    << EAM(mPropIterDeg,"PropIterDeg",true,"Prop to compute sigma in cas of iterative degre")
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
		    << EAM(mParamTestOh,"OhP",false,"Oh's method test parameter(none for now)")
		    << EAM(mXFitHom,"XCorrecHom",false,"Correct X-Pax using homologous point")
		    << EAM(mXFitModele,"XCorrecOri",false,"Correct X-Pax using orient and Z=average")
		    << EAM(mXFitL2,"XCorrecL2",false,"L1/L2 Correction for X-Pax")
		    << EAM(mNameOut,"Out",false,"To spcecify names of results")
		    << EAM(mGenereImageDirEpip,"ImDir",false,"Generate image of direction of epipolar")
		    << EAM(mBoxTerrain,"BoxTerrain",false,"Box ter to limit size of created epip")
		    /*
		    */
    );


    mXFit = (mXFitHom||mXFitModele);
    if (! EAMIsInit(&mXFitL2))
      mXFitL2  = mXFitModele;


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

         if (EAMIsInit(&mVecIterDeg))
         {
            // No meaning to have only one degree
            ELISE_ASSERT(mVecIterDeg.size()>=2,"Bad size for Iter Degree");
            ELISE_ASSERT(! EAMIsInit(&mDegre),"Degree and VecIterDeg both init");
            mDegre = mVecIterDeg.back();
         }
         else if (EAMIsInit(&mDegre))
         {
            mVecIterDeg = std::vector<int>({mDegre});
         }
         else 
         {
            mDegre = mWithOri ? 9 : 2;
            // fix nbellaiche 2021/06/17: initialisation du vecteur
            if (mVecIterDeg.empty())  mVecIterDeg = std::vector<int>({mDegre});
         }
                    // << EAM(mVecIterDeg,"VecIterDeg",true,"Vector of degree in case of iterative approach")
                    // << EAM(mPropIterDeg,"PropIterDeg",true,"Prop to compute sigma in cas of iterative degre")
         std::cout << "Deegreee: " << mDegre << " WithOri:" << mWithOri << "\n";


	 if (EAMIsInit(&mBoxTerrain))
	 {
              ELISE_ASSERT(mIntervZIsDef,"Interval Z required for box terrain");
              ELISE_ASSERT(mGenI1&&mGenI2,"Modeles required for box terrain");
	      Pt2dr  aTabC[4];
              mBoxTerrain.Corners(aTabC);
	      for (int aK=0 ; aK<4; aK++)
	      {
                  mPtsLimTerr.push_back(Pt3dr(aTabC[aK],mZMin));
                  mPtsLimTerr.push_back(Pt3dr(aTabC[aK],mZMax));
	      }
	 }


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

     const char * aCarHom = 0; // Because ImEpip requires a char * with 0 defaulta value as argument
     if (EAMIsInit(&mNameHom))
        aCarHom = mNameHom.c_str();

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
	bool aDoReech = true;

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
		  			  << EAM(aDoReech,"DoReech",true,"Do resampling, Def=true")
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

	if (aDoReech)
	{
    
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
	}
	else
	{
		cElHomographie  aH12 = cElHomographie::RansacInitH(aPack,50,2000);
		cElHomographie  aH21 = aH12.Inverse();


        std::cout << "H12=" << "\n";
        aH12.Show();
        std::cout << "H21=" << "\n";
        aH21.Show();
	}


	return EXIT_SUCCESS;
}

class cAppliReechDepl    : public ElDistortion22_Gen
{
    public :
        cAppliReechDepl(int argc,char ** argv);
        void DoReech();
    private :
        Pt2dr Direct(Pt2dr) const;
        bool OwnInverse(Pt2dr &) const ;

        bool IsInside(Pt2dr& ) const;

        cElemAppliSetFile mEASF;


        std::string mFullNameI1;
        /* Direct displacement */
        std::string mFullNameI1Px1;
        std::string mFullNameI1Px2;
        /* Inverse displacement */
        std::string mFullNameI2Px1;
        std::string mFullNameI2Px2;

        std::string mNameI1;
        std::string mNameI1Px1;
        std::string mNameI1Px2;
        std::string mNameI2Px1;
        std::string mNameI2Px2;


        std::string mNameI2Redr;
        std::string mPostMasq;


        /* Direct map */
        TIm2D<REAL4,REAL8> * mPx1TIm_1To2;
        TIm2D<REAL4,REAL8> * mPx2TIm_1To2;
        /* Inverse map */
        TIm2D<REAL4,REAL8> * mPx1TIm_2To1;
        TIm2D<REAL4,REAL8> * mPx2TIm_2To1;

        Pt2di           mSzIn;
        double          mScaleReech;
        int             mKernel;
};

bool cAppliReechDepl::IsInside(Pt2dr& aP) const
{
    //-2 to allow bilinear trafo at real positions
    if ((aP.x<(mSzIn.x-2)) && (aP.y<(mSzIn.y-2))) 
        return true;
    else return false;
}
    
Pt2dr cAppliReechDepl::Direct(Pt2dr aP) const
{
    if (IsInside(aP))
        return aP + Pt2dr(mPx1TIm_2To1->getr(aP),mPx2TIm_2To1->getr(aP));
    else
        return aP;
}

bool cAppliReechDepl::OwnInverse(Pt2dr & aP) const 
{
    if (IsInside(aP))
    {
        aP = Pt2dr(mPx1TIm_1To2->getr(aP),mPx2TIm_1To2->getr(aP)); 

    }

    return true;
    
}

cAppliReechDepl::cAppliReechDepl(int argc,char ** argv) : 
    mPostMasq ("Masq"),
    mKernel(5)
{

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(mFullNameI1,"Name of image", eSAM_IsExistFile)
                      << EAMC(mFullNameI1Px1,"Name of inverse displacemnt image in X (Px1)", eSAM_IsExistFile)
                      << EAMC(mFullNameI1Px2,"Name of inverse displacemnt image in Y (Px2)", eSAM_IsExistFile)
                      << EAMC(mFullNameI2Px1,"Name of direct displacemnt image in X (Px1)", eSAM_IsExistFile)
                      << EAMC(mFullNameI2Px2,"Name of direct displacemnt image in Y (Px2)", eSAM_IsExistFile)
                      << EAMC(mNameI2Redr,"Name of resulting image", eSAM_IsExistFile),
          LArgMain()  << EAM (mPostMasq,"PostMasq",true,"Name of Masq , Def = \"Masq\"")
                      << EAM (mScaleReech,"ScaleReech",true,"Scale Resampling, used for interpolator when downsizing")
		      << EAM (mKernel,"Kern",true,"Kernel size, Def=5")
    );


    mNameI1 = NameWithoutDir(mFullNameI1);
    mNameI1Px1 = NameWithoutDir(mFullNameI1Px1);
    mNameI1Px2 = NameWithoutDir(mFullNameI1Px2);
    mNameI2Px1 = NameWithoutDir(mFullNameI2Px1);
    mNameI2Px2 = NameWithoutDir(mFullNameI2Px2);
   
    mEASF.Init(mFullNameI1);


    cMetaDataPhoto aMTD1 = cMetaDataPhoto::CreateExiv2(mFullNameI1);
    mSzIn = Pt2di(aMTD1.TifSzIm());


    /* Read displacements */
    Tiff_Im     aPx1Tif_1To2( mNameI1Px1.c_str());
    mPx1TIm_1To2 = new TIm2D<REAL4,REAL8>(mSzIn);
    ELISE_COPY
    (         
               //rectangle(Pt2di(mKernelHalf,mKernelHalf),aSzIn),
               mPx1TIm_1To2->all_pts(),
               trans(aPx1Tif_1To2.in(),Pt2di(0,0)),
               mPx1TIm_1To2->out()
    );
    Tiff_Im     aPx2Tif_1To2( mNameI1Px2.c_str());
    mPx2TIm_1To2 = new TIm2D<REAL4,REAL8>(mSzIn);
    ELISE_COPY
    (
               //rectangle(Pt2di(mKernelHalf,mKernelHalf),aSzIn),
               mPx2TIm_1To2->all_pts(),
               trans(aPx2Tif_1To2.in(),Pt2di(0,0)),
               mPx2TIm_1To2->out()
    );
    Tiff_Im     aPx1Tif_2To1( mNameI2Px1.c_str());
    mPx1TIm_2To1 = new  TIm2D<REAL4,REAL8> (mSzIn);
    ELISE_COPY
    (
               //rectangle(Pt2di(mKernelHalf,mKernelHalf),aSzIn),
               mPx1TIm_2To1->all_pts(),
               trans(aPx1Tif_2To1.in(),Pt2di(0,0)),
               mPx1TIm_2To1->out()
    );
    Tiff_Im     aPx2Tif_2To1(       mNameI2Px2.c_str());
    mPx2TIm_2To1 = new  TIm2D<REAL4,REAL8> (mSzIn);
    ELISE_COPY
    (
               //rectangle(Pt2di(mKernelHalf,mKernelHalf),aSzIn),
               mPx2TIm_2To1->all_pts(),
               trans(aPx2Tif_2To1.in(),Pt2di(0,0)),
               mPx2TIm_2To1->out()
    );



    ReechFichier
    (
          false,
          mFullNameI1,
          Box2dr(Pt2dr(0,0),
                 Pt2dr(mSzIn.x,mSzIn.y)),
          this,
          Box2dr(Pt2dr(0,0),
                 Pt2dr(mSzIn.x,mSzIn.y)),
          1.0,
          mNameI2Redr,
          mPostMasq,
          mKernel,
          1000
     );

     delete mPx1TIm_1To2;
     delete mPx2TIm_1To2; 
     delete mPx1TIm_2To1;
     delete mPx2TIm_2To1;
}

int CPP_ReechDepl(int argc,char ** argv)
{

    cAppliReechDepl aApRechDepl(argc,argv);
    return EXIT_SUCCESS;
}

std::string BatchReechDeplMakeCmd(const std::string aNameI1,
                                  const std::string aNameI1Px1,
                                  const std::string aNameI1Px2,
                                  const std::string aNameI2Px1,
                                  const std::string aNameI2Px2,
                                  const std::string aNameI2Redr,
                                  const std::string aPostMasq,
                                  const int aScaleReech,
                                  const int aKernel
                                  )
{

    return  MMBinFile(MM3DStr) + "TestLib OneReechDepl " + aNameI1 + BLANK
                                                          + aNameI1Px1 + BLANK
                                                          + aNameI1Px2 + BLANK
                                                          + aNameI2Px1 + BLANK
                                                          + aNameI2Px2 + BLANK
                                                          + aNameI2Redr + BLANK 
                                                          + "PostMasq=" + aPostMasq + BLANK 
                                                          + "Kern=" + ToString(aKernel) + BLANK 
                                                          + "ScaleReech=" + ToString(aScaleReech)  ;
}

int CPP_BatchReechDepl(int argc,char ** argv)
{
    cElemAppliSetFile aEASF;
    std::string aPatNameI1;
    std::string aPatNameI1Px1, aPatNameI1Px2, aPatNameI2Px1, aPatNameI2Px2;
    std::string aPatNameI2Redr;
    std::string aPostMasq="Masq";
    int         aScaleReech=1, aKernel=5;
    bool        aExe=true;

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aPatNameI1,"Name of image", eSAM_IsExistFile)
                      << EAMC(aPatNameI1Px1,"Name of inverse displacemnt image in X (Px1)", eSAM_IsExistFile)
                      << EAMC(aPatNameI1Px2,"Name of inverse displacemnt image in Y (Px2)", eSAM_IsExistFile)
                      << EAMC(aPatNameI2Px1,"Name of direct displacemnt image in X (Px1)", eSAM_IsExistFile)
                      << EAMC(aPatNameI2Px2,"Name of direct displacemnt image in Y (Px2)", eSAM_IsExistFile)
                      << EAMC(aPatNameI2Redr,"Name of resulting image", eSAM_IsExistFile),
          LArgMain()  << EAM (aPostMasq,"PostMasq",true,"Name of Masq , Def = \"Masq\"")
                      << EAM (aScaleReech,"ScaleReech",true,"Scale Resampling, used for interpolator when downsizing")
					  << EAM (aKernel,"Kern",true,"Kernel size, Def=5")
                      << EAM (aExe,"Exe",true,"Execute the commend, Def=true")
    );

    aEASF.Init(aPatNameI1);

    std::list<std::string> aLCom;

    if (aEASF.SetIm()->size()>1)
    {
        std::cout << "N patches, N displacement mappings.\n";

        cElRegex  anAutom(aPatNameI1.c_str(),10);

        for (size_t aKIm=0  ; aKIm< aEASF.SetIm()->size() ; aKIm++)
        {
        
            std::string aNameIm = (*aEASF.SetIm())[aKIm];
            std::string aNameI1Px1  =  MatchAndReplace(anAutom,aNameIm,aPatNameI1Px1);
            std::string aNameI1Px2  =  MatchAndReplace(anAutom,aNameIm,aPatNameI1Px2);
            std::string aNameI2Px1  =  MatchAndReplace(anAutom,aNameIm,aPatNameI2Px1);
            std::string aNameI2Px2  =  MatchAndReplace(anAutom,aNameIm,aPatNameI2Px2);
            std::string aNameI2Redr =  MatchAndReplace(anAutom,aNameIm,aPatNameI2Redr);


            std::string aCom =  BatchReechDeplMakeCmd (aNameIm,
                                                       aNameI1Px1,
                                                       aNameI1Px2,
                                                       aNameI2Px1,
                                                       aNameI2Px2,
                                                       aNameI2Redr,
                                                       aPostMasq,
                                                       aScaleReech,
                                                       aKernel) ;
            aLCom.push_back(aCom);
        }

        if (aExe)
            cEl_GPAO::DoComInParal(aLCom);
        else 
        {
            for (auto cmd : aLCom)
                std::cout << cmd << "\n";
        }
    }
    else
    {
        aEASF.Init(aPatNameI1Px1);
        
        if (aEASF.SetIm()->size()>1)
        {
            std::cout << "1 patch, N displacement mapppings.\n";
            
            cElRegex  anAutom(aPatNameI1Px1.c_str(),10);
            std::string aNameIm = aPatNameI1;

            for (size_t aKIm=0  ; aKIm< aEASF.SetIm()->size() ; aKIm++)
            {


                std::string aNameI1Px1  = (*aEASF.SetIm())[aKIm];
                std::string aNameI1Px2  =  MatchAndReplace(anAutom,aNameI1Px1,aPatNameI1Px2);
                std::string aNameI2Px1  =  MatchAndReplace(anAutom,aNameI1Px1,aPatNameI2Px1);
                std::string aNameI2Px2  =  MatchAndReplace(anAutom,aNameI1Px1,aPatNameI2Px2);
                std::string aNameI2Redr =  MatchAndReplace(anAutom,aNameI1Px1,aPatNameI2Redr);
             
                std::string aCom =  BatchReechDeplMakeCmd (aNameIm,
                                                           aNameI1Px1,
                                                           aNameI1Px2,
                                                           aNameI2Px1,
                                                           aNameI2Px2,
                                                           aNameI2Redr,
                                                           aPostMasq,
                                                           aScaleReech,
                                                           aKernel);
                
                aLCom.push_back(aCom);

            }

            if (aExe)
                cEl_GPAO::DoComInParal(aLCom);
            else 
            {
                for (auto cmd : aLCom)
                    std::cout << cmd << "\n";
            }
        }
        else 
        {
            std::cout << "1 patch, 1 displacement mapping.\n";
            
            std::string aCom =  BatchReechDeplMakeCmd (aPatNameI1,
                                                       aPatNameI1Px1,
                                                       aPatNameI1Px2,
                                                       aPatNameI2Px1,
                                                       aPatNameI2Px2,
                                                       aPatNameI2Redr,
                                                       aPostMasq,
                                                       aScaleReech,
                                                       aKernel);


            if (aExe)
                System(aCom,true,true);
            else
                std::cout << "CMD: " << aCom << "\n";

            return 1.0;
        }

    }




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
