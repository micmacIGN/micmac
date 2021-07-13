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

// #define _ELISE_ALGO_GEOM_QDT_H 1
// #define _ELISE_ALGO_GEOM_QDT_IMPLEM_H 1
// #define __QDT_INSERTOBJ__ 1



#include "cPose.h"

    /* ========== cStructRigidInit ============*/

cStructRigidInit::cStructRigidInit(cPoseCam * RigidMere,const ElRotation3D & aR) :
  mCMere  (RigidMere),
  mR0m1L0 (aR)
{
}


static const int NbMinCreateIm = 200;

//class cPtAVGR;
//class cAperoVisuGlobRes;


/***************** classes moved to cPose.h *******************/
     /*===========     cPtAVGR  ===========*/



cPtAVGR::cPtAVGR(const Pt3dr & aP,double aRes) :
   mPt  (Pt3df::P3ToThisT(aP)),
   mRes (aRes)
{
}

/*class cFoncPtOfPtAVGR
{
   public :
       Pt2dr operator () (cPtAVGR * aP) {return  Pt2dr(aP->mPt.x,aP->mPt.y);}
};*/

     /*===========     cAperoVisuGlobRes  ===========*/

/*typedef enum
{
    eBAVGR_X,
    eBAVGR_Y,
    eBAVGR_Z,
    eBAVGR_Res
} eBoxAVGR;

class cAperoVisuGlobRes
{
    public :
       void AddResidu(const Pt3dr & aP,double aRes);
       void DoResidu(const std::string & aDir,int aNbMes);

       cAperoVisuGlobRes();
       
    private :
       Interval  CalculBox(double & aVMil,double & aResol,eBoxAVGR aMode,double PropElim,double Rab);
       Box2dr    CalculBox_XY(double PropElim,double Rab);
       double    ToEcartStd(double anE) const;
       double    FromEcartStd(double anE) const;
       Pt3di     ColOfEcart(double anE);

       typedef ElQT<cPtAVGR *,Pt2dr,cFoncPtOfPtAVGR> tQtTiepT;

       int                  mNbPts;
       std::list<cPtAVGR *> mLpt;
       tQtTiepT *           mQt;
       double               mResol;
       double               mResolX;
       double               mResolY;
       double               mSigRes;
       double               mMoyRes;
       cPlyCloud            mPC;
       cPlyCloud            mPCLeg;  // Legende
       double               mVMilZ;
};*/


cAperoVisuGlobRes::cAperoVisuGlobRes() :
   mNbPts  (0),
   mQt     (0)
{
}



void cAperoVisuGlobRes::AddResidu(const Pt3dr & aP,double aRes)
{
     cPtAVGR * aPVG = new cPtAVGR(aP,aRes);

     mLpt.push_back(aPVG);
     mNbPts++;
}


Interval cAperoVisuGlobRes::CalculBox(double & aVMil,double & aResol,eBoxAVGR aMode,double aPropElim,double aPropRab)
{
    std::vector<float> aVVals;
    // double aSomV = 0;
    // double aSomV2 = 0;

    for (auto iT=mLpt.begin() ; iT!=mLpt.end() ; iT++)
    {
        double aVal = 0.0;
        if       (aMode==eBAVGR_Res)     aVal = (*iT)->mRes  ;
        else if  (aMode==eBAVGR_X)       aVal = (*iT)->mPt.x;
        else if  (aMode==eBAVGR_Y)       aVal = (*iT)->mPt.y;
        else if  (aMode==eBAVGR_Z)       aVal = (*iT)->mPt.z;
        else
        {
                 aVal = (*iT)->mPt.z;
                ELISE_ASSERT(false,"cAperoVisuGlobRes::CalculBox");
        }
        aVVals.push_back(aVal);
        // aSomV += aVal;
        // aSomV2 += ElSquare(aVal);
    }
    // aSomV /= mNbPts;
    // aSomV2 /= mNbPts;
    // aSomV2 -= ElSquare(aSomV);
    // Sigma = Larg / srqt(12) pour une distrib uniforme
    // double Larg =  sqrt(12*aSomV2);
    double aV25 = KthValProp(aVVals,0.25);  
    double aV75 = KthValProp(aVVals,0.75);
    aVMil = (aV75+aV25) / 2.0;
    double Larg = 2 * (aV75 - aV25);
    aResol = Larg / mNbPts;

    double aVMin = KthValProp(aVVals,aPropElim);  
    double aVMax = KthValProp(aVVals,1-aPropElim);  

    double aRab = (aVMax-aVMin) * aPropRab;

    aVMin -= aRab ;
    aVMax += aRab ;

    return Interval(aVMin,aVMax);
}

Box2dr  cAperoVisuGlobRes::CalculBox_XY(double aPropElim,double aPropRab)
{
    double aVMilX,aVMilY;
    Interval aIntX = CalculBox(aVMilX,mResolX,eBAVGR_X,aPropElim,aPropRab);
    Interval aIntY = CalculBox(aVMilY,mResolY,eBAVGR_Y,aPropElim,aPropRab);

    mResol = euclid(Pt2dr(mResolX,mResolY));

    return Box2dr(Pt2dr(aIntX._v0,aIntY._v0), Pt2dr(aIntX._v1,aIntY._v1));
}

double DirErFonc(double x);
double InvErrFonc(double x);

double  cAperoVisuGlobRes::ToEcartStd(double anE) const
{
     return DirErFonc((anE-mMoyRes)/mSigRes); 
}

double  cAperoVisuGlobRes::FromEcartStd(double anE) const
{
     return mMoyRes + mSigRes *InvErrFonc(anE);
}



Pt3di     cAperoVisuGlobRes::ColOfEcart(double anE)
{
    Elise_colour aCol = Elise_colour::its(0.5,(1-anE)/3.0,1.0);

    return Pt3di (AdjUC(aCol.r()*255),AdjUC(aCol.g()*255),AdjUC(aCol.b()*255));
}

template <class Type> bool gen_std_isnan(const Type & aP)
{
   return std_isnan(aP.x) || std_isnan(aP.y) ;
}

void cAperoVisuGlobRes::DoResidu(const std::string & aDir,int aNbMes)
{
    {
       double aRR;
       CalculBox(mVMilZ,aRR,eBAVGR_Z,0.16,0.0);
    }

    Box2dr aBoxQt = CalculBox_XY(0.02,0.3);
    cFoncPtOfPtAVGR aFoncP;
    
    mQt = new tQtTiepT(aFoncP,aBoxQt,10,mResol*5);

    for (auto iT=mLpt.begin() ; iT!=mLpt.end() ; iT++)
    {
        (*iT)->mInQt=   mQt->insert(*iT,true);
    }

    
    for (auto iTGlob=mLpt.begin() ; iTGlob!=mLpt.end() ; iTGlob++)
    {
        if ((*iTGlob)->mInQt)
        {
           // std::set<cPtAVGR *> aSet;
           Pt2dr aP = aFoncP(*iTGlob);
           // std::list<cPtAVGR *> aLVois = mQt->KPPVois(aP,aNbMes,1.5*mResol*sqrt(aNbMes));
           double aDist = mResol*sqrt(aNbMes);
           std::set<cPtAVGR *> aLVois;  
           mQt->RVoisins(aLVois,aP,mResol*sqrt(aNbMes));
           double aSomP=0.0;
           double aSomPRes=0.0;
           for (auto itV=aLVois.begin() ; itV!=aLVois.end() ; itV++)
           {
               if (euclid((*iTGlob)->mPt-(*itV)->mPt) < aDist)
               {
                   double aPds = 1.0;
                   aSomP += aPds;
                   aSomPRes += aPds * (*itV)->mRes;
               }
           }
           double aRes = aSomPRes / aSomP;
           (*iTGlob)->mResFiltr = aRes;
        }
    }
 
    double aRR; // Inutilise
    // Pour une gaussienne 68 % compris dans [-Sig,Sig]
    double aVMilRes;
    Interval  aIntRes = CalculBox(aVMilRes,aRR,eBAVGR_Res,0.16,0.0);
    mSigRes = (aIntRes._v1 - aIntRes._v0) / 2.0;
    mMoyRes = (aIntRes._v1 + aIntRes._v0) / 2.0;

    for (auto iTGlob=mLpt.begin() ; iTGlob!=mLpt.end() ; iTGlob++)
    {
        if ((*iTGlob)->mInQt)
        {
             double anEcart = ToEcartStd( (*iTGlob)->mResFiltr);
             Pt3di aCol = ColOfEcart(anEcart);
             mPC.AddPt(aCol,Pt3dr::P3ToThisT((*iTGlob)->mPt));
        }
    }

    mPC.PutFile(aDir+"CloudResidual.ply");

    Pt2dr aP0 = aBoxQt._p0 - Pt2dr(0,-40) * mResol;

    double aPasLeg = mResol *100;

    double aZLeg = mVMilZ;
    int aNbLeg = 10;
    double aLargX = 10;
    int    aLargY = 20;
    for (int aCpt=0 ; aCpt<=aNbLeg; aCpt++)
    {
        double aEcartStd =  (aCpt-aNbLeg*0.5) /(0.5+aNbLeg*0.5);
        double aRes = FromEcartStd(aEcartStd);
        if (aRes >0)
        {
           char aBuf[100];
           sprintf(aBuf,"%.2f",aRes);
           std::string aStrRes (aBuf);
           
           mPCLeg.PutString
           (
                aStrRes,
                Pt3dr(aP0.x,aP0.y,aZLeg)+Pt3dr(0,aCpt*aPasLeg*aLargY,0) , 
                Pt3dr(1,0,0),
                Pt3dr(0,1,0),
                Pt3di(255,255,255),
                aLargX* aPasLeg,  //  La largeur des caractere
                aPasLeg,  // espacement
                4  // carre de 4x4
            );


        }
    }

    int DensLeg = 20;
    for (int aCpt= -DensLeg/2 ; aCpt<=((DensLeg*aNbLeg) + DensLeg/2) ; aCpt++)
    {
        double aCptN  =  aCpt/double(DensLeg);
        double aEcartStd =  (aCptN-aNbLeg*0.5) /(0.5+aNbLeg*0.5);
        Pt3di aCol = ColOfEcart(aEcartStd);

        
        Pt3dr aPLine  = Pt3dr(aP0.x,aP0.y,aZLeg)+Pt3dr(0,aCptN*aPasLeg*aLargY,0) ; 

        for (int anX=0 ; anX<=DensLeg ; anX++)
        {
            Pt3dr aP = aPLine + Pt3dr(-(anX/double(DensLeg))*aPasLeg*aLargY,0,0);
            mPCLeg.AddPt(aCol,aP);
        }
    }
    mPCLeg.PutFile(aDir+"CloudResidual_Leg.ply");


/*
    for (int aK=-10 ; aK<10 ; aK++)
    {
        double aV= aK*0.2;
        std::cout << "EErr " << aV << " " << erfcc(aV) << " " <<DirErFonc(aV) <<  " Dif=" <<  InvErrFonc(DirErFonc(aV)) - aV << "\n";
    }
    getchar();
*/
}


static cAperoVisuGlobRes mAVGR;


//============================================

/*class cInfoAccumRes
{
     public :
       cInfoAccumRes(const Pt2dr & aPt,double aPds,double aResidu,const Pt2dr & aDir);

       Pt2dr  mPt;
       double mPds;
       double mResidu;
       Pt2dr  mDir;
};*/


cInfoAccumRes::cInfoAccumRes(const Pt2dr & aPt,double aPds,double aResidu,const Pt2dr & aDir) :
   mPt      (aPt),
   mPds     (aPds),
   mResidu  (aResidu),
   mDir     (aDir)
{
}

/*class cAccumResidu
{
    public :
       void Accum(const cInfoAccumRes &);
       cAccumResidu(Pt2di aSz,double aRed,bool OnlySign,int aDegPol);

       const Pt2di & SzRed() {return mSzRed;}

       void Export(const std::string & aDir,const std::string & aName,const cUseExportImageResidu &,FILE * );
       void ExportResXY(TIm2D<REAL4,REAL8>* aTResX,TIm2D<REAL4,REAL8>* aTResY);
       void ExportResXY(const Pt2di&,Pt2dr& aRes);
    private :
       void AccumInImage(const cInfoAccumRes &);

       std::list<cInfoAccumRes> mLIAR;
       int                      mNbInfo;
       double                   mSomPds;
       bool                     mOnlySign;
       double                   mResol;
       Pt2di                    mSz;
       Pt2di                    mSzRed;

       Im2D_REAL4               mPds;
       TIm2D<REAL4,REAL8>       mTPds;
       Im2D_REAL4               mMoySign;
       TIm2D<REAL4,REAL8>       mTMoySign;
       Im2D_REAL4               mMoyAbs;
       TIm2D<REAL4,REAL8>       mTMoyAbs;
       bool                     mInit;
       int                      mDegPol;
       L2SysSurResol *          mSys;
};*/

cAccumResidu::cAccumResidu(Pt2di aSz,double aResol,bool OnlySign,int aDegPol) :
   mNbInfo (0),
   mSomPds   (0.0),
   mOnlySign (OnlySign),
   mResol    (aResol),
   mSz       (aSz),
   mSzRed    (round_up(Pt2dr(aSz)/mResol)),
   mPds      (1,1),
   mTPds     (mPds),
   mMoySign  (1,1),
   mTMoySign (mMoySign),
   mMoyAbs   (1,1),
   mTMoyAbs  (mMoyAbs),
   mInit     (false),
   mDegPol   (aDegPol),
   mSys      (0)
{
}

void cAccumResidu::Export(const std::string & aDir,const std::string & aName,const cUseExportImageResidu & aUEIR,FILE * aFP ) 
{
   if (! mInit) return;

   fprintf(aFP,"=== %s =======\n",aName.c_str());
   fprintf(aFP,"  Nb=%d  WAver=%f\n",mNbInfo,mSomPds/mNbInfo);

   Tiff_Im::CreateFromFonc
   (
        aDir+"RawWeight-"+aName+".tif",
        mSzRed,
        mPds.in() * (mNbInfo/mSomPds),
        GenIm::real4
   );

   int aNbPixel = mSzRed.x * mSzRed.y;
   double aNbMesByCase = mNbInfo / double(aNbPixel);
   double aTargetNbMesByC = aUEIR.NbMesByCase().Val();
   if (aTargetNbMesByC >= 0)
   {
        double aSigma = sqrt(aTargetNbMesByC / aNbMesByCase);
        FilterGauss(mPds,aSigma);
        FilterGauss(mMoySign,aSigma);
        FilterGauss(mMoyAbs,aSigma);
   }

   Tiff_Im::CreateFromFonc
   (
        aDir+"ResSign-"+aName+".tif",
        mSzRed,
        mMoySign.in() / Max(mPds.in(),1e-4),
        GenIm::real4
   );

   double aMoySign,aMoyAbsSign,aSomPds;
   ELISE_COPY
   (
        mPds.all_pts(),
        Virgule(mPds.in(),mMoySign.in(),Abs(mMoySign.in())),
        Virgule(sigma(aSomPds),sigma(aMoySign),sigma(aMoyAbsSign))
   );
   fprintf(aFP,"  AverSign=%f  AverAbsSign=%f\n",aMoySign/aSomPds,aMoyAbsSign/aSomPds);

   if (!mOnlySign)
   {
      Tiff_Im::CreateFromFonc
      (
           aDir+"ResAbs-"+aName+".tif",
           mSzRed,
           mMoyAbs.in() / Max(mPds.in(),1e-4),
           GenIm::real4
      );
   }

    if (mSys)
    {
        bool aOk;
        Im1D_REAL8  aSol = mSys->Solve(&aOk);
        if (aOk)
        {
            double * aDS = aSol.data();

            Im2D_REAL4               aResX(mSzRed.x,mSzRed.y);
            TIm2D<REAL4,REAL8>       aTRx(aResX);
            Im2D_REAL4               aResY(mSzRed.x,mSzRed.y);
            TIm2D<REAL4,REAL8>       aTRy(aResY);

            Pt2di aPInd;
            for (aPInd.x=0 ; aPInd.x<mSzRed.x ; aPInd.x++)
            {
                for (aPInd.y=0 ; aPInd.y<mSzRed.y ; aPInd.y++)
                {
                    Pt2dr aPFulRes = Pt2dr(aPInd) * mResol;
                    Pt2dr aSzN = mSz/2.0;
                    double  aX = (aPFulRes.x-aSzN.x) / aSzN.x;
                    double  aY = (aPFulRes.y-aSzN.y) / aSzN.y;

                    std::vector<double> aVMx; // Monome Xn
                    std::vector<double> aVMy; // Monome Yn
                    aVMx.push_back(1.0);
                    aVMy.push_back(1.0);
                    for (int aD=0 ; aD< mDegPol ; aD++)
                    {
                      aVMx.push_back(aVMx.back() * aX);
                      aVMy.push_back(aVMy.back() * aY);
                    }


                    int anIndEq = 0;
                    double aSX=0 ;
                    double aSY=0 ;
                    for (int aDx=0 ; aDx<= mDegPol ; aDx++)
                    {
                       for (int aDy=0 ; aDy<= mDegPol - aDx ; aDy++)
                       {
                            double aMonXY = aVMx[aDx] * aVMy[aDy]; // X ^ Dx * Y ^ Dy
                            aSX += aDS[anIndEq++] * aMonXY;
                            aSY += aDS[anIndEq++] * aMonXY;
                       }
                    }
                    aTRx.oset(aPInd,aSX);
                    aTRy.oset(aPInd,aSY);
                }
            }
            Tiff_Im::CreateFromFonc
            (
                 aDir+"ResX-"+aName+".tif",
                 mSzRed,
                 aResX.in(),
                 GenIm::real4
            );
            Tiff_Im::CreateFromFonc
            (
                 aDir+"ResY-"+aName+".tif",
                 mSzRed,
                 aResY.in(),
                 GenIm::real4
            );
        }
    }
}

void cAccumResidu::ExportResXY(const Pt2di& aPt,Pt2dr& aRes)
{
    if (mSys)
    {
        bool aOk;
        Im1D_REAL8  aSol = mSys->Solve(&aOk);
        if (aOk)
        {
            double * aDS = aSol.data();

            std::vector<double> aVMx;
            std::vector<double> aVMy;
            aVMx.push_back(1.0);
            aVMy.push_back(1.0);

            Pt2dr aPFulRes = Pt2dr(aPt) * mResol;
            Pt2dr aSzN = mSz/2.0;
            double  aX = (aPFulRes.x-aSzN.x) / aSzN.x;
            double  aY = (aPFulRes.y-aSzN.y) / aSzN.y;

            for (int aD=0 ; aD< mDegPol ; aD++)
            {
              aVMx.push_back(aVMx.back() * aX);
              aVMy.push_back(aVMy.back() * aY);
            }

            int anIndEq = 0;
            double aSX=0 ;
            double aSY=0 ;
            for (int aDx=0 ; aDx<= mDegPol ; aDx++)
            {
               for (int aDy=0 ; aDy<= mDegPol - aDx ; aDy++)
               {
                    double aMonXY = aVMx[aDx] * aVMy[aDy];
                    aSX += aDS[anIndEq++] * aMonXY;
                    aSY += aDS[anIndEq++] * aMonXY;
               }
            }

            aRes.x = aSX;
            aRes.y = aSY;
        }
    }
}


void cAccumResidu::ExportResXY(TIm2D<REAL4,REAL8>* aTRx,TIm2D<REAL4,REAL8>* aTRy)
{
    if (mSys)
    {
        bool aOk;
        Im1D_REAL8  aSol = mSys->Solve(&aOk);
        if (aOk)
        {
            double * aDS = aSol.data();

            Pt2di aPInd;
            for (aPInd.x=0 ; aPInd.x<mSzRed.x ; aPInd.x++)
            {
                for (aPInd.y=0 ; aPInd.y<mSzRed.y ; aPInd.y++)
                {
                    Pt2dr aPFulRes = Pt2dr(aPInd) * mResol;
                    Pt2dr aSzN = mSz/2.0;
                    double  aX = (aPFulRes.x-aSzN.x) / aSzN.x;
                    double  aY = (aPFulRes.y-aSzN.y) / aSzN.y;

                    std::vector<double> aVMx; 
                    std::vector<double> aVMy;  
                    aVMx.push_back(1.0);
                    aVMy.push_back(1.0);
                    for (int aD=0 ; aD< mDegPol ; aD++)
                    {
                      aVMx.push_back(aVMx.back() * aX);
                      aVMy.push_back(aVMy.back() * aY);
                    }


                    int anIndEq = 0;
                    double aSX=0 ;
                    double aSY=0 ;
                    for (int aDx=0 ; aDx<= mDegPol ; aDx++)
                    {
                       for (int aDy=0 ; aDy<= mDegPol - aDx ; aDy++)
                       {
                            double aMonXY = aVMx[aDx] * aVMy[aDy]; 
                            aSX += aDS[anIndEq++] * aMonXY;
                            aSY += aDS[anIndEq++] * aMonXY;
                       }
                    }
                    aTRx->oset(aPInd,aSX);
                    aTRy->oset(aPInd,aSY);
                }
            }
        }
    }
}


void cAccumResidu::AccumInImage(const cInfoAccumRes & anInfo)
{
    Pt2dr aP = anInfo.mPt / mResol;
    mTPds.incr(aP,anInfo.mPds);
    mTMoySign.incr(aP,anInfo.mPds * anInfo.mResidu);

    if (!mOnlySign)
    {
         mTMoyAbs.incr(aP,anInfo.mPds*ElAbs(anInfo.mResidu));
    }
    if (mSys)
    {
        // 
        Pt2dr aSzN = mSz/2.0;
        Pt2dr aN = anInfo.mDir * Pt2dr(0,1);
        // Pour precision matrice, mieux vaut coordonnees normalisees
        double  aX = (anInfo.mPt.x-aSzN.x) / aSzN.x;
        double  aY = (anInfo.mPt.y-aSzN.y) / aSzN.y;

        std::vector<double> aVMx; // Monome Xn
        std::vector<double> aVMy; // Monome Xn
        aVMx.push_back(1.0);
        aVMy.push_back(1.0);
        for (int aD=0 ; aD< mDegPol ; aD++)
        {
          aVMx.push_back(aVMx.back() * aX);
          aVMy.push_back(aVMy.back() * aY);
        }

        std::vector<double> anEq;

        for (int aDx=0 ; aDx<= mDegPol ; aDx++)
        {
           for (int aDy=0 ; aDy<= mDegPol - aDx ; aDy++)
           {
                double aMonXY = aVMx[aDx] * aVMy[aDy]; // X ^ Dx * Y ^ Dy
                anEq.push_back(aMonXY* aN.x);
                anEq.push_back(aMonXY* aN.y);

//    std::cout << " eq " << aDx << " " << aDx << " " << aVMx[aDx] << " " << aVMy[aDy] << " " << aN.x << " " << aN.y << " " << anInfo.mDir <<"\n";
  //  std::cout << " eq " << aMonXY* aN.x << " " << aMonXY* aN.y << "\n";
           }
        }
        mSys->AddEquation(anInfo.mPds,VData(anEq),anInfo.mResidu);
    }
}
void cAccumResidu::Accum(const cInfoAccumRes & anInfo)
{
   mSomPds += anInfo.mPds;
   mNbInfo++;
   if (mNbInfo<NbMinCreateIm)
   {
      mLIAR.push_back(anInfo);
   }
   else if (mNbInfo==NbMinCreateIm)
   {
       mInit = true;
       mLIAR.push_back(anInfo);
       mPds = Im2D_REAL4(mSzRed.x,mSzRed.y,0.0);
       mTPds =  TIm2D<REAL4,REAL8>(mPds);
       mMoySign = Im2D_REAL4(mSzRed.x,mSzRed.y,0.0);
       mTMoySign =  TIm2D<REAL4,REAL8>(mMoySign);
       if (! mOnlySign)
       {
           mMoyAbs = Im2D_REAL4(mSzRed.x,mSzRed.y,0.0);
           mTMoyAbs =  TIm2D<REAL4,REAL8>(mMoyAbs);
       }
       if (mDegPol >=0)
       {
          mSys =  new L2SysSurResol((1+mDegPol)*(mDegPol+2));
       }
       for (std::list<cInfoAccumRes>::const_iterator itI=mLIAR.begin() ;itI!=mLIAR.end() ; itI++)
       {
           AccumInImage(*itI);
       }
       mLIAR.clear();
   }
   else
   {
        AccumInImage(anInfo);
   }
}

       // cAccumResidu(Pt2di aSz,double aRed,bool OnlySign);
void cAppliApero::AddOneInfoImageResidu
     (
         const cInfoAccumRes & anInfo,
         const std::string &   aName,
         Pt2di                 aSz,
         double                aSzRed,
         bool                  OnlySign,
         int                   aDegPol
     )
{

  if (aSzRed <=0) return;

  double aFactRed = dist8(aSz) / aSzRed;

  cAccumResidu * & aRef = mMapAR[aName];
  if (aRef==0)
  {
     aRef = new cAccumResidu(aSz,aFactRed,OnlySign,aDegPol);
  }

  aRef->Accum(anInfo);
}

void cAppliApero::AddInfoImageResidu
     (
         const Pt3dr &                 aPt,
         const  cNupletPtsHomologues & aNupl,
         const std::vector<cGenPoseCam *> aVP,
         const std::vector<double> &  aVpds
     )
{
  if ((!Param().UseExportImageResidu().IsInit()) || (! IsLastEtapeOfLastIter()))
     return;

  const cUseExportImageResidu & aUEIR = Param().UseExportImageResidu().Val();


  double aSomPds     = 0.0;
  double aSomPdsRes  = 0.0;

  for (int aK1=0 ; aK1< aNupl.NbPts() ; aK1++)
  {
      double aPds1 = aVpds[aK1];
      if (aPds1>0)
      {
         for (int aK2=0 ; aK2< aNupl.NbPts() ; aK2++)
         {
             double aPds2 = aVpds[aK2];
             if ((aK1!=aK2) && (aPds2>0))
             {
                const cBasicGeomCap3D * aCam1 = aVP[aK1]->GenCurCam();
                const cBasicGeomCap3D * aCam2 = aVP[aK2]->GenCurCam();
                Pt2di aSzCam1 = aCam1->SzBasicCapt3D();
                Pt2dr aP1 = aNupl.PK(aK1);
                Pt2dr aP2 = aNupl.PK(aK2);
                   
                Pt2dr aDir;
                double aRes = aCam1->EpipolarEcart(aP1,*aCam2,aP2,&aDir);
                cInfoAccumRes anInfo(aP1,ElMin(aPds1,aPds2),aRes,aDir);

                double aPds = aPds1 * aPds2;
                aSomPds    += aPds;
                aSomPdsRes += aPds * ElAbs(aRes);

                if (aK1<aK2)
                {
                    std::string aNamePair = "Pair-"+aVP[aK1]->Name() + "-" + aVP[aK2]->Name();
                    AddOneInfoImageResidu(anInfo,aNamePair,aSzCam1,aUEIR.SzByPair().Val(),true,-1);
                }
                AddOneInfoImageResidu(anInfo,"Pose-"+aVP[aK1]->Name(),aSzCam1,aUEIR.SzByPose().Val(),false,5);

                cCalibCam *  aCC1 = aVP[aK1]->CalibCam();
                if (aCC1)
                {
                   AddOneInfoImageResidu(anInfo,"Cam-"+aCC1->KeyId(),aSzCam1,aUEIR.SzByCam().Val(),false,10);
                }
             }
         }
      }
  }

  if (aSomPds)
  {
       mAVGR.AddResidu(aPt,aSomPdsRes/aSomPds);
  }
}

void cAppliApero::ExportImageResidu(const std::string & aName,const cAccumResidu & anAccum) 
{
    const cUseExportImageResidu & aUEIR = Param().UseExportImageResidu().Val();
    // ELISE_fp::MkDirRec(mDirExportImRes);

    const_cast<cAccumResidu &>(anAccum).Export(mDirExportImRes,aName,aUEIR,mFileExpImRes);
}

void cAppliApero::ExportImageResidu() 
{
  if ((!Param().UseExportImageResidu().IsInit()) )
     return;

  const cUseExportImageResidu & aUEIR = Param().UseExportImageResidu().Val();
  mDirExportImRes =  DC() + "Ori" + aUEIR.AeroExport() + "/ImResidu/";
  ELISE_fp::MkDirRec(mDirExportImRes);

  mFileExpImRes = FopenNN(mDirExportImRes+"StatRes.txt","w","cAppliApero::ExportImageResidu");

  for (auto it=mMapAR.begin() ; it!=mMapAR.end() ; it++)
  {
       ExportImageResidu(it->first,*(it->second));
  }
  fclose(mFileExpImRes);

   mAVGR.DoResidu(mDirExportImRes,aUEIR.NbMesByCase().Val());

}

//============================================


int PROF_UNDEF() { return -1; }


int cPoseCam::theCpt = 0;

int  TheDefProf2Init = 1000000;

void cPoseCam::SetNameCalib(const std::string & aNameC)
{
   mNameCalib = aNameC;
}

static int theNumCreate =0;


cStructRigidInit * cPoseCam::GetSRI(bool SVP) const 
{
   if (!SVP && (mSRI==0))
   {
       ELISE_ASSERT(false,"NO SRI");
   }
   return mSRI;
}
void  cPoseCam::SetSRI(cStructRigidInit * aSRI) 
{
    ELISE_ASSERT(mSRI==0,"Muliple SRI set");
    mSRI = aSRI;
}

cPreCompBloc * cPoseCam::GetPreCompBloc(bool SVP) const 
{
   if (!SVP && (mBlocCam==0))
   {
       ELISE_ASSERT(false,"NO Boc Cam");
   }
   return mBlocCam;
}
void  cPoseCam::SetPreCompBloc(cPreCompBloc * aBloc) 
{
    ELISE_ASSERT(mBlocCam==0,"Muliple Bloc set");
    mBlocCam = aBloc;
}

cPreCB1Pose * cPoseCam::GetPreCB1Pose(bool SVP) const 
{
   if (!SVP && (mPoseInBlocCam==0))
   {
       ELISE_ASSERT(false,"NO Pose in Boc Cam");
   }
   return mPoseInBlocCam;
}
void  cPoseCam::SetPreCB1Pose(cPreCB1Pose * aPoseInBloc) 
{
    ELISE_ASSERT(mPoseInBlocCam==0,"Muliple Bloc set");
    mPoseInBlocCam = aPoseInBloc;
}


void cPoseCam::SetOrInt(const cTplValGesInit<cSetOrientationInterne> & aTplSI)
{
  if (! aTplSI.IsInit()) return;

  const cSetOrientationInterne & aSOI = aTplSI.Val();

   cSetName *  aSelector = mAppli.ICNM()->KeyOrPatSelector(aSOI.PatternSel());

   if (! aSelector->IsSetIn(mName))
      return;

  std::string aNameFile =  mAppli.DC() + mAppli.ICNM()->Assoc1To1(aSOI.KeyFile(),mName,true);

   cAffinitePlane aXmlAff = StdGetObjFromFile<cAffinitePlane>
                            (
                                 aNameFile,
                                 StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                 aSOI.Tag().Val(),
                                 "AffinitePlane"
                            );

    ElAffin2D  anAffM2C = Xml2EL(aXmlAff);

   if (! aSOI.M2C())
      anAffM2C = anAffM2C.inv();

   if (aSOI.AddToCur())
      mOrIntM2C= anAffM2C * mOrIntM2C ; 
   else
      mOrIntM2C= anAffM2C;

   //  Si on le fait avec les marques fiduciaires ca ecrase le reste

   mOrIntC2M = mOrIntM2C.inv();
}

cPoseCam * cPoseCam::DownCastPoseCamSVP() {return this;}
const cPoseCam * cPoseCam::DownCastPoseCamSVP() const {return this;}

cCalibCam *  cPoseCam::CalibCam() const {return mCalib;}
cGenPDVFormelle *   cPoseCam::PDVF()  {return  mCF;}
const cGenPDVFormelle *   cPoseCam::PDVF() const  {return  mCF;}


cPoseCam::cPoseCam
(
     cAppliApero & anAppli,
     const cPoseCameraInc & aPCI,
     const std::string & aNamePose,
     const std::string & aNameCalib,
     cPoseCam *             aPRat,
     cCompileAOI  *         aCompAOI
)   :
    cGenPoseCam (anAppli,aNamePose),
    // mAppli   (anAppli),
    mNameCalib (aNameCalib),
    // mName    (aNamePose),
    mCpt     (-1),
    mProf2Init (TheDefProf2Init),
    mPdsTmpMST (0.0),
    mPCI     (&aPCI),
	mCalib(NULL),
    mPoseRat (aPRat),
    mPoseInitMST1 (0),
    mPoseInitMST2 (0),
    mCamRF   (mPoseRat ? mPoseRat->mCF : 0),
	mCF(NULL),
	mRF(NULL),
    mAltiSol     (ALTISOL_UNDEF()),
    mProfondeur  (PROF_UNDEF()),
    mTime        (TIME_UNDEF()),
    mPrioSetAlPr (-1),
    mLastCP      (0),
    mCompAOI     (aCompAOI),
    mFirstBoxImSet (false),
    mImageLoaded (false),
    mBoxIm       (Pt2di(0,0),Pt2di(0,0)),
    mIm          (1,1),
    mTIm         (mIm),
    // mMasqH       (mAppli.MasqHom(aNamePose)),
    // mTMasqH      (mMasqH ? new  TIm2DBits<1>(*mMasqH) : 0),
    // mObsCentre   (0,0,0),
    mHasObsOnCentre (false),
    mHasObsOnVitesse (false),
    mLastItereHasUsedObsOnCentre (false),
    mNumBande     (0),
    mPrec         (NULL),
    mNext         (NULL),
    mNumCreate    (theNumCreate++),
    mNbPosOfInit  (-1),
    mFidExist     (false),
    mCamNonOrtho         (0),
    mEqOffsetGPS         (0),
    mSRI                 (nullptr),
    mBlocCam             (nullptr),
    mNumTimeBloc         (-1),
    mPoseInBlocCam       (nullptr),
    mUseRappelPose       (false),
    mRotURP              (ElRotation3D::Id)
{
    mPrec = this;
    mNext = this;

    std::string anIdGlob =  mAppli.GetNewIdIma(aNamePose);
    mCalib = mAppli.CalibFromName(aNameCalib,this);
    // mCF	=  mCalib->PIF().NewCam(cNameSpaceEqF::eRotLibre,ElRotation3D::Id,mCamRF,aNamePose,true,false,mAppli.HasEqDr());
    mCF	=  mCalib->PIF().NewCam(cNameSpaceEqF::eRotLibre,ElRotation3D::Id,mCamRF,anIdGlob,true,false,mAppli.HasEqDr());
    mCF->SetNameIm(aNamePose);
    mRF = &mCF->RF();

   SetOrInt(mAppli.Param().GlobOrInterne());
   SetOrInt(aPCI.OrInterne());

   std::pair<std::string,std::string> aPair = mAppli.ICNM()->Assoc2To1("Key-Assoc-STD-Orientation-Interne",mName,true);
   std::string aNamePtsCam = aPair.first;
   std::string aNamePtsIm  = aPair.second;

   if ((aPair.first!="NONE") && (ELISE_fp::exist_file(mAppli.OutputDirectory()+ aNamePtsIm)))
   {
       cMesureAppuiFlottant1Im aMesCam = mAppli.StdGetOneMAF(aNamePtsCam);

       // Correction du bug apparu lors du stage de Tibaut Sauter, lorsque les marques ont une origine tres
       // loin de zero, la box images est tres differente de par ex [0,0] [23x23], du coup le distorsion n'est pas coupe
       // au bon endroit
       Pt2dr aPMin(1e9,1e9);
       for 
       (
           std::list<cOneMesureAF1I>::iterator itAp = aMesCam.OneMesureAF1I().begin(); 
           itAp != aMesCam.OneMesureAF1I().end();
           itAp++
       )
       {
           aPMin = Inf(aPMin,itAp->PtIm());
       }

       for 
       (
           std::list<cOneMesureAF1I>::iterator itAp = aMesCam.OneMesureAF1I().begin(); 
           itAp != aMesCam.OneMesureAF1I().end();
           itAp++
       )
       {
           itAp->PtIm() = itAp->PtIm()-aPMin;
       }



       cMesureAppuiFlottant1Im aMesIm  = mAppli.StdGetOneMAF(aNamePtsIm);

       ElPackHomologue  aPack = PackFromCplAPF(aMesIm,aMesCam);
       ElAffin2D anAf = ElAffin2D::L2Fit(aPack);

       mOrIntM2C = anAf.inv();
       mFidExist = true;
   }

   mOrIntC2M = mOrIntM2C.inv();
   InitAvantCompens();

   if (aPCI.IdOffsetGPS().IsInit())
   {
       cAperoOffsetGPS * anOfs = mAppli.OffsetNNOfName(aPCI.IdOffsetGPS().Val());
       mEqOffsetGPS =  mAppli.SetEq().NewEqOffsetGPS(*mCF,*(anOfs->BaseUnk()));
   }

}

cEqOffsetGPS *   cPoseCam::EqOffsetGPS()
{
   return mEqOffsetGPS;
}

int cPoseCam::NumCreate() const
{
   return mNumCreate;
}


bool cPoseCam::AcceptPoint(const Pt2dr & aP) const
{


    if (mCalib-> CamInit().IsScanned())
    {
        Pt2dr aSz = Pt2dr(mCalib->SzIm());
        if ((aP.x<=0) || (aP.y <=0) || (aP.x>=aSz.x) || (aP.y>=aSz.y))
           return false;
    }

    return true;
}

bool cPoseCam::FidExist() const
{
   return mFidExist;
}


void cPoseCam::SetNumTimeBloc(int aNum)
{
   mNumTimeBloc = aNum;
}

int cPoseCam::DifBlocInf1(const cPoseCam & aPC) const
{
   if ((mNumTimeBloc==-1) || (aPC.mNumTimeBloc==-1)) return 1000;
   return ElAbs(mNumTimeBloc-aPC.mNumTimeBloc);
}



bool cPoseCam::IsInZoneU(const Pt2dr & aP) const
{
   return mCalib->IsInZoneU(aP);
}


Pt2di  cPoseCam::SzCalib() const
{
   return Calib()->SzIm();
}

void cPoseCam::SetLink(cPoseCam * aPrec,bool OK)
{
   mNumBande =  aPrec->mNumBande;
   if (OK) 
   {
      mPrec     = aPrec;
      aPrec->mNext = this;
   }
   else
   {
      mNumBande++;
   }
/*
    mNumBande =  aNumBande;
    mPrec     = aPrec;
    if (aPrec) aPrec->mNext = this;
*/
}


double  GuimbalAnalyse(const ElRotation3D & aR,bool show)
{
    // aR = aR.inv();
// aR  = ElRotation3D(Pt3dr(0,0,0),0,1.57,0);
    double aTeta01 = aR.teta01();
    double aTeta02 = aR.teta02();
    double aTeta12 = aR.teta12();

    double aEps = 1e-2;
    ElMatrix<double> aM1 = (ElRotation3D(aR.tr(),aTeta01+aEps,aTeta02,aTeta12).Mat()-aR.Mat())*(1/aEps);
    ElMatrix<double> aM2 = (ElRotation3D(aR.tr(),aTeta01,aTeta02+aEps,aTeta12).Mat()-aR.Mat())*(1/aEps);
    ElMatrix<double> aM3 = (ElRotation3D(aR.tr(),aTeta01,aTeta02,aTeta12+aEps).Mat()-aR.Mat())*(1/aEps);

    aM1 = aM1 * (1/sqrt(aM1.L2()));
    aM2 = aM2 * (1/sqrt(aM2.L2()));
    aM3 = aM3 * (1/sqrt(aM3.L2()));

    ElMatrix<double> aU1 = aM1;
    ElMatrix<double> aU2 = aM2 -aU1*aU1.scal(aM2);
    double aD2 = sqrt(aU2.L2());
    aU2 = aU2 *(1/aD2);


    ElMatrix<double> aU3 = aM3 -aU1*aU1.scal(aM3) -aU2*aU2.scal(aM3);

    double aD3 = sqrt(aU3.L2());

    if (show)
    {
        std::cout << aU2.scal(aU1) << " " << aU3.scal(aU1) << " " << aD2*aD3 << "\n";
    

        ShowMatr("M1",aM1);
        ShowMatr("M2",aM2);
        ShowMatr("M3",aM3);
        getchar();
    }
    return aD2 * aD3;
}

void cPoseCam::Trace() const
{
    if (! mAppli.TracePose(*this))
       return;

    std::cout   << mName ;

    if (RotIsInit())
    {
        ElRotation3D  aR = CurRot() ;
        std::cout <<  " C=" << aR.tr() ;
        std::cout <<  " Teta=" << aR.teta01() << " " << aR.teta02() <<  " " << aR.teta12()  ;
       
        if (mAppli.Param().TraceGimbalLock().Val())
        {
            std::cout << " GL-Score=" <<  GuimbalAnalyse(aR,false);
        }
    }
    std::cout << "\n";
}


double & cPoseCam::PdsTmpMST()
{
   return mPdsTmpMST;
}

void cPoseCam::Set0Prof2Init()
{
    mProf2Init = 0;
}

bool cPoseCam::ProfIsInit() const {return mProf2Init != TheDefProf2Init ;}


double cPoseCam::Time() const
{
  return mTime;
}

int cPoseCam::Prof2Init() const
{
    return mProf2Init;
}

void cPoseCam::UpdateHeriteProf2Init(const  cPoseCam & aC2)
{
    if (mProf2Init==TheDefProf2Init)
         mProf2Init = 0;

    ElSetMax(mProf2Init,1+aC2.mProf2Init);
}



cPoseCam * cPoseCam::PoseInitMST1()
{
   return mPoseInitMST1;
}
void cPoseCam::SetPoseInitMST1(cPoseCam * aPoseInitMST1)
{
   mPoseInitMST1 = aPoseInitMST1;
}


cPoseCam * cPoseCam::PoseInitMST2()
{
   return mPoseInitMST2;
}
void cPoseCam::SetPoseInitMST2(cPoseCam * aPoseInitMST2)
{
   mPoseInitMST2 = aPoseInitMST2;
}



std::string  cPoseCam::CalNameFromL(const cLiaisonsInit & aLI)
{
     std::string aName2 = aLI.NameCam();
     if (aLI.NameCamIsKeyCalc().Val())
     {
         aName2 =  mAppli.ICNM()->Assoc1To1(aName2,mName,aLI.KeyCalcIsIDir().Val());
     }

     return aName2;
}


cRotationFormelle & cPoseCam::RF()
{
   return *mRF;
}

ElRotation3D cPoseCam::CurRot() const
{
   return mRF->CurRot();
}



const std::string &  cPoseCam::NameCalib() const
{
   return mCalib->CCI().Name();
}



void cPoseCam::SetRattach(const std::string & aNameRat)
{
   if ((mPoseRat==0) || (mPoseRat != mAppli.PoseFromName(aNameRat)))
   {
       std::cout << mPoseRat << "\n";
       std::cout << "Pour rattacher " << mName << " a " << aNameRat << "\n";
       std::cout << "(Momentanement) : le ratachement doit etre prevu a l'initialisation";
       ELISE_ASSERT(false,"Rattachement impossible");
   }
}

void    cPoseCam::VirtualInitAvantCompens()
{
    mLastItereHasUsedObsOnCentre = false;
    AssertHasNotCamNonOrtho();
}

const CamStenope *  cPoseCam::CurCam() const
{
   if (mCamNonOrtho) return mCamNonOrtho;
   return  mCF->CameraCourante() ;
}
CamStenope *  cPoseCam::NC_CurCam() 
{
   if (mCamNonOrtho) return mCamNonOrtho;
   return  mCF->NC_CameraCourante() ;
}

const cBasicGeomCap3D * cPoseCam::GenCurCam () const { return CurCam(); }
cBasicGeomCap3D * cPoseCam::GenCurCam () { return NC_CurCam(); }





CamStenope *  cPoseCam::DupCurCam() const 
{
   if (mCamNonOrtho) return mCamNonOrtho;
   return  mCF->DuplicataCameraCourante() ;
}



double cPoseCam::GetProfDyn(int & Ok) const
{
    Ok = true;

    if (PMoyIsInit())
    {
        Ok =1 ;
        return ProfMoyHarmonik();
    }
    if (mLastEstimProfIsInit)
    {
       Ok =2 ;
       return mLasEstimtProf;
    }

    if (mProfondeur != PROF_UNDEF())
    {
       Ok = 3 ;
       return mProfondeur;
    }


    Ok = 0;
    return 0;
}






void cPoseCam::ActiveContrainte(bool Stricte)
{
    mAppli.SetEq().AddContrainte(mRF->StdContraintes(),Stricte);
}

void ShowResMepRelCoplan(cResMepRelCoplan aRMRC)
{
    std::vector<cElemMepRelCoplan> aV =  aRMRC.VElOk();
    std::cout << "  NB SOL = " << aV.size() << "\n";
    for (int aKS=0; aKS<int(aV.size()) ; aKS++)
    {
        cElemMepRelCoplan aS = aV[aKS];
        std::cout << "ANGLE " << aKS << " " << aS.AngTot() << "\n";
        ElRotation3D aR = aS.Rot();
        std::cout << aR.ImRecAff(Pt3dr(0,0,0)) << " "
                  << aR.teta01() << " "
                  << aR.teta02() << " "
                  << aR.teta12() << " "
                  << "\n";
    }
}


/*
void ShowPrecisionLiaison(ElPackHomologue  aPack,ElRotation3D aR1,ElRotation3D aR2)
{
   CamStenopeIdeale aC1(1.0,Pt2dr(0.0,0.0));
   aC1.SetOrientation(aR1.inv());
   CamStenopeIdeale aC2(1.0,Pt2dr(0.0,0.0));
   aC2.SetOrientation(aR2.inv());

   double aSD=0;
   double aSP=0;
   for (ElPackHomologue::iterator itP= aPack.begin(); itP!=aPack.end() ; itP++)
   {
        ElSeg3D aS1 = aC1.F2toRayonR3(itP->P1());
        ElSeg3D aS2 = aC2.F2toRayonR3(itP->P2());

	Pt3dr aPTer = aS1.PseudoInter(aS2);

	Pt2dr aQ1 = aC1.R3toF2(aPTer);
	Pt2dr aQ2 = aC2.R3toF2(aPTer);

	std::cout << aQ1 << aQ2 << euclid(aQ1,itP->P1()) << " " <<  euclid(aQ2,itP->P2())  << "\n";
	aSP += 2;
	aSD += euclid(aQ1,itP->P1()) + euclid(aQ2,itP->P2());

   }

   std::cout << "DMOYENNE " << aSD/aSP * 1e4 << "/10000\n";
}
*/
 

ElMatrix<double> RFrom3P(const Pt3dr & aP1, const Pt3dr & aP2, const Pt3dr & aP3)
{
    Pt3dr aU = aP2-aP1;
    aU = aU / euclid(aU);
    Pt3dr aW = aU ^(aP3-aP1);
    aW = aW / euclid(aW);
    Pt3dr aV =  aW ^aU;
    return ElMatrix<double>::Rotation(aU,aV,aW);
}

void  TTTT (ElPackHomologue & aPack)
{
//
   int aCpt=0;
   Pt2dr aS1(0,0);
   Pt2dr aS2(0,0);
   double aSP=0;
   std::cout << " ccccccccccc " << aPack.size() << "\n";
   for 
   (
       ElPackHomologue::iterator itP=aPack.begin() ;
       itP!=aPack.end();
       itP++
   )
   {
       aS1 = aS1 + itP->P1() * itP->Pds();
       aS2 = aS2 + itP->P2() * itP->Pds();
       aSP += itP->Pds();
       aCpt++;
       if (euclid(itP->P1()) > 1e5)
          std::cout << itP->P1() << itP->P2() << itP->Pds()  << "\n";
   }
   std::cout << "SOMES ::  "<< (aS1/aSP) << " " << (aS2/aSP) << "\n";
}

void  TTTT (cElImPackHom & anIP)
{
    ElPackHomologue aPack = anIP.ToPackH(0);
    TTTT(aPack);
}

void  cPoseCam::TenteInitAltiProf
      (
           int    aPrio,
           double anAlti,
	   double aProf
      )
{
   if ((aProf != PROF_UNDEF()) && (aPrio > mPrioSetAlPr))
   {
         mProfondeur = aProf;
         mAltiSol = anAlti;
	 mPrioSetAlPr = aPrio;
   }
}
extern bool AllowUnsortedVarIn_SetMappingCur;

cPoseCam * cPoseCam::Alloc
           (
               cAppliApero & anAppli,
               const cPoseCameraInc & aPCI,
               const std::string & aNamePose,
               const std::string & aNameCalib,
               cCompileAOI * aCompAOI
           )
{

    cPoseCam * aPRat=0;

    if (aPCI.PosesDeRattachement().IsInit())
    {
        // En fait le tri sur les variables n'avait pas d'effet puisque la fonction est symetrique !!
        AllowUnsortedVarIn_SetMappingCur = true;
        aPRat = anAppli.PoseCSFromNameGen
                (
                    aPCI.PosesDeRattachement().Val(),
                    aPCI.NoErroOnRat().Val()
                );

    }

    cPoseCam * aRes = new cPoseCam(anAppli,aPCI,aNamePose,aNameCalib,aPRat,aCompAOI);
    return aRes;
}

int   cPoseCam::NbPosOfInit(int aDef)
{
   return (mNbPosOfInit>=0) ? mNbPosOfInit : aDef;
}

void  cPoseCam::SetNbPosOfInit(int aNbPosOfInit)
{
   mNbPosOfInit = aNbPosOfInit;
}


void cPoseCam::DoInitIfNow()
{

    mPreInit = true;


    mAppli.AddRotPreInit();
    if (mPCI->InitNow().Val())
    {
       InitRot();
    }
}

/*
void  cPoseCam::AddLink(cPoseCam * aPC)
{
   if (! BoolFind(mVPLinked,aPC))
      mVPLinked.push_back(aPC);
}
*/



void cPoseCam::PCSetCurRot(const ElRotation3D & aRot)
{
    ELISE_ASSERT(!mUseRappelPose,"Internam Error, probaly bascule + UseRappelPose");


    AssertHasNotCamNonOrtho();
    mCF->SetCurRot(aRot,aRot);
}


void  cPoseCam::SetBascRig(const cSolBasculeRig & aSBR)
{
 
    //  Correc MPD 20/05/21 : put PCSetCurRot  after aP =  aSBR(aP);
    // else the bascule on the point is done twice and altisol is bad ...
    // PCSetCurRot(aSBR.TransformOriC2M(CurRot()));

    Pt3dr aP;
    if (mSomPM)
    {
       aP = mPMoy / mSomPM;
       mSomPM = 0;
    }
    else
    {
        const CamStenope *  aCS = CurCam() ;
        if (mProfondeur == PROF_UNDEF())
        {
            std::cout << "Warn : NoProfInBasc For camera =" << mName << "\n";
            PCSetCurRot(aSBR.TransformOriC2M(CurRot()));
            return;
            // ELISE_ASSERT( false,"No Profondeur in cPoseCam::SetBascRig");
        }
        else
        {
            aP =  aCS->ImEtProf2Terrain(aCS->Sz()/2.0,mProfondeur);
            aP =  aSBR(aP);
        }
    }
    PCSetCurRot(aSBR.TransformOriC2M(CurRot()));



    const CamStenope *  aCS = CurCam() ;
    mAltiSol = aP.z;
    mProfondeur = aCS->ProfondeurDeChamps(aP);
    mLasEstimtProf = mProfondeur;
}


void cPoseCam::VirtualAddPMoy
     (
           const Pt2dr & aPIm,
           const Pt3dr & aP,
           int aKPoseThis,
           const std::vector<double> * aVPds,
           const std::vector<cGenPoseCam*>*
     )
{
  if (mAppli.UsePdsCalib())
      mCalib->AddPds(aPIm,(*aVPds)[aKPoseThis]);

}


void cPoseCam::BeforeCompens()
{
   mRF->ReactuFcteurRapCoU();
}

void cPoseCam::InitCpt()
{
    if (mCpt<0)
    {
       mCpt  = theCpt++;
       mAppli.AddPoseInit(mCpt,this);
    }
}

bool cPoseCam::HasObsOnCentre() const
{
   return mHasObsOnCentre;
}

bool  cPoseCam::LastItereHasUsedObsOnCentre() const
{
    return mLastItereHasUsedObsOnCentre;
}

void cPoseCam::AssertHasObsCentre() const
{
   if (!mHasObsOnCentre)
   {
       std::cout << "Name Pose = " << mName << "\n";
       ELISE_ASSERT
       (
             false,
             "Observation on centre (GPS) has no been associated to camera"
       );
   }
}

bool cPoseCam::HasObsOnVitesse() const
{
   return mHasObsOnVitesse;
}

void cPoseCam::AssertHasObsVitesse() const
{
    AssertHasObsCentre();
    if (! mObsCentre.mVitesse.IsInit())
    {
       std::cout << "Name Pose = " << mName << "\n";
       ELISE_ASSERT
       (
             false,
             "No speed has  been associated to camera"
       );
    }
}

const Pt3dr  & cPoseCam::ObsCentre() const
{
   AssertHasObsCentre();
 
   return mObsCentre.mCentre;
}

Pt3dr   cPoseCam::Vitesse() const
{
    AssertHasObsVitesse();
    return mObsCentre.mVitesse.Val();
}


bool cPoseCam::IsId(const ElAffin2D & anAff) const
{
    Pt2dr aSz =  Pt2dr(mCalib->SzIm());
    Box2dr aBox(Pt2dr(0,0),aSz);
    Pt2dr aCoins[4];
    aBox.Corners(aCoins);
    double aDiag = euclid(aSz);
    // double aDiag2 = 0;

    double aDMax = 0.0;

    for (int aK=0 ; aK<4 ; aK++)
    {
         Pt2dr aP1 = aCoins[aK];
         Pt2dr aP2 = anAff(aP1);
         double aD = euclid(aP1,aP2) / aDiag;
         ElSetMax(aDMax,aD);
    }

    return aDMax < 1e-3;
}
/*
*/

double DistanceMatr(const ElRotation3D & aR1,const ElRotation3D & aR2)
{
   ElMatrix<double> aMatr = aR1.inv().Mat() * aR2.Mat(); 
   ElMatrix<double> aId(3,true);

   return aId.L2(aMatr);
}

/*class cTransfo3DIdent : public cTransfo3D
{
     public :
          std::vector<Pt3dr> Src2Cibl(const std::vector<Pt3dr> & aSrc) const {return aSrc;}

};*/


extern bool DebugOFPA;
extern int aCPTOkOFA ;
extern int aCPTNotOkOFA ;

void   cPoseCam::InitRot()
{
   const cLiaisonsInit * theLiasInit = 0;
   mNumInit =  mAppli.NbRotInit();
   if (mAppli.ShowMes())
      std::cout << "NUM " << mNumInit << " FOR " << mName<< "\n";
/*
{

if (mNumInit==90)
{
   BugFE=true;
}
else
{
   std::cout << "END BUG FE \n";
   BugFE=false;
}
 BugFE=true;
}
*/


   if (mPCI->IdBDCentre().IsInit())
   {

//  std::cout << "CCCcccC : " << mName <<  " " << mAppli.HasObsCentre(mPCI->IdBDCentre().Val(),mName) << "\n";
      if (mAppli.HasObsCentre(mPCI->IdBDCentre().Val(),mName))
      {
          mObsCentre = *( mAppli.ObsCentre(mPCI->IdBDCentre().Val(),mName).mVals );
          mHasObsOnCentre = mObsCentre.mHasObsC;
          mHasObsOnVitesse = mHasObsOnCentre && mObsCentre.mVitFiable && mObsCentre.mVitesse.IsInit();

//   std::cout << "NameBDDCCC " << mName << " HasC " << mHasObsOnCentre << "\n";
      }
   }

  
    // std::cout << mName << "::Prof=" << mProf2Init << "\n";
// std::cout <<  "Init Pose " << aNamePose << "\n";

   InitCpt();


    double aProfI = mPCI->ProfSceneImage().ValWithDef(mAppli.Param().ProfSceneChantier().Val());
    ElRotation3D aRot(Pt3dr(0,0,0),0,0,0);
    std::string aNZPl ="";
    double aDZPl = -1;
    double aDZPl2 = -1;
    cPoseCam * aCam2PL = 0;
    double aLambdaRot=1;

    CamStenope* aCS1 =    (mCalib->PIF().CurPIF());

    double  aProfPose = -1;
    double  anAltiSol = ALTISOL_UNDEF();


   bool isMST = mPCI->MEP_SPEC_MST().IsInit();
   if (isMST)
   {
       ELISE_ASSERT
       (
           mPCI->PoseFromLiaisons().IsInit(),
           "MST requires PoseFromLiaisons"
       );
   }

    bool isAPC =  mAppli.Param().IsAperiCloud().Val();
    bool isForISec =  mAppli.Param().IsChoixImSec().Val();
    bool initFromBD = false;


    if  (mSRI)
    {
        ElRotation3D aR1 = mSRI->mCMere->CurRot()  ;  // R1 to M
        ElRotation3D aL1Bis = aR1 * mSRI->mR0m1L0;
        aRot = aL1Bis;
    }
    else if (mPCI->PosId().IsInit())
    {
         aRot =  ElRotation3D(Pt3dr(0,0,0),0,0,-PI);
    }
    else if (mPCI->PosFromBlockRigid().IsInit())
    {
         mAppli.PreInitBloc(mPCI->PosFromBlockRigid().Val());
         aRot = GetPreCB1Pose(false)->mRot;
    }
    else if(mPCI->PosFromBDOrient().IsInit())
    {
        initFromBD = true;
	const std::string & anId = mPCI->PosFromBDOrient().Val();
        aRot =mAppli.Orient(anId,mName);
	cObserv1Im<cTypeEnglob_Orient>  &  anObs = mAppli.ObsOrient(anId,mName);

         bool Ok1 = IsId(anObs.mOrIntC2M);
         bool Ok2 = IsId(anObs.mOrIntC2M * mOrIntC2M.inv());
         ELISE_ASSERT
         (
             Ok1 || Ok2,
             "Specicied Internal Orientation is incompatible with Fiducial marks"
         );
         

	aProfPose = anObs.mProfondeur;
	anAltiSol =  anObs.mAltiSol;
        mTime     =  anObs.mTime;
    }
    else if (mPCI->PoseInitFromReperePlan().IsInit())
    {
       cPoseInitFromReperePlan aPP = mPCI->PoseInitFromReperePlan().Val();
       ElPackHomologue aPack;
       std::string  aNamePose2 = aPP.NameCam();
       std::string  aNameCal2 = mAppli.NameCalOfPose(aNamePose2);

       std::string aTestNameFile = mAppli.DC()+aPP.IdBD();
       if (ELISE_fp::exist_file(aTestNameFile))
       {
            ELISE_ASSERT(false,"Obsolet Init Form repere plan");
            // Onsolete, pas cohrent avec orient interen
            // aPack = ElPackHomologue::- FromFile(aTestNameFile);
       }
       else 
           mAppli.InitPack(aPP.IdBD(),aPack,mName,aPP.NameCam());
       CamStenope & aCS2 = mAppli.CalibFromName(aNameCal2,0)->CamInit();

// TTTT(aPack);



       aPack = aCS1->F2toPtDirRayonL3(aPack,&aCS2);
       cResMepRelCoplan aRMRC = aPack.MepRelCoplan(1.0,aPP.L2EstimPlan().Val());
       cElemMepRelCoplan & aSP = aRMRC.RefBestSol();

       // aM1 aM2 aM3 -> coordonnees monde, specifiees par l'utilisateur
       // aC1 aC2 aC3 -> coordonnees monde

       Pt3dr aM1,aM2,aM3;
       Pt2dr aIm1,aIm2,aIm3;
       Pt3dr aDirPl;
       bool aModeDir = false;
       if (aPP.MesurePIFRP().IsInit())
       {
           aM1 = aPP.Ap1().Ter();
           aM2 = aPP.Ap2().Ter();
           aM3 = aPP.Ap3().Ter();
           aIm1 = aCS1->F2toPtDirRayonL3(aPP.Ap1().Im());
           aIm2 = aCS1->F2toPtDirRayonL3(aPP.Ap2().Im());
           aIm3 = aCS1->F2toPtDirRayonL3(aPP.Ap3().Im());
       }
       else if (aPP.DirPlan().IsInit())
       {
           aModeDir = true;
           aDirPl = aPP.DirPlan().Val();
           aM1 = Pt3dr(0,0,0);

           Pt3dr aW = vunit(aPP.DirPlan().Val());
           aM2 = OneDirOrtho(aW);
           aM3 = aW ^ aM2;
           
           // aIm1 = aCS1.Sz() /2.0;
           // aIm2 = aIm1 + Pt2dr(1.0,0.0);
           // aIm3 = aIm1 + Pt2dr(0.0,1.0);
           aIm1 = Pt2dr(0,0);
           aIm2 = aIm1 + Pt2dr(0.1,0.0);
           aIm3 = aIm1 + Pt2dr(0.0,0.1);
       }



       Pt3dr aC1 = aSP.ImCam1(aIm1);
       Pt3dr aC2 = aSP.ImCam1(aIm2);
       Pt3dr aC3 = aSP.ImCam1(aIm3);

       double aFMult=0;


       if (aPP.DEuclidPlan().IsInit())
           aFMult = aPP.DEuclidPlan().Val() / aSP.DistanceEuclid();
       else
           aFMult = euclid(aM2-aM1)/euclid(aC2-aC1);
        aDZPl = aSP.DPlan() * aFMult;
        aNZPl = aPP.OnZonePlane();

        aC1 = aC1 * aFMult;
        aC2 = aC2 * aFMult;
        aC3 = aC3 * aFMult;

	ElMatrix<double> aMatrM = RFrom3P(aM1,aM2,aM3);
	ElMatrix<double> aMatrC = RFrom3P(aC1,aC2,aC3);
        ElMatrix<double>  aMatrC2M = aMatrM * gaussj(aMatrC);

	Pt3dr aTr = aM1 - aMatrC2M * aC1;

        if (aModeDir)
        {
            Pt3dr aC1 = aMatrC2M*Pt3dr(1,0,0);
            Pt3dr aC2 = aMatrC2M*Pt3dr(0,1,0);
            Pt3dr aC3 = aMatrC2M*Pt3dr(0,0,1);
            double aScal = scal(aC3,aDirPl);

            // std::cout << "CSAL = " << aScal << "\n";
            // std::cout << aMatrC2M*Pt3dr(1,0,0) << aMatrC2M*Pt3dr(0,1,0) <<  aMatrC2M*Pt3dr(0,0,1) << "\n";

            if (aScal <0)
            {
                 aMatrC2M = ElMatrix<double>::Rotation(aC1,aC2*-1,aC3*-1);
            }
            // std::cout << aMatrC2M*Pt3dr(1,0,0) << aMatrC2M*Pt3dr(0,1,0) <<  aMatrC2M*Pt3dr(0,0,1) << "\n";
        }

	aRot = ElRotation3D(aTr,aMatrC2M,true);

        anAltiSol = aM1.z;
        aProfPose = euclid(aC1);
	mAppli.AddPlan(aNZPl,aM1,aM2,aM3,false);
	//Pt3dr  uM = 
    }
    else if(mPCI->PosFromBDAppuis().IsInit())
    {
         // DebugOFPA = (mName=="Im00523.png");
         if (mAppli.ShowMes() || DebugOFPA)
            std::cout << "InitByAppuis " << mName  << "\n\n";
         const cPosFromBDAppuis & aPFA = mPCI->PosFromBDAppuis().Val();
	 const std::string & anId = aPFA.Id();


         std::list<Appar23> aL = mAppli.AppuisPghrm(anId,mName,mCalib);
         tParamAFocal aNoPAF;
         CamStenopeIdeale aCamId(true,1.0,Pt2dr(0.0,0.0),aNoPAF);
	 double aDMin;

         Pt3dr aDirApprox;
         Pt3dr * aPtrDirApprox=0;
         if (aPFA.DirApprox().IsInit())
         {
               aDirApprox = aPFA.DirApprox().Val();
               aPtrDirApprox = &aDirApprox;
         }

	 // aRot = aCamId.CombinatoireOFPA(anAppli.Param().NbMaxAppuisInit().Val(),aL,&aDMin);
	 aRot = aCamId.RansacOFPA(true,aPFA.NbTestRansac(),aL,&aDMin,aPtrDirApprox);

/*
*/
	 aRot = aRot.inv();

         if (mAppli.ShowMes() || DebugOFPA)
            std::cout << mName << " DIST-MIN  = " << aDMin << aRot.ImAff(Pt3dr(0,0,0)) << " "  <<  aRot.ImVect(Pt3dr(0,0,1))  << " OK " << aCPTOkOFA << " NotOk " << aCPTNotOkOFA << "\n\n";
	 // cObserv1Im<cTypeEnglob_Appuis>  &  anObs = mAppli.ObsAppuis(anId,mName);
	 // Pt3dr aCdg =  anObs.mBarryTer;
         Pt3dr aCdg = BarryImTer(aL).pter;
	 anAltiSol = aCdg.z;
	 Pt3dr aDirVisee = aRot.ImVect(Pt3dr(0,0,1));

	 aProfPose = scal(aDirVisee,aCdg-aRot.ImAff(Pt3dr(0,0,0)));
         if (DebugOFPA) getchar();
         DebugOFPA = false;
    }
    else if (mPCI->PoseFromLiaisons().IsInit())
    {
         cResolvAmbiBase * aRAB=0;
         const std::vector<cLiaisonsInit> & aVL = mPCI->PoseFromLiaisons().Val().LiaisonsInit();
	 // bool   aNormIsFixed = false;
	 int aNbL = (int)aVL.size();

         
	 ElPackHomologue  aPack0;
	 ElRotation3D     anOrAbs0 (Pt3dr(0,0,0),0,0,0);

         Pt3dr aP0Pl,aP1Pl,aP2Pl;
	 double aMultPl=1.0;
         
         
	 std::vector<cPoseCam *> aVPC;
	 //cPoseCam * aCam00 = 0;
         if (isMST)
         {
             if (PoseInitMST1()==0)
             {
                std::cout << "For : " << mName << "\n";
                ELISE_ASSERT(false,"MST1 Incoh");
             }
             aNbL = PoseInitMST2() ? 2 : 1;
// std::cout << "isMST " << aNbL << " " << mName << "\n";
             if (aVL[0].OnZonePlane().IsInit())
                aNbL=1;
         }

         bool aRPure = false;
         const cLiaisonsInit * pLI0 = 0;
	 for (int aK=0 ; aK<aNbL ; aK++)
	 {
            const cLiaisonsInit & aLI = aVL[isMST?0:aK];
            if (aK==0) 
            {
               pLI0 = &aLI;
               theLiasInit = pLI0;
            }

	    std::string  aName2;
	    cPoseCam * aCam2 = 0;
            if (isMST)
            {
                aCam2 = (aK==0) ? PoseInitMST1() : PoseInitMST2();
                aName2 = aCam2->Name();

                ELISE_ASSERT
                (
                    aLI.IdBD()==mAppli.SymbPack0(),
                    "MST must be used with first Pack "
                );
            }
            else
            {
	       aName2 = CalNameFromL(aLI);
	       aCam2 = mAppli.PoseFromName(aName2);
            }
            // ElSetMax(mProf2Init,1+aCam2->mProf2Init);


            if (! aCam2->RotIsInit())
            {
               std::cout << "For " << mName << "/" << aName2;
               ELISE_ASSERT(false,"Incohernce : Init based on Cam not init");
            }
	    aVPC.push_back(aCam2);
	    CamStenope &  aCS2 = aCam2->Calib()->CamInit();

	    bool aBaseFixee = false;
	    double aLBase=1.0;
	    if (aLI.LongueurBase().IsInit() && ((!isMST) || (aNbL==1)))
	    {
	      // Ce serait incoherent puisque les liaisons multiples servent a
	      // fixer la longueur de la base
	        ELISE_ASSERT(aNbL==1,"Ne peut fixe la longueur de base avec plusieurs liaisons");
		aBaseFixee = true;
		aLBase  = aLI.LongueurBase().Val();
	    }


	    // if (aK==0) aCam0 = aCam;

            // ElPackHomologue aPack = anAppli.PackPhgrmFromCple(&aCS1,aNamePose,&aCS2,aName2);
	    ElPackHomologue aPack;
	    mAppli.InitPackPhgrm(aLI.IdBD(),aPack,mName,aCS1,aName2,&aCS2);
	    double aProfC = aLI.ProfSceneCouple().ValWithDef(aProfI);

            if (aK==0)
	    {
               ElRotation3D aOrRel0(Pt3dr(0,0,0),0,0,0);
	       aPack0 = aPack;
	       anOrAbs0 = aCam2->CurRot() ;
               if (aLI.InitOrientPure().Val())
               {
                  ELISE_ASSERT(aNbL==1,"Multiple Liaison with InitOrientPure");
                  ElMatrix<REAL> aMat =  aPack.MepRelCocentrique(aLI.NbTestRansacOrPure().Val(),aLI.NbPtsRansacOrPure().Val());
                  aOrRel0 = ElRotation3D(Pt3dr(0,0,0),aMat,true);
                  aRPure = true;
                  anAltiSol = aCam2->AltiSol();
                  aProfPose = aCam2->Profondeur();
               }
               else if (aLI.OnZonePlane().IsInit())
	       {
	          aNZPl = aLI.OnZonePlane().Val();
	          cResMepRelCoplan aRMRC = aPack.MepRelCoplan(aLBase,aLI.L2EstimPlan().Val());
		  cElemMepRelCoplan & aSP = aRMRC.RefBestSol();


		  aP0Pl = aSP.P0();
		  aP1Pl = aSP.P1();
		  aP2Pl = aSP.P2();
		  aOrRel0 = aSP.Rot();

		  if (aNbL==1)
		  {
		      double aMul=0;
		      if (DicBoolFind(aCam2->mDZP,aNZPl))
		      {
	                   ELISE_ASSERT(!aBaseFixee,"Ne peut fixe la longueur de base avec liaison plane");
			   aMul = aCam2->mDZP[aNZPl] /aSP.DPlan2();
		      }
		      else
		      {
		          if (aBaseFixee)
			     aMul = 1;
			  else
			  {
			     aMul = aProfC / aPack.Profondeur(aOrRel0);
                          }
                      }
		      aOrRel0.tr() = aOrRel0.tr() * aMul;
		      aDZPl = aSP.DPlan() * aMul;
		      aCam2->mDZP[aNZPl] = aSP.DPlan2() * aMul;
		      aMultPl = aMul;
		  }
		  else
		  {
		       // Sinon il faudra, une fois connu le multiplicateur donne
		       // par les autres liaisons mettre a jour le plan
		       aDZPl = aSP.DPlan() ;
		       // Et eventuellement initialiser Plan2
		       if (! DicBoolFind(aCam2->mDZP,aNZPl))
		       {
		           aDZPl2 = aSP.DPlan2();
			   aCam2PL = aCam2;
		       }
		  }
	       }
	       else
	       {
                   if ((aNbL<2) && (NbPosOfInit(mAppli.NbRotInit()) >=2))
                   {
                       ELISE_ASSERT 
                       (
                           mAppli.Param().AutoriseToujoursUneSeuleLiaison().Val(),
                           "Une seule liaison pour initialiser la pose au dela de 3"
                       );
                   }
	           bool L2 = aPack.size() > mAppli.Param().SeuilL1EstimMatrEss().Val();
                   double aDGen;
/*
std::cout << "TEST MEPS STD " << mName  << " L2 " << L2  
          << " " <<  mAppli.Param().SeuilL1EstimMatrEss().Val()<< "\n";
*/
	           aOrRel0 = aLI.TestSolPlane().Val()               ? 
                              aPack.MepRelGenSsOpt(aLBase,L2,aDGen) :
                             aPack.MepRelPhysStd(aLBase,L2)         ;
		   if (aNbL==1 && (! aBaseFixee))
                   {
		      aPack.SetProfondeur(aOrRel0,aProfC);
                   }
	       }

	       aOrRel0 = aCam2->CurRot() * aOrRel0;
	       aRot = aOrRel0;
	       aRAB = new cResolvAmbiBase(aCam2->CurRot(),aOrRel0);
            }
	    else
	    {
                
	         ELISE_ASSERT
                 (
                         (!aLI.OnZonePlane().IsInit()) || isMST,
                         "Seule la premiere liaison peut etre plane"
                 );
	         // ELISE_ASSERT(false,"Do not handle multi Liaison");
		 aRAB->AddHom(aPack,aCam2->CurRot());
	    }
	 }

	 if (aNbL > 1)
	 {
	      aRot = aRAB->SolOrient(aLambdaRot);
	      aMultPl= aLambdaRot;
	 }
         delete aRAB;

         // Calcul de l'alti et de la prof
         if (aRPure)
         {
         }
         else
	 {
	     CamStenopeIdeale aC1 = CamStenopeIdeale::CameraId(true,aRot.inv());
	     CamStenopeIdeale aC2 = CamStenopeIdeale::CameraId(true,anOrAbs0.inv());
	     double aD;
	     Pt3dr aCdg = aC1.CdgPseudoInter(aPack0,aC2,aD);


             anAltiSol = aCdg.z;
             aProfPose = aC1.ProfondeurDeChamps(aCdg) ;

	     for (int aK=0 ; aK<int(aVPC.size())  ; aK++)
	     {
	         CamStenopeIdeale aCK = CamStenopeIdeale::CameraId(true,aVPC[aK]->CurRot().inv());
		 double aPrK = aCK.ProfondeurDeChamps(aCdg) ;
		 aVPC[aK]->TenteInitAltiProf
		 (
		     (aK==0) ? 1 : 0,
		     aCdg.z,
		     aPrK
		 );
	     }
	 }

         int aNbRAp = pLI0->NbRansacSolAppui().Val();
         if (aNbRAp>0)
         {
             cObsLiaisonMultiple * anOLM = mAppli.PackMulOfIndAndNale(pLI0->IdBD(),mName);
             anOLM->TestMEPAppuis(mAppli.ZuUseInInit(),aRot,aNbRAp,*pLI0);
         }


	 if (aNZPl!="")
	 {
	    mAppli.AddPlan
	    (
	        aNZPl,
		aRot.ImAff(aP0Pl*aMultPl),
		aRot.ImAff(aP1Pl*aMultPl),
		aRot.ImAff(aP2Pl*aMultPl),
                true
	    );
	 }

    }
    else
    {
       ELISE_ASSERT(false,"cPoseCam::Alloc");
    }

{
if  (mSRI && MPD_MM())
{
#if (0)
   ElRotation3D aR1 = mSRI->mCMere->CurRot()  ;  // R1 to M
   ElRotation3D aL1Bis = aR1 * mSRI->mR0m1L0;
   aRot = aL1Bis;


   std::string aNameOri = "MPD-CmpPolygBlinis";
   CamStenope *  aCamR1 = mAppli.ICNM()->StdCamStenOfNames(mSRI->mCMere->Name(),aNameOri);
   CamStenope *  aCamL1 = mAppli.ICNM()->StdCamStenOfNames(Name(),aNameOri);

   ElRotation3D aRefRotL1 = aCamL1->Orient().inv();
   ElRotation3D aRefRotR1 = aCamR1->Orient().inv();

/*
   ElRotation3D aL1Bis =  aRotR1  * mSRI->mR0m1L0;

   ElRotation3D aDif = aRotL1 * aL1Bis.inv();
   ElMatrix<double> anId(3,true);
   if (anId.L2(aDif.Mat()) > -1)
   {
       std::cout  << "NAME= " << mName << " " <<  euclid(aRotL1.tr()-aL1Bis.tr()) <<  " " << anId.L2(aDif.Mat()) << "\n";
       getchar();
   }
*/
   

   ElRotation3D aL1 = aRot;
   ElRotation3D aR1 = mSRI->mCMere->CurRot()  ;  // R1 to M
   ElRotation3D aL1Bis = aR1 * mSRI->mR0m1L0;

   ElRotation3D aPassL1 =  aL1 *  aRefRotL1.inv() ; //                L1 to M
   ElRotation3D aPassR1 =  aR1 * aRefRotR1.inv()  ; //                L1 to M

   static ElRotation3D FirsrtPass = aPassR1;
   
   // ElRotation3D aDif = aL1.inv() * aRot;
   ElMatrix<double> anId(3,true);
   ElRotation3D aDif = aL1 * aL1Bis.inv();
   if (1) // anId.L2(aDif.Mat()) > 0.01)
   {
       std::cout  << "NAME= " << mName  << " " << mRotIsInit  
                 << " mere: " << mSRI->mCMere->mName << " " <<  mSRI->mCMere->mRotIsInit << "\n";

       std::cout << " TR:" <<  euclid(aL1.tr()-aL1Bis.tr()) 
                  <<  " MAT:" << anId.L2(aDif.Mat()) 
                  <<  " PassRL " << DistanceMatr(aPassL1,aPassR1)
                  <<  " PassFirst-R " << DistanceMatr(FirsrtPass,aPassR1)
                  <<  " PassFirst-L " << DistanceMatr(FirsrtPass,aPassL1)
                  << "\n";
               

       // getchar();
   }
   aRot = aL1Bis;
/*
   else
      aRot = aL1;
*/
#endif
}
}


//GUIMBAL

    if (isForISec && (! aRot.IsTrueRot()))
    {
        CamStenope* aCS =    (mCalib->PIF().DupCurPIF());
        aCS->UnNormalize();

        aCS->SetProfondeur(aProfPose);
        aCS->SetAltiSol(anAltiSol);
        aCS->SetOrientation(aRot.inv());
        aCS->SetIdentCam(mName);
        std::vector<ElCamera *> aVCam;
        aVCam.push_back(aCS);
        cTransfo3DIdent aTransfo;
        ElCamera::ChangeSys(aVCam,aTransfo,true,true);

        ElRotation3D aRMod = aCS->Orient();
        aRot = aRMod.inv();

// ShowMatr("Entree",aRot.Mat());
// ShowMatr("Sortie",aRMod.inv().Mat());
// getchar();
    }


    mRotURP = aRot;
    mUseRappelPose = mAppli.PtrRP()  &&  mAppli.PtrRP()->PatternApply()->Match(mName);
    if (mUseRappelPose)
    {
        CamStenope * aCS = mAppli.ICNM()->StdCamStenOfNames(mName,mAppli.PtrRP()->IdOrient());
        mRotURP = aCS->Orient().inv();
    }

    double aLMG = mAppli.Param().LimModeGL().Val();
    double aGVal = GuimbalAnalyse(aRot,false);
    if (((aLMG>0) && (aGVal<aLMG)) || mUseRappelPose)
    {
       std::cout << "GUIMBAL-INIT " << mName << " " << aGVal<< "\n";
       mCF->SetGL(true,mRotURP);
    }
    else
    {
       std::cout << "NO GUIMBAL " << mName  << " " << aGVal<< "\n";
    }

    mCF->SetCurRot(aRot,mRotURP);




    if (isAPC)
    {
       CamStenope* aCS =    (mCalib->PIF().DupCurPIF());
       aCS->SetOrientation(aRot.inv());
       SetCamNonOrtho(aCS);
       ELISE_ASSERT(initFromBD,"IsAperiCloud requires init from BD");
    }


    if (aNZPl!="")
    {
       mDZP[aNZPl] = aDZPl * aLambdaRot;
       if (aCam2PL)
       {
           if (! DicBoolFind(aCam2PL->mDZP,aNZPl))
               aCam2PL->mDZP[aNZPl] = aDZPl2 * aLambdaRot;
       }
    }
    TenteInitAltiProf(2,anAltiSol,aProfPose);
    mRotIsInit = true;


/*
    {
        ElRotation3D  aR = CurRot() ;
        const CamStenope * aCS = mCF->CameraCourante() ;
        std::cout << " " << mName <<  " " << aR.tr() <<  "\n";
        std::cout << " " << aCS-> R3toF2(Pt3dr(0,0,10)) <<  " " <<  aCS->F2AndZtoR3(Pt2dr(1000,1000),6) <<  "\n";
        
        getchar();
    }
*/

    if (mCompAOI)
    {
        AffineRot();
    }
    mAppli.AddRotInit();

    mCF->ResiduM2C() = mOrIntM2C;

    Trace();


    if (theLiasInit)
    {
             mAppli.CheckInit(theLiasInit,this);
    }
}


void cPoseCam::UseRappelOnPose() const 
{
   if (! mUseRappelPose) return;

   double aPdsC  = 1/ElSquare(mAppli.PtrRP()->SigmaC());
   Pt3dr aPtPdsC(aPdsC,aPdsC,aPdsC);
   double aPdsR  = 1/ElSquare(mAppli.PtrRP()->SigmaR());
   Pt3dr aPtPdsR (aPdsR,aPdsR,aPdsR);
   mRF->AddRappOnRot(mRotURP,aPtPdsC,aPtPdsR);

   // std::cout << "NAME RAPPELE ON POSE =" << mName << "\n";
}


void cPoseCam::AffineRot()
{
   for (int aK=0 ; aK<int(mCompAOI->mPats.size()) ; aK++)
   {
      if (    (mCompAOI->mPats[aK]->Match(mName))
           || (mCompAOI->mPats[aK]->Match(ToString(mCpt)))
         )
      {
           vector<cPoseCam *> aVC;
           aVC.push_back(this);
           vector<eTypeContraintePoseCamera> aVT;
           aVT.push_back(mCompAOI->mCstr[aK]);
std::cout << " Opt " << mName << " :: " << mCpt << "\n";
            mAppli.PowelOptimize(mCompAOI->mParam,aVC,aVT);


           return;
      }
   }
}



void cPoseCam::SetFigee()
{
    mRF->SetTolAng(-1);
    mRF->SetTolCentre(-1);
    mRF->SetModeRot(cNameSpaceEqF::eRotFigee);
}

void cPoseCam::SetDeFigee()
{
   if (mLastCP)
      SetContrainte(*mLastCP);
   else
      mRF->SetModeRot(cNameSpaceEqF::eRotLibre);
}

void cPoseCam::SetContrainte(const cContraintesPoses & aCP)
{
   mLastCP = & aCP;
   switch(aCP.Val())
   {
      case ePoseLibre :
          cElWarning::ToleranceSurPoseLibre.AddWarn("",__LINE__,__FILE__);
/*
          ELISE_ASSERT
	  (
	       (aCP.TolAng().Val()<=0)&&(aCP.TolCoord().Val()<=0),
	       "Tolerance inutile avec ePoseLibre"
	  );
*/
          mRF->SetModeRot(cNameSpaceEqF::eRotLibre);
      break;

      case ePoseFigee :
           mRF->SetTolAng(aCP.TolAng().Val());
           mRF->SetTolCentre(aCP.TolCoord().Val());
	   mRF->SetModeRot(cNameSpaceEqF::eRotFigee);
      break;


      case eCentreFige :
           ELISE_ASSERT
	   (
	       (aCP.TolAng().Val()<=0),
	       "Tolerance angulaire avec eCentreFige"
	   );
           mRF->SetTolCentre(aCP.TolCoord().Val());
	   mRF->SetModeRot(cNameSpaceEqF::eRotCOptFige);
      break;

      case eAnglesFiges :
           ELISE_ASSERT
	   (
	       (aCP.TolCoord().Val()<=0),
	       "Tolerance angulaire avec eCentreFige"
	   );
           mRF->SetTolAng(aCP.TolAng().Val());
	   mRF->SetModeRot(cNameSpaceEqF::eRotAngleFige);
      break;




      case ePoseBaseNormee :
      case ePoseVraieBaseNormee :
           ELISE_ASSERT
	   (
	       aCP.PoseRattachement().IsInit(),
	       "Rattachement non initialise !"
	   );
          ELISE_ASSERT
	  (
	       (aCP.TolAng().Val()<=0),
	       "Tolerance angle inutile avec ePoseBaseNormee"
	  );

           mRF->SetTolCentre(aCP.TolCoord().Val());
           if (aCP.Val()==ePoseVraieBaseNormee) 
           {
                SetRattach(aCP.PoseRattachement().Val());
	       mRF->SetModeRot(cNameSpaceEqF::eRotBaseU);
           }
           else
           {
               cPoseCam * aPR  = mAppli.PoseFromName(aCP.PoseRattachement().Val());
               mRF->SetRotPseudoBaseU(aPR->mRF);
	       mRF->SetModeRot(cNameSpaceEqF::eRotPseudoBaseU);
           }
      break;

   }
}


    //   Gestion image

void cPoseCam::InitIm()
{
    mFirstBoxImSet = false;
    mImageLoaded = false;

}


bool cPoseCam::PtForIm(const Pt3dr & aPTer,const Pt2di & aRab,bool Add)
{
    const CamStenope * aCS = CurCam() ;
    Pt2dr aPIm =  aCS->R3toF2(aPTer);
    
    Box2di aCurBIm(round_down(aPIm)-aRab,round_up(aPIm)+aRab);
    Box2di aFulBox(Pt2di(0,0),aCS->Sz());
   
    if (! aCurBIm.include_in(aFulBox)) 
       return false;

    if (Add)
    {
       ELISE_ASSERT(!mImageLoaded,"cPoseCam::PtForIm : Im Loaded in Add");
       if (mFirstBoxImSet)
       {
          mBoxIm = Sup(mBoxIm,aCurBIm);
       }
       else
       {
          mBoxIm = aCurBIm;
          mFirstBoxImSet = true;
       }
    }

    return true;
}



bool cPoseCam::ImageLoaded() const
{
   return mImageLoaded;
}

void cPoseCam::AssertImL() const
{
    if (!mImageLoaded)
    {
       std::cout << "For cam=" << mName << "\n";
       ELISE_ASSERT(false,"Image not Loaded");
    }
}

const Box2di & cPoseCam::BoxIm()
{
   AssertImL();
   return mBoxIm;
}

Im2D_U_INT2  cPoseCam::Im()
{
   AssertImL();
   return mIm;
}


void cPoseCam::CloseAndLoadIm(const Pt2di & aRab)
{
    if (! mFirstBoxImSet) 
       return;
    mImageLoaded = true;

    {
       Pt2di aP0 = Sup(mBoxIm._p0-aRab,Pt2di(0,0));
       Pt2di aP1 = Inf(mBoxIm._p1+aRab,CurCam()->Sz());
       mBoxIm = Box2di(aP0,aP1);
    }
     

    Pt2di aSz = mBoxIm.sz();
    mIm.Resize(aSz);
    mTIm = TIm2D<U_INT2,INT>(mIm);
    Tiff_Im aTF = Tiff_Im::StdConvGen(mAppli.DC()+mName,1,true,false);

    ELISE_COPY
    (
        mIm.all_pts(),
        trans(aTF.in_proj(),mBoxIm._p0),
        mIm.out()
    );
}




    //   ACCESSEURS 

cCalibCam * cPoseCam::Calib() const { return mCalib;}
cCameraFormelle * cPoseCam::CamF() {return mCF;}
double cPoseCam::AltiSol() const {return mAltiSol;}
double cPoseCam::Profondeur() const {return mProfondeur;}

int  cPoseCam::NumInit() const {return mNumInit;}

/*
bool &   cPoseCam::MMSelected() { return mMMSelected;}
double & cPoseCam::MMGain()     { return  mMMGain;}
double & cPoseCam::MMAngle()    { return mMMAngle;}
Pt3dr  & cPoseCam::MMDir()      { return mMMDir;}
Pt2dr  & cPoseCam::MMDir2D()      { return mMMDir2D;}

std::vector<double> &cPoseCam::MMGainTeta() {return mMMGainTeta;}
double & cPoseCam::MMNbPts()    { return  mMMNbPts;}
double & cPoseCam::MMGainAng()  { return  mMMGainAng;}
*/


/*********************************************************/
/*                                                       */
/*                 cAppliApero                           */
/*                                                       */
/*********************************************************/


void   cAppliApero::LoadImageForPtsMul
       (
          Pt2di aRabIncl,
          Pt2di aRabFinal,
          const std::list<cOnePtsMult *> & aLMul
       )
{
    for (int aK=0; aK<int(mVecPose.size()) ; aK++)
    {
        mVecPose[aK]->InitIm();
    } 

    for
    (
        std::list<cOnePtsMult *>::const_iterator itPM=aLMul.begin();
        itPM!=aLMul.end();
        itPM++
    )
    {
         std::vector<double> aVPds;
         const cResiduP3Inc * aRes = (*itPM)->ComputeInter(1.0,aVPds);
         if (aRes)
         {
             for (int aK=0; aK<int(mVecPose.size()) ; aK++)
             {
                 mVecPose[aK]->PtForIm(aRes->mPTer,aRabIncl,true);
             } 
         }
    }

    mVecLoadedPose.clear();
    for (int aK=0; aK<int(mVecPose.size()) ; aK++)
    {
        mVecPose[aK]->CloseAndLoadIm(aRabFinal);
        if (mVecPose[aK]->ImageLoaded())
           mVecLoadedPose.push_back(mVecPose[aK]);
    } 
}


const std::vector<cPoseCam*> &  cAppliApero::VecLoadedPose()
{
    return mVecLoadedPose;
}


std::vector<cPoseCam*>  cAppliApero::VecLoadedPose(const cOnePtsMult & aPM,int aSz)
{
    std::vector<cPoseCam*> aRes;
    std::vector<cPoseCam*> aRes2;

    std::vector<double> aVPds;
    const cResiduP3Inc * aResidu = aPM.ComputeInter(1.0,aVPds);
    Pt2di aPRab(aSz,aSz);

    if (aResidu)
    {
       for (int aKp=0 ; aKp<int(mVecLoadedPose.size()) ; aKp++)
       {
           if (mVecLoadedPose[aKp]->PtForIm(aResidu->mPTer,aPRab,false))
           {
               if (mVecLoadedPose[aKp]==aPM.GenPose0()->DownCastPoseCamNN())
               {
                   aRes.push_back(mVecLoadedPose[aKp]);
               }
               else
               {
                   aRes2.push_back(mVecLoadedPose[aKp]);
               }
           }
       }
       std::cout << "NB -------  " << aRes.size() << " ## " << aRes2.size() << "\n";
       if ((int(aRes.size())==1) && (int(aRes2.size())>= 1))
       {
          for (int aKp=0 ; aKp<int(aRes2.size()) ; aKp++)
          {
              aRes.push_back(aRes2[aKp]);
          }
       }
       else
       {
          aRes.clear();
       }
    }

    return aRes;
}

bool   cPoseCam::DoAddObsCentre(const cObsCentrePDV & anObs)
{
   if (! mHasObsOnCentre)
      return false;

   if (  
          (anObs.PatternApply().IsInit())
       && (!anObs.PatternApply().Val()->Match(mName))
      )
      return false;

   return true;
}

Pt3dr cPoseCam::CurCentre() const
{
    return  CurRot().ImAff(Pt3dr(0,0,0));
}

Pt3dr cPoseCam::CurCentreOfPt(const Pt2dr & ) const
{
   return CurCentre();
}

void cPoseCam::AddObsPlaneOneCentre(const cXml_ObsPlaneOnPose & aXmlOPOO ,const double & aWeight)
{
    cRotationFormelle & aRF = RF();
    const  cIncIntervale & aII = aRF. IncInterv();

    std::vector<int>  aVIndexe;
    for (int aK=3; aK<6 ; aK++)
         aVIndexe.push_back(aII.I0Alloc()+aK);

    for (const auto & a1ObsPl : aXmlOPOO.Obs1Plane() )
    {
        Pt3dr aVU = vunit(a1ObsPl.Vect());
        std::vector<double> aVCoeff =  aVU.ToTab();

        double aRes = mAppli.SetEq().AddEqLineaire(aWeight/ElSquare(a1ObsPl.Sigma()),aVIndexe,aVCoeff,a1ObsPl.Cste());
        std::cout << "RRReesss ddObsPlaneOneCentre " << aRes << "\n";
        // getchar();
    }
/*
    cRotationFormelle & aRF = RF();
    const  cIncIntervale & aII = aRF. IncInterv();

    std::cout << "NAME= " << mName << "\n";
    std::cout << "   ALLOC " <<  aII.I0Alloc() << " " <<  aII.I1Alloc() << "\n";
    for (int I= aII.I0Alloc() ; I<aII.I1Alloc() ; I++)
        std::cout << " VARRR= " <<       mAppli.SetEq().Alloc().GetVar(I) << "\n";

getchar();
*/
    // std::cout << "   SOLVE " <<  aII.I0Solve() << " " <<  aII.I1Solve() << "\n";
    
/*
    aII.I0Alloc();
    class cRotationFormelle : public cElemEqFormelle,
   const  cIncIntervale & IncInterv()

    INT I0Alloc()  const ;
    INT I1Alloc()  const ;


  INT I0Solve()  const ;
       INT I1Solve()  const ;


Equation en delta par rapport a la valeur courante

cSetEqFormelles
 double AddEqLineaire (
                          double aPds, const std::vector<int>  &    indexe,
                          const std::vector<double>  & aCoeff,double aB);
*/

}


Pt3dr  cPoseCam::AddObsCentre
      (
           const cObsCentrePDV & anObs,
           const cPonderateur &  aPondPlani,
           const cPonderateur &  aPondAlti,
           cStatObs & aSO
      )
{
// cObsCentre aRes;
// mIncOnC 


   mLastItereHasUsedObsOnCentre = true;
   ELISE_ASSERT(DoAddObsCentre(anObs),"cPoseCam::AddObsCentre");

   if (mEqOffsetGPS)
   {
       Pt3dr aResidu = mEqOffsetGPS->Residu(mObsCentre.mCentre);
       double aNormLA = euclid(mEqOffsetGPS->Base()->ValueBase());
       std::cout << "Lever Arm, Cam: " << mName << " Residual " << aResidu  << " LA: " <<  mEqOffsetGPS->Base()->ValueBase() << ", " << aNormLA << "\n";

       double aPdsPX = aPondPlani.PdsOfError(euclid(Pt2dr(aResidu.x,aResidu.y))/sqrt(2.));
       double aPdsPY = aPdsPX;
       double aPdsZ  = aPondAlti.PdsOfError(ElAbs(aResidu.z));
//std::cout << " cPoseCam::AddObsCentre " << mObsCentre.mIncOnC  << " " <<  aPdsP << " " << aPdsZ  <<  "\n";
       Pt3dr aPInc = mObsCentre.mIncertOnC;
       // Si il y a une incertitude
       if (aPInc.x >0)
       {
          aPdsPX *= ElSquare(1/aPInc.x);
       }
       if (aPInc.y >0)
       {
          aPdsPY *= ElSquare(1/aPInc.y);
       }
       if (aPInc.z >0)
       {
          aPdsZ *= ElSquare(1/aPInc.z);
       }

/*
       if (1)
       {
          cBaseGPS * aBG = mEqOffsetGPS->Base();
          Pt3d<Fonc_Num>  BaseInc();
       }
*/

       return mEqOffsetGPS->AddObs(mObsCentre.mCentre,Pt3dr(aPdsPX,aPdsPY,aPdsZ));
   }



   Pt3dr aC0 = CurRot().ImAff(Pt3dr(0,0,0));
   Pt3dr aDif = aC0 - mObsCentre.mCentre;
   Pt2dr aDifPlani(aDif.x,aDif.y);

   double aPdsP  = aPondPlani.PdsOfError(euclid(aDifPlani)/sqrt(2.));
   double aPdsZ  = aPondAlti.PdsOfError(ElAbs(aDif.z));

   Pt3dr aPPds = aSO.AddEq() ? Pt3dr(aPdsP,aPdsP,aPdsZ) : Pt3dr(0,0,0) ; 
   Pt3dr aRAC = mRF->AddRappOnCentre(mObsCentre.mCentre,aPPds ,false);


   double aSEP = aPdsP*(ElSquare(aRAC.x)+ElSquare(aRAC.y))+aPdsZ*ElSquare(aRAC.z);
   //  std::cout << "========= SEP " << aSEP << "\n";
   aSO.AddSEP(aSEP);


   mAppli.AddResiducentre(aDif);

   if (aPondPlani.PPM().Show().Val() >=eNSM_Indiv)
   {
        std::cout << mName << " DeltaC " <<  euclid(aDif) << " " << aDif << "\n";
   }

   if (anObs.ShowTestVitesse().Val())
   {
      // Montre le linkage GPS
      {
          ElRotation3D  aR = CurRot() ;
          mAppli.COUT() << aR.teta01()  << " ";
          if (mPrec != mNext)
          {
              Pt3dr aVC = mNext->CurCentre()- mPrec->CurCentre() ;
              double aDT = mNext->mTime -  mPrec->mTime;
              Pt2dr aV2C(aVC.x,aVC.y);

              Pt3dr aVG = mNext->mObsCentre.mCentre - mPrec->mObsCentre.mCentre ;
              Pt3dr  aVitG = aVG / aDT;

              Pt3dr aGC = CurCentre() -mObsCentre.mCentre;

              double aRetard = scal(aGC,aVitG) /scal(aVitG,aVitG);

              mAppli.AddRetard(aRetard);

              Pt2dr aV2G(aVG.x,aVG.y);
              mAppli.COUT() << " Retard " << aRetard
                            << " VIt " << euclid(aVitG)
                            << " Traj " << (aR.teta01() - atan2(aV2C.y,aV2C.x) -PI/2) << " "
                             << " TGps " << (aR.teta01() - atan2(aV2G.y,aV2G.x) -PI/2) << " ";
           
          }
          mAppli.COUT()  << "\n";
      }
/*
      mAppli.COUT().precision(10);
      mAppli.COUT() << "RESIDU CENTRE " << aDif << " pour " << mName 
                     << " AERO=" <<  aC0 << " GPS=" << mObsCentre<< "\n";
*/
   }

   return aDif;
    
}



/************************************************************/
/*                                                          */
/*              cClassEquivPose                             */
/*              cRelEquivPose                               */
/*                                                          */
/************************************************************/

             // =======   cClassEquivPose  ====

cClassEquivPose::cClassEquivPose(const std::string & anId) :
   mId (anId)
{
}

void cClassEquivPose::AddAPose(cGenPoseCam * aPC)
{
    if (BoolFind(mGrp,aPC))
    {
        std::cout << "For Pose : " << aPC->Name() << "\n";
        ELISE_ASSERT(false,"cClassEquivPose::AddAPose multiple name");
    }
    mGrp.push_back(aPC);
}

const std::vector<cGenPoseCam *> &   cClassEquivPose::Grp() const
{
   return mGrp;
}

const std::string & cClassEquivPose::Id() const
{
    return mId;
}


             // =======   cRelEquivPose  ====

/*
cRelEquivPose::cRelEquivPose(int aNum) :
   mNum (aNum)
{
}
*/
cRelEquivPose::cRelEquivPose() 
{
}


cClassEquivPose * cRelEquivPose::AddAPose(cPoseCam * aPC,const std::string & aName)
{
   cClassEquivPose * & aCEP = mMap[aName];
   if (aCEP==0) 
      aCEP = new cClassEquivPose(aName);
   aCEP->AddAPose(aPC);

   mPos2C[aPC->Name()] = aCEP;
   return aCEP;
}

cClassEquivPose &  cRelEquivPose::ClassOfPose(const cGenPoseCam & aPC)
{
   cClassEquivPose * aCEP = mPos2C[aPC.Name()];
   if (aCEP==0)
   {
       std::cout << "For Pose " << aPC.Name() << "\n";
       ELISE_ASSERT(false,"Can get Class in cRelEquivPose::ClassOfPose");
   }
   return *aCEP;
}

bool cRelEquivPose::SameClass(const cGenPoseCam & aPC1,const cGenPoseCam & aPC2)
{
   return ClassOfPose(aPC1).Id() == ClassOfPose(aPC2).Id();
}



const std::map<std::string,cClassEquivPose *> &  cRelEquivPose::Map() const
{
   return mMap;
}

void cRelEquivPose::Show()
{
    std::cout << "========== REL NUM ==================\n";

   for 
   (
        std::map<std::string,cClassEquivPose *>::const_iterator itM=mMap.begin();
        itM!=mMap.end();
        itM++
   )
   {
          const cClassEquivPose& aCl = *(itM->second);
          const std::vector<cGenPoseCam *> & aGrp = aCl.Grp() ;

          if (aGrp.size() == 1)
             std::cout << aCl.Id() << " ::  " << aGrp[0]->Name()<< "\n";
          else 
          {
             std::cout << "## " << aCl.Id() << " ##\n";
             for (int aK=0 ; aK<int(aGrp.size()) ; aK++)
                std::cout <<  "  --- "  << aGrp[aK]->Name()<< "\n";
          }
          
   }
}
  

             // =======   cAppliApero  ====

cRelEquivPose * cAppliApero::RelFromId(const std::string & anId)
{
   cRelEquivPose * aRes = mRels[anId];
   if (aRes ==0)
   {
      std::cout << "For Id = " << anId << "\n";
      ELISE_ASSERT(false,"cAppliApero::RelFromId do not exist");
   }

   return aRes;
}


bool cAppliApero::SameClass(const std::string& anId,const cGenPoseCam & aPC1,const cGenPoseCam & aPC2)
{
   return RelFromId(anId)->SameClass(aPC1,aPC2);
}



void cAppliApero::AddObservationsRigidGrp(const cObsRigidGrpImage & anORGI,bool IsLastIter,cStatObs & aSO)
{
   cRelEquivPose * aREP = RelFromId(anORGI.RefGrp());
   const std::map<std::string,cClassEquivPose *> &  aMap = aREP->Map();
   for 
   ( 
       std::map<std::string,cClassEquivPose *>::const_iterator itG=aMap.begin();
       itG!=aMap.end();
       itG++
   )
   {
        const std::vector<cGenPoseCam *> & aGrp = itG->second->Grp();
        int aNb = (int)aGrp.size();
        if (aNb>=2)
        {
            for (int aK1=0 ; aK1<aNb ; aK1++)
            {
                for (int aK2=aK1+1 ; aK2<aNb ; aK2++)
                {
                    cPoseCam * aCP1 = aGrp[aK1]->DownCastPoseCamSVP();
                    cPoseCam * aCP2 = aGrp[aK2]->DownCastPoseCamSVP();


                    if ((! aCP1) || (!aCP2))
                    {
                        std::cout << "For " << aGrp[aK1]->Name() << " "<< aGrp[aK2]->Name() << "\n";
                        ELISE_ASSERT(false,"Non stenope cam use in ddObservationsRigidGrp");
                    }

                    cRotationFormelle & aRF1 =  aCP1->RF();
                    cRotationFormelle & aRF2 =  aCP2->RF();
                    if (anORGI.ORGI_CentreCommun().IsInit())
                    {
                       Pt3dr aPInc = anORGI.ORGI_CentreCommun().Val().Incertitude();
                       double anInc[3];
                       aPInc.to_tab(anInc);
                       for (int aD=0 ; aD<3 ; aD++)
                       {
                           if (anInc[aD]>0)
                           {
                              double aR = mSetEq.AddEqEqualVar(ElSquare(1.0/anInc[aD]),aRF1.NumCentre(aD),aRF2.NumCentre(aD),true);
                              aSO.AddSEP(aR);
                              
                           }
                       }
                    }
                    if (anORGI.ORGI_TetaCommun().IsInit())
                    {
                       Pt3dr aPInc = anORGI.ORGI_TetaCommun().Val().Incertitude();
                       double anInc[3];
                       aPInc.to_tab(anInc);
                       for (int aD=0 ; aD<3 ; aD++)
                       {
                           if (anInc[aD]>0)
                           {
                              double aR  = mSetEq.AddEqEqualVar(ElSquare(1.0/anInc[aD]),aRF1.NumTeta(aD),aRF2.NumTeta(aD),true);
                              aSO.AddSEP(aR);
                           }
                       }
                    }
                }
            }
        }
   }
}

/***********************************************************/
/*                                                         */
/*                                                         */
/*                                                         */
/***********************************************************/

/*

class cIBC_ImsOneTime
{
    public :
        cIBC_ImsOneTime(int aNbCam,int aNum,const std::string& aNameTime) ;
        void  AddPose(cPoseCam *, int aNum);

    private :

        std::vector<cPoseCam *> mCams;
        int                     mNum;
        std::string             mNameTime;
};


class cIBC_OneCam
{
      public :
          cIBC_OneCam(const std::string & ,int aNum);
          const int & Num() const;
      private :
          std::string mNameCam;
          int         mNum;
};



class cImplemBlockCam
{
    public :
         static cImplemBlockCam * AllocNew(cAppliApero &,const cStructBlockCam);
    private :
         cImplemBlockCam(cAppliApero & anAppli,const cStructBlockCam );

         cAppliApero &               mAppli;
         const cStructBlockCam &     mSBC;
         cRelEquivPose               mRelGrp;
         cRelEquivPose               mRelId;

         std::map<std::string,cIBC_OneCam *>   mName2Cam;
         std::vector<cIBC_OneCam *>            mNum2Cam;
         int                                   mNbCam;

         std::map<std::string,cIBC_ImsOneTime *> mName2ITime;
         std::vector<cIBC_ImsOneTime *>          mNum2ITime;
};

    // =================================
    //              cIBC_ImsOneTime
    // =================================

cIBC_ImsOneTime::cIBC_ImsOneTime(int aNb,int aNum,const std::string & aNameTime) :
       mCams     (aNb),
       mNum      (aNum),
       mNameTime (aNameTime)
{
}

void  cIBC_ImsOneTime::AddPose(cPoseCam * aPC, int aNum) 
{
    cPoseCam * aPC0 =  mCams.at(aNum);
    if (aPC0 != 0)
    {
         std::cout <<  "For cameras " << aPC->Name() <<  "  and  " << aPC0->Name() << "\n";
         ELISE_ASSERT(false,"Conflicting name from KeyIm2TimeCam ");
    }
    
    mCams[aNum] = aPC;
}

    // =================================
    //              cIBC_OneCam 
    // =================================

cIBC_OneCam::cIBC_OneCam(const std::string & aNameCam ,int aNum) :
    mNameCam (aNameCam ),
    mNum     (aNum)
{
}

const int & cIBC_OneCam::Num() const {return mNum;}

    // =================================
    //       cImplemBlockCam
    // =================================

cImplemBlockCam::cImplemBlockCam(cAppliApero & anAppli,const cStructBlockCam aSBC) :
      mAppli (anAppli),
      mSBC   (aSBC)
{
    const std::vector<cPoseCam*> & aVP = mAppli.VecAllPose();
   

    // On initialise les camera
    for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
    {
          cPoseCam * aPC = aVP[aKP];
          std::string aNamePose = aPC->Name();
          std::pair<std::string,std::string> aPair =   mAppli.ICNM()->Assoc2To1(mSBC.KeyIm2TimeCam(),aNamePose,true);
          std::string aNameCam = aPair.second;
          if (! DicBoolFind(mName2Cam,aNameCam))
          {
               cIBC_OneCam *  aCam = new cIBC_OneCam(aNameCam,mNum2Cam.size());
               mName2Cam[aNameCam] = aCam;
               mNum2Cam.push_back(aCam); 
          }
    }
    mNbCam  = mNum2Cam.size();

    
    // On regroupe les images prises au meme temps
    for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
    {
          cPoseCam * aPC = aVP[aKP];
          std::string aNamePose = aPC->Name();
          std::pair<std::string,std::string> aPair =   mAppli.ICNM()->Assoc2To1(mSBC.KeyIm2TimeCam(),aNamePose,true);
          std::string aNameTime = aPair.first;
          std::string aNameCam = aPair.second;
          
          cIBC_ImsOneTime * aIms =  mName2ITime[aNameTime];
          if (aIms==0)
          {
               aIms = new cIBC_ImsOneTime(mNbCam,mNum2ITime.size(),aNameTime);
               mName2ITime[aNameTime] = aIms;
               mNum2ITime.push_back(aIms);
          }
          cIBC_OneCam * aCam = mName2Cam[aNameCam];
          aIms->AddPose(aPC,aCam->Num());
    }
}

*/

void cPoseCam::AddMajick(cMajickChek & aMC) const
{
    aMC.Add(CurRot());
}

   //   CamNonOrtho 

void  cPoseCam::SetCamNonOrtho(CamStenope * aCS)
{
    AssertHasNotCamNonOrtho();
    mCamNonOrtho = aCS;
}
CamStenope *  cPoseCam::GetCamNonOrtho() const
{
   AssertHasCamNonOrtho();
   return mCamNonOrtho;
}
bool cPoseCam::HasCamNonOrtho() const
{
   return mCamNonOrtho != 0;
}
void  cPoseCam::AssertHasCamNonOrtho() const
{
    ELISE_ASSERT(HasCamNonOrtho(),"Camera Non Ortho expected");
}
void  cPoseCam::AssertHasNotCamNonOrtho() const
{
    ELISE_ASSERT(! HasCamNonOrtho(),"Unexpected Camera Non Ortho");
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
