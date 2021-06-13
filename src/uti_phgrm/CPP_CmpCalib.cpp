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



#define DEF_OFSET -12349876

class cCapteurCmpCal
{
    public :
        cCapteurCmpCal(const std::string & aName) :
             mName (aName)
    {
    }

    static cCapteurCmpCal & StdAlloc(const std::string & aName);

        const std::string & Name() { return mName; }
        virtual Pt2dr P0()                               = 0;
        virtual Pt2dr P1()                               = 0;
    virtual double  Focale()                         = 0;
        virtual Pt2dr ToPDirL3(const Pt2dr&  aP) const   = 0;
    private :
        std::string   mName;
};


class cGridCmpCal : public cCapteurCmpCal
{
    public :
        cGridCmpCal(const std::string & aName)  :
             cCapteurCmpCal(aName)
        {
             SplitDirAndFile(mDir,mFile,aName);
             std::cout << "[" << mDir << "][" << mFile << "]\n";
             cDbleGrid::cXMLMode aXM;
             mGr = new cDbleGrid(aXM,mDir,mFile);
        }

        //  const std::string & Name() { return mName; }
        Pt2dr P0() {return mGr->GrDir().P0();}
        Pt2dr P1() {return mGr->GrDir().P1();}
    double  Focale() {return mGr->Focale();}
        Pt2dr ToPDirL3(const Pt2dr&  aP) const {return mGr->Direct(aP);}


   private :
        std::string   mDir;
        std::string   mFile;
        cDbleGrid   * mGr;
};

class cCamStenopeCmpCal : public cCapteurCmpCal
{
    public :
        cCamStenopeCmpCal(const std::string & aName)  :
             cCapteurCmpCal(aName),
         mCam (Std_Cal_From_File(aName))
        {
        std::cout << aName << "\n";
        }


        //  const std::string & Name() { return mName; }
        Pt2dr P0() {return Pt2dr(0,0);}
        Pt2dr P1() {return Pt2dr(mCam->Sz());}
    double  Focale() {return  mCam->Focale();}
        Pt2dr ToPDirL3(const Pt2dr&  aP) const {return mCam->F2toPtDirRayonL3(aP);}


   private :
       CamStenope * mCam;
};

cCapteurCmpCal & cCapteurCmpCal::StdAlloc(const std::string & aName)
{
   cElXMLTree aTr(aName);

   if (aTr.Get("sensor"))
       return  * (new cGridCmpCal(aName));

   return * (new cCamStenopeCmpCal(aName));
}

class cAppliCmpCal
{
     public :
          cAppliCmpCal
          (
                 const std::string & aName1,
                 const std::string & aName2,
                 bool  aModeL1,
                 int   aSzW,
                 double aDynV,
                 const std::string Out
          ) :
             mGr1   (cCapteurCmpCal::StdAlloc(aName1)),
             mGr2   (cCapteurCmpCal::StdAlloc(aName2)),
             mSetEq   (aModeL1 ? cNameSpaceEqF::eSysL1Barrodale : cNameSpaceEqF::eSysPlein,1000),
             mEqORV (mSetEq.NewEqObsRotVect()),
             mRotF  (mEqORV->RotF()),
             mNBP   (30) ,
             mBrd   (0.02),

             mP0    (Inf(mGr1.P0(), mGr2.P0())),
             mP1    (Inf(mGr1.P1(), mGr2.P1())),
             mSz     (mP1-mP0),
             mRatioW (aSzW/ElMax(mSz.x,mSz.y)),

             mBox    (mP0,mP1),
             mMil    (mBox.milieu()),
             mRay    (euclid(mP0,mMil)),
             mFocale ( (mGr1.Focale() + mGr2.Focale()) / 2.0),
             mSzW    (round_ni(mSz*mRatioW)),
             mW      (aSzW>0 ? Video_Win::PtrWStd(mSzW) : 0),
             mDynVisu (aDynV),
             mRotCur  (Pt3dr(0,0,0),0,0,0)
          {
std::cout <<  mGr1.P0() << mGr1.P1() << "\n";
std::cout <<  mGr2.P0() << mGr2.P1() << "\n";
              mSetEq.SetClosed();
          }

          void OneItere(bool First,bool Last,const std::string Out);
          void OneItere2(bool First,bool Last,const std::string Out,const std::string mXmlG);

          double K2Pds(int aK)
          {
              return mBrd +  (1-2*mBrd) * aK/double(mNBP);
          }
          void InitNormales(Pt2dr aPIm);
          Pt2dr EcartNormaleCorr();
          Pt2dr ENC_From_PIm(Pt2dr aPIm);
          double EcartFromRay(double aR);

     private :
          cCapteurCmpCal & mGr1;
          cCapteurCmpCal & mGr2;
          cSetEqFormelles  mSetEq;
          cEqObsRotVect *  mEqORV;
          cRotationFormelle & mRotF;
          int               mNBP;
          double            mBrd;
          Pt2dr             mP0;
          Pt2dr             mP1;
          Pt2dr             mSz;
          double            mRatioW;
          Box2dr            mBox;
          Pt2dr             mMil;
          double            mRay;
          double            mFocale;
          Pt2di             mSzW;
          Video_Win *       mW;
          double            mDynVisu;
          ElRotation3D      mRotCur;
          Pt3dr             mN1;
          Pt3dr             mN2;
};

void cAppliCmpCal::InitNormales(Pt2dr aPIm)
{
    Pt2dr aPPH1 =  mGr1.ToPDirL3(aPIm);
    Pt2dr aPPH2 =  mGr2.ToPDirL3(aPIm);
    mN1 = PZ1(aPPH1);
    mN2 = PZ1(aPPH2);
    mN1 = mN1 / euclid(mN1);
    mN2 = mN2 / euclid(mN2);
}

Pt2dr cAppliCmpCal::EcartNormaleCorr()
{
   return (ProjStenope(mRotCur.ImVect(mN1))-ProjStenope(mN2))*mFocale;
}

Pt2dr cAppliCmpCal::ENC_From_PIm(Pt2dr aPIm)
{
    InitNormales(aPIm);
    return EcartNormaleCorr();
}

double cAppliCmpCal::EcartFromRay(double aR)
{
    int aNbDir = 3000;

    double aS1=0, aSD=0;
    for (int aK=0 ; aK<aNbDir ; aK++)
    {
        Pt2dr aP = mMil + Pt2dr::FromPolar(aR,(2*PI*aK)/aNbDir);
        if (mBox.inside(aP))
        {
            aS1++;
            double aD = euclid(ENC_From_PIm(aP));
            if (aD>1.0)
            {
                 // std::cout << "P= " << aP << ENC_From_PIm(aP)  << "\n";
                 // getchar();
            }
            aSD += aD;
        }
    }
    return (aS1>0) ? (aSD/aS1) : -1;
}


void cAppliCmpCal::OneItere(bool First,bool Last,const std::string Out)
{

    double aS1=0, aSD=0;
    mSetEq.AddContrainte(mRotF.StdContraintes(),true);
    mSetEq.SetPhaseEquation();
    mRotCur =  mRotF.CurRot();

    FILE * aFP=0;

    if (Last)
    {
        if(Out!="")
		{
			std::string aName = Out;
			aFP = ElFopen(aName.c_str(),"w");
		}
		else
		{
			std::string aName = StdPrefix(mGr1.Name()) + "_Ecarts.txt";
			aFP = ElFopen(aName.c_str(),"w");
		}
			
       //std::string aName = StdPrefix(mGr1.Name()) + "_Ecarts.txt";
       //aFP = ElFopen(aName.c_str(),"w");
       double aStep = 200;
       fprintf(aFP,"--------------  Ecart radiaux  -----------\n");
       fprintf(aFP," Rayon   Ecart\n");
       for( double aR = 0 ; aR<(mRay+aStep) ; aR+= aStep)
       {
           double aRM = ElMin(aR,mRay-10.0);
           double EC =  EcartFromRay(aRM);
           std::cout << "Ray=" << aRM
                     << " ; Ecart=" << EC << "\n";
           fprintf(aFP," %lf %lf\n",aRM,EC);
       }
    }

    if (Last)
    {
       fprintf(aFP,"--------------  Ecart plani  -----------\n");
       fprintf(aFP,"Im.X Im.Y  PhG.X Phg.Y Ec\n");
    }
    for (int aKX=0 ; aKX<=mNBP ; aKX++)
		for (int aKY=0 ; aKY<=mNBP ; aKY++)
		{
			double aPdsX = K2Pds(aKX);
			double aPdsY = K2Pds(aKY);
			
			Pt2dr  aPIm(
						mP0.x * aPdsX+mP1.x * (1-aPdsX),
						mP0.y * aPdsY+mP1.y * (1-aPdsY)
						);

// std::cout << "PIMMM " << aPIm << "\n";
			InitNormales (aPIm);
			
			mEqORV->AddObservation(mN1,mN2);
			
			Pt2dr U = EcartNormaleCorr();
			aS1++;
			
			aSD += euclid(U);

            if ((mW!=0) && (First || Last ))
            {
                // Pt2dr aP0 (mSzW*aPdsX,mSzW*aPdsY);
               //  Pt2dr aP0 = (aPIm- mP0)  * mRatioW;
                Pt2dr aP0 = (aPIm-mP0)  * mRatioW;


                mW->draw_circle_loc(aP0,2.0,mW->pdisc()(P8COL::green));
                int aCoul = First ? P8COL::blue : P8COL::red;

                mW->draw_seg
                (
                   aP0,
                   aP0 + U* mDynVisu,
                   mW->pdisc()(aCoul)
                );
            }
            if (aFP)
               fprintf(aFP,"%lf %lf %lf %lf %lf\n",aPIm.x,aPIm.y,U.x,U.y,euclid(U));
       }

       std::cout << (aSD/aS1) << " "  << (aSD/aS1) * (1e6/mFocale) << " MicroRadians " << "\n";
       mSetEq.SolveResetUpdate();

       if (aFP)
          ElFclose(aFP);
}

void cAppliCmpCal::OneItere2(bool First,bool Last,const std::string Out,const std::string mXmlG)
{
	cXmlTNR_TestCalibReport aCalib;
	aCalib.CalibName() = mGr1.Name();
	
    double aS1=0, aSD=0;
    mSetEq.AddContrainte(mRotF.StdContraintes(),true);
    mSetEq.SetPhaseEquation();
    mRotCur =  mRotF.CurRot();

    FILE * aFP=0;
    double SEp = 0;
    double SUx = 0;
    double SUy = 0;
    double SE = 0;

    if (Last)
    {
        if(Out!="")
		{
			std::string aName = Out;
			aFP = ElFopen(aName.c_str(),"w");
		}
		else
		{
			std::string aName = StdPrefix(mGr1.Name()) + "_Ecarts.txt";
			aFP = ElFopen(aName.c_str(),"w");
		}
			
       //std::string aName = StdPrefix(mGr1.Name()) + "_Ecarts.txt";
       //aFP = ElFopen(aName.c_str(),"w");
       double aStep = 200;
       fprintf(aFP,"--------------  Ecart radiaux  -----------\n");
       fprintf(aFP," Rayon   Ecart\n");
       for( double aR = 0 ; aR<(mRay+aStep) ; aR+= aStep)
       {
           double aRM = ElMin(aR,mRay-10.0);
           double EC =  EcartFromRay(aRM);
           std::cout << "Ray=" << aRM
                     << " ; Ecart=" << EC << "\n";
           SEp += EC;
           fprintf(aFP," %lf %lf\n",aRM,EC);
           Pt2dr aER(aRM,EC);
           aCalib.EcartsRadiaux().push_back(aER);
       }
    }

    if (Last)
    {
       fprintf(aFP,"--------------  Ecart plani  -----------\n");
       fprintf(aFP,"Im.X Im.Y  PhG.X Phg.Y Ec\n");
    }
    
    for (int aKX=0 ; aKX<=mNBP ; aKX++)
	{
       for (int aKY=0 ; aKY<=mNBP ; aKY++)
       {
           double aPdsX = K2Pds(aKX);
           double aPdsY = K2Pds(aKY);

           Pt2dr  aPIm (
                      mP0.x * aPdsX+mP1.x * (1-aPdsX),
                      mP0.y * aPdsY+mP1.y * (1-aPdsY)
                 );

           InitNormales (aPIm);

           mEqORV->AddObservation(mN1,mN2);
           Pt2dr U = EcartNormaleCorr();
           aS1++;
           aSD += euclid(U);

            if ((mW!=0) && (First || Last ))
            {
                // Pt2dr aP0 (mSzW*aPdsX,mSzW*aPdsY);
               //  Pt2dr aP0 = (aPIm- mP0)  * mRatioW;
                Pt2dr aP0 = (aPIm-mP0)  * mRatioW;


                mW->draw_circle_loc(aP0,2.0,mW->pdisc()(P8COL::green));
                int aCoul = First ? P8COL::blue : P8COL::red;

                mW->draw_seg
                (
                   aP0,
                   aP0 + U* mDynVisu,
                   mW->pdisc()(aCoul)
                );
            }
            if (aFP)
               fprintf(aFP,"%lf %lf %lf %lf %lf\n",aPIm.x,aPIm.y,U.x,U.y,euclid(U));
            SUx += U.x;
            SUy += U.y;
            SE += euclid(U);
            Pt2dr aCoord(aPIm.x,aPIm.y);
            Pt3dr aR(U.x,U.y,euclid(U));
            cEcartsPlani aEP;
            aEP.CoordPx() = aCoord;
            aEP.UxUyE() = aR;
            aCalib.EcartsPlani().push_back(aEP);
            
       }
	}

       std::cout << (aSD/aS1) << " "  << (aSD/aS1) * (1e6/mFocale) << " MicroRadians " << "\n";
       mSetEq.SolveResetUpdate();

       if (aFP)
          ElFclose(aFP);
if(SEp==0&&SUx==0&&SUy==0&&SE==0){aCalib.TestCalibDiff() = true;}
else{aCalib.TestCalibDiff() = false;}
MakeFileXML(aCalib, mXmlG);
}


int CmpCalib_main(int argc,char ** argv)
{
    double  aTeta01 = 0.0;
    double  aTeta02 = 0.0;
    double  aTeta12 = 0.0;
    int     aL1 = 0;
    int     aSzW= 700;
    double  aDynV = 100.0;
    bool 	DispW = false;
    std::string	mXmlG ="";
    std::string aName1;
    std::string aName2;
    std::string Out="";

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aName1, "First calibration file",  eSAM_IsExistFile)
                << EAMC(aName2, "Second calibration file", eSAM_IsExistFile),
    LArgMain()  << EAM(aTeta01,"Teta01",true)
                << EAM(aTeta02,"Teta02",true)
                << EAM(aTeta12,"Teta12",true)
                << EAM(aL1,"L1",true)
                << EAM(aSzW,"SzW",true)
                << EAM(aDynV,"DynV",true)
                << EAM(Out,"Out",true,"Result (Def=Name1_ecarts.txt)", eSAM_IsOutputFile)
                << EAM(DispW,"DispW",true,"Display window")
                << EAM(mXmlG,"XmlG",true,"Generate Xml")
    );
    if (!MMVisualMode)
    {
        cAppliCmpCal aCmpC(aName1,aName2, (aL1!=0),aSzW,aDynV,Out);
        int aNbStep = 5;
        for (int aK=0 ; aK< 5 ; aK++)
			if(mXmlG!=""){aCmpC.OneItere2(aK==0,aK==(aNbStep-1),Out,mXmlG);}
			else{aCmpC.OneItere(aK==0,aK==(aNbStep-1),Out);}
		if(DispW)
		{
			getchar();
		}
        return EXIT_SUCCESS;
    }
    else return EXIT_SUCCESS;
}

//=================================================================

class cLibertOfCalib
{
    public :
      cLibertOfCalib(int aDR,int aDG,int aDC,bool HasCD,bool PPCDLie) :
           mDegRad    (aDR),
           mDegGen    (aDG),
           mDDecentr  (aDC),
           mHasCD     (HasCD),
           mPPCDLie   (PPCDLie)
      {
      }


      int   mDegRad;
      int   mDegGen;
      int   mDDecentr;
      bool  mHasCD;
      bool  mPPCDLie;
};


cLibertOfCalib GetDefDegreeOfCalib(const cCalibDistortion & aCalib )
{
    if (aCalib.ModNoDist().IsInit())
    {
           return cLibertOfCalib(0,0,0,false,true);
    }
    if (aCalib.ModRad().IsInit())
    {
         const cCalibrationInterneRadiale & aCIR = aCalib.ModRad().Val();
         return cLibertOfCalib((int)aCIR.CoeffDist().size(), 0, 0, true, false);
    }
    if (aCalib.ModPhgrStd().IsInit())
    {
         const cCalibrationInternePghrStd & aCIR = aCalib.ModPhgrStd().Val();
         const cCalibrationInterneRadiale & aCIP = aCIR.RadialePart();
         return cLibertOfCalib((int)aCIP.CoeffDist().size(), 1, 1, true, false);
    }

    if (aCalib.ModUnif().IsInit())
    {
        eModelesCalibUnif aMode = aCalib.ModUnif().Val().TypeModele();

        if (aMode==eModeleEbner) return cLibertOfCalib(0,4,0,true ,false);
        if (aMode==eModeleDCBrown) return cLibertOfCalib(0,4,0,true ,false);
        if ((aMode>=eModelePolyDeg2) && (aMode<=eModelePolyDeg7))
        {
             int aDeg = int(aMode) - int(eModelePolyDeg2) + 2;
             return cLibertOfCalib(0,aDeg,0,false,false);
        }

        if ((aMode == eModele_FishEye_10_5_5)  || (aMode==eModele_EquiSolid_FishEye_10_5_5))
        {
           return cLibertOfCalib(10,3,4,true,false);
        }

        if (aMode==eModele_DRad_PPaEqPPs) return cLibertOfCalib(2,0,0,true ,true);

        if ((aMode>=eModeleRadFour7x2) && (aMode<=eModeleRadFour19x2))
        {
            int aDegRad = 3 + 2* (int(aMode) - int(eModeleRadFour7x2));
            return cLibertOfCalib(2,aDegRad,0,false,false);
        }
    }


    ELISE_ASSERT(false,"GetDefDegreeOfCalib");
    return cLibertOfCalib(0,0,0,false,true);
}


void GenerateMesure2D3D(cBasicGeomCap3D * aCamIn,int aNbXY,int aNbProf,const std::string & aNameIm,cDicoAppuisFlottant & aDAF,cMesureAppuiFlottant1Im & aMAF)
{

// std::cout << "CONVCALL " << __LINE__ << "\n";
    double aProfGlob = 1.0;
    CamStenope * aCS = aCamIn->DownCastCS() ;

    if (aCS && aCS->ProfIsDef())
    {
         aProfGlob = aCS->GetProfondeur();
    }

/*
std::cout << "Ppppppppppp= " << aCamIn->GetVeryRoughInterProf()  
         << " " << aCamIn->ProfondeurDeChamps(aCamIn->SzPixel()/2.0) << "\n";
*/

   Pt2dr aSzPix = aCamIn->SzPixel();

   Pt2di aPInt;
   double aEps = 1e-2;
   double anInc = 1/ euclid(aSzPix);
   aMAF.NameIm() = aNameIm;
   for (aPInt.x=0 ; aPInt.x<= aNbXY ; aPInt.x++)
   {
       for (aPInt.y=0 ; aPInt.y<= aNbXY ; aPInt.y++)
       {
           Pt2dr aPds(aPInt.x/double(aNbXY),aPInt.y/double(aNbXY));
           aPds = aPds * (1-2*aEps) + Pt2dr(aEps,aEps);
           Pt2dr aPIm = aSzPix.mcbyc(aPds);
           for (int aKP=0; aKP < aNbProf ; aKP++)
           {
               //Pt2dr aPCheck = aCamIn->R3toF2(aPGround);
               //std::cout << aPInt << " => " << aPIm << " " << aPCheck<< "\n";
               std::string aNamePt = "Pt_"+ ToString(aPInt.x)
                                     + "_"+ ToString(aPInt.y)
                                     + "_"+ ToString(aKP);

              // GCP generation
               cOneAppuisDAF anAp;
               double aMult =  1;
               if (aNbProf > 1)
               {
                   double aNbP1 = aNbProf-1.0;
                   aMult =  pow(2.0, 0.3 * (aKP-aNbP1/2.0) / aNbP1);
               }
               double aProf = aProfGlob * aMult;
               Pt3dr aPGround = aCamIn->ImEtProf2Terrain(aPIm,aProf);
               anAp.Pt() = aPGround;
               anAp.Incertitude() = Pt3dr(anInc,anInc,anInc);
               anAp.NamePt() = aNamePt;
               aDAF.OneAppuisDAF().push_back(anAp);

              // Image measurement
              cOneMesureAF1I aM1;
              aM1.NamePt() = aNamePt;
              aM1.PtIm() = aPIm;
               aMAF.OneMesureAF1I().push_back(aM1);

           }
       }
   }
}

class cAppli_GenerateAppuisLiaison : public cAppliWithSetImage
{
    public :
        cAppli_GenerateAppuisLiaison(int argc, char** argv);

        std::string mNameIm,mNameOri;
        int mNbXY,mNbProf;
};
 
std::string NameGenepi(const std::string & aName,bool Is3D)
{
    return  "Genepi-" + aName +  std::string("-Mes") + std::string(Is3D ? "3": "2") + std::string("D.xml");
}


cAppli_GenerateAppuisLiaison::cAppli_GenerateAppuisLiaison(int argc, char** argv) :
      cAppliWithSetImage (argc-1,argv+1,0),
      mNbXY    (20),
      mNbProf  (3)
{

    ElInitArgMain
    (
       argc,argv,
       LArgMain()  << EAMC(mNameIm, "Image",eSAM_IsExistFile)
                   << EAMC(mNameOri, "Orientation",eSAM_IsExistFile),
       LArgMain()  << EAM(mNbXY,"NbXY",true,"Number of point of the Grid")
                   << EAM(mNbProf,"NbProf",true,"Number of depth")
    );
    // std::string aPref = "Genepi-";

     for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
     {
          cImaMM * anIm = mVSoms[aK]->attr().mIma;
          cDicoAppuisFlottant aDAF;
          cMesureAppuiFlottant1Im aMAF;
          GenerateMesure2D3D(anIm->CamGen(),mNbXY,mNbProf,anIm->mNameIm,aDAF,aMAF);
          cSetOfMesureAppuisFlottants  aSMAF;
          aSMAF.MesureAppuiFlottant1Im().push_back(aMAF);
          MakeFileXML(aDAF,  NameGenepi(anIm->mNameIm,true));
          MakeFileXML(aSMAF, NameGenepi(anIm->mNameIm,false));
          // MakeFileXML(aDAF, aPref+ anIm->mNameIm + "-Mes3D.xml");
          // MakeFileXML(aSMAF, aPref+ anIm->mNameIm + "-Mes2D.xml");
        
     }
}




int GenerateAppLiLiaison_main(int argc, char** argv)
{
     cAppli_GenerateAppuisLiaison anAppli(argc,argv);
/*
    std::string aNameCam,aNameOri;
    int aNbXY = 20,aNbProf = 3;

    ElInitArgMain
    (
       argc,argv,
       LArgMain()  << EAMC(aNameIm, "Image",eSAM_IsExistFile),
                   << EAMC(aNameOri, "Orientation",eSAM_IsExistFile),
       LArgMain()  << EAM(aNbXY,"NbXY",true,"Number of point of the Grid")
                   << EAM(aNbProf,"NbProf",true,"Number of depth")
    );

    CamStenope * aCam = BasicCamOrientGenFromFile(aNameCam);
    cDicoAppuisFlottant aDAF;
    cMesureAppuiFlottant1Im aMAF;
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    GenerateMesure2D3D(aCam,aNbXY,aNbProf,aNameIm,aDAF,aMAF);
*/



    std::cout << "*******************************\n";
    std::cout << "*                             *\n";
    std::cout << "*     GENE-rate               *\n";
    std::cout << "*     P-oints 3D and          *\n";
    std::cout << "*     I-mage projectctio,     *\n";
    std::cout << "*                             *\n";
    std::cout << "*******************************\n";

    return EXIT_SUCCESS;
}

int ConvertCalib_main(int argc, char** argv)
{
   // virtual  Pt3dr ImEtProf2Terrain(const Pt2dr & aP,double aZ) const;
   // for (int aK=0 ; aK<argc ; aK++)
   //     std::cout << " # " << argv[aK] << "\n";

    std::string aCalibIn;
    std::string aCalibOut;
    int aNbXY=20;
    int aNbProf=2;
    int aDRMax;
    int aDegGen;
    std::string aNameCalibOut =  "Out-ConvCal.xml";
    bool PPFree = true;
    bool CDFree = true;
    bool FocFree = true;
    bool DecFree = true;

    

    ElInitArgMain
    (
       argc,argv,
       LArgMain()  << EAMC(aCalibIn, "Input Calibration",eSAM_IsExistFile)
                   << EAMC(aCalibOut,"Output calibration",eSAM_IsExistFile),
       LArgMain()  << EAM(aNbXY,"NbXY",true,"Number of point of the Grid")
                   << EAM(aNbProf,"NbProf",true,"Number of depth")
                   << EAM(aDRMax,"DRMax",true,"Max degree of radial dist (def=depend Output calibration)")
                   << EAM(aDegGen,"DegGen",true,"Max degree of generik polynom (def=depend Output calibration)")
                   << EAM(PPFree,"PPFree",true,"Principal point free (Def=true)")
                   << EAM(CDFree,"CDFree",true,"Distorsion center free (def=true)")
                   << EAM(FocFree,"FocFree",true,"Focal free (def=true)")
                   << EAM(DecFree,"DecFree",true,"Decentrik free (def=true when appliable)")
    );


    if (MMVisualMode) return EXIT_SUCCESS;

   std::string aNameImage = aCalibOut;
   std::string aDirTmp = DirOfFile(aCalibIn) + "Ori-ConvCalib/";
   ELISE_fp::MkDir(aDirTmp);

   CamStenope * aCamIn =  Std_Cal_From_File(aCalibIn);
   cDicoAppuisFlottant aDAF;
   cMesureAppuiFlottant1Im aMAF;

std::cout << "CONVCALL " << __LINE__ << "\n";
   GenerateMesure2D3D(aCamIn,aNbXY,aNbProf,aNameImage,aDAF,aMAF);
std::cout << "CONVCALL " << __LINE__ << "\n";
   cCalibrationInternConique aCICOut = StdGetFromPCP(aCalibOut,CalibrationInternConique);
   cLibertOfCalib  aLOC = GetDefDegreeOfCalib(aCICOut.CalibDistortion().back());
   if (!EAMIsInit(&aDRMax) ) aDRMax = aLOC.mDegRad;
   if (!EAMIsInit(&aDegGen)) aDegGen = aLOC.mDegGen;

   cSetOfMesureAppuisFlottants aSMAF;
   aSMAF.MesureAppuiFlottant1Im().push_back(aMAF);

   MakeFileXML(aDAF, aDirTmp + "Mes3D.xml");
   MakeFileXML(aSMAF, aDirTmp + "Mes2D.xml");

   cOrientationConique  anOC = aCamIn->StdExportCalibGlob();
   MakeFileXML(anOC, aDirTmp + "Orientation-" + aNameImage + ".xml");

   std::string aCom =    MM3dBinFile("Apero")
                       + XML_MM_File("Apero-ConvCal.xml")
                       + " DirectoryChantier=" +DirOfFile(aCalibIn)
                       + " +AeroIn=ConvCalib"
                       + " +CalibIn="  + aCalibOut
                       + " +CalibOut=" + aNameCalibOut
                       + " +FocFree="  + ToString(FocFree)
                       + " +PPFree="   + ToString(PPFree)
                       + " +CDFree="   + ToString(CDFree)
                       + " +DRMax=" + ToString(aDRMax)
                       + " +DegGen=" + ToString(aDegGen)
                       + " +LibDec=" + ToString(DecFree)
                      ;


   std::cout << "COM= " << aCom << "\n";
   System(aCom);


// "/opt/micmac/culture3d/bin/mm3d"

//Apero Apero-ConvCal.xml  DirectoryChantier=/home/prof/Bureau/ConvertCali/ +DRMax=0

   // std::cout << "CalIn=" << aCalibIn << " Foc " << aCamIn->Focale() << "\n";
   return EXIT_SUCCESS;
}

class cAppli_ConvOriCalib
{
   public :
     cAppli_ConvOriCalib(int argc, char** argv);

     std::string         mPat;
     std::string         mDir;
     std::string         mOriIn;
     std::string         mOriCalib;
     std::string         mOriOut;
     cElemAppliSetFile   mEASF;

};

cAppli_ConvOriCalib::cAppli_ConvOriCalib(int argc, char** argv) 
{
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mPat, "Pat or file")
                    << EAMC(mOriIn, "Input global orientation")
                    << EAMC(mOriCalib, "Targeted   internal orientatin")
                    << EAMC(mOriOut, "Ouptut prientation folder"),
        LArgMain()  
    );

    mEASF.Init(mPat);
    mDir = mEASF.mDir;
    StdCorrecNameOrient(mOriIn,mDir);
    StdCorrecNameOrient(mOriCalib,mDir);
 
    for (const auto & aName : (*mEASF.SetIm()))
    {
        std::string aCom = MM3dBinFile_quotes( "Genepi " )
                           + " " + aName 
                           + " " + mOriIn
                           + "\n"; 
        System(aCom);
        aCom =  MM3dBinFile_quotes( "Aspro ")
                           + " " + aName
                           + " " + mOriCalib
                           + " " + NameGenepi(aName,true)
                           + " " + NameGenepi(aName,false)
                           + " Out=" + mOriOut;
        System(aCom);
        ELISE_fp::RmFile(mDir+NameGenepi(aName,true));
        ELISE_fp::RmFile(mDir+NameGenepi(aName,false));
        // std::cout << "COM : [ " << aCom << " ]\n";
        // std::string 
        // std::cout << "COM : [ " << aCom << " ]\n";

    }
}

int ConvertOriCalib_main(int argc,char ** argv)
{
   cAppli_ConvOriCalib anAppli(argc,argv);
   return EXIT_SUCCESS;
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
