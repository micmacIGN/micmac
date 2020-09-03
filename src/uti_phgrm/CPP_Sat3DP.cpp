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

#if ELISE_QT
    #include "general/visual_mainwindow.h"
#endif

#include "StdAfx.h"
#include <algorithm>
#include "Apero/cCameraRPC.h"


class cSomSat;
class cGraphHomSat;
class cAppliSat3DPipeline;
class cCommonAppliSat3D;
class cAppliCreateEpi;
class cAppliRecalRPC;
class cAppliMM1P;



/****************************************/
/********* cCommonAppliSat3D ************/
/****************************************/

class cCommonAppliSat3D
{
	public:

		cCommonAppliSat3D();

		LArgMain &     ArgCom();
        std::string    ComParamPairs();
        std::string    ComParamEpip();
		std::string    ComParamRPC();
        std::string    ComParamMatch();
		std::string	   ComParamFuse();


		cInterfChantierNameManipulateur * mICNM;
		
		
		/* Common parameters */
		bool							  mExe;
		std::string 					  mDir;
		std::string 					  mSH;
		bool 							  mExpTxt;


		/* Pairs param */
		std::string mFilePairs;

        
        /* Epip param */
        bool 				mDoIm;
        bool 				mDegreEpi;
        Pt2dr 				mDir1;
		Pt2dr				mDir2;
        int 				mNbZ;
	    int					mNbZRand;
	    Pt2dr				mIntZ;
	    int					mNbXY;
	    Pt2di				mNbCalcDir;
	    std::vector<double>	mExpCurve;
	    std::vector<double>	mOhP;

        /* Convert orientation => à verifier */
        // images and Appuis generés par CreateEpip, mOutRPC, Degre, ChSys
		std::string mOutRPC;
		int			mDegreRPC;
		std::string mChSys;

        /* Match param */
        int 	mZoom0;
	   	int 	mZoomF;
		bool 	mCMS;
		bool 	mDoPly;
		double  mRegul;
		double  mDefCor;
		bool 	ExpTxt;
		Pt2di   mSzW0;
		bool	mCensusQ;

        /* Bascule param */
        // Malt UrbanMNE to create destination frame
        // NuageBascule -> obligatory params not user
		std::string mNameEpiLOF;

        /* SMDM param */
		std::string mOutSMDM;

	private:
		LArgMain * mArgCommon;

		cCommonAppliSat3D(const cCommonAppliSat3D&) ; // N.I.

};

cCommonAppliSat3D::cCommonAppliSat3D() :
	mExe(true),
	mDir("./"),
	mSH(""),
	mExpTxt(false),
	mFilePairs("Pairs.xml"),
	mOutRPC("EpiRPC"),
	mDegreRPC(0),
	mNameEpiLOF("EpiListOfFile.xml"),
	mOutSMDM("Fusion/"),
	mArgCommon(new LArgMain)
{
	*mArgCommon
			<< EAM(mDir,"Dir",true,"Current directory, Def=./")
			<< EAM(mFilePairs,"Pairs",true,"File with overlapping pairs, Def=Pairs.xml")
			<< EAM(mDoIm,"DoEpi",true,"Epipolar rectification, Def=true")
			<< EAM(mDegreEpi,"DegreeEpi",true,"Epipolar rectification: polynomial degree, Def=9")
			<< EAM(mDir1,"Dir1",true,"Epipolar rectification: Direction of Epip one (when Ori=NONE)")
			<< EAM(mDir2,"Dir2",true,"Epipolar rectification: Direction of Epip two (when Ori=NONE)")
			<< EAM(mSH,"SH",true,"Epipolar rectification: Prefix Homologue (when Ori=NONE), Def=\"\"")
			<< EAM(mExpTxt,"ExpTxt",true,"Epipolar rectification: Homol in text format? (when Ori=NONE), Def=\"false\"")
			<< EAM(mNbZ,"NbZ",true,"Epipolar rectification: Number of Z, def=1 (NbLayer=1+2*NbZ)")
			<< EAM(mNbZRand,"NbZRand",true,"Epipolar rectification: Number of additional random Z in each bundle, Def=1")
			<< EAM(mIntZ,"IntZ",true,"Epipolar rectification: Z interval, for test or correct interval of RPC")
			<< EAM(mNbXY,"NbXY",true,"Epipolar rectification: Number of point / line or col, def=100")
			<< EAM(mNbCalcDir,"NbCalcDir",true,"Epipolar rectification: Calc directions : Nbts / NbEchDir")
			<< EAM(mExpCurve,"ExpCurve",true,"Epipolar rectification: 0-SzIm ,1-Number of Line,2- Larg (in [0 1]),3-Exag deform,4-ShowOut")
			<< EAM(mOhP,"OhP",true,"Epipolar rectification: Oh's method test parameter")
			<< EAM(mOutRPC,"OutRPC",true,"RPC recalculation: Output RPC orientation directory (after rectification)")
			<< EAM(mDegreRPC,"DegreRPC",true,"RPC recalculation: Degree of RPC polynomial correction, Def=0")
			<< EAM(mChSys,"ChSys",true,"RPC recalculation: File specifying a euclidean projection system of your zone")
			<< EAM(mZoom0,"Zoom0",true,"Image matching: Zoom Init (Def=64)")
			<< EAM(mZoomF,"ZoomF",true,"Image matching: Zoom Final (Def=1)")
			<< EAM(mCMS,"CMS",true,"Image matching: Multi Scale Correl (Def=ByEpip)")
			<< EAM(mDoPly,"DoPly",true,"Image matching: Generate Ply")
			<< EAM(mRegul,"ZReg",true,"Image matching: Regularisation factor (Def=0.05)")
			<< EAM(mDefCor,"DefCor",true,"Image matching: Def cor (Def=0.5)")
			<< EAM(mSzW0,"SzW0",true,"Image matching: Sz first Windows, def depend of NbS (1 MS, 2 no MS)")
			<< EAM(mCensusQ,"CensusQ",true,"Image matching: Use Census Quantitative")
			<< EAM(mOutSMDM,"OutSMDM",true,"Depth map fusion: Name of the output folder, Def=Fusion/")
			<< EAM(mExe,"Exe",true,"Execute all, Def=true")
			;

	mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);

	StdCorrecNameOrient(mOutRPC,mDir,true);
}

LArgMain & cCommonAppliSat3D::ArgCom()
{
	return * mArgCommon;
}


std::string cCommonAppliSat3D::ComParamPairs()
{
	std::string aCom;
	aCom += aCom + " Out=" + mFilePairs;

	return aCom;
}

std::string cCommonAppliSat3D::ComParamEpip()
{
    std::string aCom;
    if (EAMIsInit(&mExpTxt))    aCom += aCom   + " ExpTxt=" + ToString(mExpTxt);
    if (EAMIsInit(&mSH))        aCom +=  " NameH=" + mSH;
    if (EAMIsInit(&mDegreEpi))  aCom +=  " Degre=" + ToString(mDegreEpi);
    if (EAMIsInit(&mDegreEpi))  aCom +=  " Degre=" + ToString(mDegreEpi);
    if (EAMIsInit(&mDir1))      aCom +=  " Dir1=" + ToString(mDir1);
    if (EAMIsInit(&mDir2))      aCom +=  " Dir2=" + ToString(mDir2);

    if (EAMIsInit(&mNbZ))       aCom +=  " NbZ=" + ToString(mNbZ);
    if (EAMIsInit(&mNbZRand))   aCom +=  " NbZRand=" + ToString(mNbZRand);
    if (EAMIsInit(&mIntZ))      aCom +=  " IntZ=" + ToString(mIntZ);
    if (EAMIsInit(&mNbXY))      aCom +=  " NbXY=" + ToString(mNbXY);
    if (EAMIsInit(&mNbCalcDir)) aCom +=  " NbCalcDir=" + ToString(mNbCalcDir);
    if (EAMIsInit(&mExpCurve))  aCom +=  " ExpCurve=" + ToString(mExpCurve);
    if (EAMIsInit(&mOhP))       aCom +=  " Oh=" + ToString(mOhP);

    return aCom;
}

std::string cCommonAppliSat3D::ComParamRPC()
{
	std::string aCom;
	if (EAMIsInit(&mDegreRPC))  aCom += aCom  + " Degre=" + ToString(mDegreRPC);
	if (EAMIsInit(&mChSys))     aCom += " ChSys=" + mChSys;
//	if (EAMIsInit(&mOutRPC))     aCom += " OutRPC=" + mOutRPC;

	return aCom;
}
 

std::string cCommonAppliSat3D::ComParamMatch()
{
	std::string aCom;
    if (EAMIsInit(&mExpTxt))  aCom += aCom  + " ExpTxt=" + ToString(mExpTxt);
    if (EAMIsInit(&mZoom0))   aCom +=  " Zoom0=" + ToString(mZoom0);
    if (EAMIsInit(&mZoomF))   aCom +=  " ZoomF=" + ToString(mZoomF);
    if (EAMIsInit(&mCMS))     aCom +=  " CMS=" + ToString(mCMS);
    if (EAMIsInit(&mDoPly))   aCom +=  " DoPly=" + ToString(mDoPly);
    if (EAMIsInit(&mRegul))   aCom +=  " Regul=" + ToString(mRegul);
    if (EAMIsInit(&mDefCor))  aCom +=  " DefCor=" + ToString(mDefCor);
    if (EAMIsInit(&mSzW0))    aCom +=  " SzW0=" + ToString(mSzW0);
    if (EAMIsInit(&mCensusQ)) aCom +=  " CensusQ=" + ToString(mCensusQ);

	return aCom;
}

std::string cCommonAppliSat3D::ComParamFuse()
{
	std::string aCom;
	if (EAMIsInit(&mOutRPC))     aCom += " OutRPC=" + mOutRPC;


	return aCom;
}

/****************************************/
/*********** cSomSat       **************/
/****************************************/



class cSomSat
{
     public :
       cSomSat(const cGraphHomSat & aGH,const std::string & aName,CameraRPC * aCam,Pt3dr aC) :
          mGH   (aGH),
          mName (aName),
          mCam (aCam),
          mC   (aC)
       {
       }

       const cGraphHomSat & mGH;
       std::string          mName;
       CameraRPC *          mCam;
       Pt3dr                mC;

       bool HasInter(const cSomSat & aS2) const;

};

bool cSomSat::HasInter(const cSomSat & aS2) const
{
    if (mCam && (aS2.mCam))
    {
         const cElPolygone &  aPol1= mCam->EmpriseSol();
         const cElPolygone &  aPol2= aS2.mCam->EmpriseSol();
         const cElPolygone &  aInter = aPol1 * aPol2; 
	

         if (aInter.Surf() <= 0) 
		 	return false;

    }

    return true;
}

/****************************************/
/*********** cGraphHomSat  **************/
/****************************************/

class cGraphHomSat
{
    public :

        friend class cSomSat;

        cGraphHomSat(int argc,char** argv);
        void DoAll();

    private :

        std::string mDir;
        std::string mPat;
        std::string mOri;
        cInterfChantierNameManipulateur * mICNM;

        std::string mOut;

        std::list<std::string>  mLFile;
        std::vector<cSomSat *>    mVC;
        int                    mNbSom;
        double                 mAltiSol;

};


cGraphHomSat::cGraphHomSat(int argc,char ** argv) :
      mOut       ("Pairs.xml"),
      mAltiSol   (0)
{

	

    ElInitArgMain
    (
    	argc,argv,
        LArgMain()  << EAMC(mDir,"Directory", eSAM_IsDir)
                    << EAMC(mPat,"Images pattern", eSAM_IsPatFile)
                    << EAMC(mOri,"Orientation dir", eSAM_IsExistFile),
        LArgMain()  << EAM(mAltiSol,"AltiSol",true)
                    << EAM(mOut,"Out",true)

    );
    if (!MMVisualMode)
    {
		cTplValGesInit<std::string>  aTplFCND;
        mICNM = cInterfChantierNameManipulateur::StdAlloc(argc,argv,mDir,aTplFCND);
 
		StdCorrecNameOrient(mOri,mDir,true);
 
        mLFile =  mICNM->StdGetListOfFile(mPat,1);
 
        mNbSom =  (int)mLFile.size();
 
        std::cout << "Nb Images = " <<  mNbSom << "\n";

        int aCpt = 0;
		for
        (
             std::list<std::string>::const_iterator itS=mLFile.begin();
             itS!=mLFile.end();
             itS++
        )
        {

			
			CameraRPC * aCam = new CameraRPC(mICNM->StdNameCamGenOfNames(mOri,*itS));
			Pt3dr aC = aCam->OrigineProf();

			mVC.push_back(new cSomSat(*this,*itS,aCam,aC));
            std::cout << "Load  : remain " << (mNbSom-aCpt) << " to do\n";


			aCpt++;
		}



		cSauvegardeNamedRel aRel;
        for (int aK1=0 ; aK1<mNbSom ; aK1++)
        {
            for (int aK2=aK1+1 ; aK2<mNbSom ; aK2++)
            {
                 if (mVC[aK1]->HasInter(*(mVC[aK2])))
                 {
                    aRel.Cple().push_back(cCpleString(mVC[aK1]->mName,mVC[aK2]->mName));
                 }

            }
            std::cout << "Graphe : remain " << (mNbSom-aK1) << " to do\n";
        }
        MakeFileXML(aRel,mDir+mOut);


	}
}

int GraphHomSat_main(int argc,char** argv)
{
	cGraphHomSat aGHS(argc,argv);

	return EXIT_SUCCESS;
}

/****************************************/
/********* cAppliCreateEpi **************/
/****************************************/

class cAppliCreateEpi : cCommonAppliSat3D
{
	public:
		cAppliCreateEpi(int argc, char** argv);

	private:
		cCommonAppliSat3D mCAS3D;
		std::string 	  mFilePairs;
		std::string 	  mOri;

};

cAppliCreateEpi::cAppliCreateEpi(int argc, char** argv)
{
	ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mFilePairs,"List of overlapping image pairs",eSAM_IsExistFile)
                     << EAMC(mOri,"Orientation directory, (NONE if rectification from tie-points)",eSAM_IsDir),
         LArgMain()
                     << mCAS3D.ArgCom()
    );

	
	StdCorrecNameOrient(mOri,mCAS3D.mDir,true);

	cSauvegardeNamedRel aPairs = StdGetFromPCP(mCAS3D.mDir+mFilePairs,SauvegardeNamedRel);
	
	std::list<std::string> aLCom;

	for (auto itP : aPairs.Cple())
	{
		std::string aComTmp = MMBinFile(MM3DStr) + " CreateEpip " 
						      + itP.N1() + BLANK + itP.N2() + BLANK + mOri + BLANK 
							  + mCAS3D.ComParamEpip();

		aLCom.push_back(aComTmp);
	}


	if (mCAS3D.mExe)
		cEl_GPAO::DoComInParal(aLCom);
    else
	{
		for (auto iCmd : aLCom)
      		std::cout << "SUBCOM= " << iCmd << "\n";
	}



}

int CPP_AppliCreateEpi_main(int argc,char ** argv)
{
	cAppliCreateEpi anAppCreatEpi(argc,argv);

	return EXIT_SUCCESS;
}


/****************************************/
/********* cAppliRecalRPC  **************/
/****************************************/

class cAppliRecalRPC : cCommonAppliSat3D
{
	public:
		cAppliRecalRPC(int argc,char ** argv);

	private:
		cCommonAppliSat3D mCAS3D;
		std::string 	  mOri;
};					


cAppliRecalRPC::cAppliRecalRPC(int argc, char** argv)
{
    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mCAS3D.mFilePairs,"List of overlapping image pairs",eSAM_IsExistFile)
		 			 << EAMC(mOri,"RPC original orientation (dir)",eSAM_IsDir), 
					 //mOri needed to recover the names of the Epi images

         LArgMain()
                     << mCAS3D.ArgCom()
    );

	StdCorrecNameOrient(mOri,mCAS3D.mDir,true);
    
	cSauvegardeNamedRel aPairs = StdGetFromPCP(mCAS3D.mDir+mCAS3D.mFilePairs,SauvegardeNamedRel);

    std::list<std::string> aLCom;

    for (auto itP : aPairs.Cple())
    {
        std::string aNAppuisI1 = mCAS3D.mICNM->NameAppuiEpip(mOri,itP.N1(),itP.N2());
        std::string aNAppuisI2 = mCAS3D.mICNM->NameAppuiEpip(mOri,itP.N2(),itP.N1());

        std::string aNI1 = mCAS3D.mICNM->NameImEpip(mOri,itP.N1(),itP.N2());
        std::string aNI2 = mCAS3D.mICNM->NameImEpip(mOri,itP.N2(),itP.N1());

		

        std::string aComI1 = MMBinFile(MM3DStr) + " Convert2GenBundle "
                              + aNI1 + BLANK + aNAppuisI1 + BLANK
							  + mCAS3D.mOutRPC.substr(0,mCAS3D.mOutRPC.size()-1) 
                              + mCAS3D.ComParamRPC();

        std::string aComI2 = MMBinFile(MM3DStr) + " Convert2GenBundle "
                              + aNI2 + BLANK + aNAppuisI2 + BLANK
							  + mCAS3D.mOutRPC.substr(0,mCAS3D.mOutRPC.size()-1)
                              + mCAS3D.ComParamRPC();


		std::string aKeyGB = "NKS-Assoc-Im2GBOrient@-" +  mCAS3D.mOutRPC.substr(0,mCAS3D.mOutRPC.size()-1);

		std::string aNameGBI1 = mCAS3D.mICNM->Assoc1To1(aKeyGB,aNI1,true);
		std::string aNameGBI2 = mCAS3D.mICNM->Assoc1To1(aKeyGB,aNI2,true);

		std::string aComConv1 = MMBinFile(MM3DStr) + "Satelib RecalRPC " 
							  + aNameGBI1 + " OriOut=" + mCAS3D.mOutRPC.substr(mCAS3D.mOutRPC.size()-1);
		std::string aComConv2 = MMBinFile(MM3DStr) + "Satelib RecalRPC " 
							  + aNameGBI2 + " OriOut=" + mCAS3D.mOutRPC.substr(mCAS3D.mOutRPC.size()-1);



        aLCom.push_back(aComI1);
        aLCom.push_back(aComI2);
		aLCom.push_back(aComConv1);
		aLCom.push_back(aComConv2);

		// remove tmp dir
		aLCom.push_back("rm -r Ori-" + mCAS3D.mOutRPC.substr(0,mCAS3D.mOutRPC.size()-1));
    }


    if (mCAS3D.mExe)
        cEl_GPAO::DoComInSerie(aLCom);
    else
    {
        for (auto iCmd : aLCom)
            std::cout << "SUBCOM= " << iCmd << "\n";
    }
}

int CPP_AppliRecalRPC_main(int argc,char ** argv)
{
    cAppliRecalRPC anAppRRPC(argc,argv);

    return EXIT_SUCCESS;
}


/****************************************/
/********* cAppliMM1P      **************/
/****************************************/

class cAppliMM1P : cCommonAppliSat3D
{
	public:
		cAppliMM1P(int argc, char** argv);

	private:
		cCommonAppliSat3D mCAS3D;
        std::string 	  mFilePairs;
        std::string 	  mOri;

};

cAppliMM1P::cAppliMM1P(int argc, char** argv)
{
	ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mFilePairs,"List of overlapping image pairs",eSAM_IsExistFile)
                     << EAMC(mOri,"RPC original orientation",eSAM_IsDir),
					 //mOri needed to recover the names of the Epi images
         LArgMain()
                     << mCAS3D.ArgCom()
    );

	StdCorrecNameOrient(mOri,mCAS3D.mDir,true);
    
	cSauvegardeNamedRel aPairs = StdGetFromPCP(mCAS3D.mDir+mFilePairs,SauvegardeNamedRel);

    std::list<std::string> aLCom;

    for (auto itP : aPairs.Cple())
    {
		std::string aNI1 = mCAS3D.mICNM->NameImEpip(mOri,itP.N1(),itP.N2());
		std::string aNI2 = mCAS3D.mICNM->NameImEpip(mOri,itP.N2(),itP.N1());
        
		std::string aComTmp = MMBinFile(MM3DStr) + " MM1P "
                              + aNI1 + BLANK + aNI2 + " NONE "
                              + mCAS3D.ComParamMatch();

        aLCom.push_back(aComTmp);
    }

    if (mCAS3D.mExe)
        cEl_GPAO::DoComInSerie(aLCom);
    else
    {
        for (auto iCmd : aLCom)
            std::cout << "SUBCOM= " << iCmd << "\n";
    }


}

int CPP_AppliMM1P_main(int argc,char ** argv)
{
	cAppliMM1P anAppMM1P(argc,argv);
	
	return EXIT_SUCCESS;
}

/****************************************/
/********* cAppliFusion    **************/
/****************************************/

class cAppliFusion
{
	public:
		cAppliFusion(int argc,char ** argv);

		void DoAll();

		cCommonAppliSat3D mCAS3D;

	private:
		std::string mFilePairs;
		std::string mOri;

};

cAppliFusion::cAppliFusion(int argc,char ** argv)
{
	ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mFilePairs,"List of overlapping image pairs",eSAM_IsExistFile)
                     << EAMC(mOri,"RPC original orientation",eSAM_IsDir),
                     //mOri needed to recover the names of the Epi images
         LArgMain()
                     << mCAS3D.ArgCom()
    );

    StdCorrecNameOrient(mOri,mCAS3D.mDir,true);
}

void cAppliFusion::DoAll()
{
	cSauvegardeNamedRel aPairs = StdGetFromPCP(mCAS3D.mDir+mFilePairs,SauvegardeNamedRel);

	/* Create xml file list with the concerned epipolar images */	
	cListOfName 				aLON;
	std::map<std::string,int>	aMEp;
	std::list<std::string>      aLP;

	int aCpt=0;
    for (auto itP : aPairs.Cple())
    {
        std::string aNI1 = mCAS3D.mICNM->NameImEpip(mOri,itP.N1(),itP.N2());
        std::string aNI2 = mCAS3D.mICNM->NameImEpip(mOri,itP.N2(),itP.N1());

		aLP.push_back(aNI1+"-"+aNI2);
		aLP.push_back(aNI2+"-"+aNI1);


		if (!DicBoolFind(aMEp,aNI1))
		{
			aMEp[aNI1] = aCpt;
    		aLON.Name().push_back(aNI1);
		}
		aCpt++;
		
		if (!DicBoolFind(aMEp,aNI2))
		{
			aMEp[aNI2] = aCpt;
    		aLON.Name().push_back(aNI2);
		}
		aCpt++;


	}
    MakeFileXML(aLON,mCAS3D.mNameEpiLOF);


	/* Define the global frame of the reconstruction */
	std::string aCom = MMBinFile(MM3DStr) + " Malt UrbanMNE " 
			             + "NKS-Set-OfFile@" + mCAS3D.mNameEpiLOF + BLANK 
						 + mCAS3D.mOutRPC + " DoMEC=0";

	std::cout << "COM= " << aCom << "\n";
	System(aCom);


	/* Transform individual surfaces to global frame */
	std::list<std::string> aLCom;

	std::string aPref = "DSM_Pair";
	ELISE_fp::MkDirSvp(mCAS3D.mOutSMDM);	
	aCpt=0;
	for (auto itP : aLP)
	{
		std::string aComFuse = MMBinFile(MM3DStr) + " NuageBascule " 
				             + "MEC2Im-" + itP + "/"
							 + "NuageImProf_LeChantier_Etape_7.xml " 
							 + "MEC-Malt/NuageImProf_STD-MALT_Etape_8.xml " 
							 + mCAS3D.mOutSMDM + aPref + ToString(aCpt) + ".xml"; 

		aCpt++;

		std::cout << aComFuse << "\n";
		aLCom.push_back(aComFuse);
	}	
	cEl_GPAO::DoComInParal(aLCom);

	/* Merge */
	std::string aComMerge = MMBinFile(MM3DStr) + " SMDM " + mCAS3D.mOutSMDM + "/" + aPref + ".*xml";
	std::cout << aComMerge << "\n";

	System(aComMerge);
}

int CPP_AppliFusion_main(int argc,char ** argv)
{

	cAppliFusion aAppFus(argc,argv);
	aAppFus.DoAll();

	return EXIT_SUCCESS;
}

/*******************************************/
/********* cAppliSat3DPipeline  ************/
/*******************************************/


class cAppliSat3DPipeline : cCommonAppliSat3D
{
	public:
		cAppliSat3DPipeline(int argc, char** argv);
		
		void DoAll();

	private:
		void StdCom(const std::string & aCom,const std::string & aPost="");

		cCommonAppliSat3D mCAS3D;

		std::string mPat;
		std::string mOri;

		bool		mDebug;

		ElTimer     mChrono;
};

cAppliSat3DPipeline::cAppliSat3DPipeline(int argc,char** argv) :
	mDebug(false)

{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mPat,"Pattern of images",eSAM_IsPatFile)
					<< EAMC(mOri,"Orientation directory",eSAM_IsDir),  
        LArgMain()

                    << EAM(mDebug, "Debug", true, "Debug mode, def false")
					<< mCAS3D.ArgCom()
   );

	StdCorrecNameOrient(mOri,mCAS3D.mDir,true);
}


void cAppliSat3DPipeline::StdCom(const std::string & aCom,const std::string & aPost)
{
	std::string  aFullCom = MMBinFile(MM3DStr) + BLANK + aCom + BLANK;
    aFullCom = aFullCom + aPost;


    if (mCAS3D.mExe)
       System(aFullCom);
    else
       std::cout << "COM= " << aFullCom << "\n";

    std::cout << " DONE " << aCom << " in time " << mChrono.uval() << "\n";
}

void cAppliSat3DPipeline::DoAll()
{


	/********************************/
	/* 1- Calculate the image pairs */
	/********************************/
	if (! EAMIsInit(&mCAS3D.mFilePairs)) 
	{
		StdCom("TestLib Sat3D_Pairs ", 
			    mCAS3D.mDir + BLANK + QUOTE(mPat) + BLANK + mOri + BLANK + 
				mCAS3D.ComParamPairs());
	}
	else
		std::cout << mCAS3D.mFilePairs << " used." << "\n";





	/****************************************************/
	/* 2- Rectify pairs of images to epipolar geometry  */
	/****************************************************/
	if (mCAS3D.mDoIm == true)
		StdCom("TestLib Sat3D_CreateEpip ", 
				mCAS3D.mFilePairs + BLANK + mOri + BLANK + 
				mCAS3D.ComParamEpip());
	else
		std::cout << "No epipolar image creation." << "\n";



	/**************************************/
	/* 3- Recalculate the RPC orientation */
	/**************************************/
	StdCom("TestLib Sat3D_EpiRPC ", mCAS3D.mFilePairs + BLANK + mOri
								    + mCAS3D.ComParamRPC());
	


	/******************************************************/
	/* 4- Perform dense image matching per pair of images */
	/******************************************************/
	StdCom("TestLib Sat3D_MM1P ", 
			mCAS3D.mFilePairs + BLANK + mOri + BLANK + mCAS3D.ComParamMatch());



	/**************************************************************************/
	/* 5- Transform the per-pair reconstructions to a commont reference frame 
	 *    and do the 3D fusion */
	/**************************************************************************/
	StdCom("TestLib Sat3D_Fuse ", mCAS3D.mFilePairs + BLANK + mOri 
								 + mCAS3D.ComParamFuse());  



}

int Sat3D_main(int argc, char ** argv)
{
	cAppliSat3DPipeline aAppliSat3D(argc,argv);
	aAppliSat3D.DoAll();

	return EXIT_SUCCESS;
}


//TODO:
//testing of OutRPC bc i think it does not take it into account in EpiRPC
//fix pb with NuageBascule
//automated calcul of Etape
//
//
//
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
