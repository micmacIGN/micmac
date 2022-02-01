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


#include "SAT4GEO.h"


cCommonAppliSat3D::cCommonAppliSat3D() :
	mExe(true),
	mDir("./"),
	mSH(""),
	mExpTxt(false),
	mNbProc(8),
	mFilePairs("Pairs.xml"),
	mFPairsDirMEC("PairsDirMEC.xml"),
	mBtoHLim   (Pt2dr(0.01,0.3)),
	mDoIm(true),
        mXCorrecOri(true),
	mOutRPC("EpiRPC"),
	mDegreRPC(0),
	mZoom0(64),
	mRegul(0.2),
	mSzW(3),
        mMMVII(0),
	mMMVII_mode("MMV1"),
	mMMVII_ImName("Px1_MMVII.tif"),
	mMMVII_SzTile(Pt2di(1024,1024)),
        mMMVII_NbProc(8),
        //mZoomF(1),
	//mHasVeg(true),
	//mHasSBG(false),
	mNameEpiLOF("EpiListOfFile.xml"),
	mOutSMDM("Fusion/"),
	mArgBasic(new LArgMain),
	mArgEpip(new LArgMain),
	mArgRPC(new LArgMain),
	mArgMM1P(new LArgMain),
	mArgFuse(new LArgMain)

{
	*mArgBasic
			<< EAM(mDir,"Dir",true,"Current directory, Def=./")
			<< EAM(mExe,"Exe",true,"Execute all, Def=true")
			<< EAM(mNbProc,"NbP",true,"Num of parallel processes, Def=8")
			<< EAM(mFilePairs,"Pairs",true,"File with overlapping pairs, Def=Pairs.xml")
			<< EAM(mFPairsDirMEC,"PairsDirMEC",true,"File with DirMECc of overlapping pairs, Def=PairsDirMEC.xml")
			<< EAM(mBtoHLim,"BH",true,"Base to height ratio limits, def=[0.01,0.3]");
	

	*mArgEpip
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
			<< EAM(mXCorrecOri,"XCorrecOri",true,"Epipolar rectification: Correct X-Pax using orient and Z=average,Def=true")
			<< EAM(mXCorrecHom,"XCorrecHom",true,"Epipolar rectification: Correct X-Pax using homologous point")
			<< EAM(mXCorrecL2,"XCorrecL2",true,"Epipolar rectification: L1/L2 Correction for X-Pax");



	*mArgRPC
			<< EAM(mDegreRPC,"DegreRPC",true,"RPC recalculation: Degree of RPC polynomial correction, Def=0")
			<< EAM(mChSys,"ChSys",true,"RPC recalculation: File specifying a euclidean projection system of your zone");


	*mArgMM1P
			<< EAM(mZoom0,"Zoom0",true,"Image matching: Zoom Init (Def=64)")
//			<< EAM(mZoomF,"ZoomF",true,"Image matching: Zoom Final (Def=1)")
//			<< EAM(mCMS,"CMS",true,"Image matching: Multi Scale Correl (Def=ByEpip)")
//			<< EAM(mHasVeg,"HasVeg",true,"Image matching: Has vegetation, Def= false")
//			<< EAM(mHasSBG,"HasSBG",true,"Image matching: Has Sky Background , Def= true")
			<< EAM(mEZA,"EZA",true,"Image matching: Export Z absolute (Def=false)")
			<< EAM(mDoPly,"DoPly",true,"Image matching: Generate Ply")
			<< EAM(mInc,"Inc",true,"Image matching: Sigma Pixel for coherence (Def=1.5)")
			<< EAM(mRegul,"Regul",true,"Image matching: Regularisation factor (Def=0.2)")
			<< EAM(mSzW,"SzW",true,"Image matching: matching window size (Def=3)")
//			<< EAM(mDefCor,"DefCor",true,"Image matching: Def cor (Def=0.5)")
//			<< EAM(mSzW0,"SzW0",true,"Image matching: Sz first Windows, def depend of NbS (1 MS, 2 no MS)")
//			<< EAM(mCensusQ,"CensusQ",true,"Image matching: Use Census Quantitative")
			<< EAM(mMMVII,"MMVII",true,"Image matching: Use MMVII matching")
            << EAM(mMMVII_mode,"MMVII_mode",true,"Image matching: if MMVII==1, {MMV1,PSMNet,DeepPruner} Def=MMV1")
            << EAM(mMMVII_ModePad,"MMVII_ModePad",true,"Image matching: if MMVII==1, {NoPad PxPos PxNeg SzEq}")
            << EAM(mMMVII_ImName,"MMVII_ImName",true,"Image matching: if MMVII==1, name of depth map")
            << EAM(mMMVII_SzTile,"MMVII_SzTile",true,"Image matching: if MMVII==1, Size of tiling used to split computation, Def=[1024,1024]")
            << EAM(mMMVII_NbProc,"MMVII_NbProc",true,"Image matching: if MMVII==1, Nb of cores for II processing in MMVII, Def=8");

	*mArgFuse
			<< EAM(mOutRPC,"OutRPC",true,"RPC recal/Depth map fusion: RPC orientation directory (corresp. to epipolar images)")
			<< EAM(mOutSMDM,"OutSMDM",true,"Depth map fusion: Name of the output folder, Def=Fusion/");

	mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);

	StdCorrecNameOrient(mOutRPC,mDir,true);

}

LArgMain & cCommonAppliSat3D::ArgFuse()
{
	return * mArgFuse;
}

LArgMain & cCommonAppliSat3D::ArgBasic()
{
	return * mArgBasic;
}

LArgMain & cCommonAppliSat3D::ArgEpip()
{
	return * mArgEpip;
}

LArgMain & cCommonAppliSat3D::ArgRPC()
{
	return * mArgRPC;
}

LArgMain & cCommonAppliSat3D::ArgMM1P()
{
	return * mArgMM1P;
}


std::string cCommonAppliSat3D::ComParamPairs()
{
	std::string aCom;
	aCom += aCom + " Out=" + mFilePairs;
	aCom += " PairsDirMEC=" + mFPairsDirMEC;  
	aCom += " BH=" + ToString(mBtoHLim); 

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
    aCom +=  " XCorrecOri=" + ToString(mXCorrecOri);
    if (EAMIsInit(&mXCorrecHom)) aCom +=  " XCorrecHom=" + ToString(mXCorrecHom);
    if (EAMIsInit(&mXCorrecL2)) aCom +=  " XCorrecL2=" + ToString(mXCorrecL2);
    return aCom;
}

std::string cCommonAppliSat3D::ComParamRPC_Basic()
{
    std::string aCom;
    if (EAMIsInit(&mDegreRPC))  aCom += aCom  + " Degre=" + ToString(mDegreRPC);
    if (EAMIsInit(&mChSys))     aCom += " ChSys=" + mChSys;

    return aCom;
}

std::string cCommonAppliSat3D::ComParamRPC()
{
	std::string aCom = ComParamRPC_Basic();
	aCom += " OutRPC=" + mOutRPC;

	return aCom;
}
 

std::string cCommonAppliSat3D::ComParamMatch()
{
    std::string aCom;

    if (mMMVII)
    {
        aCom += BLANK + "MMVII=" + ToString(mMMVII);
        if (EAMIsInit(&mMMVII_mode))    aCom += BLANK + "MMVII_mode=" + mMMVII_mode;
        if (EAMIsInit(&mMMVII_ImName))  aCom += BLANK + "MMVII_ImName=" + mMMVII_ImName;
	if (EAMIsInit(&mMMVII_SzTile))  aCom += BLANK + "MMVII_SzTile=" + ToString(mMMVII_SzTile);
        if (EAMIsInit(&mMMVII_NbProc))  aCom += BLANK + "MMVII_NbProc=" + ToString(mMMVII_NbProc); 
    }
    else
    {
        if (EAMIsInit(&mExpTxt))  aCom += aCom  + " ExpTxt=" + ToString(mExpTxt);
        if (EAMIsInit(&mZoom0))   aCom +=  " Zoom0=" + ToString(mZoom0);
        //if (EAMIsInit(&mZoomF))   aCom +=  " ZoomF=" + ToString(mZoomF);
        //if (EAMIsInit(&mCMS))     aCom +=  " CMS=" + ToString(mCMS);
        if (EAMIsInit(&mEZA))     aCom +=  " EZA=" + ToString(mEZA);
            //if (EAMIsInit(&mHasVeg))  aCom +=  " HasVeg=" + ToString(mHasVeg);
            //if (EAMIsInit(&mHasSBG))  aCom +=  " HasSBG=" + ToString(mHasSBG);
        if (EAMIsInit(&mInc))  aCom +=  " Inc=" + ToString(mInc);
        if (EAMIsInit(&mDoPly))   aCom +=  " DoPly=" + ToString(mDoPly);
        //if (EAMIsInit(&mDefCor))  aCom +=  " DefCor=" + ToString(mDefCor);
        if (EAMIsInit(&mRegul))   aCom +=  " Regul=" + ToString(mRegul);
        if (EAMIsInit(&mSzW))    aCom +=  " SzW=" + ToString(mSzW);
        //if (EAMIsInit(&mDefCor))  aCom +=  " DefCor=" + ToString(mDefCor);
        //if (EAMIsInit(&mSzW0))    aCom +=  " SzW0=" + ToString(mSzW0);
        //if (EAMIsInit(&mCensusQ)) aCom +=  " CensusQ=" + ToString(mCensusQ);
        if (EAMIsInit(&mNbProc))    aCom +=  " NbP=" + ToString(mNbProc);

    }
	return aCom;
}

std::string cCommonAppliSat3D::ComParamFuse()
{
	std::string aCom;
	aCom += " OutRPC=" + mOutRPC;
    if (EAMIsInit(&mNbProc))    aCom +=  " NbP=" + ToString(mNbProc);


	return aCom;
}


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



cGraphHomSat::cGraphHomSat(int argc,char ** argv) :
      mOut       ("Pairs.xml"),
      mAltiSol   (0),
	  mBtoHLim   (Pt2dr(0.01,0.3))
{

	std::string aFPairsDirMEC = "PairsDirMEC.xml";

    ElInitArgMain
    (
    	argc,argv,
        LArgMain()  << EAMC(mDir,"Directory", eSAM_IsDir)
                    << EAMC(mPat,"Images pattern", eSAM_IsPatFile)
                    << EAMC(mOri,"Orientation dir", eSAM_IsExistFile),
        LArgMain()  << EAM(mAltiSol,"AltiSol",true, "Ground altitutde")
					<< EAM(mBtoHLim,"BH",true,"Base to height ratio limits, def=[0.01,0.3]")
                    << EAM(mOut,"Out",true,"Output file name")
					<< EAM(aFPairsDirMEC,"PairsDirMEC",true,"File with DirMECc of overlapping pairs, Def=PairsDirMEC.xml")

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



		/* Get overlapping images *****************
		 *  + verify if images intersect in 3D
		 *  + check if within bh limits 	 *  */
		cSauvegardeNamedRel aRel;
		cListOfName         aLDirMEC;
		std::list< std::string >  aLDM;
        for (int aK1=0 ; aK1<mNbSom ; aK1++)
        {
            for (int aK2=aK1+1 ; aK2<mNbSom ; aK2++)
            {
                 if (mVC[aK1]->HasInter(*(mVC[aK2])))
                 {
					double aBH = CalcBtoH(mVC[aK1]->mCam,mVC[aK2]->mCam);
				
					if ( (aBH>mBtoHLim.x) && (aBH<mBtoHLim.y))
					{
                    	aRel.Cple().push_back(cCpleString(mVC[aK1]->mName,mVC[aK2]->mName));
						aLDM.push_back("MEC-Cple_" + ToString(aK1) + "-" + ToString(aK2) + "/");
					}
                 }

            }
            std::cout << "Graphe : remain " << (mNbSom-aK1) << " to do\n";
        }
		aLDirMEC.Name() = aLDM;
        MakeFileXML(aRel,mDir+mOut);
		MakeFileXML(aLDirMEC,mDir+aFPairsDirMEC);

	}
}

double cGraphHomSat::CalcBtoH(const CameraRPC * aCam1, const CameraRPC * aCam2)
{
	Pt2dr aCentIm1(double(aCam1->SzBasicCapt3D().x)/2,double(aCam1->SzBasicCapt3D().y)/2);
    Pt3dr aTer 		  = aCam1->ImEtZ2Terrain(aCentIm1, aCam1->GetAltiSol());
    Pt3dr aCenter1Ter = aCam1->OpticalCenterOfPixel(aCentIm1);

	Pt2dr aTerBPrj 	  = aCam2->Ter2Capteur(aTer);
    Pt3dr aCenter2Ter = aCam2->OpticalCenterOfPixel(aTerBPrj);

    //H within the "epipolar plane"
    double aA = sqrt(std::pow(aCenter1Ter.x - aTer.x,2) + std::pow(aCenter1Ter.y - aTer.y,2) + std::pow(aCenter1Ter.z - aTer.z,2) );
    double aB = sqrt(std::pow(aCenter2Ter.x - aTer.x,2) + std::pow(aCenter2Ter.y - aTer.y,2) + std::pow(aCenter2Ter.z - aTer.z,2) );
    double aC = sqrt(std::pow(aCenter2Ter.x - aCenter1Ter.x,2)  + std::pow(aCenter2Ter.y - aCenter1Ter.y,2)  + std::pow(aCenter2Ter.z - aCenter1Ter.z,2)  );
    double aH = sqrt( aA*aB*(aA+aB+aC)*(aA+aB-aC) )/(aA+aB);

	return (aC/aH);

}

int GraphHomSat_main(int argc,char** argv)
{
	cGraphHomSat aGHS(argc,argv);

	return EXIT_SUCCESS;
}


cAppliCreateEpi::cAppliCreateEpi(int argc, char** argv)
{
	ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mFilePairs,"List of overlapping image pairs",eSAM_IsExistFile)
                     << EAMC(mOri,"Orientation directory, (NONE if rectification from tie-points)",eSAM_IsDir),
         LArgMain()
                     << mCAS3D.ArgBasic()
                     << mCAS3D.ArgEpip()
    );

	
	StdCorrecNameOrient(mOri,mCAS3D.mDir,true);

	cSauvegardeNamedRel aPairs = StdGetFromPCP(mCAS3D.mDir+mFilePairs,SauvegardeNamedRel);
	
	std::list<std::string> aLCom;

	for (auto itP : aPairs.Cple())
	{
		std::string aComTmp = MMBinFile(MM3DStr) + "CreateEpip " 
						      + itP.N1() + BLANK + itP.N2() + BLANK + mOri  
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



cAppliRecalRPC::cAppliRecalRPC(int argc, char** argv)
{
    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mCAS3D.mFilePairs,"List of overlapping image pairs",eSAM_IsExistFile)
		 			 << EAMC(mOri,"RPC original orientation (dir)",eSAM_IsDir), 
					 //mOri needed to recover the names of the Epi images

         LArgMain()
                     << mCAS3D.ArgBasic()
                     << mCAS3D.ArgRPC()
                     << mCAS3D.ArgFuse()
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

		

        std::string aComI1 = MMBinFile(MM3DStr) + "Convert2GenBundle "
                              + aNI1 + BLANK + aNAppuisI1 + BLANK
							  + mCAS3D.mOutRPC 
                              + mCAS3D.ComParamRPC_Basic();

        std::string aComI2 = MMBinFile(MM3DStr) + "Convert2GenBundle "
                              + aNI2 + BLANK + aNAppuisI2 + BLANK
							  + mCAS3D.mOutRPC
                              + mCAS3D.ComParamRPC_Basic();


		std::string aKeyGB = "NKS-Assoc-Im2GBOrient@-" +  mCAS3D.mOutRPC;

		std::string aNameGBI1 = mCAS3D.mICNM->Assoc1To1(aKeyGB,aNI1,true);
		std::string aNameGBI2 = mCAS3D.mICNM->Assoc1To1(aKeyGB,aNI2,true);

		std::string aComConv1 = MMBinFile(MM3DStr) + "Satelib RecalRPC " 
							  + aNameGBI1 + " OriOut=" + mCAS3D.mOutRPC;
		std::string aComConv2 = MMBinFile(MM3DStr) + "Satelib RecalRPC " 
							  + aNameGBI2 + " OriOut=" + mCAS3D.mOutRPC;



        aLCom.push_back(aComI1);
        aLCom.push_back(aComI2);
		aLCom.push_back(aComConv1);
		aLCom.push_back(aComConv2);


    }


    if (mCAS3D.mExe)
        cEl_GPAO::DoComInSerie(aLCom);
    else
    {
        for (auto iCmd : aLCom)
            std::cout << "SUBCOM_RPC= " << iCmd << "\n";
    }
}

int CPP_AppliRecalRPC_main(int argc,char ** argv)
{
    cAppliRecalRPC anAppRRPC(argc,argv);

    return EXIT_SUCCESS;
}



/*
 * If mOri is not initialized, then no convention and
 *        epipolar image names (EINs) are taken from mFilePairs 
 * If mOri is initialized, then MicMac convention and 
 *        EINs will be deduced from their original names in mFilesPairs + Ori
 * */
cAppliMM1P::cAppliMM1P(int argc, char** argv)
{
	ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mFilePairs,"List of overlapping image pairs",eSAM_IsExistFile),
         LArgMain()  << EAM(mOri,"Ori",true,"RPC original orientation")
					 //mOri maye be needed to recover the names of the Epi images
                     << mCAS3D.ArgBasic()
                     << mCAS3D.ArgMM1P()
    );

	if (EAMIsInit(&mOri))
		StdCorrecNameOrient(mOri,mCAS3D.mDir,true);
    
	cSauvegardeNamedRel aPairs = StdGetFromPCP(mCAS3D.mDir+mFilePairs,SauvegardeNamedRel);
	cListOfName         aLDirMec = StdGetFromPCP(mCAS3D.mFPairsDirMEC,ListOfName);

	ELISE_ASSERT(int(aPairs.Cple().size())==(int)aLDirMec.Name().size(),"In TestLib SAT4GEO_MM1P the PairsDirMEC.xml must contain as many elements as there are the matching couples in Pairs.xml!")


    std::list<std::string> aLCom;

	auto aDir_it = aLDirMec.Name().begin();

    for (auto itP : aPairs.Cple())
    {
		std::string aNI1;
		std::string aNI2;

		if (EAMIsInit(&mOri))
		{
			aNI1 = mCAS3D.mICNM->NameImEpip(mOri,itP.N1(),itP.N2());
			aNI2 = mCAS3D.mICNM->NameImEpip(mOri,itP.N2(),itP.N1());
		}
		else
		{
			aNI1 = itP.N1();
			aNI2 = itP.N2();
		}

		std::string aComTmp;
		
		if (mCAS3D.mMMVII)
		{
			ELISE_fp::MkDirSvp((*aDir_it));	
			aComTmp = "MMVII DenseMatchEpipGen" + BLANK + mCAS3D.mMMVII_mode
                                                     + BLANK + aNI1 + BLANK + aNI2 
                                                     + BLANK + "Out=" + (*aDir_it++) + mCAS3D.mMMVII_ImName
                                                     + ((EAMIsInit(&mCAS3D.mMMVII_SzTile)) ? (BLANK + "SzTile=" + ToString(mCAS3D.mMMVII_SzTile)) : "") 
                                                     + ((EAMIsInit(&mCAS3D.mMMVII_NbProc)) ? (BLANK + "NbProc=" + ToString(mCAS3D.mMMVII_NbProc)) : ""); 
		}
		else
		{
			aComTmp = MMBinFile(MM3DStr) + "MMAI4Geo " + mCAS3D.mDir + BLANK
                              + aNI1 + BLANK + aNI2 + BLANK
							  + "DirMEC=" + (*aDir_it++)
                              + mCAS3D.ComParamMatch();
		}

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


cAppliFusion::cAppliFusion(int argc,char ** argv)
{
	ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mFilePairs,"List of overlapping image pairs",eSAM_IsExistFile),
         LArgMain()  << EAM(mOri,"Ori",true,"RPC original orientation")
                     //mOri may be needed to recover the names of the Epi images
                     << EAM(mCAS3D.mMMVII,"MMVII",true,"True if mathing done in MMVII, Def=false")
                     << EAM(mCAS3D.mMMVII_ImName,"MMVII_ImName",true,"if MMVII==1, name of the depth map, Def=Px1_MMVII.tif")
                     << mCAS3D.ArgBasic()
                     << mCAS3D.ArgFuse()
    );

	if (EAMIsInit(&mOri))
    	StdCorrecNameOrient(mOri,mCAS3D.mDir,true);

	ELISE_ASSERT(EAMIsInit(&mCAS3D.mOutRPC),"In TestLib SAT4GEO_Fuse the \"OutRPC\" must be initializes. We need to know the geolocation of your input depth maps to move them to a common reference frame!")
}


std::string cAppliFusion::PxZName(const std::string & aInPx)
{
	return StdPrefix(aInPx) + AddFilePostFix() + ".tif";
}

std::string cAppliFusion::NuageZName(const std::string & aInNuageProf)
{
	return StdPrefix(aInNuageProf) + AddFilePostFix() + ".xml";
}

std::string  cAppliFusion::MaskZName(const std::string & aInMask)
{
	return StdPrefix(aInMask) + AddFilePostFix() + ".tif";
}

std::string cAppliFusion::AddFilePostFix()
{
	return "_FIm1ZTerr";
}

/*
 * If mOri is not initialized, then no convention and
 *        epipolar image names (EINs) are taken from mFilePairs
 * If mOri is initialized, then MicMac convention and
 *        EINs will be deduced from their original names in mFilesPairs + Ori
 * */
void cAppliFusion::DoAll()
{
	/* List of pairs */
	cSauvegardeNamedRel aPairs = StdGetFromPCP(mCAS3D.mDir+mFilePairs,SauvegardeNamedRel);

	/* List of MEC-Dirs*/
    cListOfName         aLDirMec = StdGetFromPCP(mCAS3D.mFPairsDirMEC,ListOfName);
    ELISE_ASSERT(int(aPairs.Cple().size())==(int)aLDirMec.Name().size(),"In TestLib SAT4GEO_MM1P the PairsDirMEC.xml must contain as many elements as there are the matching couples in Pairs.xml!")
    auto aDir_it = aLDirMec.Name().begin();


	/* Key to retrieve MEC2Im directory name */
	std::string aKeyMEC2Im = "Key-Assoc-MEC-Dir";


	/* Create xml file list with the concerned epipolar images */	
	cListOfName 				aLON;
	std::map<std::string,int>	aMEp;//serves to save uniquely all epip images
	std::list<std::pair<std::string,std::string>>  aLP;

	int aCpt=0;
    for (auto itP : aPairs.Cple())
    {
        std::string aNI1;
        std::string aNI2;
	
	 	if (EAMIsInit(&mOri))
		{	
			aNI1 = mCAS3D.mICNM->NameImEpip(mOri,itP.N1(),itP.N2());
        	aNI2 = mCAS3D.mICNM->NameImEpip(mOri,itP.N2(),itP.N1());
		}
		else
		{
			aNI1 = itP.N1();
			aNI2 = itP.N2();
		}

		aLP.push_back(std::make_pair(aNI1,aNI2));
		//aLP.push_back(std::make_pair(aNI2,aNI1));


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
	if (mCAS3D.mExe)
    	MakeFileXML(aLON,mCAS3D.mNameEpiLOF);


	/* Define the global frame of the reconstruction */
	std::string aCom = MMBinFile(MM3DStr) + "Malt UrbanMNE " 
			             + "NKS-Set-OfFile@" + mCAS3D.mNameEpiLOF + BLANK 
						 + mCAS3D.mOutRPC + " DoMEC=0";

	if (EAMIsInit(&mCAS3D.mEZA))
		aCom += " EZA=" + ToString(mCAS3D.mEZA);
	
    if (mCAS3D.mExe)
		if ((int)aLP.size()>1)
			System(aCom);
		else
			std::cout << "TestLib SAT4GEO_Fuse, there is only 1 image pair, I'm not defining the global frame.";
    else
    {
        std::cout << "SUBCOM1= " << aCom << "\n";
    }
	


	/* Transform surfaces geometries 
	 * from eGeomPxBiDim to  eGeomMNTFaisceauIm1ZTerrain_Px1D */
	std::list<std::string> aLCTG;

	std::string aNuageInName = "MMLastNuage.xml";

	for (auto itP : aLP)
    {
		//std::string aMECDir1to2 = mCAS3D.mICNM->Assoc1To2(aKeyMEC2Im,itP.first,itP.second,true);
		//std::string aMECDir2to1 = mCAS3D.mICNM->Assoc1To2(aKeyMEC2Im,itP.second,itP.first,true);
		std::string aMECBasic = (*aDir_it++); 
	

		//collect cmd to do conversion in parallel
		std::string aCTG1to2 = MMBinFile(MM3DStr) + "TestLib TransGeom "
				         + mCAS3D.mDir + " " 
						 + itP.first + " "
						 + itP.second + " " 
						 + mCAS3D.mOutRPC + " "
						 + aMECBasic+aNuageInName + " "
						 + AddFilePostFix() + " " 
						 + "NbP=" + ToString(mCAS3D.mNbProc) + " "
						 + "Exe=" + ToString(mCAS3D.mExe) + " " 
                         + ((mCAS3D.mMMVII) ? ("MMVII=" + ToString(mCAS3D.mMMVII) + " ") : "") 
                         + ((EAMIsInit(&mCAS3D.mMMVII_ImName)) ? ("MMVII_ImName=" + mCAS3D.mMMVII_ImName) : "");

		/*std::string aCTG2to1 = MMBinFile(MM3DStr) + "TestLib TransGeom "
				         + mCAS3D.mDir + " " 
						 + itP.second + " " 
						 + itP.first + " "
						 + mCAS3D.mOutRPC + " "
						 + aMECDir2to1+aNuageInName + " "
						 + AddFilePostFix() + " "
						 + "Exe=" + ToString(mCAS3D.mExe);*/



		aLCTG.push_back(aCTG1to2);
		//aLCTG.push_back(aCTG2to1);

	}

    if (mCAS3D.mExe)
		if ((int)aLP.size()>1)
			cEl_GPAO::DoComInSerie(aLCTG);
		else
			std::cout << "TestLib SAT4GEO_Fuse, there is only 1 image pair. I'm not transforming depths to Z.";
    else
    {
        for (auto iCmd : aLCTG)
            std::cout << "SUBCOM2= " << iCmd << "\n";
    }



	/* Transform individual surfaces to global frame */
	std::list<std::string> aLCom;

	std::string aNuageOutName = "MMLastNuage.xml";
	std::string aPref = "DSM_Pair";
	if (mCAS3D.mExe)
		if ((int)aLP.size()>1)
			ELISE_fp::MkDirSvp(mCAS3D.mOutSMDM);	
	aCpt=0;

	// reset the iterator to MECDirs
    aDir_it = aLDirMec.Name().begin();

	for (auto itP : aLP)
	{
		//std::string aMECDir1to2 = mCAS3D.mICNM->Assoc1To2(aKeyMEC2Im,itP.first,itP.second,true);
		//std::string aMECDir2to1 = mCAS3D.mICNM->Assoc1To2(aKeyMEC2Im,itP.second,itP.first,true);
		std::string aMECBasic = (*aDir_it++);
		
		std::string aComFuse1to2 = MMBinFile(MM3DStr) + "NuageBascule " 
				                 + aMECBasic + StdPrefix(aNuageInName) + AddFilePostFix() + ".xml" + " " 
							     + "MEC-Malt/" + aNuageOutName + " " 
							     + mCAS3D.mOutSMDM + aPref + ToString(aCpt) + ".xml"; 
		aCpt++;
		
		/*std::string aComFuse2to1 = MMBinFile(MM3DStr) + "NuageBascule " 
				                 + aMECDir2to1 + StdPrefix(aNuageInName) + AddFilePostFix() + ".xml" + " " 
							     + "MEC-Malt/" + aNuageOutName + " " 
							     + mCAS3D.mOutSMDM + aPref + ToString(aCpt) + ".xml"; 
		aCpt++;*/




		aLCom.push_back(aComFuse1to2);
		//aLCom.push_back(aComFuse2to1);
	}	

    if (mCAS3D.mExe)
		if ((int)aLP.size()>1)
			cEl_GPAO::DoComInSerie(aLCom);
		else
			std::cout << "TestLib SAT4GEO_Fuse, there is only 1 image pair. I'm not transforming from image to reference frame.";
    else
    {
        for (auto iCmd : aLCom)
            std::cout << "SUBCOM3= " << iCmd << "\n";
    }



	/* Fusion */
	std::string aComMerge = MMBinFile(MM3DStr) + "SMDM " 
			             + mCAS3D.mOutSMDM + aPref + ".*xml";


    if (mCAS3D.mExe)
		if ((int)aLP.size()>1)
			System(aComMerge);
		else
			std::cout << "TestLib SAT4GEO_Fuse, there is only 1 image pair, and there is nothing to fuse.";
    else
    {
        std::cout << "SUBCOM4= " << aComMerge << "\n";
    }

}

int CPP_AppliFusion_main(int argc,char ** argv)
{

	cAppliFusion aAppFus(argc,argv);
	aAppFus.DoAll();

	return EXIT_SUCCESS;
}


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
                    << mCAS3D.ArgBasic()
					<< mCAS3D.ArgEpip()
					<< mCAS3D.ArgRPC()
					<< mCAS3D.ArgMM1P()
					<< mCAS3D.ArgFuse()
   );

	StdCorrecNameOrient(mOri,mCAS3D.mDir,true);

	ELISE_ASSERT((EAMIsInit(&mCAS3D.mChSys)),"cAppliSat3DPipeline, you must indicate the ChSys parameter.");
}


void cAppliSat3DPipeline::StdCom(const std::string & aCom,const std::string & aPost)
{
	std::string  aFullCom = MMBinFile(MM3DStr) +  aCom + BLANK;
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
		StdCom("TestLib SAT4GEO_Pairs", 
			    mCAS3D.mDir + BLANK + QUOTE(mPat) + BLANK + mOri +  
				mCAS3D.ComParamPairs());
	}
	else
		std::cout << mCAS3D.mFilePairs << " used." << "\n";





	/****************************************************/
	/* 2- Rectify pairs of images to epipolar geometry  */
	/****************************************************/
	if (mCAS3D.mDoIm == true)
		StdCom("TestLib SAT4GEO_CreateEpip", 
				mCAS3D.mFilePairs + BLANK + mOri + BLANK + 
				mCAS3D.ComParamEpip());
	else
		std::cout << "No epipolar image creation." << "\n";



	/**************************************/
	/* 3- Recalculate the RPC orientation */
	/**************************************/
	StdCom("TestLib SAT4GEO_EpiRPC", mCAS3D.mFilePairs + BLANK + mOri
								    + mCAS3D.ComParamRPC());
	


	/******************************************************/
	/* 4- Perform dense image matching per pair of images */
	/******************************************************/
	StdCom("TestLib SAT4GEO_MM1P", 
			mCAS3D.mFilePairs + BLANK + "Ori=" + mOri
                              //+ ((mCAS3D.mMMVII) ? (BLANK + "MMVII=" + ToString(mCAS3D.mMMVII)) : "")
                              //+ ((mCAS3D.mMMVII) ? (BLANK + "MMVII_mode=" + mCAS3D.mMMVII_mode) : "")
                              //+ ((EAMIsInit(&mCAS3D.mMMVII_ImName)) ? (BLANK + "MMVII_ImName=" + mCAS3D.mMMVII_ImName) : "")
                              //+ ((mCAS3D.mMMVII) ? (BLANK + "MMVII_SzTile=" + ToString(mCAS3D.mMMVII_SzTile)) : "")
			      + mCAS3D.ComParamMatch() 
			      + BLANK + "PairsDirMEC=" + mCAS3D.mFPairsDirMEC);



	/**************************************************************************/
	/* 5- Transform the per-pair reconstructions to a commont reference frame 
	 *    and do the 3D fusion */
	/**************************************************************************/

	StdCom("TestLib SAT4GEO_Fuse", mCAS3D.mFilePairs + BLANK + "Ori=" + mOri 
		  				           + mCAS3D.ComParamFuse()
								   + BLANK + "PairsDirMEC=" + mCAS3D.mFPairsDirMEC
                                   + ((mCAS3D.mMMVII) ? (BLANK + "MMVII=" + ToString(mCAS3D.mMMVII)) : "")
                                   + ((EAMIsInit(&mCAS3D.mMMVII_ImName)) ? (BLANK + "MMVII_ImName=" + mCAS3D.mMMVII_ImName) : ""));

}



/*******************************************/
/********* CPP_TransformGeom    ************/
/*******************************************/
int CPP_TransformGeom_main(int argc, char ** argv)
{

	std::string aDir;
	std::string aOri;
	std::string aNuageName;
	std::string aPostFix;
	std::string aNuageNameOut;
	std::string aIm1;
	std::string aIm2;
	std::string aPx1NameOut;
	std::string aPx1MasqName;
	std::string aPx1MasqNameOut;
	bool InParal = true;
	bool CalleByP = false;
	int aSzDecoup = 2000;
	int aMaxNbProc = 8;
	Box2di aBoxOut;
	bool aExe=true;
    bool aMMVII=false;
    std::string aMMVII_ImProfName = "Px1.tif";

	ElInitArgMain
   	(
        argc,argv,
        LArgMain()  << EAMC(aDir,"Current directory")
					<< EAMC(aIm1,"First (left) image")
					<< EAMC(aIm2,"Second (right) image")
                    << EAMC(aOri,"Orientation directory",eSAM_IsDir)
	   				<< EAMC(aNuageName,"XML NuageImProf file")
					<< EAMC(aPostFix,"New file PostFix"),
        LArgMain()  << EAM(InParal,"InParal",true,"Compute in parallel (Def=true)")
					<< EAM(aSzDecoup,"SzDec",true,"Max size of the tile for parallel proc (Def=2000)")
					<< EAM(aMaxNbProc,"NbP",true,"Max nb of parallel processes (Def=8)")
                    << EAM(CalleByP,"CalleByP",true,"Internal Use", eSAM_InternalUse)
                    << EAM(aBoxOut,"BoxOut",true,"Internal Use", eSAM_InternalUse)
                    << EAM(aMMVII,"MMVII",true,"True if matching done with MMVII, Def=false")
                    << EAM(aMMVII_ImProfName,"MMVII_ImName",true,"If MMVII=1, name of depth image, Def=Px1.tif")
					<< EAM(aExe,"Exe",true,"Execute, def=true")

   	);



	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    StdCorrecNameOrient(aOri,aDir,true);

    // Directory of the depth map
    std::string aDirMEC = DirOfFile(aNuageName);


    //if MMVII create the NuageLast.xml file (if it does not exist)
    if (aMMVII && (!ELISE_fp::exist_file(aNuageName)))
    {
        //read the depth map to read the image size 
		Tiff_Im     aImProfMMVII( (aDirMEC + aMMVII_ImProfName).c_str());

        //fill the XML_ParamNuage3DMaille structure 
        cXML_ParamNuage3DMaille aNMVII;
        aNMVII.SsResolRef() = 1;
        aNMVII.NbPixel() = aImProfMMVII.sz();

        cPN3M_Nuage aPN;
        cImage_Profondeur  aImP;
        aImP.Image() = aMMVII_ImProfName;

        std::string aMasqPre; 
        std::vector<std::string> aMasqPosts;
        SplitInNArroundCar(aIm1,'.',aMasqPre,aMasqPosts);
        aImP.Masq() = aMasqPre ;
        for (int el=0; el<int(aMasqPosts.size()-1); el++)
            aImP.Masq() += "." + aMasqPosts[el];
       
        aImP.Masq() += "_Masq.tif";
          
        aImP.OrigineAlti() = 0;
        aImP.ResolutionAlti() = 1;
        aImP.GeomRestit() = eGeomPxBiDim;
        
        aPN.Image_Profondeur() = aImP;
        aNMVII.PN3M_Nuage() = aPN;


        cOrientationConique aOc;
        cAffinitePlane aAP;
        aAP.I00() = Pt2dr(0,0);
        aAP.V10() = Pt2dr(1,0);
        aAP.V01() = Pt2dr(0,1);
        aOc.OrIntImaM2C() = aAP;

        aOc.ZoneUtileInPixel() = true;
        aOc.TypeProj() = eProjOrthographique;

        cOrientationExterneRigide aEOR;
        aEOR.Centre() = Pt3dr(0,0,0);
        cRotationVect aRotVec;
        cTypeCodageMatr aRotCode;
        aRotCode.L1() = Pt3dr(1,0,0);
        aRotCode.L2() = Pt3dr(0,1,0);
        aRotCode.L3() = Pt3dr(0,0,1);
        aRotVec.CodageMatr() = aRotCode;

        aEOR.ParamRotation() = aRotVec;

        aOc.Externe() = aEOR;

        cConvOri aCOri;
        aCOri.KnownConv() = eConvApero_DistM2C; 
        aOc.ConvOri() = aCOri;

        aNMVII.Orientation() = aOc;


        aNMVII.RatioResolAltiPlani() = 1;

        cPM3D_ParamSpecifs aParSpec;
        cModeFaisceauxImage aMFI;
        aMFI.DirFaisceaux() = Pt3dr(0,0,1);
        aMFI.ZIsInverse() = false;
        aMFI.IsSpherik() = false;
        aParSpec.ModeFaisceauxImage() = aMFI;
        aNMVII.PM3D_ParamSpecifs() = aParSpec;
        
        MakeFileXML(aNMVII,aNuageName);
        
    }



    /* Read the depth map */
    cXML_ParamNuage3DMaille  aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
                                        (
                                            aNuageName,
                                            StdGetFileXMLSpec("SuperposImage.xml"),
                                            "XML_ParamNuage3DMaille",
                                            "XML_ParamNuage3DMaille"
                                        );
	cImage_Profondeur aImProfPx   = aNuageIn.Image_Profondeur().Val();
	Pt2di 		aSz 			  = aNuageIn.NbPixel();
	ElAffin2D   aAfM2CGlob 		  = Xml2EL(aNuageIn.Orientation().OrIntImaM2C()); 
	//double aResAlti = aNuageIn.ResolutionAlti();

	/* Names of new files */
    aNuageNameOut   = StdPrefix(aNuageName) + aPostFix + ".xml";
    aPx1NameOut     = aDirMEC + StdPrefix(aImProfPx.Image()) + aPostFix + ".tif";
	aPx1MasqName    = aDirMEC + aImProfPx.Masq();
	aPx1MasqNameOut = aDirMEC + StdPrefix(aImProfPx.Masq()) + aPostFix + ".tif";


	/* Create new depth map if needed */
	bool isModified;
    Tiff_Im aImProfZTif = Tiff_Im::CreateIfNeeded(
                            isModified,
                            aPx1NameOut.c_str(),
                            aSz,
                            GenIm::real4,
                            Tiff_Im::No_Compr,
                            Tiff_Im::BlackIsZero
                         );




	Tiff_Im  aImMasqZTif =  Tiff_Im::CreateIfNeeded(
                            isModified,
                            aPx1MasqNameOut.c_str(),
                            aSz,
                            GenIm::bits1_msbf,
                            Tiff_Im::No_Compr,
                            Tiff_Im::BlackIsZero
                      );



	if (CalleByP)
	{
		std::cout << "TG Box p0=[" << aBoxOut._p0 << "], p1=[" << aBoxOut._p1 << "] \n";

		Pt2di aSzOut = aBoxOut.sz();
    
		std::string aImName = aDirMEC + aImProfPx.Image();
		
		Tiff_Im     aImProfPxTif(aImName.c_str());

    	/* Read the input depth map */
		TIm2D<float,double> aTImProfPx(aSzOut);
		ELISE_COPY
        (
               aTImProfPx.all_pts(),
               trans(aImProfPxTif.in(),aBoxOut._p0),
               aTImProfPx.out()
        );
  
    

    
		/* Read the mask */
		Im2D_Bits<1> aMasq(aSzOut.x,aSzOut.y,1);
		if (EAMIsInit(&aPx1MasqName))
		{
			ELISE_COPY
			(			aMasq.all_pts(),
						trans(Tiff_Im(aPx1MasqName.c_str()).in(),aBoxOut._p0),
					   	aMasq.out()
			);
		}
		TIm2DBits<1> aTMasq(aMasq);
    

    
		/* Create the depth map & mask to which we will write */
		//TIm2D<float,double> aTImProfZ(aSz);
  		//Im2D_Bits<1>        aTMasqZ(aSz.x,aSz.y,0);
		TIm2D<float,double> aTImProfZ(aSzOut);
  		Im2D_Bits<1>        aTMasqZ(aSzOut.x,aSzOut.y,0);


		/* Read cameras */
		cBasicGeomCap3D * aCamI1 = aICNM->StdCamGenerikOfNames(aOri,aIm1);	
		cBasicGeomCap3D * aCamI2 = aICNM->StdCamGenerikOfNames(aOri,aIm2);	

		/* Plani resolution */
		ElAffin2D   aAfM2CCur     = ElAffin2D::trans(-Pt2dr(aBoxOut._p0)) * aAfM2CGlob ;
		ElAffin2D   aAfC2MCur     = aAfM2CCur.inv();
		double aResolPlaniReel    = (euclid(aAfC2MCur.I10()) + euclid(aAfC2MCur.I01()))/2.0;
		double aResolPlaniEquiAlt = aResolPlaniReel * aNuageIn.RatioResolAltiPlani().Val();

		/* Triangulate and fill in aTImProfZ */
		Pt2di aPt1;
		for (aPt1.x=0; aPt1.x<aSzOut.x; aPt1.x++)
		{
			for (aPt1.y=0; aPt1.y<aSzOut.y; aPt1.y++)
			{
				Pt2di aPt1InFul(aPt1.x + aBoxOut._p0.x,aPt1.y + aBoxOut._p0.y);


				if (aTMasq.get(aPt1))
				{

					Pt2dr aPt2InFul(aPt1InFul.x + aTImProfPx.get(aPt1), aPt1InFul.y);

                    
					ElSeg3D aSeg1 = aCamI1->Capteur2RayTer(Pt2dr(aPt1InFul.x,aPt1InFul.y)*aResolPlaniReel); 
					ElSeg3D aSeg2 = aCamI2->Capteur2RayTer(aPt2InFul*aResolPlaniReel); 
                    
					std::vector<ElSeg3D> aVSeg = {aSeg1,aSeg2};
					
					Pt3dr aRes =  ElSeg3D::L2InterFaisceaux(0,aVSeg,0);

					//aTImProfZ.oset(aPt1InFul,aRes.z/aResolPlaniEquiAlt);
					aTImProfZ.oset(aPt1,aRes.z/aResolPlaniEquiAlt);
				
					//aTMasqZ.set(aPt1InFul.x,aPt1InFul.y,1);
					aTMasqZ.set(aPt1.x,aPt1.y,1);
				}
			}
		}

    

		/* Write box to new depth map */
        /*ELISE_COPY
        (
            rectangle(aBoxOut._p0,aBoxOut._p1), 
            //trans(aTImProfZ.in(),aBoxOut._p0),
            aTImProfZ.in(),
            aImProfZTif.out()
        );*/
        ELISE_COPY
        (
            rectangle(aBoxOut._p0,aBoxOut._p1), 
            trans(aTImProfZ.in(),-aBoxOut._p0),
            aImProfZTif.out()
        );
   	
		/* Write box to the new mask */
		/*ELISE_COPY
		(
			rectangle(aBoxOut._p0,aBoxOut._p1),
			aTMasqZ.in(),
			aImMasqZTif.out()	
		);*/
		ELISE_COPY
		(
			rectangle(aBoxOut._p0,aBoxOut._p1),
			trans(aTMasqZ.in(),-aBoxOut._p0),
			aImMasqZTif.out()	
		);




		/*Disc_Pal  Pdisc = Disc_Pal::P8COL();
		Gray_Pal  Pgr (30);
		Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
       	RGB_Pal   Prgb  (5,5,5);
		Elise_Set_Of_Palette SOP
        (    NewLElPal(Pdisc)
            + Elise_Palette(Pgr)
            + Elise_Palette(Prgb)
            + Elise_Palette(Pcirc)  );

		
        Video_Display Ecr((char *) NULL);
        Ecr.load(SOP);

		Pt2di aSzWin(-110+aSzOut.x,-50+aSzOut.y);
	    Video_Win   W  (Ecr,SOP,Pt2di(50,50),aSzOut);
		ELISE_COPY
        (
            rectangle(Pt2di(110,50),aSzWin),
            trans(aTImProfZ.in(),aBoxOut._p0),
            W.out(Pgr)
        );*/

		//getchar();
   		

	}
	else
	{

	    cDecoupageInterv2D aDecoup  = cDecoupageInterv2D::SimpleDec(aSz,aSzDecoup,0);

		
		// To avoid occupying too many cluster nodes
		int aReduCPU=1;
		while (aDecoup.NbInterv()>(aMaxNbProc))
		{
			aSzDecoup = ceil(sqrt(double(aSz.x)*double(aSz.y)/(aMaxNbProc-(aReduCPU++))));

			aDecoup  = cDecoupageInterv2D::SimpleDec(aSz,aSzDecoup,0);

		}



		std::list<std::string> aLCom;

		std::string aComBase = MMBinFile(MM3DStr) + "TestLib TransGeom "
			                 + aDir + BLANK 
							 + aIm1 + BLANK
                         	 + aIm2 + BLANK
                         	 + aOri + BLANK
                         	 + aNuageName + BLANK
							 + aPostFix + BLANK
                        	 //+ aNuageNameOut + BLANK
                         	 //+ aPx1NameOut + BLANK
							 //+ "Mask=" + aPx1MasqName + BLANK
							 //+ "MaskOut=" + aPx1MasqNameOut + BLANK 
							 + "InParal=" + ToString(InParal);





		for (int aK=0; aK<aDecoup.NbInterv(); aK++)
		{
			std::string aCom = aComBase + " CalleByP=true " 
					         + "BoxOut=" + ToString(aDecoup.KthIntervIn(aK));
			aLCom.push_back(aCom);
		}



		if (aExe)
        	cEl_GPAO::DoComInParal(aLCom);
	    else
    	{
        	for (auto iCmd : aLCom)
            	std::cout << "SUBCOM_TG= " << iCmd << "\n";
    	}
	
		if (aExe)
		{
			/* Update aNuageIn */
            aImProfPx.Image()           = NameWithoutDir(aPx1NameOut);
			aImProfPx.Masq()            = NameWithoutDir(aPx1MasqNameOut);
            aImProfPx.GeomRestit()      = eGeomMNTFaisceauIm1ZTerrain_Px1D;
            aNuageIn.Image_Profondeur() = aImProfPx;
            aNuageIn.NameOri() = aICNM->StdNameCamGenOfNames(aOri,aIm1);
            
            MakeFileXML(aNuageIn,aNuageNameOut);
		}
	
	}

	return EXIT_SUCCESS;
}


//TODO:
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
