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

/*



Bascule ".*.jpg" RadialExtended B2   ImRep=00000002.jpg  P1Rep=[463,1406] P2Rep=[610,284] Teta=90
Bascule ".*.jpg" RadialExtended R2.xml   ImRep=00000002.jpg  P1Rep=[463,1406] P2Rep=[610,284] Teta=45


Bascule ".*.jpg" RadialExtended B2 MesureIm=OutAligned-Im.xml Teta=90
Tarama ".*.jpg" B2

Bascule ".*.jpg" RadialExtended R2.xml MesureIm=OutAligned.xml Teta=180
Tarama ".*.jpg" RadialExtended Repere=R2.xml

Bascule ".*.jpg" RadialExtended R3.xml MesureIm=OutAligned.xml Teta=180

*/

#define DEF_OFSET -12349876

//  ===============================================================================================
int Bascule_main(int argc,char ** argv)
{
    NoInit = "NoP1P2";
    aNoPt = Pt2dr(123456,-8765432);

    MemoArg(argc,argv);
    std::string  aDir,aPat,aFullDir;
    int ExpTxt=0;


    std::string AeroOut;
    std::string AeroIn;
    std::string PostPlan="_Masq";
    std::vector<std::string> ImPl;
    bool AllPlanExist = false;
    bool UserKeyPlan = false;

    Pt2dr aP1Rep = aNoPt;
    Pt2dr aP2Rep = aNoPt;
    Pt2dr aAxeRep(1,0);
    double aNoVal = 1.234e30;
    double  aTetaRep = aNoVal;
    std::string  aImRep=NoInit;

    std::string  aFileMesureIm = "";

    bool  OrthoCyl = false;

    std::string FileOrthoCyl="EmptyXML.xml";

    double DistFE;
    Pt3dr  Normal;
    Pt3dr  SNormal;
    double aLimBsH;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)" , eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(AeroOut,"Out: orientation or local repair (if postfixed by \"xml\")", eSAM_IsExistFile),
    LArgMain()
                    << EAM(ImPl,"ImPl",true)
                    << EAM(ExpTxt,"ExpTxt",true)
                    << EAM(PostPlan,"PostPlan",true)
                    << EAM(AllPlanExist,"AllPl",true)
                    << EAM(UserKeyPlan,"UserKeyPlan",true)
                    << EAM(aP1Rep,"P1Rep",true,"P1Rep", eSAM_NoInit)
                    << EAM(aP2Rep,"P2Rep",true,"P2Rep", eSAM_NoInit)
                    << EAM(aAxeRep,"AxeRep",true)
                    << EAM(aImRep,"ImRep",true)
                    << EAM(aTetaRep,"Teta",true, "Angle (degree)")
                    << EAM(aFileMesureIm,"MesureIm",true,"Image measure file", eSAM_IsExistFile)
                    << EAM(OrthoCyl,"OrthoCyl",true,"Generate a local repair of orthocyl mode")
                    << EAM(DistFE,"DistFS",true,"Distance between to fix scale, if not given no scaling", eSAM_NoInit)
                    << EAM(Normal,"Norm",true,"Target normal for the plane", eSAM_NoInit)
                    << EAM(SNormal,"SNorm",true,"\"Symbolic Normal\" (must be X, Y or Z)", eSAM_NoInit)
                    << EAM(aLimBsH,"LimBsH",true,"Limit ratio base to high (Def=1e-2)", eSAM_NoInit)

    );

    if (!MMVisualMode)
    {
        if (aTetaRep != aNoVal)
        {
           aAxeRep = Pt2dr::FromPolar(1.0,aTetaRep * (PI/180.0));
        }

        #if (ELISE_windows)
            replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
        #endif
        SplitDirAndFile(aDir,aPat,aFullDir);
		StdCorrecNameOrient(AeroIn, aDir);

        bool ModeRepere = IsPostfixed(AeroOut) && (StdPostfix(AeroOut) == "xml");

        std::string FileOriInPlan  = "EmptyXML.xml";
        std::string FilePlan="Bascule-Plan.xml";
        std::string FileExport = "Export-Orient-Std.xml";

        std::string FileFixScale  = "EmptyXML.xml";

        std::string aPatternPlan;

        std::string aPrefOut="-";

        std::string aStrRep;
        if (! ImPl.empty())
        {
             aPatternPlan = ".*(";
             aPatternPlan = aPatternPlan + ImPl[0];
             for (int aK=1; aK<int(ImPl.size()) ; aK++)
             {
                aPatternPlan = aPatternPlan + "|" + ImPl[aK];
             }
             aPatternPlan =  aPatternPlan+").*";
        }
        else if (UserKeyPlan)
        {
             aPatternPlan  = "Loc-Key-SetImagePlan";  // Plan Utilisateur
        }
        else
        {
             aPatternPlan  =  "NKS-Set-OfExistingPlan@"+ aPat + "@"+  PostPlan ;  // Plan Utilisateur
        }


        ELISE_ASSERT
        (
                ((aP1Rep !=aNoPt) == (aP2Rep !=aNoPt))
             && ((aP1Rep !=aNoPt) == (aImRep !=NoInit))
           ,"Incoh (Pt) in rep cart"
        );

        bool OrientInPlan = false;

        if (aP1Rep !=aNoPt)
        {
             OrientInPlan = true;
        }

        if (aFileMesureIm!="")
        {
            ELISE_ASSERT
            (
                     IsPostfixed(aFileMesureIm)
                 &&( StdPostfix(aFileMesureIm)== "xml"),
                 "Image Measure File has no xml extension"
            );
            aImRep = aFileMesureIm;

           aStrRep = aStrRep + " +FileMesure="+aFileMesureIm;
        }

        if (OrthoCyl)
        {
             ELISE_ASSERT(ModeRepere,"Bad export file for OrthoCyl");
        }

        if (true)
        {
            aStrRep =       aStrRep
                          +  " +X1RC=" + ToString(aP1Rep.x)
                          +  " +Y1RC=" + ToString(aP1Rep.y)
                          +  " +X2RC=" + ToString(aP2Rep.x)
                          +  " +Y2RC=" + ToString(aP2Rep.y)
                          +  " +XAxeRC=" + ToString(aAxeRep.x)
                          +  " +YAxeRC=" + ToString(aAxeRep.y)
                          +  " +ImP1P2="+aImRep
                       ;


    // aFileMesureIm
            if (ModeRepere)
            {
                 FilePlan = "EmptyXML.xml";
                 FileExport  =   "Export-Repere-Cart.xml";
                 if (OrthoCyl)
                    FileOrthoCyl = "Export-OrthoCyl.xml" ;
                 aPrefOut= "";
            }
            else
            {
                if (OrientInPlan)
                {
                    FileOriInPlan  = "Bascule-InternePlan.xml";
                }

                if (aFileMesureIm!="")
                {
                    FileOriInPlan  = "Bascule-InternePlan-ByFile.xml";
                }
            }
        }

       std::string aStrFixS = "";
       if (EAMIsInit(&DistFE))
       {
           FileFixScale = "Bascule-Fixe-Scale.xml";
           aStrFixS =  " +DistFS=" + ToString(DistFE);
       }

    #ifdef ELISE_windows
        std::string aCom =   MMDir() + std::string("bin" ELISE_STR_DIR "Apero ")
                           + MMDir() + std::string("include" ELISE_STR_DIR "XML_MicMac" ELISE_STR_DIR "Apero-Bascule.xml ")
    #else
        std::string aCom =   MMDir() + std::string("bin" ELISE_STR_DIR "Apero ")
                           + MMDir() + std::string("include" ELISE_STR_DIR "XML_MicMac" ELISE_STR_DIR "Apero-Bascule.xml ")
    #endif
                           + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                           + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                           + std::string(" +AeroOut=") + aPrefOut + AeroOut
                           + std::string(" +Ext=") + (ExpTxt?"txt":"dat")
                           + std::string(" +AeroIn=-") + AeroIn
                           + std::string(" +FileBasculePlan=") + FilePlan
                           + std::string(" +FileOriInPlan=") + FileOriInPlan
                           + std::string(" +FileFixScale=") + FileFixScale  + aStrFixS
                           + std::string(" +FileExport=") + FileExport
                           + std::string(" +FileOrthoCyl=") + FileOrthoCyl
                           + std::string(" +PostMasq=") + PostPlan
                           + aStrRep
                        ;


       if (EAMIsInit(&aLimBsH))
           aCom = aCom + std::string(" +LimBsH=") + ToString(aLimBsH);


       if (aPatternPlan!="")
       {
            aCom = aCom + std::string(" +PatternPlan=") + QUOTE(aPatternPlan);
       }



       std::cout << "Com = " << aCom << "\n";
       int aRes = system_call(aCom.c_str());


       return aRes;
   }
   else
       return EXIT_SUCCESS;
}


//  ===============================================================================================
int BasculePtsInRepCam_main(int argc,char ** argv)
{
    std::string aNameCam,aNamePts;
    std::string aNamePts_Out;
    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aNameCam,"Name Camera" )
                     << EAMC(aNamePts,"Name GGP In"),
          LArgMain()
                     << EAM(aNamePts_Out,"Out", true )
    );

    if (! EAMIsInit(&aNamePts_Out))
    {
        aNamePts_Out =     DirOfFile(aNamePts)
                        + "Basc-" + StdPrefix(NameWithoutDir(aNameCam)) + "-"
                        +  NameWithoutDir(aNamePts);
    }

    CamStenope * aCamIn =  BasicCamOrientGenFromFile(aNameCam);
    ElRotation3D aR = aCamIn->Orient();
    cDicoAppuisFlottant aDAFIn =  StdGetFromPCP(aNamePts,DicoAppuisFlottant);
    for 
    (
         std::list<cOneAppuisDAF>::iterator itO=aDAFIn.OneAppuisDAF().begin();
         itO!=aDAFIn.OneAppuisDAF().end();
         itO++

    )
    {
        itO->Pt()  = aR.ImAff(itO->Pt() );
    }

    MakeFileXML(aDAFIn,aNamePts_Out);

    
    return EXIT_SUCCESS;
}


//  ===============================================================================================
class cAppliBasculeCamsInRepCam_main : public cAppliWithSetImage
{
    public :
          cAppliBasculeCamsInRepCam_main(int argc,char ** argv);
    private :
         std::string mOriFull;
         std::string mCamC;
         std::string mOriOut;
};

cAppliBasculeCamsInRepCam_main::cAppliBasculeCamsInRepCam_main(int argc,char ** argv) :
    cAppliWithSetImage(argc-1,argv+1,0)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mEASF.mFullName,"Full Name (Dir+Pattern)",eSAM_IsPatFile)
                    << EAMC(mOriFull,"Orientation", eSAM_IsExistDirOri)
                    << EAMC(mCamC,"Central camera")
                    << EAMC(mOriOut,"Output"),
        LArgMain()
   );

   tSomAWSI * aS0 = mDicIm[mCamC];
   // CamStenope * aCS0 = aS0->attr().mIma.mCam;  /// MtoC0
   
   // La bascule n'a pas d'interet sur push broom, donc on s'embete pas 
   ElRotation3D anOri0 = aS0->attr().mIma->CamSNN()->Orient();
   ELISE_ASSERT(aS0!=0,"cAppliBasculeCamsInRepCam_main No CamC");

   std::string aKey = "NKS-Assoc-Im2Orient@-" + mOriOut;

   // MtoC * MtoC0  
   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
       CamStenope * aCS = mVSoms[aK]->attr().mIma->CamSNN();  // MtoC
       aCS->SetOrientation(aCS->Orient() * anOri0.inv());
       cOrientationConique  anOC = aCS->StdExportCalibGlob();
       std::string aNameOut = mEASF.mICNM->Assoc1To1(aKey,mVSoms[aK]->attr().mIma->mNameIm,true);
       MakeFileXML(anOC,aNameOut);
   }
}

int BasculeCamsInRepCam_main(int argc,char ** argv)
{
    cAppliBasculeCamsInRepCam_main anAppli(argc,argv);
    return EXIT_SUCCESS;
}


//  ===============================================================================================
class cAppliEstimLA_main : public cAppliWithSetImage
{
	public :
		cAppliEstimLA_main(int argc,char ** argv);
	private :
		std::string mFullName;
		std::string mOri1;
		std::string mOri2;
		std::string mOriOut;
};

cAppliEstimLA_main::cAppliEstimLA_main(int argc,char ** argv) :
	cAppliWithSetImage(argc-1,argv+1,0)
{
	bool aExportResTxt=false;
	bool aExportXML=false;
	
	ElInitArgMain
	(
		argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pattern)",eSAM_IsPatFile)
                    << EAMC(mOri1,"Directory orientation of images", eSAM_IsExistDirOri)
                    << EAMC(mOri2,"Directory positions of GPS trajectory", eSAM_IsExistDirOri),
        LArgMain()  << EAM(mOriOut,"OriOut",true,"Output Ori Name of corrected mandatory Ori ; Def=OriName-CorrLA")
		    << EAM(aExportResTxt,"ResTxt",false,"Export residuals in a .txt file ; Def=false")
		    << EAM(aExportXML,"ExportXML",false,"Export corrected trajectory as GPS .xml file ; Def=false")
	);
	
	std::string aDir, aPat;
	
	SplitDirAndFile(aDir,aPat,mFullName);
	
    StdCorrecNameOrient(mOri1,aDir);
    StdCorrecNameOrient(mOri2,aDir);
    
    if(mOriOut == "")
    {
		mOriOut = mOri2 + "-CorrLA" ;
	}
	
	cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPat));
    
    std::vector<CamStenope *> vCSO1;
	std::vector<CamStenope *> vCSO2;
    for(unsigned aC=0; aC<aSetIm.size(); aC++)
    {
		CamStenope * aCam1 = CamOrientGenFromFile("Ori-" + mOri1 + "/" + "Orientation-" + aSetIm[aC] + ".xml", aICNM);
		vCSO1.push_back(aCam1);
				
		CamStenope * aCam2 = CamOrientGenFromFile("Ori-" + mOri2 + "/" + "Orientation-" + aSetIm[aC] + ".xml", aICNM);
		vCSO2.push_back(aCam2);
	}
	
	ELISE_ASSERT((vCSO1.size() ==  vCSO2.size()),"The size can't be different");
	
	//read Ori1 : Positions et Rotations
	//read Ori2 : Positions
	//construct the system : estimate LA
	L2SysSurResol aSys(3);
	double* aData = NULL;

	for (unsigned int aK=0 ; aK<vCSO1.size() ; aK++)
	{
       Pt3dr aC1 = vCSO1[aK]->PseudoOpticalCenter();
       //std::cout << "aC1 = " << aC1 << std::endl;
       ElMatrix<double>  aM1 = vCSO1[aK]->Orient().Mat();
       
       Pt3dr aC2 = vCSO2[aK]->PseudoOpticalCenter();
       //std::cout << "aC2 = " << aC2 << std::endl;
       
 
	   double coeffX[3] = {aM1(0,0),aM1(0,1),aM1(0,2)};
	   double coeffY[3] = {aM1(1,0),aM1(1,1),aM1(1,2)};
	   double coeffZ[3] = {aM1(2,0),aM1(2,1),aM1(2,2)};
				
	   double ResX = aC2.x - aC1.x;
	   double ResY = aC2.y - aC1.y;
	   double ResZ = aC2.z - aC1.z;
				
	   aSys.AddEquation(1.0, coeffX, ResX);
	   aSys.AddEquation(1.0, coeffY, ResY);
	   aSys.AddEquation(1.0, coeffZ, ResZ);
	   
	}

	bool solveOK = true;
    Im1D_REAL8 aResol1 = aSys.GSSR_Solve(&solveOK);
    aData = aResol1.data();
    
    if (solveOK != false)
    {
		cout<<"Estimed value : Lx Ly Lz = "<<aData[0]<<" "<<aData[1]<<" "<<aData[2]<<" "<<endl;
	}
	
	std::string aKey = "NKS-Assoc-Im2Orient@-" + mOriOut;
	
	//compute residuals & export them in a file
	std::vector<Pt3dr> aVRes;
	std::vector<Pt3dr> aVG2C;
	std::vector<double> aVRes3D;
	for(unsigned int aP=0; aP<vCSO1.size(); aP++)
	{
		
		Pt3dr aC1 = vCSO1[aP]->PseudoOpticalCenter();
		ElMatrix<double>  aM1 = vCSO1[aP]->Orient().Mat();
		Pt3dr aC2 = vCSO2[aP]->PseudoOpticalCenter();
		
		double ResX = aC2.x - aC1.x - aM1(0,0)*aData[0] - aM1(0,1)*aData[1] - aM1(0,2)*aData[2];
		double ResY = aC2.y - aC1.y - aM1(1,0)*aData[0] - aM1(1,1)*aData[1] - aM1(1,2)*aData[2];
		double ResZ = aC2.z - aC1.z - aM1(2,0)*aData[0] - aM1(2,1)*aData[1] - aM1(2,2)*aData[2];
		
		double aRes3D = sqrt(ResX*ResX + ResY*ResY + ResZ*ResZ);

		std::cout << "Res - Image " << aSetIm[aP] << " = [" << ResX << "," << ResY << "," << ResZ << "] ; Dist = " << aRes3D << std::endl;
		
		Pt3dr aRes(ResX,ResY,ResZ);
		aVRes.push_back(aRes);
		
		double aG2CX = aC2.x - aM1(0,0)*aData[0] - aM1(0,1)*aData[1] - aM1(0,2)*aData[2];
		double aG2CY = aC2.y - aM1(1,0)*aData[0] - aM1(1,1)*aData[1] - aM1(1,2)*aData[2];
		double aG2CZ = aC2.z - aM1(2,0)*aData[0] - aM1(2,1)*aData[1] - aM1(2,2)*aData[2];
		
		Pt3dr aG2C(aG2CX,aG2CY,aG2CZ);
		aVRes3D.push_back(aRes3D);
		aVG2C.push_back(aG2C);
		
		//export Ori Gps corrected from LA
		CamStenope * aCam =  aICNM->StdCamStenOfNames(aSetIm[aP],mOri2);
		ElRotation3D anOriC(-aG2C,0,0,0);
		aCam->SetOrientation(anOriC);
		cOrientationConique  anOC = aCam->StdExportCalibGlob();
		std::string aNameOut = aICNM->Assoc1To1(aKey,aSetIm[aP],true);
		MakeFileXML(anOC,aNameOut);
		
	}
	
	//generate a new GPS trajectory corrected from LA (in ori format and in .xml format)
	//now GPS positions are given at optical camera center
	if(aExportResTxt)
	{
		std::string aOutTxt = "Res_Estime_LA.txt";
		FILE * aFP = FopenNN(aOutTxt,"w","EstimLA_main");
		cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOutTxt);
		for (unsigned int aK=0 ; aK<aVRes.size() ; aK++)
		{
			fprintf(aFP,"%lf %lf %lf %lf \n",aVRes.at(aK).x,aVRes.at(aK).y,aVRes.at(aK).z,aVRes3D.at(aK));
		}
		ElFclose(aFP);
	}
		
	if(aExportXML)
	{
		std::string aXmlOut = "Gps_Traj_Corr_LA.xml";
		cDicoGpsFlottant  aDico;
		for (unsigned int aKP=0; aKP<aVG2C.size(); aKP++)
		{
			cOneGpsDGF aOAD;
			aOAD.Pt() = aVG2C[aKP];
			aOAD.NamePt() = "";
			Pt3dr aInc(1,1,1);
			aOAD.Incertitude() = aInc;
			aOAD.TagPt() = 0;
			aOAD.TimePt() = aKP;

			aDico.OneGpsDGF().push_back(aOAD);
		}

		MakeFileXML(aDico,aXmlOut);
	}
}

int EstimLA_main(int argc,char ** argv)
{
	cAppliEstimLA_main anAppli(argc,argv);
	return EXIT_SUCCESS;
}


//  ===============================================================================================
class cAppliCorrecCamFromLA_main : public cAppliWithSetImage
{
    public :
          cAppliCorrecCamFromLA_main(int argc,char ** argv);
    private :
         std::string mDir;
         std::string mPat;
         std::string mOriFull;
         Pt3dr mLA;
         std::string mOriOut;
         std::string mOri2;
         std::string mOri2Out;
         cInterfChantierNameManipulateur * mICNM2;
};

cAppliCorrecCamFromLA_main::cAppliCorrecCamFromLA_main(int argc,char ** argv) :
    cAppliWithSetImage(argc-1,argv+1,0)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mEASF.mFullName,"Full Name (Dir+Pattern)",eSAM_IsPatFile)
                    << EAMC(mOriFull,"Directory orientation", eSAM_IsExistDirOri)
                    << EAMC(mLA,"Lever-Arm value"),
        LArgMain()  << EAM(mOriOut,"OriOut",true,"Output Ori Name of corrected mandatory Ori ; Def=OriName-CorrLA")
					<< EAM(mOri2,"Ori2",true,"Ori2 directory to apply LA correction to centers")
					<< EAM(mOri2Out,"Ori2Out",true,"Output Ori Name of corrected Ori2 ; Def=Ori2Name-CorrLA")  
   );

   SplitDirAndFile(mDir,mPat,mEASF.mFullName);
   
   StdCorrecNameOrient(mOriFull,mDir);
   
   if(mOriOut == "")
   {
		mOriOut = mOriFull + "-CorrLA" ;
   }
   
   std::string aKey = "NKS-Assoc-Im2Orient@-" + mOriOut;
   
   std::vector<Pt3dr> aCorr;
 
   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
       // Lever Arm, uniquement pour camera stenope
       CamStenope * aCS = mVSoms[aK]->attr().mIma->CamSNN();
       Pt3dr aCorr2Add = aCS->Orient().IRecVect(mLA);
       aCorr.push_back(aCorr2Add);
       ElRotation3D anOriC(aCorr2Add - aCS->PseudoOpticalCenter(),0,0,0);
       aCS->SetOrientation(anOriC);
       cOrientationConique  anOC = aCS->StdExportCalibGlob();
       std::string aNameOut = mEASF.mICNM->Assoc1To1(aKey,mVSoms[aK]->attr().mIma->mNameIm,true);
       MakeFileXML(anOC,aNameOut);
   }
   
   mICNM2 = mEASF.mICNM;
   
   if(EAMIsInit(&mOri2))
   {
       mICNM2 = cInterfChantierNameManipulateur::BasicAlloc(mOri2);
   }
   
   
   StdCorrecNameOrient(mOri2,mDir);
   if(mOri2Out == "")
   {
	   mOri2Out = mOri2 + "-CorrLa";
   }
   
   
   std::string aKey2 = "NKS-Assoc-Im2Orient@-" + mOri2Out;
   
      
   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
       cImaMM * anIm = mVSoms[aK]->attr().mIma;
       CamStenope * aCam =  mICNM2->StdCamStenOfNames(anIm->mNameIm,mOri2);
	   ElRotation3D anOriC(aCorr[aK] - aCam->PseudoOpticalCenter(),0,0,0);
	   aCam->SetOrientation(anOriC);
	   cOrientationConique  anOC = aCam->StdExportCalibGlob();
       std::string aNameOut = mEASF.mICNM->Assoc1To1(aKey2,mVSoms[aK]->attr().mIma->mNameIm,true);
       MakeFileXML(anOC,aNameOut);
   }
   
}

int CorrLA_main(int argc,char ** argv)
{
    cAppliCorrecCamFromLA_main anAppli(argc,argv);
    return EXIT_SUCCESS;
}


//  ===============================================================================================
class cAppliCorrOriFromBias_main : public cAppliWithSetImage
{
	public :
		cAppliCorrOriFromBias_main(int argc,char ** argv);
	private :
		std::string mDir;
		std::string mPat;
		std::string mOriFull;
		Pt3dr mBias;
		std::string mOriOut;
		cInterfChantierNameManipulateur * mICNM2;
};

cAppliCorrOriFromBias_main::cAppliCorrOriFromBias_main(int argc,char ** argv) :
	cAppliWithSetImage(argc-1,argv+1,0)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mEASF.mFullName,"Full Name (Dir+Pattern)",eSAM_IsPatFile)
                    << EAMC(mOriFull,"Directory orientation", eSAM_IsExistDirOri)
                    << EAMC(mBias,"Bias value"),
        LArgMain()  << EAM(mOriOut,"OriOut",true,"Output Ori Name of corrected mandatory Ori ; Def=OriName-Corr")
   );
   
   SplitDirAndFile(mDir,mPat,mEASF.mFullName);
   
   StdCorrecNameOrient(mOriFull,mDir);
   
   if(mOriOut == "")
   {
		mOriOut = mOriFull + "-Corr" ;
   }
   
   std::string aKey = "NKS-Assoc-Im2Orient@-" + mOriOut;
   
   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
       CamStenope * aCS = mVSoms[aK]->attr().mIma->CamSNN();
       
       Pt3dr aPtCorr;
       aPtCorr.x = mBias.x + aCS->PseudoOpticalCenter().x;
       aPtCorr.y = mBias.y + aCS->PseudoOpticalCenter().y;
       aPtCorr.z = mBias.z + aCS->PseudoOpticalCenter().z;
       
       ElRotation3D anOriC(aPtCorr,aCS->Orient().Mat().transpose(),false);
       aCS->SetOrientation(anOriC.inv());
       cOrientationConique  anOC = aCS->StdExportCalibGlob();
       std::string aNameOut = mEASF.mICNM->Assoc1To1(aKey,mVSoms[aK]->attr().mIma->mNameIm,true);
       MakeFileXML(anOC,aNameOut);
   }
}

int CorrOri_main(int argc,char ** argv)
{
	cAppliCorrOriFromBias_main anAppli(argc,argv);
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
