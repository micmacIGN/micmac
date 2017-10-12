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

#define  NbModes  8
#define  NbFreqs  5
#define  NbSols   3
#define  NbIono   11
#define  NbTropo  5
#define  NbEphe   5
#define  NbNavSys 6
#define  NbAmbRes 4
#define  NbSolOutF 4
#define  NbTimeOutSys 3
#define  NbOutGeoid 5
#define  NbAntPosType 6

//positionning modes
const char * Modes[NbModes] = {
                                   "single",         // 0
                                   "dgps",  		 // 1
                                   "kinematic",      // 2
                                   "static",     	 // 3
                                   "movingbase",     // 4
                                   "fixed",          // 5
                                   "ppp-kine",       // 6
                                   "ppp-static"      // 7
                              };

//frequencies used in processing
const char * Freq[NbFreqs] = {
									"l1",				// 0
									"l1+l2",			// 1
									"l1+l2+l5",			// 2
									"l1+l2+l5+l6",		// 3
									"l1+l2+l5+l6+l7"	// 4
							 };

//solution type
const char * SolTypes[NbSols] = {
									"forward",			// 0
									"backward",			// 1
									"combined"			// 2
								};

//iono
const char * Iono[NbIono] = {
									"off",				// 0
									"brdc",				// 1
									"sbas",				// 2
									"dual-freq",		// 3
									"est-stec",			// 4
									"ionex-tec",		// 5
									"qzs-brdc",			// 6
									"qzs-lex",			// 7
									"vtec_sf",			// 8
									"vtec_ef",			// 9
									"gtec"				//10
							};

//tropo
const char * Tropo[NbTropo] = {
									"off",				// 0
									"saas",				// 1
									"sbas",				// 2
									"est-ztd",			// 3
									"est-ztdgrad"		// 4
							  };

//ephemerides
const char * Ephe[NbEphe] = {
									"brdc",				// 0
									"precise",			// 1
									"brdc+sbas",		// 2
									"brdc+ssrapc",		// 3
									"brdc+ssrcom"		// 4
							};
							
//navigation systems
const char * NavSys[NbNavSys] = {
									"gps",
									"+sbas",
									"+glo",
									"+gal",
									"+qzs",
									"+comp"
								};
								
//Ambiguities resolution strategy
const char * AmbRes[NbAmbRes] = {
									"off",				// 0
									"continuous",		// 1
									"instantaneous",	// 2
									"fix-and-hold"		// 3
								};
								
//Solutions format output
const char * SolOutFormat[NbSolOutF] = {
											"llh",		// 0
											"xyz",		// 1
											"enu",		// 2
											"nmea"		// 3
										};
										
//Time system output
const char * TimeOutSys[NbTimeOutSys] = {
											"gpst",		// 0
											"utc",		// 1
											"jst"		// 2
										};
										
//Out Geoid
const char * OutGeoid[NbOutGeoid] = {
										"internal",		// 0
										"egm96",		// 1
										"egm08_2.5",	// 2
										"egm08_1",		// 3
										"gsi2000"		// 4
									};
									
//Antenna position type
const char * AntPosType[NbAntPosType] = {
											"llh",			// 0
											"xyz",			// 1
											"single",		// 2
											"posfile",		// 3
											"rinexhead",	// 4
											"rtcm"			// 5
										};
                                
//Display authorized values
void ShowAuthorizedValues(const char * aValues)
{
   std::cout << "\n";
   std::cout << "Authorized values : \n";
   for (unsigned int aKM=0 ; aKM<sizeof(aValues) ; aKM++)
       std::cout << "   " << aValues[aKM] << "\n";
}

int rnx2rtkp_main(int argc,char ** argv)
{
	std::string aDir, aMode, aFileR, aFileB, aFileN, aFileG, aFileP, aFileA;
	std::string aOut, aXmlOut;
	bool aAscii=false;
	std::string aStrChSys;
	std::string aFreq="l1";
	std::string aSolType="combined";
	int aElMask=15;
	bool aDyn=false;
	bool aTideCorr=false;
	std::string aIono="brdc";
	std::string aTropo="saas";
	std::string aEph="Nav";
	int aNavSys=1;
	std::string aAmbRes="fix_and_hold";
	std::string aAmbResGlo="OFF";
	int aAmbResTh=3;
	int aAmbResLockCNT=0;
	int aAmbResElvMask=0;
	int aAmbResMinFix=10;
	int aElevMaskHold=0;
	int aAmbResOutCNT=5;
	double aMaxage=30;
	double aSlipTh=0.05;
	double aRejIono=30;
	double aRejGdop=30;
	int aNbrIter=1;
	double aBaseLength=0;
	double aBaseSig=0;
	std::string aOutSolFormat="xyz";
	bool aOutHead=true;
	bool aOutOpt=true;
	std::string aOutTimeSys="gpst";
	std::string aOutTimeFormat="hms";
	int aTimeDec=6;
	std::string aDegOutFormat="deg";
	std::string aHeigth="ellipsoidal";
	std::string aGeoid="internal";
	std::string aStaticSol="one";
	int aNmeaIntv1=0;
	int aNmeaIntv2=0;
	std::string aOutStats="none";
	double aEratio1=100;
	double aEratio2=100;
	double aErrPhase=0.003;
	double aErrPhaseEl=0.003;
	double aErrPhaseBl=0;
	double aErrDoppler=1;
	double aStdBias=30;
	double aStdIono=0.03;
	double aStdTropo=0.3;
	double aAccH=10;
	double aAccV=10;
	double aPrnBias=0.0001;
	double aPrnIono=0.001;
	double aPrnTropo=0.0001;
	double aClkStab=5e-12;
	std::string aAnt1PosType="rinexhead";
	Pt3dr aAnt1Pos(0,0,0);
	std::string aAnt1Type;
	Pt3dr aAnt1Delta(0,0,0);
	std::string aAnt2PosType="rinexhead";
	Pt3dr aAnt2Pos(0,0,0);
	std::string aAnt2Type;
	Pt3dr aAnt2Delta(0,0,0);
	std::string aRcvAntFile;
	std::string aSatAntFile;
	std::string aStaPosFile;
	std::string aGeoidFile;
	std::string aIonoFile;
	std::string aDcbFile;
	std::string aTmpDir;
	std::string aGeexeFile;
	std::string aSolStatsFile;
	std::string aTraceFile;
	std::string aOutRtkParam = "rtkParamsConfigs.txt";
	
	bool aModeFile=false;
	std::string aFileMF="";
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(aDir,"Directory")
					 << EAMC(aMode,"Positionning mode",eSAM_None,ListOfVal(eNbTypeGpsMod))
					 << EAMC(aFileR,"Rinex file of rover station",  eSAM_IsExistFile)
					 << EAMC(aFileB,"Rinex file of base station ; NONE if single or PPP pos mode", eSAM_IsExistFile)
					 << EAMC(aFileN,"Navigation file for GPS satellites ephemerides", eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"output txt file name ; Def = Output_mode.txt", eSAM_IsOutputFile)
					 << EAM(aModeFile,"ModeFile",false,"if using RTKlib in mode file ; Def=false",eSAM_IsBool)
					 << EAM(aFileMF,"InputConf",false,"input name file config",eSAM_IsExistFile)
					 << EAM(aXmlOut,"Xml",false,"output xml Gps Trajectory file ; Def = Output_mode.xml", eSAM_IsOutputFile)
					 << EAM(aAscii,"Ascii",false,"ASCII file export of result ; Format = (t,X,Y,Z,Q) ; Def = Output_mode_txyz.txt", eSAM_IsBool)
					 << EAM(aStrChSys,"ChSys",true,"Change coordinate file")
					 << EAM(aFreq,"Freq",true,"Freq to be used ; Def = l1",eSAM_None,ListOfVal(eNbTypeGpsFreq))
					 << EAM(aSolType,"SolStrg",true,"Filter strategy ; Def = combined",eSAM_None,ListOfVal(eNbTypeGpsSol))
					 << EAM(aElMask,"ElevMask",true,"Elevation mask ; Def = 15 deg")
					 << EAM(aDyn,"Dynamics",true,"Dynamics ; Def = false", eSAM_IsBool)
					 << EAM(aTideCorr,"TideCorr",true,"TideCorr ; Def = false", eSAM_IsBool)
					 << EAM(aIono,"Iono",true,"Ionospheric correction ; Def = brdc", eSAM_None,ListOfVal(eNbTypeGpsIonoCorr))
					 << EAM(aTropo,"Tropo",true,"Tropospheric correction ; Def = saas", eSAM_None,ListOfVal(eNbTypeGpsTropoCorr))
					 << EAM(aEph,"Ephe",true,"Satellites Ephemerides ; Def = brdc", eSAM_None,ListOfVal(eNbTypeGpsEphe))
					 << EAM(aNavSys,"NavSys",true,"Navigation system ; Def = 1 <==> GPS")
					 << EAM(aFileG,"GloNavFile",true,"Navigation file for Glonass satellites ephemerides", eSAM_IsExistFile)
					 << EAM(aAmbRes,"AmbRes",true,"Ambiguity resolution strategy ; Def = fix-and-hold", eSAM_None,ListOfVal(eNbTypeGpsAmbRes))
					 << EAM(aAmbResGlo,"AmResGlo",true,"Ambiguity resolution for Glonass ; Def = off", eSAM_None,ListOfVal(eNbTypeGloAmbRes))
					 << EAM(aAmbResTh,"AmResTh",true,"Ambiguity resolution ratio ; Def = 3")
					 << EAM(aAmbResLockCNT,"AmbResLC",true,"Ambiguity resolution lock CNT ; Def = 0")
					 << EAM(aAmbResElvMask,"AmbResElvMask",true,"Ambiguity resolution Elevation Mask ; Def = 0 deg")
					 << EAM(aAmbResMinFix,"AmbResMinFix",true,"Ambiguity resolution minimum fix ; Def = 10")
					 << EAM(aElevMaskHold,"ElevMaskHold",true,"Elevation Mask Hold ; Def = 0")
					 << EAM(aAmbResOutCNT,"AmbResOutCNT",true,"Ambiguity resolution output CNT ; Def = 5")
					 << EAM(aMaxage,"Maxage",true,"Maxage ; Def = 30 s")
					 << EAM(aSlipTh,"SlipTh",true,"Cycle Slips Threshold ; Def = 0.05 m")
					 << EAM(aRejIono,"RejIono",true,"RejIono ; Def = 30 m")
					 << EAM(aRejGdop,"RejGdop",true,"RejGdop ; Def = 30")
					 << EAM(aNbrIter,"NbrIter",true,"Number of iterations ; Def = 1")
					 << EAM(aBaseLength,"BaseLen",true,"Base length to apply a constraint ; Def = 0 m")
					 << EAM(aBaseSig,"BaseLenInc",true,"Uncurtainty on Base length ; Def = 0 m")
					 << EAM(aOutSolFormat,"OutFormat",true,"Output solution format ; Def = xyz", eSAM_None,ListOfVal(eNbTypeGpsSolFormat))
					 << EAM(aOutHead,"OutHead",true,"Output header ; Def = true", eSAM_IsBool)
					 << EAM(aOutOpt,"OutOpt",true,"Output options ; Def = true", eSAM_IsBool)
					 << EAM(aOutTimeSys,"OutTimeSys",true,"Output time system ; Def = gpst", eSAM_None,ListOfVal(eNbTypeGpsTimeSys))
					 << EAM(aOutTimeFormat,"OutTimeFormat",true,"Output time format ; Def = hms", eSAM_None,ListOfVal(eNbTypeGpsTimeFormat))
					 << EAM(aTimeDec,"TimeDec",true,"Number of decimals for time output ; Def = 6")
					 << EAM(aDegOutFormat,"DegOutFormat",true,"Output in degree format ; Def = deg", eSAM_None, ListOfVal(eNbTypeGpsDegFormat))
					 << EAM(aHeigth,"Height",true,"Output heigth ; Def = ellipsoidal", eSAM_None, ListOfVal(eNbTypeGpsHeight))
					 << EAM(aGeoid,"Geoid",true,"Output geoid ; Def = internal", eSAM_None, ListOfVal(eNbTypeGeoid))
					 << EAM(aStaticSol,"StaticSol",true,"Static output solutions ; Def = one", eSAM_None, ListOfVal(eNbTypeGpsSolStatic))
					 << EAM(aNmeaIntv1,"nmeaintv1",true,"nmeaintv1 ; Def = 0 s")
					 << EAM(aNmeaIntv2,"nmeaintv2",true,"nmeaintv2 ; Def = 0 s")
					 << EAM(aOutStats,"OutStats",true,"Output statistics in a file ; Def = off", eSAM_None, ListOfVal(eNbTypeRtkOutStats))
					 << EAM(aEratio1,"Eratio1",true,"Eratio1 ; Def = 100")
					 << EAM(aEratio2,"Eratio2",true,"Eratio2 ; Def = 100")
					 << EAM(aErrPhase,"ErrPhase",true,"Phase error measurement ; Def = 0.003 m")
					 << EAM(aErrPhaseEl,"ErrPhaseEl",true,"Phase El error measurement ; Def = 0.003 m")
					 << EAM(aErrPhaseBl,"ErrPhaseBl",true,"Phase Bl error measurement ; Def = 0 m/10*km")
					 << EAM(aErrDoppler,"ErrDoppler",true,"Doppler error measurement ; Def = 1 Hz")
					 << EAM(aStdBias,"StdBias",true,"a priori Standard Deviation bias ; Def = 30 m")
					 << EAM(aStdIono,"StdIono",true,"a priori Standard Deviation Ionospheric delay ; Def = 0.03 m")
					 << EAM(aStdTropo,"StdTropo",true,"a priori Standard Deviation Tropospheric delay ; Def = 0.3 m")
					 << EAM(aAccH,"AccH",true,"Horizontal acceleration ; Def = 10 m/s*s")
					 << EAM(aAccV,"AccV",true,"Vertical acceleration ; Def = 10 m/s*s")
					 << EAM(aPrnBias,"PrnBias",true,"Prn Bias ; Def = 0.0001 m")
					 << EAM(aPrnIono,"PrnIono",true,"Prn Iono ; Def = 0.001 m")
					 << EAM(aPrnTropo,"PrnTropo",true,"Prn Tropo ; Def = 0.0001 m")
					 << EAM(aClkStab,"ClkStab",true,"Clock Stability ; Def = 5e-12 s/s")
					 << EAM(aAnt1PosType,"AntRPosType",true,"Antenna rover Position type ; Def = rinexhead", eSAM_None, ListOfVal(eNbGpsAntPos))
					 << EAM(aAnt1Pos,"AntRPos",true,"Antenna rover Position ; Format ==> [X,Y,Z]")
					 << EAM(aAnt1Type,"AntRType",true,"Antenna rover Type ;")
					 << EAM(aAnt1Delta,"AntRDelta",true,"Antenna rover ARP Delta ; Format ==> [dx,dy,dz]")
					 << EAM(aAnt2PosType,"AntBPosType",true,"Antenna base Position type ; Def = rinexhead", eSAM_None, ListOfVal(eNbGpsAntPos))
					 << EAM(aAnt2Pos,"AntBPos",true,"Antenna base Position ; Format ==> [X,Y,Z]")
					 << EAM(aAnt2Type,"AntBType",true,"Antenna base Type ;")
					 << EAM(aAnt2Delta,"AntBDelta",true,"Antenna base ARP Delta ; Format ==> [dx,dy,dz]")
					 << EAM(aRcvAntFile,"AntFileRCV",true,"Antenna file for receivers ; XXXXX.atx")
					 << EAM(aSatAntFile,"AntFileSATs",true,"Antenna file for satellites ; XXXXX.atx")
					 << EAM(aStaPosFile,"StaPosFile",true,"file-staposfile")
					 << EAM(aGeoidFile,"GeoidFile",true,"file-geoidfile")
					 << EAM(aIonoFile,"IonoFile",true,"file-ionofile")
					 << EAM(aDcbFile,"DcbFile",true,"file-dcbfile")
					 << EAM(aTmpDir,"TmpDir",true,"file-tempdir")
					 << EAM(aGeexeFile,"GeexeFile",true,"file-geexefile")
					 << EAM(aSolStatsFile,"SolStatsFile",true,"file-solstatfile")
					 << EAM(aTraceFile,"TraceFile",true,"file-tracefile")
					 << EAM(aOutRtkParam,"OutrtkParams",true,"Output file of rtk processing parametrs ; Def = rtkParamsConfigs.txt", eSAM_IsOutputFile)
    );
    
    //generate the good rtk params file
    bool help;
    
    eTypeGpsMod  typeM;
    StdReadEnum(help,typeM,aMode,eNbTypeGpsMod);
    
    //correction for "pos1-posmode"
    if(aMode == "ppp_kine")
		aMode = "ppp-kine";
	
	//correction for "pos1-posmode"
	if(aMode == "ppp_static")
		aMode = "ppp-static";
    
    eTypeGpsFreq typeF;
    StdReadEnum(help,typeF,aFreq,eNbTypeGpsFreq);
    
    //correction for "pos1-frequency"
    if(aFreq == "l1_l2")
		aFreq = "l1+l2";
		
	//correction for "pos1-frequency"
	if(aFreq == "l1_l2_l5")
		aFreq = "l1+l2+l5";
		
	//correction for "pos1-frequency"
	if(aFreq == "l1_l2_l5_l6")
		aFreq = "l1+l2+l5+l6";
		
	//correction for "pos1-frequency"
	if(aFreq == "l1_l2_l5_l6_l7")
		aFreq = "l1+l2+l5+l6+l7";
    
    eTypeGpsSol typeSol;
    StdReadEnum(help,typeSol,aSolType,eNbTypeGpsSol);
    
    eTypeGpsIonoCorr typeIono;
    StdReadEnum(help,typeIono,aIono,eNbTypeGpsIonoCorr);
    
    //correction for "pos1-ionoopt"
    if(aIono == "dual_freq")
		aIono = "dual-freq";
		
	//correction for "pos1-ionoopt"
	if(aIono == "est_stec")
		aIono = "est-stec";
		
	//correction for "pos1-ionoopt"
	if(aIono == "ionex_tec")
		aIono = "ionex-tec";
		
	//correction for "pos1-ionoopt"
	if(aIono == "qzs_brdc")
		aIono = "qzs-brdc";
		
	//correction for "pos1-ionoopt"
	if(aIono == "qzs_lex")
		aIono = "qzs-lex";
		
	eTypeGpsTropoCorr typeTropo;
	StdReadEnum(help,typeTropo,aTropo,eNbTypeGpsTropoCorr);
	
	//correction for "pos1-tropopt"
	if(aTropo == "null")
		aTropo = "off";
	
	//correction for "pos1-tropopt"
	if(aTropo == "SBAS")
		aTropo = "sbas";
		
	//correction for "pos1-tropopt"
	if(aTropo == "est_ztd")
		aTropo = "est-ztd";
		
	//correction for "pos1-tropopt"
	if(aTropo == "est_ztdgrad")
		aTropo = "est-ztdgrad";
    
    eTypeGpsEphe typeEphe;
    StdReadEnum(help,typeEphe,aEph,eNbTypeGpsEphe);
    
    //correction for "pos1-sateph"
    if(aEph == "Nav")
		aEph = "brdc";
	
	//correction for "pos1-sateph"
	if(aEph == "brdc_sbas")
		aEph = "brdc+sbas";
	
	//correction for "pos1-sateph"
	if(aEph == "brdc_ssrapc")
		aEph = "brdc+ssrapc";
	
	//correction for "pos1-sateph"
	if(aEph == "brdc_ssrcom")
		aEph = "brdc+ssrcom";
	
	eTypeGpsAmbRes typeAR;
	StdReadEnum(help,typeAR,aAmbRes,eNbTypeGpsAmbRes);
	
	//correction for "pos2-armode"
	if(aAmbRes == "NONE")
		aAmbRes = "off";
	
	//correction for "pos2-armode"
	if(aAmbRes == "fix_and_hold")
		aAmbRes = "fix-and-hold";
		
	eTypeGloAmbRes typeGAR;
	StdReadEnum(help,typeGAR,aAmbResGlo,eNbTypeGloAmbRes);
	
	//correction for "pos2-gloarmode"
	if(aAmbResGlo == "OFF")
		aAmbResGlo = "off";
		
	eTypeGpsSolFormat typeSF;
	StdReadEnum(help,typeSF,aOutSolFormat,eNbTypeGpsSolFormat);
	
	eTypeGpsTimeSys typeTS;
	StdReadEnum(help,typeTS,aOutTimeSys,eNbTypeGpsTimeSys);
	
	eTypeGpsTimeFormat typeTF;
	StdReadEnum(help,typeTF,aOutTimeFormat,eNbTypeGpsTimeFormat);
	
	eTypeGpsDegFormat typeDF;
	StdReadEnum(help,typeDF,aDegOutFormat,eNbTypeGpsDegFormat);
	
	eTypeGpsHeight typeH;
	StdReadEnum(help,typeH,aHeigth,eNbTypeGpsHeight);
	
	eTypeGeoid typeG;
	StdReadEnum(help,typeG,aGeoid,eNbTypeGeoid);
	
	//correction for "out-geoid"
	if(aGeoid == "egm08_2_5")
		aGeoid = "egm08_2.5";
		
	eTypeGpsStaticSol typeSolS;
	StdReadEnum(help,typeSolS,aStaticSol,eNbTypeGpsSolStatic);
	
	//correction for "out-solstatic"
	if(aStaticSol == "one")
		aStaticSol = "single";
	
	eTypeRtkOutStats typeROS;
	StdReadEnum(help,typeROS,aOutStats,eNbTypeRtkOutStats);
	
	//correction for "out-outstat"
	if(aOutStats == "none")
		aOutStats = "off";
		
	eTypeGpsAntPos typeA1P;
	StdReadEnum(help,typeA1P,aAnt1PosType,eNbGpsAntPos);
	
	//correction for "ant1-postype"
	if(aAnt1PosType == "LLH")
		aAnt1PosType = "llh";
		
	//correction for "ant1-postype"
	if(aAnt1PosType == "XYZ")
		aAnt1PosType = "xyz";
		
	//correction for "ant1-postype"
	if(aAnt1PosType == "code")
		aAnt1PosType = "single";
	
	eTypeGpsAntPos typeA2P;
	StdReadEnum(help,typeA2P,aAnt2PosType,eNbGpsAntPos);
	
	//correction for "ant2-postype"
	if(aAnt2PosType == "LLH")
		aAnt2PosType = "llh";
		
	//correction for "ant2-postype"
	if(aAnt2PosType == "XYZ")
		aAnt2PosType = "xyz";
		
	//correction for "ant2-postype"
	if(aAnt2PosType == "code")
		aAnt2PosType = "single";
			
	//****************************************************************//
	//****************************************************************//
	//lanch gps processing
	std::string aCom, aCom1;
	if (aOut == "")
	{
		aOut = "Output_" + aMode + ".txt";
	}
	
	if(aModeFile)
	{
		aCom = g_externalToolHandler.get( "Rnx2rtkp" ).callName()
				+ std::string(" ") + aFileR
				+ std::string(" ") + aFileB
				+ std::string(" ") + aFileN
				+ std::string(" ") + aFileG
				+ string(" -o ")
				+ aOut
				+ string(" -k ")
				+ aFileMF;
		std::cout << "aCom = " << aCom << std::endl;
		system_call(aCom.c_str());
	}
	else
	{
	
		//if a station file is given
		if(aStaPosFile != "")
		{
			//read the .xml file : one position only
			cDicoGpsFlottant aStationFile =  StdGetFromPCP(aStaPosFile,DicoGpsFlottant);
					
			if(aStationFile.OneGpsDGF().size() != 1)
			{
				ELISE_ASSERT(0,"Your station file does not containe a unique position");
			}
			else
			{
				// std::list <cOneGpsDGF> & aVPS = aStationFile.OneGpsDGF();
				// for(std::list<cOneGpsDGF>::iterator iT=aVPS.begin(); iT!=aVPS.end(); iT++)
				for(auto iT=aStationFile.OneGpsDGF().begin(); iT!=aStationFile.OneGpsDGF().end(); iT++)
				{
					aAnt2Pos.x = iT->Pt().x;
					aAnt2Pos.y = iT->Pt().y;
					aAnt2Pos.z = iT->Pt().z;
				}
			}
			
			//give coordinates
		}
		
		 
		if (!MMVisualMode)
		{
			FILE * aFP = FopenNN(aOutRtkParam,"w","rnx2rtkp_main");
			
			cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOutRtkParam);
					
			fprintf(aFP,"pos1-posmode=%s\n",aMode.c_str());
			fprintf(aFP,"pos1-frequency=%s\n",aFreq.c_str());
			fprintf(aFP,"pos1-soltype=%s\n",aSolType.c_str());
			fprintf(aFP,"pos1-elmask=%d\n",aElMask);
			fprintf(aFP,"pos1-snrmask_r=off\n");
			fprintf(aFP,"pos1-snrmask_b=off\n");
			fprintf(aFP,"pos1-snrmask_b=off\n");
			fprintf(aFP,"pos1-snrmask_L1=0,0,0,0,0,0,0,0,0\n");
			fprintf(aFP,"pos1-snrmask_L2=0,0,0,0,0,0,0,0,0\n");
			fprintf(aFP,"pos1-snrmask_L5=0,0,0,0,0,0,0,0,0\n");
			if(aDyn)
				fprintf(aFP,"pos1-dynamics=on\n");
			else
				fprintf(aFP,"pos1-dynamics=off\n");
			
			if(aTideCorr)
				fprintf(aFP,"pos1-tidecorr=on\n");
			else
				fprintf(aFP,"pos1-tidecorr=off\n");
				
			fprintf(aFP,"pos1-ionoopt=%s\n",aIono.c_str());
			fprintf(aFP,"pos1-tropopt=%s\n",aTropo.c_str());
			fprintf(aFP,"pos1-sateph=%s\n",aEph.c_str());
			fprintf(aFP,"pos1-exclsats=\n");
			fprintf(aFP,"pos1-navsys=%d\n",aNavSys);
			fprintf(aFP,"pos2-armode=%s\n",aAmbRes.c_str());
			fprintf(aFP,"pos2-gloarmode=%s\n",aAmbResGlo.c_str());
			fprintf(aFP,"pos2-arthres=%d\n",aAmbResTh);
			fprintf(aFP,"pos2-arlockcnt=%d\n",aAmbResLockCNT);
			fprintf(aFP,"pos2-arelmask=%d\n",aAmbResElvMask);
			fprintf(aFP,"pos2-arminfix=%d\n",aAmbResMinFix);
			fprintf(aFP,"pos2-elmaskhold=%d\n",aElevMaskHold);
			fprintf(aFP,"pos2-aroutcnt=%d\n",aAmbResOutCNT);
			fprintf(aFP,"pos2-maxage=%f\n",aMaxage);
			fprintf(aFP,"pos2-slipthres=%f\n",aSlipTh);
			fprintf(aFP,"pos2-rejionno=%f\n",aRejIono);
			fprintf(aFP,"pos2-rejgdop=%f\n",aRejGdop);
			fprintf(aFP,"pos2-niter=%d\n",aNbrIter);
			fprintf(aFP,"pos2-baselen=%f\n",aBaseLength);
			fprintf(aFP,"pos2-basesig=%f\n",aBaseSig);
			fprintf(aFP,"out-solformat=%s\n",aOutSolFormat.c_str());
			if(aOutHead)
				fprintf(aFP,"out-outhead=on\n");
			else
				fprintf(aFP,"out-outhead=off\n");
				
			if(aOutOpt)
				fprintf(aFP,"out-outopt=on\n");
			else
				fprintf(aFP,"out-outopt=off\n");
			
			fprintf(aFP,"out-timesys=%s\n",aOutTimeSys.c_str());
			fprintf(aFP,"out-timeform=%s\n",aOutTimeFormat.c_str());
			fprintf(aFP,"out-timendec=%d\n",aTimeDec);
			fprintf(aFP,"out-degform=%s\n",aDegOutFormat.c_str());
			fprintf(aFP,"out-fieldsep=\n");
			fprintf(aFP,"out-height=%s\n",aHeigth.c_str());
			fprintf(aFP,"out-geoid=%s\n",aGeoid.c_str());
			fprintf(aFP,"out-solstatic=%s\n",aStaticSol.c_str());
			fprintf(aFP,"out-nmeaintv1=%d\n",aNmeaIntv1);
			fprintf(aFP,"out-nmeaintv2=%d\n",aNmeaIntv2);
			fprintf(aFP,"out-outstat=%s\n",aOutStats.c_str());
			fprintf(aFP,"stats-eratio1=%f\n",aEratio1);
			fprintf(aFP,"stats-eratio2=%f\n",aEratio2);
			fprintf(aFP,"stats-errphase=%f\n",aErrPhase);
			fprintf(aFP,"stats-errphaseel=%f\n",aErrPhaseEl);
			fprintf(aFP,"stats-errphasebl=%f\n",aErrPhaseBl);
			fprintf(aFP,"stats-errdoppler=%f\n",aErrDoppler);
			fprintf(aFP,"stats-stdbias=%f\n",aStdBias);
			fprintf(aFP,"stats-stdiono=%f\n",aStdIono);
			fprintf(aFP,"stats-prnaccelh=%f\n",aAccH);
			fprintf(aFP,"stats-prnaccelv=%f\n",aAccV);
			fprintf(aFP,"stats-prnbias=%f\n",aPrnBias);
			fprintf(aFP,"stats-prniono=%f\n",aPrnIono);
			fprintf(aFP,"stats-prntrop=%f\n",aPrnTropo);
			fprintf(aFP,"stats-clkstab=%f\n",aClkStab);
			fprintf(aFP,"ant1-postype=%s\n",aAnt1PosType.c_str());
			fprintf(aFP,"ant1-pos1=%f\n",aAnt1Pos.x);
			fprintf(aFP,"ant1-pos2=%f\n",aAnt1Pos.y);
			fprintf(aFP,"ant1-pos3=%f\n",aAnt1Pos.z);
			fprintf(aFP,"ant1-anttype=%s\n",aAnt1Type.c_str());
			fprintf(aFP,"ant1-antdele=%f\n",aAnt1Delta.x);
			fprintf(aFP,"ant1-antdeln=%f\n",aAnt1Delta.y);
			fprintf(aFP,"ant1-antdelu=%f\n",aAnt1Delta.z);
			fprintf(aFP,"ant2-postype=%s\n",aAnt2PosType.c_str());
			fprintf(aFP,"ant2-pos1=%f\n",aAnt2Pos.x);
			fprintf(aFP,"ant2-pos2=%f\n",aAnt2Pos.y);
			fprintf(aFP,"ant2-pos3=%f\n",aAnt2Pos.z);
			fprintf(aFP,"ant2-anttype=%s\n",aAnt2Type.c_str());
			fprintf(aFP,"ant2-antdele=%f\n",aAnt2Delta.x);
			fprintf(aFP,"ant2-antdeln=%f\n",aAnt2Delta.y);
			fprintf(aFP,"ant2-antdelu=%f\n",aAnt2Delta.z);
			fprintf(aFP,"misc-timeinterp=on\n");
			fprintf(aFP,"misc-sbasatsel=0\n");
			fprintf(aFP,"file-rcvantfile=%s\n",aRcvAntFile.c_str());
			fprintf(aFP,"file-satantfile=%s\n",aRcvAntFile.c_str());
			fprintf(aFP,"file-staposfile=\n");
			fprintf(aFP,"file-geoidfile=%s\n",aGeoidFile.c_str());
			fprintf(aFP,"file-ionofile=%s\n",aIonoFile.c_str());
			fprintf(aFP,"file-dcbfile=%s\n",aDcbFile.c_str());
			fprintf(aFP,"file-tempdir=%s\n",aTmpDir.c_str());
			fprintf(aFP,"file-geexefile=%s\n",aGeexeFile.c_str());
			fprintf(aFP,"file-solstatfile=%s\n",aSolStatsFile.c_str());
			fprintf(aFP,"file-tracefile=%s\n",aTraceFile.c_str());
			
				
			ElFclose(aFP);
				
		}
		
		//if mode is single/ppp-kine/ppp-static no need of base station
		if(aMode == "single" || aMode == "ppp-static" || aMode == "ppp-kine")
		{
			aCom = g_externalToolHandler.get( "Rnx2rtkp" ).callName()
					+ std::string(" ") + aFileR
					+ std::string(" ") + aFileN
					+ string(" -o ")
					+ aOut
					+ string(" -k ")
					+ aOutRtkParam;
		}
		else if((aNavSys == 5) && (aFileG == ""))							//if Glonass satellites are used
		{
			ELISE_ASSERT(0,"No Glonass Navigation file given");
		}
		else if((aNavSys == 5) && (aFileG != ""))
		{
			aCom = g_externalToolHandler.get( "Rnx2rtkp" ).callName()
					+ std::string(" ") + aFileR
					+ std::string(" ") + aFileB
					+ std::string(" ") + aFileN
					+ std::string(" ") + aFileG
					+ string(" -o ")
					+ aOut
					+ string(" -k ")
					+ aOutRtkParam;
		}
		else
		{
			aCom = g_externalToolHandler.get( "Rnx2rtkp" ).callName()
				+ std::string(" ") + aFileR 
				+ std::string(" ") + aFileB 
				+ std::string(" ") + aFileN 
				+ string(" -o ") 
				+ aOut 
				+ string(" -k ")
				+ aOutRtkParam;
		}
		
		std::cout << "aCom = " << aCom << std::endl;
		system_call(aCom.c_str());
		 
		 //lanch conversion to .xml Gps Trajectory format
		 if (aXmlOut == "")
		 {
			 aXmlOut = "Output_" + aMode + ".xml";
		 }
		 
		 //dealing with changing system
		 if(aStrChSys == "")
		 {
			aCom1 = MMDir() 
					+ std::string("bin/mm3d ")
					+ "TestLib ConvRTK "
					+ aDir + std::string(" ")
					+ aOut;
		 }
		 else
		 {
			aCom1 = MMDir() 
					+ std::string("bin/mm3d ")
					+ "TestLib ConvRTK "
					+ aDir + std::string(" ")
					+ aOut + std::string(" ")
					+ std::string("ChSys=")
					+ aStrChSys;
		 }
		 
		 if(aAscii)
		 {
			 aCom1 = aCom1 + std::string(" ") + std::string("tXYZQ=true");
		 }
		 
		 std::cout << "aCom = " << aCom1 << std::endl;
		 system_call(aCom1.c_str());
	 }
    
   	return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
