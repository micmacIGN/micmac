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

int Recover_Main(int argc, char ** argv);

int XLib_Main(int argc, char ** argv);

const cArgLogCom cArgLogCom::NoLog(-1);

// MPD : suspecte un problème d'écrasement mutuel entre processus dans le logfile, inhibe temporairement pour
// valider / invalider le diagnostic
static bool DOLOG_MM3d = true;

FILE * FileLogMM3d(const std::string & aDir)
{
	// return  FopenNN(aDir+"mm3d-LogFile.txt","a+","Log File");
	std::string aName = aDir + "mm3d-LogFile.txt";
	FILE * aRes = 0;
	int aCpt = 0;
	int aCptMax = 20;
	while (aRes == 0)
	{
		aRes = fopen(aName.c_str(), "a+");
		if (aRes == 0)
		{
			int aModulo = 1000;
			int aPId = mm_getpid();

			double aTimeSleep = ((aPId%aModulo) *  ((aCpt + 1) / double(aCptMax))) / double(aModulo * 20.0);
			SleepProcess(aTimeSleep);
		}
		aCpt++;
		ELISE_ASSERT(aCpt<aCptMax, (string("FileLogMM3d: cannot open file for writing in [") + aDir + "]").c_str());
	}
	return aRes;
}

#include <ctime>

void LogTime(FILE * aFp, const std::string & aMes)
{

	time_t rawtime;
	struct tm * timeinfo;

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	fprintf(aFp, " PID : %d ;   %s %s", mm_getpid(), aMes.c_str(), asctime(timeinfo));
}

void LogIn(int  argc, char **  argv, const std::string & aDir, int aNumArgDir)
{
	if (!DOLOG_MM3d) return;
	FILE * aFp = FileLogMM3d(aDir);

	fprintf(aFp, "=================================================================\n");
	for (int aK = 0; aK< argc; aK++)
	{
		// MPD : je l'avais deja fait il y a 15 jours, ai pas du commite !!!!  Ca facilite copier-coller sur commande
		fprintf(aFp, "\"%s\" ", argv[aK]);
	}
	fprintf(aFp, "\n");
	LogTime(aFp, "[Beginning at ]");

	fclose(aFp);
}

void LogOut(int aRes, const std::string & aDir)
{
	if (!DOLOG_MM3d) return;

	FILE * aFp = FileLogMM3d(aDir);
	std::string aMes;
	if (aRes == 0)
		aMes = "[Ending correctly at]";
	else
		aMes = std::string("[Failing with code ") + ToString(aRes) + " at ]";
	LogTime(aFp, aMes);
	fclose(aFp);
}

class cCmpMMCom
{
public:

	cCmpMMCom() {}

	// Comparison; not case sensitive.
	bool operator ()(const cMMCom & aArg0, const cMMCom & aArg1)
	{
		string first = aArg0.mName;
		string second = aArg1.mName;

		unsigned int i = 0;
		while ((i < first.length()) && (i < second.length()))
		{
			if (tolower(first[i]) < tolower(second[i])) return true;
			else if (tolower(first[i]) > tolower(second[i])) return false;
			i++;
		}

		if (first.length() < second.length()) return true;
		else return false;
	}
};

extern int TiepTriFar_Main(int argc, char ** argv);
extern int PHO_MI_main(int, char **);
extern int TiePByMesh_main(int, char **);
extern int MeshProjOnImg_main(int, char **);
extern int CCL_main(int, char **);
extern int ReprojImg_main(int, char **);
extern int TestRegEx_main(int, char **);
extern int PatFromOri_main(int argc, char ** argv);
extern int GenFilePairs_main(int argc, char ** argv);
extern int CleanPatByOri_main(int argc, char ** argv);
extern int InitOriLinear_main(int, char **);
extern int ExtractMesure2D_main(int, char **);
extern int ExtractAppui3D_main(int, char **);
extern int TestElParseDir_main(int, char **);
extern int Kugelhupf_main(int, char **);
extern int FFTKugelhupf_main(int, char **);
extern int MasqMaker_main(int, char **);
extern int SimplePredict_main(int, char **);
extern int ProjImPtOnOtherImages_main(int, char **);
extern int schnaps_main(int, char **);
extern int mergeHomol_main(int, char **);
extern int Zlimit_main(int, char **);
extern int Homol2GND_main(int, char **);
extern int SimpleFusionCarte_main(int, char **);


extern int CPP_Martini_main(int, char **);
extern int CPP_MartiniGin_main(int, char **);

extern int CPP_SetExif(int argc, char **argv);
extern int CPP_SetGpsExif(int argc, char **argv);
extern int GenerateAppLiLiaison_main(int argc, char **argv);
extern int TestNameCalib_main(int argc, char ** argv);
extern int Init3App_Main(int argc, char ** argv);

extern int  CPP_CmpOriCam_main(int argc, char** argv);


extern int CPP_ConvertBundleGen(int argc, char ** argv);

extern int AlphaGet27_main(int argc, char ** argv);

extern int mergeSOMAF_main(int argc, char ** argv);


int TiePMS_main(int argc, char ** argv);
int TiePLine_main(int argc, char ** argv);
int TiePAll_main(int argc, char ** argv);

int  OneReechFid_main(int argc, char ** argv);

int TNR_main(int argc, char ** argv);
int Apero2NVM_main(int argc, char ** argv);

int Vino_Main(int argc, char ** argv);
int XifDate2Txt_main(int argc, char ** argv);
int XifGps2Xml_main(int argc, char ** argv);
int XifGps2Txt_main(int argc, char ** argv);
int CPP_GCP2D3D2Xml_main(int argc, char ** argv);
int RedTieP_main(int argc, char **argv);
int OriRedTie_main(int argc, char **argv);
int HomFusionPDVUnik_main(int argc, char **argv);
int TestDistM2C_main(int argc, char ** argv);
int TestDistortion_main(int argc, char ** argv);

int Blinis_main(int argc, char ** argv);
int OrientFromBlock_main(int argc, char ** argv);
int Contrast_main(int argc, char ** argv);
int Nikrup_main(int argc, char ** argv);
int TournIm_main(int argc,char ** argv);


int TestCamRPC(int argc, char** argv);
int TestBundleInter_main(int argc, char ** argv);
int GenerateBorderCam_main(int argc, char ** argv);

int Ratafia_Main(int argc, char ** argv);

int FusionDepl_Main(int argc, char ** argv);

int BasculeRobuste_main(int argc, char ** argv);
int CPP_CalcMapAnalitik(int argc, char** argv);
int CPP_CalcMapXYT(int argc, char** argv);
int CPP_ReechImMap(int argc, char** argv);
int CPP_DenseMapToHom(int argc, char** argv);
int CPP_CmpDenseMap(int argc, char** argv);
int CPP_FermDenseMap(int argc, char** argv);
int CPP_SampleMap2D(int argc, char** argv);
int ScalePat_main(int argc, char** argv);
int CPP_MakeMapEvolOfT(int argc, char ** argv);
int CPP_PolynOfImage(int argc, char ** argv);
int CPP_PolynOfImageStd(int argc, char ** argv);
int MSD_main(int argc,char ** argv);

int GCP_Fusion(int argc, char ** argv);

int SimuLib_Main(int argc, char ** argv);

extern int CPP_ProfilImage(int argc, char ** argv);

extern int ExtractRaw_main(int argc, char ** argv);
extern int CPP_Extrac_StdRaw(int argc, char ** argv);

extern int CPP_MMRename(int argc, char**argv);
extern int  CPP_EditSet(int argc, char**argv);

int CPP_MMHelp(int argc, char ** argv);
int ConvertOriCalib_main(int argc, char ** argv);

int DroneFootPrint(int argc,char ** argv);

int Image_Vide(int argc,char ** argv);
int  PPMD_MatEss2Orient(int argc,char ** argv);

int GrapheStereopolis_main(int argc,char ** argv);
int CheckGCPStereopolis_main(int argc,char ** argv);
int AnalyseTrajStereopolis_main(int argc,char ** argv);
int CPP_CalcImScale(int argc,char** argv);




std::vector<cMMCom>&  AddLib(std::vector<cMMCom> & aVC, const std::string & aLib)
{
	for (int aK = 0; aK<int(aVC.size()); aK++)
		aVC[aK].mLib = aLib;
	return aVC;
}

int CPP_AimeApprent(int argc, char ** argv);
int CPP_StatPHom(int argc, char ** argv);
int CPP_PHom_RenameRef(int argc, char ** argv);
int CPP_PHom_ApprentBinaire(int argc, char ** argv);
int CPP_FitsMatch1Im(int argc, char ** argv);
int HackToF(int argc,char ** argv);
int CPP_GenPrime(int argc,char ** argv);
int CPP_GIMMI_main(int argc,char ** argv);
int HomolFromProfEtPx_main(int argc,char ** argv);
int Line2Line_main(int argc,char ** argv);
int CoronaRessample_main(int argc,char ** argv);
int DivFilters_main(int argc,char ** argv);
int AnalysePxFrac_Main(int argc,char ** argv);

int CPP_ConvertRTKlib2Micmac(int argc,char ** argv);
int CPP_YannViewIntersect(int argc,char ** argv);
int CPP_YannEstimHomog(int argc,char ** argv);
int CPP_YannApplyHomog(int argc,char ** argv);
int CPP_YannInvHomolHomog(int argc,char ** argv);
int CPP_YannExcludeSats(int argc,char ** argv);
int CPP_YannSetTimestamps(int argc,char ** argv);
int CPP_YannSkyMask(int argc,char ** argv);
int CPP_YannExport2Colmap(int argc,char ** argv);
int CPP_YannScript(int argc,char ** argv);

int CPP_GCP2MeasureLine2D(int argc,char ** argv);
int CPP_MeasureL2D2L3D(int argc,char ** argv);
int CPP_L3D2Ply(int argc,char ** argv);
int CPP_DebugAI4GeoMasq (int argc,char ** argv);
int CPP_MMBasic4IGeo(int argc,char ** argv);
int CPP_MMBasicTestDeep(int argc,char ** argv);

const std::vector<cMMCom> & getAvailableCommands()
{
	static std::vector<cMMCom> aRes;
	if (aRes.empty())
	{
		aRes.push_back(cMMCom("InterView",CPP_YannViewIntersect, "Field of view intersections for tie points computation "));
		aRes.push_back(cMMCom("EstimHomog",CPP_YannEstimHomog, "Homographie estimation from GCPs and image measurements "));
		aRes.push_back(cMMCom("ApplyHomog",CPP_YannApplyHomog, "Homographie application on images "));
		aRes.push_back(cMMCom("InvHomolHomog",CPP_YannInvHomolHomog, "Homographie application on images "));
		aRes.push_back(cMMCom("PPMD_MatEss2Orient", PPMD_MatEss2Orient, "transform essential matrix as list of orient "));
		aRes.push_back(cMMCom("Help", CPP_MMHelp, "Help on existing MicMac commands "));
		aRes.push_back(cMMCom("BAR", BasculeRobuste_main, "Bascule robutse "));

		aRes.push_back(cMMCom("CalcMapAnalytik", CPP_CalcMapAnalitik, "Compute map2d between images using various model "));
		aRes.push_back(cMMCom("CalcMapXYT", CPP_CalcMapXYT, "Compute map2d evol of T "));
		aRes.push_back(cMMCom("CalcMapOfT", CPP_MakeMapEvolOfT, "Compute value of map evol for a given T "));
		aRes.push_back(cMMCom("ReechImMap", CPP_ReechImMap, "Resample image using 2d map "));
		aRes.push_back(cMMCom("DMatch2Hom", CPP_DenseMapToHom, "Dense matching 2 homologues "));
		aRes.push_back(cMMCom("CmpDenseMap", CPP_CmpDenseMap, "comparison of dense map "));
		aRes.push_back(cMMCom("FermDenseMap", CPP_FermDenseMap, "Consistency by closing on dense maps "));
		aRes.push_back(cMMCom("SampleMap", CPP_SampleMap2D, "Test values of maps on few points"));

		aRes.push_back(cMMCom("FusionDepl", FusionDepl_Main, "Fusion carte de deplacement "));
		aRes.push_back(cMMCom("TestPbRPC", TestCamRPC, "Test possible Problems on RPC ", cArgLogCom(2)));
		aRes.push_back(cMMCom("TestBundleInter", TestBundleInter_main, "Block Initialisation "));
		aRes.push_back(cMMCom("Blinis", Blinis_main, "Block Initialisation ", cArgLogCom(2)));
		aRes.push_back(cMMCom("OriFromBlock", OrientFromBlock_main, "Use Rigid Block to complete orientation ", cArgLogCom(2)));
		aRes.push_back(cMMCom("ContrastFilter", Contrast_main, "Some contrast filtering "));
        aRes.push_back(cMMCom("Nikrup", Nikrup_main,/*(*/ "Generik image filter, using invert polish like notation ;-) ",cArgLogCom(3)));
		aRes.push_back(cMMCom("Turn90Im", TournIm_main, "Turn image of 90 degre"));
		aRes.push_back(cMMCom("RedTieP", RedTieP_main, "Test tie points filtering "));
		aRes.push_back(cMMCom("OriRedTieP", OriRedTie_main, "Tie points filtering, using Martini results "));
		aRes.push_back(cMMCom("Vino", Vino_Main, "Image Viewer"));
		aRes.push_back(cMMCom("TripleSec", TNR_main, "Test Non Regression"));
		aRes.push_back(cMMCom("Ratafia", Ratafia_Main, "Tie point reduction", cArgLogCom(2)));
		aRes.push_back(cMMCom("TiePMS", TiePMS_main, " matches points of interest of two images"));
		aRes.push_back(cMMCom("TiePLine", TiePLine_main, " matches points of interest of two images"));
		aRes.push_back(cMMCom("TiePAll", TiePAll_main, " matches points of interest of two images"));

		aRes.push_back(cMMCom("Ann", Ann_main, " matches points of interest of two images"));
		aRes.push_back(cMMCom("AperiCloud", AperiCloud_main, " Visualization of camera in ply file", cArgLogCom(2)));
		aRes.push_back(cMMCom("Apero", Apero_main, " Compute external and internal orientations"));
		aRes.push_back(cMMCom("Arsenic", Arsenic_main, " IN DEV : Radiometric equalization from tie points"));
		aRes.push_back(cMMCom("Digeo", Digeo_main, " In development- Will compute tie points "));
        aRes.push_back(cMMCom("MSD", MSD_main, " In development- Will compute tie points "));
		aRes.push_back(cMMCom("AperoChImSecMM", AperoChImMM_main, " Select secondary images for MicMac", cArgLogCom(2)));
		aRes.push_back(cMMCom("Apero2PMVS", Apero2PMVS_main, " Convert Orientation from Apero-Micmac workflow to PMVS format"));
		aRes.push_back(cMMCom("Apero2Meshlab", Apero2Meshlab_main, "Convert Orientation from Apero-Micmac workflow to a meshlab-compatible format"));
		aRes.push_back(cMMCom("Bascule", Bascule_main, " Generate orientations coherent with some physical information on the scene", cArgLogCom(2)));
		aRes.push_back(cMMCom("BatchFDC", BatchFDC_main, " Tool for batching a set of commands"));
		aRes.push_back(cMMCom("Campari", Campari_main, " Interface to Apero, for compensation of heterogeneous measures", cArgLogCom(2)));
		aRes.push_back(cMMCom("ChgSysCo", ChgSysCo_main, " Change coordinate system of orientation", cArgLogCom(2)));
		aRes.push_back(cMMCom("CmpCalib", CmpCalib_main, " Compare two  calibrations"));
		aRes.push_back(cMMCom("CmpOri", CPP_CmpOriCam_main, " Compare two sets of orientation"));
		aRes.push_back(cMMCom("ConvertCalib", ConvertCalib_main, " Conversion of calibration from one model 2 the other"));
		aRes.push_back(cMMCom("ConvertOriCalib", ConvertOriCalib_main, "Convert external orientation with new internal orientation "));
		aRes.push_back(cMMCom("ReprojImg", ReprojImg_main, " Reproject an image into geometry of another"));

		aRes.push_back(cMMCom("TestRegEx", TestRegEx_main, " Test regular expression"));
		aRes.push_back(cMMCom("PatFromOri", PatFromOri_main, "Get pattern of images from Ori folder"));
		aRes.push_back(cMMCom("GenPairsFile", GenFilePairs_main, "Generate pairs files between one image and a pattern"));
		aRes.push_back(cMMCom("CleanPatByOri", CleanPatByOri_main, "Clean a pattern of image by Ori-XXX folder"));
		aRes.push_back(cMMCom("InitOriLinear", InitOriLinear_main, " Initialize orientation for linear acquisition"));
		aRes.push_back(cMMCom("PHO_MI", PHO_MI_main, " Filter homologue points from initial orientation to reduce number of observations"));
		aRes.push_back(cMMCom("TiePByMesh", TiePByMesh_main, " Raffiner pts homologue par mesh"));
		aRes.push_back(cMMCom("MeshProjOnImg", MeshProjOnImg_main, " Reproject mesh on image"));
		aRes.push_back(cMMCom("ExtractMesure2D", ExtractMesure2D_main, " Extract points from a 2D measures xml file"));
		aRes.push_back(cMMCom("ExtractAppui3D", ExtractAppui3D_main, " Extract points from a 3D appui points xml file"));
		aRes.push_back(cMMCom("Kugelhupf", Kugelhupf_main, " Semi-automatic fiducial points determination"));
		aRes.push_back(cMMCom("FFTKugelhupf", FFTKugelhupf_main, " Version of Kugelhupf using FFT, expecetd faster when it works (if ever ...)"));
		aRes.push_back(cMMCom("SimplePredict", SimplePredict_main, " Project ground points on oriented cameras"));
		aRes.push_back(cMMCom("Schnaps", schnaps_main, " Reduction of homologue points in image geometry"));
		aRes.push_back(cMMCom("MergeHomol", mergeHomol_main, " Merge Homol dir"));
		aRes.push_back(cMMCom("Zlimit", Zlimit_main, " Crop Depth image (or DEM) in Z"));
		aRes.push_back(cMMCom("Homol2GND", Homol2GND_main, " Creates fake ground points for aerotriangulation wedge"));
		aRes.push_back(cMMCom("MasqMaker", MasqMaker_main, " Create Mask form image values"));
		aRes.push_back(cMMCom("cod", cod_main, " Do some stuff"));
		aRes.push_back(cMMCom("vic", vicod_main, " Do some stuff"));
		aRes.push_back(cMMCom("genmail", genmail_main, " Do some stuff"));
		aRes.push_back(cMMCom("CreateEpip", CreateEpip_main, " Create epipolar images",cArgLogCom(2)));
		aRes.push_back(cMMCom("CoherEpip", CoherEpi_main, " Test coherence between conjugate epipolar depth-map"));
		aRes.push_back(cMMCom("Dequant", Dequant_main, " Tool for dequantifying an image"));
		aRes.push_back(cMMCom("Devlop", Devlop_main, " Do some stuff"));
		aRes.push_back(cMMCom("TifDev", TiffDev_main, " Develop raw-jpg-tif, in suitable tiff file"));
        aRes.push_back(cMMCom("Drunk", Drunk_main, " Images distortion removing tool"));
		aRes.push_back(cMMCom("ElDcraw", ElDcraw_main, " Do some stuff"));
		aRes.push_back(cMMCom("GCPBascule", GCPBascule_main, " Relative to absolute using GCP", cArgLogCom(2)));
		aRes.push_back(cMMCom("GCPCtrl", GCPCtrl_main, " Control accuracy with GCP", cArgLogCom(2)));
		aRes.push_back(cMMCom("GCPMerge", GCP_Fusion, " Merging of different GCP files", cArgLogCom(2)));
		aRes.push_back(cMMCom("GCPVisib", GCPVisib_main, " Print a list of GCP visibility in images"));

		aRes.push_back(cMMCom("CenterBascule", CentreBascule_main, " Relative to absolute using embedded GPS", cArgLogCom(2)));

		aRes.push_back(cMMCom("GrapheHom", GrapheHom_main, "Compute XML-Visibility graph from approximate orientation", cArgLogCom(3)));
		aRes.push_back(cMMCom("GrapheStereopolis", GrapheStereopolis_main,"Compute Pair of Image for Stereopolis", cArgLogCom(2)));
		aRes.push_back(cMMCom("CheckGCPStereopolis", CheckGCPStereopolis_main,"Check GCP with strategy optimized for Stereopolis-like acquisition", cArgLogCom(2)));
		aRes.push_back(cMMCom("CalcImScale", CPP_CalcImScale,"Calculate scale of image", cArgLogCom(2)));


		aRes.push_back(cMMCom("AnalyseTrajStereopolis", AnalyseTrajStereopolis_main,"Analyse trajectory of Stereopolis-like acquisition", cArgLogCom(2)));

		aRes.push_back(cMMCom("GCP2d3dConvert", CPP_GCP2D3D2Xml_main, "Convert GCP 2d and 3d in Txt 2 XML", cArgLogCom(3)));
		aRes.push_back(cMMCom("GCPConvert", GCP_Txt2Xml_main, "Convert GCP from Txt 2 XML", cArgLogCom(3)));
		aRes.push_back(cMMCom("OriConvert", Ori_Txt2Xml_main, "Convert Orientation from Txt 2 XML", cArgLogCom(3)));
		aRes.push_back(cMMCom("OriExport", OriExport_main, "Export orientation from XML to XML or TXT with specified convention", cArgLogCom(3)));
		aRes.push_back(cMMCom("Apero2NVM", Apero2NVM_main, "Matthieu Moneyrond's convertor to VSfM, MVE, SURE, MeshRecon ", cArgLogCom(3)));
        aRes.push_back(cMMCom("XifDate2Txt", XifDate2Txt_main, "Export embedded EXIF Date data 2 Txt", cArgLogCom(1)));
		aRes.push_back(cMMCom("XifGps2Xml", XifGps2Xml_main, "Create MicMac-Xml struct from GPS embedded in EXIF", cArgLogCom(2)));
		aRes.push_back(cMMCom("XifGps2Txt", XifGps2Txt_main, "Export embedded EXIF GPS data 2 Txt", cArgLogCom(2)));
		aRes.push_back(cMMCom("GenXML2Cpp", GenXML2Cpp_main, " Do some stuff"));
		aRes.push_back(cMMCom("GenCode", GenCode_main, " Do some stuff"));
		aRes.push_back(cMMCom("GrShade", GrShade_main, " Compute shading from depth image"));
		aRes.push_back(cMMCom("LumRas", LumRas_main, " Compute image mixing with raking light", cArgLogCom(2)));


		aRes.push_back(cMMCom("StackFlatField", EstimFlatField_main, "Basic Flat Field estimation by image stacking"));
		aRes.push_back(cMMCom("PolynOfImage", CPP_PolynOfImage, "Approximate image by polynom"));
		aRes.push_back(cMMCom("PolynOfImageV2", CPP_PolynOfImageStd, "Approximate image by polynom ver2"));
		aRes.push_back(cMMCom("Impaint", Impainting_main, "Basic Impainting"));
		aRes.push_back(cMMCom("Gri2Bin", Gri2Bin_main, " Do some stuff"));
		aRes.push_back(cMMCom("MakeGrid", MakeGrid_main, " Generate orientations in a grid format"));
		aRes.push_back(cMMCom("Malt", Malt_main, " Simplified matching (interface to MicMac)", cArgLogCom(3)));
		aRes.push_back(cMMCom("CASA", CASA_main, " Analytic Surface Estimation", cArgLogCom(2)));



		aRes.push_back(cMMCom("MMByP", MMByPair_main, " Matching By Pair of images", cArgLogCom(2)));
		aRes.push_back(cMMCom("MM1P", MMOnePair_main, " Matching One Pair of images", cArgLogCom(2)));
		aRes.push_back(cMMCom("MMAI4Geo", CPP_MMBasic4IGeo," Basic Matching for AI4Geo Satellite", cArgLogCom(2)));
		aRes.push_back(cMMCom("MMTestMMVII",CPP_MMBasicTestDeep,"Basic Matching for insert in testing MMVII ", cArgLogCom(2)));

		aRes.push_back(cMMCom("ChantierClip", ChantierClip_main, " Clip Chantier", cArgLogCom(2)));
		aRes.push_back(cMMCom("ClipIm", ClipIm_main, " Clip Chantier", cArgLogCom(2)));


		aRes.push_back(cMMCom("MapCmd", MapCmd_main, " Transforms a command working on a single file in a command working on a set of files"));
		aRes.push_back(cMMCom("Ori2Xml", Ori2XML_main, "Convert \"historical\" Matis'Ori format to xml "));
		aRes.push_back(cMMCom("MergePly", MergePly_main, " Merge ply files"));
		aRes.push_back(cMMCom("MICMAC", MICMAC_main, " Computes image matching from oriented images"));
		aRes.push_back(cMMCom("MMPyram", MMPyram_main, " Computes pyram for micmac (internal use)", cArgLogCom(2)));

		aRes.push_back(cMMCom("MMCalcSzWCor", CalcSzWCor_main, " Compute Image of Size of correlation windows (Atomic tool, for adaptive window in geom image)", cArgLogCom(2)));
		aRes.push_back(cMMCom("MpDcraw", MpDcraw_main, " Interface to dcraw"));

		aRes.push_back(cMMCom("MMTestOrient", MMTestOrient_main, " Tool for testing quality of orientation",cArgLogCom(2)));
		aRes.push_back(cMMCom("MMHomCorOri", MMHomCorOri_main, " Tool to compute homologues for correcting orientation in epip matching"));
		aRes.push_back(cMMCom("MMInitialModel", MMInitialModel_main, " Initial Model for MicMac ")); //  ,cArgLogCom(2)));
		aRes.push_back(cMMCom("MMTestAllAuto", MMAllAuto_main, " Full automatic version for 1 view point, test mode ", cArgLogCom(2)));
		aRes.push_back(cMMCom("MM2DPosSism", MM2DPostSism_Main, " Simplified interface for post 2D post sismic deformation ", cArgLogCom(2)));
		aRes.push_back(cMMCom("DistPxFrac", AnalysePxFrac_Main, "Compute distribution of fractional part of paralax ", cArgLogCom(2)));
		aRes.push_back(cMMCom("MMMergeCloud", MM_FusionNuage_main, " Merging of low resol cloud, in preparation 2 MicMac ", cArgLogCom(2)));

		aRes.push_back(cMMCom("MergeDepthMap", FusionCarteProf_main, " Merging of individual, stackable, depth maps "));
		aRes.push_back(cMMCom("SMDM", SimpleFusionCarte_main, " Simplified Merging of individual, stackable, depth maps "));
		aRes.push_back(cMMCom("MyRename", MyRename_main, " File renaming using posix regular expression "));
		aRes.push_back(cMMCom("MMRename", CPP_MMRename, " Renaming a MicMac dataset respecting MicMac convention "));


		aRes.push_back(cMMCom("Genere_Header_TiffFile", Genere_Header_TiffFile_main, " Generate Header for internal tiling format "));


		aRes.push_back(cMMCom("Nuage2Ply", Nuage2Ply_main, " Convert depth map into point cloud"));
		aRes.push_back(cMMCom("NuageBascule", NuageBascule_main, " To Change geometry of depth map "));
		aRes.push_back(cMMCom("Nuage2Homol", Nuage2Homol_main, " Create Tie Points from a depth map"));
		aRes.push_back(cMMCom("Txt2Dat", Txt2Dat_main, " Convert an ascii tie point file to binary"));



		aRes.push_back(cMMCom("Pasta", Pasta_main, " Compute external calibration and radial basic internal calibration"));
		aRes.push_back(cMMCom("PastDevlop", PastDevlop_main, " Do some stuff"));
		aRes.push_back(cMMCom("Pastis", Pastis_main, " Tie points detection"));
		//aRes.push_back(cMMCom("Poisson",Poisson_main," Mesh Poisson reconstruction by M. Khazdan"));
		aRes.push_back(cMMCom("Porto", Porto_main, " Generates a global ortho-photo"));
		aRes.push_back(cMMCom("Prep4masq", Prep4masq_main, " Generates files for making Masks (if SaisieMasq unavailable)"));
		aRes.push_back(cMMCom("Reduc2MM", Reduc2MM_main, " Do some stuff"));
		aRes.push_back(cMMCom("ReducHom", ReducHom_main, " Do some stuff"));
		aRes.push_back(cMMCom("RepLocBascule", RepLocBascule_main, " Tool to define a local repair without changing the orientation", cArgLogCom(2)));
		aRes.push_back(cMMCom("SBGlobBascule", SBGlobBascule_main, " Tool for 'scene based global' bascule", cArgLogCom(2)));
		aRes.push_back(cMMCom("HomolFilterMasq", HomFilterMasq_main, " Tool for filter homologous points according to masq", cArgLogCom(2)));
		aRes.push_back(cMMCom("HomolMergePDVUnik", HomFusionPDVUnik_main, " Tool for merge homologous point from unik point of view", cArgLogCom(2)));


		aRes.push_back(cMMCom("ScaleIm", ScaleIm_main, " Tool for image scaling"));
		aRes.push_back(cMMCom("ScalePat", ScalePat_main, " Tool for pattern scaling"));
		aRes.push_back(cMMCom("StatIm", StatIm_main, " Tool for basic stat on an image"));
		aRes.push_back(cMMCom("ConvertIm", ConvertIm_main, " Tool for convertion inside tiff-format"));
		aRes.push_back(cMMCom("PanelIm", MakePlancheImage_main, "Tool for creating a panel of images "));
		aRes.push_back(cMMCom("ScaleNuage", ScaleNuage_main, " Tool for scaling internal representation of point cloud"));
		aRes.push_back(cMMCom("Sift", Sift_main, " Tool for extracting points of interest using Lowe's SIFT method"));
		aRes.push_back(cMMCom("SysCoordPolyn", SysCoordPolyn_main, " Tool for creating a polynomial coordinate system from a set of known pair of coordinate"));

        aRes.push_back(cMMCom("OldTapas", Tapas_main, " Old Tapas", cArgLogCom(3)));
        aRes.push_back(cMMCom("Tapas", New_Tapas_main, "Interface to Apero to compute external and internal orientations", cArgLogCom(3)));
		aRes.push_back(cMMCom("NewTapas", New_Tapas_main, "Replace OldTapas - now same as Tapas", cArgLogCom(3)));

		aRes.push_back(cMMCom("Tapioca", Tapioca_main, " Interface to Pastis for tie point detection and matching", cArgLogCom(3)));

		aRes.push_back(cMMCom("Tarama", Tarama_main, " Compute a rectified image", cArgLogCom(2)));
		aRes.push_back(cMMCom("Martini", CPP_Martini_main, " New orientation initialisation (uncomplete, still in dev...) ", cArgLogCom(2)));
		aRes.push_back(cMMCom("MartiniGin", CPP_MartiniGin_main, " New orientation initialisation (uncomplete, still in dev...) ", cArgLogCom(2)));

		aRes.push_back(cMMCom("Tawny", Tawny_main, " Interface to Porto to generate ortho-image", cArgLogCom(2, "../")));
		// aRes.push_back(cMMCom("Tawny",Tawny_main," Interface to Porto to generate ortho-image"));
		aRes.push_back(cMMCom("Tequila", Tequila_main, " Texture mesh"));
		aRes.push_back(cMMCom("TiPunch", TiPunch_main, " Compute mesh"));
		aRes.push_back(cMMCom("TestCam", TestCam_main, " Test camera orientation convention"));
		aRes.push_back(cMMCom("TestDistM2C", TestDistM2C_main, " Basic Test for problematic camera "));
		aRes.push_back(cMMCom("TestDistortion", TestDistortion_main, " Basic Test of distortion formula "));
		aRes.push_back(cMMCom("TestChantier", TestChantier_main, " Test global acquisition"));

		aRes.push_back(cMMCom("TestKey", TestSet_main, " Test Keys for Sets and Assoc"));
		aRes.push_back(cMMCom("Recover", Recover_Main, " Basic tool for recover files"));
		aRes.push_back(cMMCom("TestNameCalib", TestNameCalib_main, " Test Name of calibration"));
		aRes.push_back(cMMCom("TestMTD", TestMTD_main, " Test meta data of image"));
		aRes.push_back(cMMCom("TestCmds", TestCmds_main, " Test MM3D commands on micmac_data sets"));

		aRes.push_back(cMMCom("tiff_info", tiff_info_main, " Tool for giving information about a tiff file"));
		aRes.push_back(cMMCom("to8Bits", to8Bits_main, " Tool for converting 16 or 32 bit image in a 8 bit image."));
		aRes.push_back(cMMCom("Vodka", Vignette_main, " IN DEV : Compute the vignette correction parameters from tie points", cArgLogCom(1)));
		aRes.push_back(cMMCom("mmxv", mmxv_main, " Interface to xv (due to problem in tiff lib)"));
		aRes.push_back(cMMCom("CmpIm", CmpIm_main, " Basic tool for images comparison"));
		aRes.push_back(cMMCom("ImMire", GenMire_main, " For generation of some synthetic calibration image"));
		aRes.push_back(cMMCom("ImRandGray", GrayTexture_main, " Generate Random Gray Textured Images"));
		aRes.push_back(cMMCom("Undist", Undist_main, " Tool for removing images distortion"));

		aRes.push_back(cMMCom("CheckDependencies", CheckDependencies_main, " check dependencies to third-party tools"));
		aRes.push_back(cMMCom("VV", VideoVisage_main, " A very simplified tool for 3D model of visage out of video, just for fun"));

		aRes.push_back(cMMCom("XYZ2Im", XYZ2Im_main, " tool to transform a 3D point (text file) to their 2D proj in cam or cloud"));
		aRes.push_back(cMMCom("Im2XYZ", Im2XYZ_main, " tool to transform a 2D point (text file) to their 3D cloud homologous"));
		aRes.push_back(cMMCom("SplitMPO", SplitMPO_main, "tool to develop MPO stereo format in pair of images"));
		aRes.push_back(cMMCom("ExtractRaw", ExtractRaw_main, "Convert raw image with XML descriptor to tiff "));
		aRes.push_back(cMMCom("ExtractStdRaw", CPP_Extrac_StdRaw, "Convert raw image with predefined XML descriptor (in XML_MicMac/DataBaseCameraRaw) to tiff "));


		aRes.push_back(cMMCom("Sake", Sake_main, " Simplified MicMac interface for satellite images", cArgLogCom(3)));
		aRes.push_back(cMMCom("SateLib", SateLib_main, " Library of satellite images meta-data handling - early work in progress!"));
		aRes.push_back(cMMCom("SimuLib", SimuLib_Main, " Library (almost empty now)  for simulating"));
		aRes.push_back(cMMCom("XLib", XLib_Main, " Xeres Lib - early work in progress!"));

		aRes.push_back(cMMCom("AlphaGet27", AlphaGet27_main, " Tool for relative positioning of objects on images"));
		aRes.push_back(cMMCom("MergeSOMAF", mergeSOMAF_main, " Tool for merging SetOfMesureAppuisFlottants XMLs"));

#if ELISE_QT
		aRes.push_back(cMMCom("SaisieAppuisInitQT", SaisieAppuisInitQT_main, " Interactive tool for initial capture of GCP"));
		aRes.push_back(cMMCom("SaisieAppuisPredicQT", SaisieAppuisPredicQT_main, " Interactive tool for assisted capture of GCP"));
		aRes.push_back(cMMCom("SaisieBascQT", SaisieBascQT_main, " Interactive tool to capture information on the scene"));
		aRes.push_back(cMMCom("SaisieCylQT", SaisieCylQT_main, " Interactive tool to capture information on the scene for cylinders"));
		aRes.push_back(cMMCom("SaisieMasqQT", SaisieMasqQT_main, " Interactive tool to capture masq"));
		aRes.push_back(cMMCom("SaisieBoxQT", SaisieBoxQT_main, " Interactive tool to capture 2D box"));
		aRes.push_back(cMMCom("GIMMI", CPP_GIMMI_main, "Graphical Interface MMI"));
#endif

#if (ELISE_X11)
		aRes.push_back(cMMCom("MPDtest", MPDtest_main, " My own test"));
		aRes.push_back(cMMCom("DebugAI4GeoMasq", CPP_DebugAI4GeoMasq, " For debuging masq problem appeard with AI4Geo"));
		aRes.push_back(cMMCom("SaisieAppuisInit", SaisieAppuisInit_main, " Interactive tool for initial capture of GCP", cArgLogCom(2)));
		aRes.push_back(cMMCom("SaisieAppuisPredic", SaisieAppuisPredic_main, " Interactive tool for assisted capture of GCP"));
		aRes.push_back(cMMCom("SaisieBasc", SaisieBasc_main, " Interactive tool to capture information on the scene"));
		aRes.push_back(cMMCom("SaisieCyl", SaisieCyl_main, " Interactive tool to capture information on the scene for cylinders"));
		aRes.push_back(cMMCom("SaisieMasq", SaisieMasq_main, " Interactive tool to capture masq"));
		aRes.push_back(cMMCom("SaisiePts", SaisiePts_main, " Tool to capture GCP (low level, not recommended)"));
		aRes.push_back(cMMCom("SEL", SEL_main, " Tool to visualize tie points"));
		aRes.push_back(cMMCom("MICMACSaisieLiaisons", MICMACSaisieLiaisons_main, " Low level version of SEL, not recommended"));

		aRes.push_back(cMMCom("GCP2MeasuresL2D", CPP_GCP2MeasureLine2D, " Convert a set of GCP in measure of 2D lines using convention NameLine_x with x={1,2}",cArgLogCom(2)));
		aRes.push_back(cMMCom("MeasuresL2D2L3D", CPP_MeasureL2D2L3D, " Convert a set of images measures of 2D lines to 3D lines in space",cArgLogCom(2)));
		aRes.push_back(cMMCom("L3D2Ply", CPP_L3D2Ply, " Convert a set of 3D lines in space to a ply file",cArgLogCom(2)));



		
		
#ifdef ETA_POLYGON
		aRes.push_back(cMMCom("HackToF", HackToF,"Hack ToF format "));

		//Etalonnage polygone
		aRes.push_back(cMMCom("Compens", Compens_main, " Do some stuff"));
		aRes.push_back(cMMCom("CatImSaisie", CatImSaisie_main, " Do some stuff"));
		aRes.push_back(cMMCom("CalibFinale", CalibFinale_main, " Compute Final Radial distortion model"));
		aRes.push_back(cMMCom("CalibInit", CalibInit_main, " Compute Initial Radial distortion model"));
		aRes.push_back(cMMCom("ConvertPolygone", ConvertPolygone_main, " Do some stuff"));
		aRes.push_back(cMMCom("PointeInitPolyg", PointeInitPolyg_main, " Do some stuff"));
		aRes.push_back(cMMCom("RechCibleDRad", RechCibleDRad_main, " Do some stuff"));
		aRes.push_back(cMMCom("RechCibleInit", RechCibleInit_main, " Do some stuff"));
		aRes.push_back(cMMCom("ScriptCalib", ScriptCalib_main, " Do some stuff"));

#endif

#endif
		aRes.push_back(cMMCom("TestLib", SampleLibElise_main, " To call the program illustrating the library"));
		aRes.push_back(cMMCom("FieldDep3d", ChamVec3D_main, " To export results of matching as 3D shifting"));
		aRes.push_back(cMMCom("HomProfPx", HomolFromProfEtPx_main, " Export pixel correspondences from Px1 et Px2"));
		aRes.push_back(cMMCom("L2L", Line2Line_main, " Project row/column in one image to another"));
		aRes.push_back(cMMCom("CoronaCorrect", CoronaRessample_main, " Correct some geometric defautl of corona images"));
		aRes.push_back(cMMCom("DivFilter", DivFilters_main, " Test divers filters"));
		aRes.push_back(cMMCom("SupMntIm", SupMntIm_main, " Tool for superposition of Mnt Im & level curve"));

		aRes.push_back(cMMCom("MMXmlXif", MakeMultipleXmlXifInfo_main, " Generate Xml from Xif (internal use mainly)"));
		aRes.push_back(cMMCom("Init11P", Init11Param_Main, " Init Internal & External from GCP using 11-parameters algo"));
		aRes.push_back(cMMCom("Aspro", Init3App_Main, " Init  External orientation of calibrated camera from GCP "));
		aRes.push_back(cMMCom("Genepi", GenerateAppLiLiaison_main, " Generate 3D/2d synthetical points from orientation"));


		aRes.push_back(cMMCom("DIV", Devideo_main, "Videos development (require ffmpeg)"));
		aRes.push_back(cMMCom("Liquor", Liquor_main, "Orientation specialized for linear acquisition"));
		aRes.push_back(cMMCom("Luxor", Luxor_main, "Orientation specialized for linear acquisition using a sliding window",cArgLogCom(2)));
		aRes.push_back(cMMCom("Morito", Morito_main, "Merge set of Orientations with common values"));
		aRes.push_back(cMMCom("Donuts", Donuts_main, "Cyl to Torus (Donuts like)"));
		aRes.push_back(cMMCom("C3DC", C3DC_main, "Automatic Matching from Culture 3D Cloud project"));
		aRes.push_back(cMMCom("PIMs", MPI_main, "Per Image Matchings"));
		aRes.push_back(cMMCom("PIMs2Ply", MPI2Ply_main, "Generate Ply from Per Image Matchings"));
		aRes.push_back(cMMCom("PIMs2Mnt", MPI2Mnt_main, "Generate Mnt from Per Image Matchings"));
		aRes.push_back(cMMCom("SAT4GEO", Sat3D_main, "Satellite 3D pipeline",cArgLogCom(2)));
		aRes.push_back(cMMCom("TiePHistoP", TiePHistoP_main, "Inter-date features extraction => historical images pipeline",cArgLogCom(2)));


		aRes.push_back(cMMCom("AllDev", DoAllDev_main, "Force development of all tif/xif file"));
		aRes.push_back(cMMCom("SetExif", CPP_SetExif, "Modification of exif file (requires exiv2)", cArgLogCom(2)));
		aRes.push_back(cMMCom("SetGpsExif", CPP_SetGpsExif, "Add GPS infos in images exif meta-data (requires exiv2)", cArgLogCom(2)));
		aRes.push_back(cMMCom("Convert2GenBundle", CPP_ConvertBundleGen, "Import RPC or other to MicMac format, for adjustment, matching ...", cArgLogCom(2)));
		aRes.push_back(cMMCom("ReSampFid", OneReechFid_main, "Resampling using one fiducial mark"));
		aRes.push_back(cMMCom("VisuRedHom", VisuResiduHom, " Create a visualisation of residual on tie points"));
		aRes.push_back(cMMCom("GenerateBorderCam", GenerateBorderCam_main, " Generate the polygone of image contour undistorded"));
		aRes.push_back(cMMCom("ProfilIm", CPP_ProfilImage, "Image profiling  2D->1D "));
		aRes.push_back(cMMCom("EditSet", CPP_EditSet, "Edition creation of a set of images/files"));
		aRes.push_back(cMMCom("StatPHom", CPP_StatPHom, "Stat on homologous point using orientation of 3D Model"));
		aRes.push_back(cMMCom("AimeApprent", CPP_AimeApprent, "Stat on homologous point using orientation of 3D Model"));
		aRes.push_back(cMMCom("PHom_RenameRef", CPP_PHom_RenameRef, "Rename Ref for PHom"));
		aRes.push_back(cMMCom("PHom_ApBin", CPP_PHom_ApprentBinaire, "Test Binary "));
		aRes.push_back(cMMCom("FitsMatch", CPP_FitsMatch1Im, "Test Match Images NewPHom "));
		aRes.push_back(cMMCom("GenPrime", CPP_GenPrime, "Generate prime "));

       aRes.push_back(cMMCom("DroneFootPrint",DroneFootPrint,"Draw footprint from image + orientation (drone) in PLY and QGIS format"));
   }


	cCmpMMCom CmpMMCom;
	std::sort(aRes.begin(), aRes.end(), CmpMMCom);

	return aRes;
}

class cSuggest
{
public:
	cSuggest(const std::string & aName, const std::string & aPat) :
		mName(aName),
		mPat(aPat),
		mAutom(mPat, 10)
	{
	}
	void Test(const cMMCom & aCom)
	{
		if (mAutom.Match(aCom.mLowName))
			mRes.push_back(aCom);
	}

	std::string          mName;
	std::string          mPat;
	cElRegex             mAutom;
	std::vector<cMMCom>  mRes;
};

int GenMain(int argc, char ** argv, const std::vector<cMMCom> & aVComs);

// =========================================================

//TestLib declarations
extern int  Sample_W0_main(int argc, char ** argv);
extern int  Sample_LSQ0_main(int argc, char ** argv);
extern int  Abdou_main(int argc, char ** argv);
extern int  Luc_main(int argc, char ** argv);
extern int  LucasChCloud_main(int argc, char ** argv);
extern int  ProjetInfo_main(int argc, char ** argv);
extern int  Matthieu_main(int argc, char ** argv);
extern int  TestJB_main(int argc, char ** argv);
extern int  RawCor_main(int argc, char ** argv);
extern int  CreateBlockEpip_main(int argc, char ** argv);
extern int  TD_GenereAppuis_main(int argc, char ** argv);
extern int  TD_Exemple_main(int argc, char ** argv);
extern int  TD_Sol1(int argc, char ** argv);
extern int  TD_Sol2(int argc, char ** argv);
extern int  TD_Sol3(int argc, char ** argv);

extern int  TD_Exo0(int argc, char ** argv);
extern int  TD_Exo1(int argc, char ** argv);
extern int  TD_Exo2(int argc, char ** argv);
extern int  TD_Exo3(int argc, char ** argv);
extern int  TD_Exo4(int argc, char ** argv);
extern int  TD_Exo5(int argc, char ** argv);
extern int  TD_Exo6(int argc, char ** argv);
extern int  TD_Exo7(int argc, char ** argv);
extern int  TD_Exo8(int argc, char ** argv);
extern int  TD_Exo9(int argc, char ** argv);
extern int  PPMD_Appariement_main(int argc, char ** argv);

extern int TD_Match1_main(int argc, char ** argv);
extern int TD_Match2_main(int argc, char ** argv);
extern int TD_Match3_main(int argc, char ** argv);
extern int CPP_RelMotionTest_main(int argc, char ** argv);
extern int TestER_main(int argc, char ** argv);
extern int TestER_main2(int argc, char ** argv);
extern int TestER_grille_main(int argc, char ** argv);
extern int TestER_rpc_main(int argc, char ** argv);
extern int GCPCtrlPly_main(int argc, char ** argv);
extern int TestCmpIm_Ewelina(int argc, char ** argv);
extern int TestER_hom_main(int argc, char ** argv);
extern int PFM2Tiff_main(int argc, char ** argv);
extern int BAL2OriMicMac_main(int argc, char ** argv);
extern int CPP_NewOriReadFromSfmInit(int argc, char ** argv);
extern int CPP_ImportArtsQuad(int argc, char ** argv);
extern int CPP_Bundler2MM_main(int argc, char ** argv);
extern int CPP_MM2Bundler_main(int argc, char ** argv);
extern int CPP_Strecha2MM(int argc, char ** argv);
extern int CPP_MM2OpenMVG_main(int argc, char ** argv);
extern int CPP_MM2Colmap_main(int argc, char ** argv);
extern int CPP_ExportSimilPerMotion_main(int argc, char ** argv);
extern int CPP_ImportEO_main(int argc, char ** argv);
extern int ImPts2Dir_main(int argc, char ** argv);
extern int FictiveObstest_main(int argc, char ** argv);
extern int TestFastTreeDist(int argc, char ** argv);
extern int TestPush(int argc, char ** argv);
//extern int Cillia_main(int argc,char ** argv);
extern int Homol2GCP_main(int argc, char ** argv);
extern int CPP_SiftExport_main(int argc, char ** argv);
extern int Test_Homogr_main(int argc, char ** argv);
extern int GlobToLocal_main(int argc, char ** argv);
extern int ExtractZ_main(int argc, char ** argv);
extern int XYZ_Global_main(int argc, char ** argv);
extern int HomToXML_main(int argc, char ** argv);
//extern int CilliaAss_main(int argc, char ** argv);
//extern int CilliaImgt_main(int argc, char ** argv);
extern int ImgCol_main(int argc, char ** argv);
//extern int CilliaMap_main(int argc, char ** argv);
extern int SimilComp_main(int argc, char ** argv);
extern int AffineComp_main(int argc, char ** argv);
           
extern int TestCamTOF_main(int argc,char** argv);
extern int TestMH_main(int argc,char** argv);

extern int  DocEx_Intro0_main(int, char **);
extern int  DocEx_Introd2_main(int, char **);
extern int  DocEx_Introfiltr_main(int, char **);
extern int  ImageRectification(int argc, char ** argv);
extern int  FilterFileHom_main(int argc, char ** argv);
extern int  EgalRadioOrto_main(int argc, char ** argv);
extern int  T2V_main(int argc, char ** argv);
extern int  Tapioca_IDR_main(int argc, char ** argv);
extern int  resizeImg_main(int argc, char ** argv);
extern int  resizeHomol_main(int argc, char ** argv);
extern int  ThermicTo8Bits_main(int argc, char ** argv);
extern int  main_Txt2CplImageTime(int argc, char ** argv);
// test de jo
extern int  main_test(int argc,char ** argv);
extern int  main_test2(int argc,char ** argv);
extern int  main_ero(int argc,char ** argv);
extern int  main_ascii2tif(int argc,char ** argv);
int Test_ascii2tif_BlurinessSelect(int argc,char ** argv);
int main_featheringOrtho(int argc,char ** argv);
int main_featheringOrthoBox(int argc,char ** argv);
int GCP2DMeasureConvert_main(int argc,char ** argv);
int main_densityMapPH(int argc,char ** argv);
int main_manipulateNF_PH(int argc,char ** argv);
int main_OneLionPaw(int argc,char ** argv);
int main_AllPipeline(int argc,char ** argv);


#if (ELISE_UNIX)
extern int  DocEx_Introanalyse_main(int, char **);
#endif
extern int VisuCoupeEpip_main(int, char **);
int ThermikProc_main(int argc, char ** argv);
int ExoSimulTieP_main(int argc, char** argv);
int ExoMCI_main(int argc, char** argv);
int ExoCorrelEpip_main(int argc, char ** argv);
int OptTiePSilo_main(int argc, char ** argv);
int cleanHomolByBsurH_main(int argc, char ** argv);
int GenHuginCpFromHomol_main(int argc, char ** argv);
int PseudoIntersect_main(int argc, char** argv);
int ScaleModel_main(int argc, char ** argv);
int PLY2XYZ_main(int argc, char ** argv);
int ExportXmlGcp2Txt_main(int argc, char ** argv);
int ExportXmlGps2Txt_main(int argc, char ** argv);
int ConvertRtk_main(int argc, char ** argv);
int CPP_FilterGeo3(int argc, char ** argv);
int MatchCenters_main(int argc, char ** argv);
int Panache_main(int argc, char ** argv);
int rnx2rtkp_main(int argc, char ** argv);
int GPS_Txt2Xml_main(int argc, char ** argv);
int ExportHemisTM_main(int argc, char ** argv);
int MatchinImgTM_main(int argc, char ** argv);
int CorrLA_main(int argc, char ** argv);
int EstimLA_main(int argc, char ** argv);
int InterpImgPos_main(int argc, char ** argv);
int CompareOriTieP_main(int argc, char ** argv);
int CmpOrthos_main(int argc, char ** argv);
int CorrOri_main(int argc, char ** argv);
int CalcTF_main(int argc, char ** argv);
int SplitGCPsCPs_main(int argc, char **argv);
int ConcateMAF_main(int argc, char **argv);
int CheckOri_main(int argc, char ** argv);
int NLD_main(int argc, char ** argv);
int ResToTxt_main(int argc, char ** argv);
int SelTieP_main(int argc, char ** argv);
int Ortho2TieP_main(int argc, char ** argv);
int Idem_main(int argc, char ** argv);
// int RHH_main(int argc,char **argv);
int ConvSensXml2Txt_main(int argc, char ** argv);
int CleanTxtPS_main(int argc, char ** argv);
int CheckPatCple_main(int argc, char ** argv);
int ConvPSHomol2MM_main(int argc, char ** argv);
extern int BasculePtsInRepCam_main(int argc, char ** argv);
extern int BasculeCamsInRepCam_main(int argc, char ** argv);
int SplitPatByCam_main(int argc, char ** argv);
int CheckOneOrient_main(int argc, char ** argv);
int CheckAllOrient_main(int argc, char ** argv);
int ChekBigTiff_main(int, char**);
int GenTriplet_main(int argc, char ** argv);
int CalcPatByAspro_main(int argc, char ** argv);
int CPP_GenOneHomFloat(int argc, char ** argv);
int CPP_GenAllHomFloat(int argc, char ** argv);
int CPP_GenOneImP3(int argc, char ** argv);
int CPP_GenAllImP3(int argc, char ** argv);
int CPP_OptimTriplet_main(int argc, char ** argv);
int CPP_AllOptimTriplet_main(int argc, char ** argv);
int CPP_NewSolGolInit_main(int argc, char ** argv);
int CPP_SolGlobInit_RandomDFS_main(int argc, char ** argv);
int CPP_GenOptTriplets(int argc, char ** argv);
int CPP_NewOriImage2G2O_main(int argc, char ** argv);
int CPP_FictiveObsFin_main(int argc, char ** argv);
int CPP_XmlOriRel2OriAbs_main(int argc, char ** argv);
int CPP_Rel2AbsTest_main(int argc, char ** argv);
int CPP_Rot2MatEss_main(int argc, char ** argv);
int GenOriFromOnePose_main(int argc, char ** argv);
int CPP_NewGenTriOfCple(int argc, char ** argv);
int CPP_TestBundleGen(int argc, char ** argv);
int PlyGCP_main(int argc, char ** argv);
int CmpMAF_main(int argc, char ** argv);
int DoCmpByImg_main(int argc, char ** argv);
int GenRayon3D_main(int argc, char ** argv);
int SysCalled_main(int argc, char** argv);
int SysCall_main(int argc, char** argv);
int RedImgsByN_main(int argc, char** argv);
int OptAeroProc_main(int argc, char ** argv);
int TestARCam_main(int argc, char ** argv);
int CPP_TestPhysMod_Main(int argc, char ** argv);
int MvImgsByFile_main(int argc, char** argv);
int CPP_ReechDepl(int argc, char ** argv);
int CPP_BatchReechDepl(int argc, char ** argv);
int OneReechHom_main(int argc, char ** argv);
int OneReechFromAscii_main(int argc, char ** argv);
int AllReechFromAscii_main(int argc, char ** argv);
int AllReechHom_main(int argc, char ** argv);
int OneHomMMToAerial_main(int argc,char** argv);
int AllHomMMToAerial_main(int argc,char** argv);
int RTI_main(int argc, char ** argv);
int RTIRecalRadiom_main(int argc, char ** argv);
int RTIMed_main(int argc, char ** argv);
int RTIGrad_main(int argc, char ** argv);
int RTIFiltrageGrad_main(int argc, char ** argv);
int RTI_RecalRadionmBeton_main(int argc, char ** argv);
int RTI_PosLumFromOmbre_main(int argc, char ** argv);
int GetInfosMPLF_main(int argc, char ** argv);
int TestNewMergeTieP_main(int argc, char ** argv);
int TestStephane_Main(int argc, char ** argv);
int ArboArch_main(int argc, char ** argv);

int TestDupBigTiff(int argc, char ** argv);
int Test_TomCan(int argc, char ** argv);

int TestMartini_Main(int argc, char ** argv);

int TestGiang_main(int argc, char ** argv);

int TestGiangNewHomol_Main(int argc, char ** argv);

int TestGiangDispHomol_Main(int argc, char ** argv);

int Test_Conv(int argc, char ** argv);

int Test_CtrlCloseLoop(int argc, char ** argv);

int GetSpace_main(int argc, char ** argv);

int TestDetecteur_main(int argc, char ** argv);


int CplFromHomol_main(int argc, char ** argv);

int TiepTriPrl_main(int argc, char ** argv);

int TiepTri_Main(int argc, char ** argv);

int TaskCorrel_main(int argc, char ** argv);

int TaskCorrelWithPts_main(int argc, char ** argv);

int FAST_main(int argc, char ** argv);

int Test_NewRechPH(int argc, char ** argv);

int Homol2Way_main(int argc, char ** argv);

int Homol2WayNEW_main(int argc, char ** argv);

int Test_InitBloc(int argc, char ** argv);

int HomolLSMRefine_main(int argc,char ** argv);

int UnWindows(int argc, char ** argv);

int MakePly_CamOrthoC(int argc, char ** argv);

int XMLDiffSeries_main(int argc, char ** argv);

int ZBufferRaster_main(int argc, char ** argv);

int Test_TrajectoFromOri(int argc, char ** argv);

int PlyBascule(int argc, char ** argv);

int ConvertToNewFormatHom_Main(int argc, char ** argv);
int ConvertToOldFormatHom_Main(int argc,char ** argv);

int UnionFiltragePHom_Main(int argc, char ** argv);

int TestYZ_main(int argc, char ** argv);

extern int TestLulin_main(int argc, char ** argv);
extern int SuperGlue_main(int argc, char ** argv);
extern int MergeTiePt_main(int argc, char ** argv);
extern int GetPatchPair_main(int argc, char ** argv);
extern int RANSAC_main(int argc, char ** argv);
extern int CrossCorrelation_main(int argc, char ** argv);
extern int GuidedSIFTMatch_main(int argc, char ** argv);
extern int GetOverlappedImages_main(int argc, char ** argv);
extern int DSM_Equalization_main(int argc, char ** argv);
extern int CreateGCPs_main(int argc, char ** argv);
extern int WallisFilter_main(int argc, char ** argv);
extern int TiePtEvaluation_main(int argc, char ** argv);
extern int MakeOneTrainingData_main(int argc, char ** argv);
extern int MakeTrainingData_main(int argc, char ** argv);
extern int VisuTiePtIn3D_main(int argc, char ** argv);
extern int TiePtAddWeight_main(int argc, char ** argv);
extern int EnhancedSpG_main(int argc, char ** argv);
extern int SIFT2Step_main(int argc, char ** argv);
extern int SIFT2StepFile_main(int argc, char ** argv);
//extern int D2NetMatch_main(int argc, char ** argv);
extern int Calc2DSimi_main(int argc, char ** argv);
extern int GlobalR3D_main(int argc, char ** argv);
extern int ExtractSIFT_main(int argc, char ** argv);
extern int InlierRatio_main(int argc, char ** argv);
extern int EvalOri_main(int argc, char ** argv);
extern int CoReg_GlobalR3D_main(int argc, char ** argv);
extern int PileImgs_main(int argc, char ** argv);
extern int GetOrthoHom_main(int argc, char ** argv);
extern int TransmitHelmert_main(int argc, char ** argv);
extern int TiePtPrep_main(int argc, char ** argv);
extern int CreateGCPs4Init11p_main(int argc, char ** argv);
extern int CreateGCPs4Init11pSamePts_main(int argc, char ** argv);


extern int ReechHomol_main(int argc, char ** argv);
extern int DeformAnalyse_main(int argc, char ** argv);
extern int ExtraitHomol_main(int argc, char ** argv);
extern int IntersectHomol_main(int argc, char ** argv);
extern int ReechMAF_main(int argc, char ** argv);
extern int ImgTMTxt2Xml_main(int argc, char ** argv);
extern int MoyMAF_main(int argc, char ** argv);
extern int GenerateTP_main(int argc, char ** argv);
extern int SimuRolShut_main(int argc, char ** argv);
extern int GenerateOrient_main(int argc, char ** argv);
extern int ReechRolShut_main(int argc, char ** argv);
extern int ReechRolShutV1_main(int argc, char ** argv);
extern int ExportTPM_main(int argc, char ** argv);
extern int CompMAF_main(int argc, char ** argv);
extern int GenerateOriGPS_main(int argc, char ** argv);
extern int GenerateMAF_main(int argc, char ** argv);
extern int GenImgTM_main(int argc, char ** argv);
extern int EsSim_main(int argc, char ** argv);
int ProcessThmImgs_main(int argc, char ** argv);

extern int ConvertTiePPs2MM_main(int argc, char ** argv);

extern int ConvHomolVSFM2MM_main(int argc, char ** argv);

int LSQMatch_Main(int argc, char ** argv);


extern int  TestNewOriHom1Im_main(int argc, char ** argv);
extern int  TestNewOriGpsSim_main(int argc, char ** argv);

extern int  CPP_NOGpsLoc(int argc, char ** argv);

extern int GCPRollingBasc_main(int argc, char** argv);
extern int Generate_ImagSift(int argc, char** argv);
extern int Generate_ImagePer(int argc, char** argv);

extern int  CPP_DistHistoBinaire(int argc, char ** argv);

extern int CPP_AutoCorr_CensusQuant(int argc, char ** argv);
int MosaicTFW(int argc, char** argv);

extern int ConvTiePointPix4DMM_main(int argc,char ** argv);
extern int OrthoDirectFromDenseCloud_main(int argc,char ** argv);
extern int TiepGraphByCamDist_main(int argc,char ** argv);

extern int GraphHomSat_main(int argc,char ** argv);
extern int CPP_AppliCreateEpi_main(int argc,char ** argv);
extern int CPP_AppliMM1P_main(int argc,char ** argv);
extern int CPP_AppliRecalRPC_main(int argc,char ** argv);
extern int CPP_AppliFusion_main(int argc,char ** argv);
extern int CPP_TransformGeom_main(int argc,char ** argv);

const std::vector<cMMCom> & TestLibAvailableCommands()
{
	static std::vector<cMMCom> aRes;
	if (aRes.empty())
	{

        aRes.push_back(cMMCom("TestLulin", TestLulin_main, "Explaination: TestLulin "));
        aRes.push_back(cMMCom("SuperGlue", SuperGlue_main, "Use SuperGlue to extract tie points"));
        aRes.push_back(cMMCom("MergeTiePt", MergeTiePt_main, "Merge tie points of sub images into integrated one"));
        aRes.push_back(cMMCom("GetPatchPair", GetPatchPair_main, "Divide an image pair to a number of patch pairs in order to apply learned feature matching"));

        aRes.push_back(cMMCom("RANSAC", RANSAC_main, "Tie point filter based on 2D or 3D RANSAC "));
        aRes.push_back(cMMCom("CrossCorrelation", CrossCorrelation_main, "Tie point filter based on Cross Correlation "));
        aRes.push_back(cMMCom("GuidedSIFTMatch", GuidedSIFTMatch_main, "Nearest neighbour SIFT matching, with search space narrowed down by co-registered orientation "));
        aRes.push_back(cMMCom("GetOverlappedImages", GetOverlappedImages_main, "Get Overlapped Image Pairs "));
        aRes.push_back(cMMCom("DSM_Equalization", DSM_Equalization_main, "DSM Equalization and output gray image "));
        aRes.push_back(cMMCom("CreateGCPs", CreateGCPs_main, "Create GCPs based on tie points on DSMs of 2 epochs "));
        aRes.push_back(cMMCom("Wallis", WallisFilter_main, "Apply Wallis Filter on one image"));
        aRes.push_back(cMMCom("TiePtEvaluation", TiePtEvaluation_main, "Evaluate the accuracy of tie points with ground truth DSM"));
        aRes.push_back(cMMCom("MakeOneTrainingData", MakeOneTrainingData_main, "Make training data of one patch pair for SuperGlue 512D"));
        aRes.push_back(cMMCom("MakeTrainingData", MakeTrainingData_main, "Make training data for SuperGlue 512D"));
        aRes.push_back(cMMCom("VisuTiePtIn3D", VisuTiePtIn3D_main, "Visulize tie points in image pairs together in 3D"));
        aRes.push_back(cMMCom("TiePtAddWeight", TiePtAddWeight_main, "Add weight for tie points"));
        aRes.push_back(cMMCom("EnhancedSpG", EnhancedSpG_main, "Use tiling scheme and rotation hypothesis to improve the matching performance of SuperGlue"));
        aRes.push_back(cMMCom("SIFT2Step", SIFT2Step_main, "Match an image pair by firstly applying SIFT on downsampled images without ratio test to estimate a similarity transformation, then applying SIFT on original images under guidance of the transformation"));
        aRes.push_back(cMMCom("SIFT2StepFile", SIFT2StepFile_main, "Input a xml file that contains all the image pairs to be matched, and match match with SIFT2Step method"));
        //aRes.push_back(cMMCom("D2NetMatch", D2NetMatch_main, "Input D2Net feature files \"img.d2-net\" and match them with mutual neareat neighbor"));
        aRes.push_back(cMMCom("Calc2DSimi", Calc2DSimi_main, "Input tie point file to calculate a 2D similarity transformation between them and out the parameter file"));
        aRes.push_back(cMMCom("GlobalR3D", GlobalR3D_main, "Filter tie points by running RANSAC in 3D to build a 3D Helmet transformation model that is globally consistent over the whole block"));
        aRes.push_back(cMMCom("ExtractSIFT", ExtractSIFT_main, "Extract SIFT"));
        aRes.push_back(cMMCom("InlierRatio", InlierRatio_main, "Calculate inlier ratio of tie points on DSMs or orthophotos"));
        aRes.push_back(cMMCom("EvalOri", EvalOri_main, "Input GCPs to evaluate orientations"));
        aRes.push_back(cMMCom("CoReg_GlobalR3D", CoReg_GlobalR3D_main, "Roughly co-register 2 epochs by matching individual RGB image pairs followed by GlobalR3D"));
        aRes.push_back(cMMCom("PileImgs", PileImgs_main, "Pile images on an average plane in a pseudo orthophoto style"));
        aRes.push_back(cMMCom("GetOrthoHom", GetOrthoHom_main, "project tie points on image pairs onto orthophotos"));
        aRes.push_back(cMMCom("TransmitHelmert", TransmitHelmert_main, "Input 2 sets of 3D Helmert transformation parameters (A->C and B->C), output transimtted 3D Helmert transformation parameters (A->B)"));
        aRes.push_back(cMMCom("TiePtPrep", TiePtPrep_main, "Explaination: Add weight to inter-epoch tie points, and merge them with intra-epoch tie points"));
        aRes.push_back(cMMCom("CreateGCPs4Init11p", CreateGCPs4Init11p_main, "Create virtual GCPs for command Init11p (Define grids in each image, which leads to different sets of points for each image)"));
        aRes.push_back(cMMCom("CreateGCPs4Init11pSamePts", CreateGCPs4Init11pSamePts_main, "Create virtual GCPs for command Init11p (Define grids in ground, which leads to the same sets of points for each image)"));



		aRes.push_back(cMMCom("Script",CPP_YannScript, "Fonction de script pour les tests "));		
		aRes.push_back(cMMCom("ExcludeSats",CPP_YannExcludeSats, "Excludes GNSS satellites from raw observations based on sky masks "));
		aRes.push_back(cMMCom("SkyMask",CPP_YannSkyMask, "Sky mask estimation with neural network "));
		aRes.push_back(cMMCom("SetTimestamps",CPP_YannSetTimestamps, "Add timestamps tag in image exif "));
		aRes.push_back(cMMCom("Export2Colmap",CPP_YannExport2Colmap, "Exports a Micmac orientation directory to Colmap format "));
		aRes.push_back(cMMCom("RTKlibConvert",CPP_ConvertRTKlib2Micmac, "RTKlib output file conversion to Micmac format "));


		aRes.push_back(cMMCom("Exo0", TD_Exo0, "Some stuff "));
		aRes.push_back(cMMCom("Exo1", TD_Exo1, "Some stuff "));
		aRes.push_back(cMMCom("Exo2", TD_Exo2, "Some stuff "));
		aRes.push_back(cMMCom("Exo3", TD_Exo3, "Some stuff "));
		aRes.push_back(cMMCom("Exo4", TD_Exo4, "Some stuff "));
		aRes.push_back(cMMCom("Exo5", TD_Exo5, "Some stuff "));
		aRes.push_back(cMMCom("Exo6", TD_Exo6, "Some stuff "));
		aRes.push_back(cMMCom("Exo7", TD_Exo7, "Some stuff "));
		aRes.push_back(cMMCom("Exo8", TD_Exo8, "Some stuff "));
		aRes.push_back(cMMCom("Exo9", TD_Exo9, "Some stuff "));
		aRes.push_back(cMMCom("ExoMatch", PPMD_Appariement_main, "Some stuff "));

		aRes.push_back(cMMCom("NoBill", UnWindows, "Supress the big shit in file resulting from (f**king) Windows editing"));

		aRes.push_back(cMMCom("DupBigTiff", TestDupBigTiff, "Duplicate a tiff file, handling the big tif option"));
		aRes.push_back(cMMCom("Stephane", TestStephane_Main, "In test funtction for Stephane Guinard "));
		aRes.push_back(cMMCom("TestNewMergeTieP", TestNewMergeTieP_main, "Some consitency check on Merge TieP "));
		aRes.push_back(cMMCom("TestARCam", TestARCam_main, "Some consitency check on camera "));
		aRes.push_back(cMMCom("SysCall", SysCall_main, "Some stuff "));
		aRes.push_back(cMMCom("SysCalled", SysCalled_main, "Some stuff "));

		aRes.push_back(cMMCom("PrepSift", PreparSift_Main, "Some stuff "));
		aRes.push_back(cMMCom("TD1", TD_Match1_main, "Some stuff "));
		aRes.push_back(cMMCom("TD2", TD_Match2_main, "Some stuff "));
		aRes.push_back(cMMCom("TD3", TD_Match3_main, "Some stuff "));

		aRes.push_back(cMMCom("X1", TD_Sol1, "Some stuff "));
		aRes.push_back(cMMCom("X2", TD_Sol2, "Some stuff "));
		aRes.push_back(cMMCom("X3", TD_Sol3, "Some stuff "));
		aRes.push_back(cMMCom("W0", Sample_W0_main, "Test on Graphic Windows "));
		aRes.push_back(cMMCom("LSQ0", Sample_LSQ0_main, "Basic Test on Least Square library "));
		aRes.push_back(cMMCom("Tests_Luc", Luc_main, "tests de Luc"));
		aRes.push_back(cMMCom("Abdou", Abdou_main, "Exemples fonctions abdou"));
		aRes.push_back(cMMCom("CheckOri", CheckOri_main, "Difference between two sets of orientations"));
		aRes.push_back(cMMCom("NLD", NLD_main, "test"));
		aRes.push_back(cMMCom("RTT", ResToTxt_main, "Transform residuals from GCPBascule into a readable file"));
		aRes.push_back(cMMCom("SelTieP", SelTieP_main, "Select Tie Points with favorable angles"));
		aRes.push_back(cMMCom("Ortho2TieP", Ortho2TieP_main, "Select Tie Points from the orthophotography"));
		aRes.push_back(cMMCom("Idem", Idem_main, "Interpolate DEM on GCP & CP"));
		aRes.push_back(cMMCom("TestSI", Matthieu_main, "Test SelectionInfos"));
		aRes.push_back(cMMCom("TestJB", TestJB_main, "random stuff"));
		aRes.push_back(cMMCom("TestER", CPP_RelMotionTest_main, "ER test workplace"));

		aRes.push_back(cMMCom("TestER2", TestFastTreeDist, "ER test fast tree dist"));
		aRes.push_back(cMMCom("Tif2Pfm", PFM2Tiff_main, "Tif to pfm or the other way around"));
		aRes.push_back(cMMCom("BAL2MM", BAL2OriMicMac_main, "Convert a BAL problem to MicMac"));
		aRes.push_back(cMMCom("SfmI2MM", CPP_NewOriReadFromSfmInit, "Convert the SfmInit problem to MicMac"));
		aRes.push_back(cMMCom("ArtsQuad", CPP_ImportArtsQuad, "Read ArtsQuad tracks to MicMac tie-pts"));
		aRes.push_back(cMMCom("Bundler2MM", CPP_Bundler2MM_main, "Convert the Bundler solution to MicMac"));
		aRes.push_back(cMMCom("MM2Bundler", CPP_MM2Bundler_main, "Convert the MicMac  solution to Bundler"));
		aRes.push_back(cMMCom("Str2MM", CPP_Strecha2MM, "Convert the Strecha solution to MicMac"));
		aRes.push_back(cMMCom("MM2OMVG", CPP_MM2OpenMVG_main, "Convert Homol (PMul) to OpenMVG features / matches"));
		aRes.push_back(cMMCom("MM2Colmap", CPP_MM2Colmap_main, "Convert MicMac poses to Colmap"));
		aRes.push_back(cMMCom("GlobSimPerM", CPP_ExportSimilPerMotion_main, "Export global similitude per pair/triplet motion"));
		aRes.push_back(cMMCom("GlobPoseImp", CPP_ImportEO_main, "Import global poses"));
		aRes.push_back(cMMCom("Im2Dir", ImPts2Dir_main, "Extract directions from images"));
		aRes.push_back(cMMCom("FictObs", FictiveObstest_main, "someee stuff"));
		aRes.push_back(cMMCom("CamTOFExp", TestCamTOF_main, "Export TOF camera pcd file to MicMac formats (e.g. tif, xml, ply)"));
		aRes.push_back(cMMCom("TestMH", TestMH_main, "Test Mike"));
		aRes.push_back(cMMCom("TestAT", TestPush, "AT test workplace"));
		aRes.push_back(cMMCom("ExportSIFT", CPP_SiftExport_main, "Export SIFT descriptor"));
		aRes.push_back(cMMCom("TestH", Test_Homogr_main, "TestHomogr"));

		//       aRes.push_back(cMMCom("TestCillia",Cillia_main,"cillia"));
		//aRes.push_back(cMMCom("Homol2GCP", Homol2GCP_main, "cillia"));
		//aRes.push_back(cMMCom("GlobToLocal", GlobToLocal_main, "cillia"));
		//aRes.push_back(cMMCom("ExtractZ", ExtractZ_main, "cillia"));
		//aRes.push_back(cMMCom("XYZ_Global", XYZ_Global_main, "cillia"));
		//aRes.push_back(cMMCom("HomToXML", HomToXML_main, "cillia"));
		//aRes.push_back(cMMCom("TestCilliaAss", CilliaAss_main, "cillia"));
		//aRes.push_back(cMMCom("TestCilliaImgt", CilliaImgt_main, "cillia"));
		//aRes.push_back(cMMCom("ImgCol", ImgCol_main, "cilliac"));
		//aRes.push_back(cMMCom("TestCilliaMap", CilliaMap_main, "cilliac"));
		//aRes.push_back(cMMCom("SimilComp", SimilComp_main, "cilliac"));
		//aRes.push_back(cMMCom("AffineComp", AffineComp_main, "cilliac"));

		aRes.push_back(cMMCom("PI", ProjetInfo_main, "Projet Info"));
		// aRes.push_back(cMMCom("RawCor",RawCor_main,"Test for correcting green or red RAWs"));
		aRes.push_back(cMMCom("LucasChCloud", LucasChCloud_main, "Examples functions modifying cloud"));
		aRes.push_back(cMMCom("GetDataMPLF", GetInfosMPLF_main, "Extract informations from a Mission Planner .log file"));
		aRes.push_back(cMMCom("MvImgs", MvImgsByFile_main, "Move Images in a file to a trash folder"));
		aRes.push_back(cMMCom("BlocEpip", CreateBlockEpip_main, "Epip by bloc (internal use to // epip)"));
		aRes.push_back(cMMCom("MMSMA", MMSymMasqAR_main, "Symetrise Masque Alle-Retour (internal use in MM1P)"));
		aRes.push_back(cMMCom("TD_GenApp", TD_GenereAppuis_main, "TD Generate GCP"));
		aRes.push_back(cMMCom("TD_Test", TD_Exemple_main, "Test TD "));
		aRes.push_back(cMMCom("DocI0", DocEx_Intro0_main, "Introduction 0 of example from DocElise  "));
		aRes.push_back(cMMCom("DocID2", DocEx_Introd2_main, "Introduction to D2 of example from DocElise  "));
		aRes.push_back(cMMCom("DocIntrofiltre", DocEx_Introfiltr_main, "Introduction to filter example from DocElise  "));
#if (ELISE_UNIX)
		aRes.push_back(cMMCom("DocIntroanalyse", DocEx_Introanalyse_main, "Introduction to image analysis from DocElise  "));
#endif

       aRes.push_back(cMMCom("VCE",VisuCoupeEpip_main,"Visualization of epipolar pair (cut)"));
       aRes.push_back(cMMCom("RIE",ReechInvEpip_main,"Visualization of epipolar pair (cut)"));
	   aRes.push_back(cMMCom("DoCmpByImg",DoCmpByImg_main,"Compensate Image By Image (Space Resection Mode)"));
       aRes.push_back(cMMCom("MCI",ExoMCI_main,"Exercise for multi correlation in image geometry"));
       aRes.push_back(cMMCom("ECE",ExoCorrelEpip_main,"Exercise for correlation in epipolar"));
       aRes.push_back(cMMCom("ESTP",ExoSimulTieP_main,"Tie points simulation"));
       aRes.push_back(cMMCom("TDEpi",TDEpip_main,"Test epipolar matcher"));
       aRes.push_back(cMMCom("CmpMAF",CmpMAF_main,"Compare 2 file of Image Measures",cArgLogCom(2)));
       aRes.push_back(cMMCom("ProjImPtOnOtherImages",ProjImPtOnOtherImages_main," Project image points on other images"));
	   aRes.push_back(cMMCom("ThermikProc",ThermikProc_main,"Full Process of Thermik Workflow Images",cArgLogCom(2)));
	   aRes.push_back(cMMCom("MatchImTM",MatchinImgTM_main,"Matching a Pattern of Images with a GPS TimeMark File",cArgLogCom(2)));
       aRes.push_back(cMMCom("PseudoIntersect",PseudoIntersect_main,"Pseudo Intersection of 2d points from N images",cArgLogCom(2)));
       aRes.push_back(cMMCom("Export2Ply",Export2Ply_main,"Tool to generate a ply file from TEXT or XML file, tuning",cArgLogCom(2)));
       aRes.push_back(cMMCom("ScaleModel",ScaleModel_main,"Tool for simple scaling a model",cArgLogCom(2)));
       aRes.push_back(cMMCom("Ply2Xyz",PLY2XYZ_main,"Tool to export in TxT file XYZ columns only from a .ply file",cArgLogCom(2)));
       aRes.push_back(cMMCom("XmlGcp2Txt",ExportXmlGcp2Txt_main,"Tool to export .xml GCPs file to .txt file",cArgLogCom(2)));
       aRes.push_back(cMMCom("XmlGps2Txt",ExportXmlGps2Txt_main,"Tool to export .xml GPS file to .txt file",cArgLogCom(2)));
       aRes.push_back(cMMCom("Panache",Panache_main,"Tool to export profile along axis given a line draw on Orthoimage",cArgLogCom(2)));
	   aRes.push_back(cMMCom("ConvRtk",ConvertRtk_main,"Tool to extract X_Y_Z_Ix_Iy_Iz from Rtklib output file",cArgLogCom(2)));
	   aRes.push_back(cMMCom("FilterGeo3",CPP_FilterGeo3,"Tool extract ?optimal position for a set of daily geocube obs",cArgLogCom(2)));
	   aRes.push_back(cMMCom("MatchCenters",MatchCenters_main,"Tool to match Gps positions and Camera Centers",cArgLogCom(2)));
	   aRes.push_back(cMMCom("GpsProc",rnx2rtkp_main,"Tool using rnx2rtkp from RTKlib to do GNSS processing",cArgLogCom(2)));
	   aRes.push_back(cMMCom("GPSConvert",GPS_Txt2Xml_main,"Tool to convert a GPS trajectory into xml format",cArgLogCom(2)));
	   aRes.push_back(cMMCom("CorrLA",CorrLA_main,"Tool to correct camera centers from Lever-Arm offset",cArgLogCom(2)));
	   aRes.push_back(cMMCom("EstimLA",EstimLA_main,"Tool to estimate Lever-Arm from Gps Trajectory and Ground Camera Poses",cArgLogCom(2)));
	   aRes.push_back(cMMCom("ExportHTM",ExportHemisTM_main,"Tool to export TimeMark Data from Hemisphere Bin01 file",cArgLogCom(2)));
	   aRes.push_back(cMMCom("InterpImTM",InterpImgPos_main,"Tool to interpolate image position based on TimeMark GPS trajectory",cArgLogCom(2)));
	   aRes.push_back(cMMCom("CmpTieP",CompareOriTieP_main,"Tool to compare deviations between 2 Ori-XXX folders on 3D tie points positions",cArgLogCom(2)));
	   aRes.push_back(cMMCom("CmpOrthos",CmpOrthos_main,"Tool to compute displacement vectors between 2 Orthos based on Tie Points",cArgLogCom(2)));
	   aRes.push_back(cMMCom("CorrOri",CorrOri_main,"Tool to correct images centers from a bias and generate new Ori folder",cArgLogCom(2)));
	   aRes.push_back(cMMCom("CalcTF",CalcTF_main,"Tool to compute the percentage of fixed GPS positions",cArgLogCom(2)));
	   aRes.push_back(cMMCom("SplitPts",SplitGCPsCPs_main,"Tool to split .xml ground points into GCPs and CPs",cArgLogCom(2)));
	   aRes.push_back(cMMCom("ConcateMAF",ConcateMAF_main,"Tool to concatenate .xml ground points images coordinates",cArgLogCom(2)));
	   aRes.push_back(cMMCom("MergeMAF",ConcateMAF_main,"Tool to concatenate .xml ground points images coordinates",cArgLogCom(2)));
	   aRes.push_back(cMMCom("XmlSensib2Txt",ConvSensXml2Txt_main,"Tool to convert .xml Sensibility File 2 .txt file",cArgLogCom(2)));
	   aRes.push_back(cMMCom("CleanTxtPS", CleanTxtPS_main,"Tool to clean .txt file output of PhotoScan Aero",cArgLogCom(2)));
	   aRes.push_back(cMMCom("CheckPatCple", CheckPatCple_main,"Tool to check a Pattern and an .xml File Cple",cArgLogCom(2)));
	   aRes.push_back(cMMCom("ConvPSHomol2MM", ConvPSHomol2MM_main, "Tool to convert Tie Points from PhotoScan to MicMac format",cArgLogCom(2)));
	   aRes.push_back(cMMCom("SplitPatByCam", SplitPatByCam_main, "Tool to split a Pattern based on type of camera",cArgLogCom(2)));
	   aRes.push_back(cMMCom("OptTiePSilo",OptTiePSilo_main,"Optimize Tie Points Extraction For Silo"));
	   aRes.push_back(cMMCom("GenHuginCp",GenHuginCpFromHomol_main,"Genrate Hugin Control Points from Homol",cArgLogCom(2)));
	   aRes.push_back(cMMCom("CleanHomByBH",cleanHomolByBsurH_main,"Clean Homolgues points between images based on BsurHvalues",cArgLogCom(2)));
       aRes.push_back(cMMCom("RHH",RHH_main,"In dev estimation of global 2D homography  "));
       aRes.push_back(cMMCom("RHHComputHom",RHHComputHom_main,"Internal : compute Hom for // in RHH  "));
	   aRes.push_back(cMMCom("PatAspro",CalcPatByAspro_main,"Tool to Aspro a Pattern of Imgs",cArgLogCom(2)));
       aRes.push_back(cMMCom("XmlXif",MakeOneXmlXifInfo_main,"Internal : generate Xml to accelerate Xif extraction"));
	   aRes.push_back(cMMCom("OriFromOnePose",GenOriFromOnePose_main,"Generate an Ori-XXX from one pos ; All images the same"));
       aRes.push_back(cMMCom("Xml2Dmp",Xml2Dmp_main,"Convert XML to Dump"));
       aRes.push_back(cMMCom("Dmp2Xml",Dmp2Xml_main,"Convert Dump to Xml"));
	   aRes.push_back(cMMCom("GenRayon3D",GenRayon3D_main,"Generate 3D lines in a ply format ; Visualize pseudo-intersection"));
        aRes.push_back(cMMCom("AddAffinity", AddAffinity_main, "Add an affinity, tuning"));
        aRes.push_back(cMMCom("TP2GCP",ServiceGeoSud_TP2GCP_main,"Tie Points to Ground Control Points (for GeoSud services)"));
        aRes.push_back(cMMCom("Ortho",ServiceGeoSud_Ortho_main,"Compute a basic Ortho from a DTM and a satellite image (for GeoSud services)"));
        aRes.push_back(cMMCom("GeoSud",ServiceGeoSud_GeoSud_main,""));
        aRes.push_back(cMMCom("Surf",ServiceGeoSud_Surf_main,""));
        aRes.push_back(cMMCom("ImageRectification",ImageRectification,"Rectify individual aerial images, ground is assumed to be a plane"));
        aRes.push_back(cMMCom("ThermicTo8Bits",ThermicTo8Bits_main,"Convert 16 bits tif thermic images (from variocam or optris camera) to 8 bits gray or RGB images"));
        aRes.push_back(cMMCom("jo_FFH",FilterFileHom_main,"filtrer un fichier de paire d'image"));
        aRes.push_back(cMMCom("jo_T2V",T2V_main,"appliquer une homographie a un ensemble d'im thermique pour Reg avec images visibles"));
        aRes.push_back(cMMCom("jo_test",main_test2,"test function for didro project"));
       //aRes.push_back(cMMCom("AperiCloudNF",main_manipulateNF_PH,"Generate Sparse 3D point cloud for tie point in new format (TiePMul.dat)"));
        aRes.push_back(cMMCom("IntersectBundleNF",main_manipulateNF_PH,"Compute Pseudo Intersection for tie point in new format (TiePMul.dat) and export it as 3D measurements"));
        aRes.push_back(cMMCom("AllAutoBash",main_AllPipeline,"complete photogrammetric workflow on many images blocks"));
        aRes.push_back(cMMCom("AllAuto",main_OneLionPaw,"complete photogrammetric workflow on one images blocks"));
        aRes.push_back(cMMCom("TapiocaIDR",Tapioca_IDR_main,"Utiliser Tapioca avec des Images de Résolution Différente (effectue un resample des images)"));
        aRes.push_back(cMMCom("ResizeImg",resizeImg_main,"Resize image in order to reach a specific image width"));
        aRes.push_back(cMMCom("ResizeHomol",resizeHomol_main,"Resize Homol pack"));
        aRes.push_back(cMMCom("Ero",main_ero,"Egalisation Radiometrique pour une paire d'ortho"));
        aRes.push_back(cMMCom("Eros",EgalRadioOrto_main,"Egalisation Radiometrique d'OrthoS"));
        aRes.push_back(cMMCom("Ascii2Tif",main_ascii2tif,"transform ascii file to tif file, designed for ascii from irbis or sdk direct (variocam and optris)."));
        aRes.push_back(cMMCom("Ascii2TifWithSelection",Test_ascii2tif_BlurinessSelect,"from list of ascii file from video frame, perform a selection of sharpest frame and export it in tif format."));
        aRes.push_back(cMMCom("SeamlineFeathering",main_featheringOrtho,"Perform mosaiking of orthos with a feathering around the seamline."));
        aRes.push_back(cMMCom("SeamlineFeatheringBox",main_featheringOrthoBox,"Perform mosaiking of orthos with a feathering around the seamline for one tile of the mosaic"));
        aRes.push_back(cMMCom("GCP2DMeasureConvert",GCP2DMeasureConvert_main,"Export or import 2D image marks of GCPs/Manual tie point"));
        aRes.push_back(cMMCom("DensityMapHomol",main_densityMapPH,"Compute a Density map of tie point"));
        aRes.push_back(cMMCom("Masq3Dto2D",Masq3Dto2D_main,"Create a 2D Masq from Nuage and 3D Masq "));
        aRes.push_back(cMMCom("MergeCloud",CPP_AppliMergeCloud,"Tool for merging overlapping depth maps from different view points"));
        aRes.push_back(cMMCom("MMEnvlop",MMEnveloppe_Main,"Compute initial envelope surface for MMEpi "));
        aRes.push_back(cMMCom("PlySphere",PlySphere_main,"Tool to generate a sphere of point, ply format, tuning"));
        aRes.push_back(cMMCom("PlyGCP",PlyGCP_main,"Tool to generate a visualization of ply"));
        aRes.push_back(cMMCom("San2Ply",San2Ply_main,"Generate a Ply visualisation of an Analytical Surface"));
	    aRes.push_back(cMMCom("RedImg",RedImgsByN_main,"Reduce Number of images : 1 out of N"));
        aRes.push_back(cMMCom("CASALL",CASALL_main,"Compute Analytic Surface Automatically  low level"));
        aRes.push_back(cMMCom("CalcAutoCorrel",CalcAutoCorrel_main,"Compute and Store Auto Correlation (if not already done)"));
		aRes.push_back(cMMCom("OptAeroProc",OptAeroProc_main,"Optimize Aero Processing Datatset"));
        aRes.push_back(cMMCom("CLIC",CCL_main,"Cam Light Imag Correc)"));
        aRes.push_back(cMMCom("MMEnvStatute",MMEnvStatute_main,"Envelope for mode statue"));
        aRes.push_back(cMMCom("TopoBasc",TopoSurf_main,"Topological analysis before bascule"));
		aRes.push_back(cMMCom("ArboArch",ArboArch_main,"Files organization, internal use"));

        aRes.push_back(cMMCom("Check1Hom",CheckOneHom_main,"Check One File Homologue"));
        aRes.push_back(cMMCom("CheckAllHom",CheckAllHom_main,"Check All File Homologue"));
        aRes.push_back(cMMCom("Check1Tiff",CheckOneTiff_main,"Check All File Homologue"));
        aRes.push_back(cMMCom("CheckAllTiff",CheckAllTiff_main,"Check All File Homologue"));
        aRes.push_back(cMMCom("CheckBigTiff",ChekBigTiff_main,"Check creation of a big file"));


        aRes.push_back(cMMCom("Check1Ori",CheckOneOrient_main,"Check One Orientation"));
        aRes.push_back(cMMCom("CheckAllOri",CheckAllOrient_main,"Check a Folder of Orientation"));

        aRes.push_back(cMMCom("BasculePtsInRepCam",BasculePtsInRepCam_main,"Compute GCP in cam repair"));
        aRes.push_back(cMMCom("BasculeCamsInRepCam",BasculeCamsInRepCam_main,"Compute GCP in cam repair"));

        aRes.push_back(cMMCom("NO_OriHom1Im",TestNewOriHom1Im_main,"Test New Homgr Orientation-Case 1 central Im"));
        aRes.push_back(cMMCom("NO_OriGpsSim",TestNewOriGpsSim_main,"Test New Homgr Orientation with Gps, Horizontal"));
        aRes.push_back(cMMCom("NO_GpsLoc",CPP_NOGpsLoc,"Use Gps for absolute orientation of Martini"));
        aRes.push_back(cMMCom("NO_Ori2Im",TestNewOriImage_main,"Test New Orientation"));
        aRes.push_back(cMMCom("NO_AllOri2Im",TestAllNewOriImage_main,"Test New Orientation"));
        aRes.push_back(cMMCom("NO_GenTripl",GenTriplet_main,"New Orientation : select triplet"));

        aRes.push_back(cMMCom("NO_OneHomFloat",CPP_GenOneHomFloat,"New Orientation : generate merged float point of one image"));
        aRes.push_back(cMMCom("NO_AllHomFloat",CPP_GenAllHomFloat,"New Orientation : generate float point of all image"));
        aRes.push_back(cMMCom("NO_OneImTriplet",CPP_GenOneImP3,"New Orientation : generate triple of one image"));
        aRes.push_back(cMMCom("NO_AllImTriplet",CPP_GenAllImP3,"New Orientation : generate triple of all imaget"));
        aRes.push_back(cMMCom("NO_OneImOptTrip",CPP_OptimTriplet_main,"New Orientation : otimize triplet"));
        aRes.push_back(cMMCom("NO_AllImOptTrip",CPP_AllOptimTriplet_main,"New Orientation : otimize triplet"));
        aRes.push_back(cMMCom("NO_SolInit3",CPP_NewSolGolInit_main,"New Orientation : sol init from triplet"));
        aRes.push_back(cMMCom("NO_SolInit_RndDFS",CPP_SolGlobInit_RandomDFS_main,"New Orientation : sol init by random DFS"));
        aRes.push_back(cMMCom("NO_GenPerfTripl",CPP_GenOptTriplets,"New Orientation : generate perfect triplets from InOri"));
        aRes.push_back(cMMCom("NO_ExportG2O",CPP_NewOriImage2G2O_main,"New Orientation : export triplets to g2o"));

        aRes.push_back(cMMCom("NO_GenTriOfCple",CPP_NewGenTriOfCple,"New Orientation : select triple of one edge"));
		aRes.push_back(cMMCom("NO_FicObs", CPP_FictiveObsFin_main, "New orientation : ficticious observations"));
		aRes.push_back(cMMCom("NO_XmlRel2Coniq", CPP_XmlOriRel2OriAbs_main, "New orientation : convert xml relative to conique"));
		aRes.push_back(cMMCom("NO_R2A",CPP_Rel2AbsTest_main, "New orientation : calculate the absolute orientation of a query image"));
		aRes.push_back(cMMCom("NO_Ori2MatEss", CPP_Rot2MatEss_main, "New orientation : convert Ori to essential matrix"));

        aRes.push_back(cMMCom("OriMatis2MM",MatisOri2MM_main,"Convert from Matis to MicMac"));

        aRes.push_back(cMMCom("TestBundleGen",CPP_TestBundleGen,"Unitary test for new bundle gen"));

        aRes.push_back(cMMCom("TestPhysMod",CPP_TestPhysMod_Main,"Unitary test for new bundle gen"));

        aRes.push_back(cMMCom("TestParseDir",TestElParseDir_main," Test Parse Dir"));
        aRes.push_back(cMMCom("ReechDepl",CPP_BatchReechDepl," Resample a batch of images using Px1 and Px2 displacement maps"));
        aRes.push_back(cMMCom("OneReechDepl",CPP_ReechDepl," Resample one image using Px1 and Px2 displacement maps"));
        aRes.push_back(cMMCom("OneReechFromAscii",OneReechFromAscii_main," Resample image using homography and 4 pts"));
        aRes.push_back(cMMCom("AllReechFromAscii",AllReechFromAscii_main," Resample an image pattern using homography and 4 pts"));
        aRes.push_back(cMMCom("OneReechHom",OneReechHom_main," Resample image using homography"));
        aRes.push_back(cMMCom("AllReechHom",AllReechHom_main," Resample multiple image using homography"));
        aRes.push_back(cMMCom("OneMMToAerial",OneHomMMToAerial_main," Project terrestrial image to aerial images"));
        aRes.push_back(cMMCom("AllMMToAerial",AllHomMMToAerial_main," Project a pattern of terrestrial images to aerial images"));
        aRes.push_back(cMMCom("RTI",RTI_main," RTI prototype"));
        aRes.push_back(cMMCom("RTI_RR",RTIRecalRadiom_main," RTI recalage radiom"));
        aRes.push_back(cMMCom("RTIMed",RTIMed_main," RTI calc median image"));
        aRes.push_back(cMMCom("RTIGrad",RTIGrad_main," RTI calc grad image"));
        aRes.push_back(cMMCom("RTIFilterGrad",RTIFiltrageGrad_main," RTI Filter : grad derive d'un potentiel"));
        aRes.push_back(cMMCom("RTI_RRB1",RTI_RecalRadionmBeton_main,"Recal Radiom On Image"));
        aRes.push_back(cMMCom("RTI_CLumOmbr",RTI_PosLumFromOmbre_main,"COmpute Centre Light based on shadow"));
        aRes.push_back(cMMCom("TestTomKan",Test_TomCan,"Test Tomasi Kanade"));
        aRes.push_back(cMMCom("TestMartini",TestMartini_Main,"Test Martini with simulation"));

        aRes.push_back(cMMCom("Test_Giang",TestGiangNewHomol_Main,"Test Giang"));
        aRes.push_back(cMMCom("DispHomolCom",TestGiangDispHomol_Main,"Test Giang"));
        aRes.push_back(cMMCom("GetSpace",GetSpace_main,"Delete all temporary file after treatment complete"));
        aRes.push_back(cMMCom("TiepTriPrl",TiepTriPrl_main,"Paralelliser version of TiepTri",cArgLogCom(2)));
        aRes.push_back(cMMCom("TiepTri",TiepTri_Main," Once again Test Correlation by Mesh"));
        aRes.push_back(cMMCom("TaskCorrel",TaskCorrel_main,"Creat Correlation Task XML file for TiepTri",cArgLogCom(2)));
        aRes.push_back(cMMCom("TaskCorrelGCP",TaskCorrelWithPts_main,"Creat Correlation Task XML file for GCP By Mesh",cArgLogCom(2)));
        aRes.push_back(cMMCom("FAST",FAST_main,"Some Detector interest point (FAST, FAST_NEW, DIGEO, EXTREMA)"));
        aRes.push_back(cMMCom("Homol2Way",Homol2WayNEW_main ,"Creat same pack homol in 2 way by combination 2 pack of each way"));
        aRes.push_back(cMMCom("CplFromHomol",CplFromHomol_main ,"Creat xml of pair images from Homol Folder"));
        aRes.push_back(cMMCom("LSQMatch",LSQMatch_Main ,"Giang Test LSQ"));
        aRes.push_back(cMMCom("GCPRollingBasc",GCPRollingBasc_main ,"Rolling GCPBascule"));
        aRes.push_back(cMMCom("TiepTriFar",TiepTriFar_Main ,"TestFarScene"));
        aRes.push_back(cMMCom("DetectImBlur",Test_Conv,"compute sharpness notion for each img by variance of laplacian"));
        aRes.push_back(cMMCom("CtrlCloseLoop",Test_CtrlCloseLoop,"Test Close Loop"));
        aRes.push_back(cMMCom("InitOriByBlocRigid",Test_InitBloc,"Init another camera orientation from known camera block structure and one camera ori in block"));
        aRes.push_back(cMMCom("TrajectoFromOri",Test_TrajectoFromOri,"Tracer Trajecto d'acquisition a partir de Orientation"));
        aRes.push_back(cMMCom("HomolLSMRefine",HomolLSMRefine_main,"Refine Homol Pack by Least Square Matching"));
        aRes.push_back(cMMCom("PlyBascule",PlyBascule,"Bascule PLY file with bascule XML (estimated by GCPBascule)"));
        aRes.push_back(cMMCom("ImgVide", Image_Vide, " Create image vide"));
        aRes.push_back(cMMCom("MosaicTFW", MosaicTFW, " MosaicTFW"));

        aRes.push_back(cMMCom("TestNewRechPH",Test_NewRechPH ," Test New PH"));
        aRes.push_back(cMMCom("GenTestSift",Generate_ImagSift ," Generate image with various blob"));
        aRes.push_back(cMMCom("GenImPer",Generate_ImagePer ," Generate periodic image"));
        aRes.push_back(cMMCom("MakePly_CamOrthoC",MakePly_CamOrthoC ,"Generate Ply to illustrate the long foc pb"));
        aRes.push_back(cMMCom("XMLDiffSeries",XMLDiffSeries_main ,"Generate pair images for tapioca in part c"));
        aRes.push_back(cMMCom("ZBufferRaster",ZBufferRaster_main ,"Z Buffer Raster"));


        aRes.push_back(cMMCom("ConvNewFH",ConvertToNewFormatHom_Main ,"Convert Std Tie Points to new Formats for Multiple Point"));
        aRes.push_back(cMMCom("ConvOldFH",ConvertToOldFormatHom_Main ,"Convert Multiple Tie Points to new Std Tie Points"));

        aRes.push_back(cMMCom("MergeFilterNewFH",UnionFiltragePHom_Main ,"Merge & Filter New Multiple Points"));
        aRes.push_back(cMMCom("TestYZ",TestYZ_main ,"TestYZ"));
        aRes.push_back(cMMCom("ReechHomol",ReechHomol_main ,"Apply map to homol folders to correct thermal deformation"));
        aRes.push_back(cMMCom("DeformAnalyse",DeformAnalyse_main ,"Deformation Analyse"));
        aRes.push_back(cMMCom("ExtraitHomol",ExtraitHomol_main ,"Extract certain homol files"));
        aRes.push_back(cMMCom("IntersectHomol",IntersectHomol_main ,"Pseudo-intersection for tie points"));
        aRes.push_back(cMMCom("ReechMAF",ReechMAF_main ,"Apply map to image measurement file"));
        aRes.push_back(cMMCom("ImgTMTxt2Xml",ImgTMTxt2Xml_main ,"Match tops time with image time to get GPS time"));
        aRes.push_back(cMMCom("ImgTMTxt2Xml_B",main_Txt2CplImageTime ,"Convert txt file containing camlight image name and GPS week and time into micmac format"));

        aRes.push_back(cMMCom("MoyMAF",MoyMAF_main ,"Calculate center of 4 corner points"));
        aRes.push_back(cMMCom("GenerateTP",GenerateTP_main ,"Generate simulated tie points",cArgLogCom(2)));
        aRes.push_back(cMMCom("SimuRolShut",SimuRolShut_main ,"Generate simulated tie points",cArgLogCom(2)));
        aRes.push_back(cMMCom("GenerateOrient",GenerateOrient_main,"Generate modification of orientation",cArgLogCom(2)));
        aRes.push_back(cMMCom("ReechRolShut",ReechRolShut_main ,"Resampling for rolling shutter effect correction, V2, reproj on new cam orientation",cArgLogCom(2)));
        aRes.push_back(cMMCom("ReechRolShutV1",ReechRolShutV1_main ,"Resampling for rolling shutter effect correction, V1, linear compression/dilatation"));
        aRes.push_back(cMMCom("ExportTPM",ExportTPM_main ,"Export tie point multiplicity"));
        aRes.push_back(cMMCom("CompMAF",CompMAF_main ,"Compare MAF files"));
        aRes.push_back(cMMCom("GenerateOriGPS",GenerateOriGPS_main ,"Compare MAF files"));
        aRes.push_back(cMMCom("GenerateMAF",GenerateMAF_main ,"Generate simulated MAF",cArgLogCom(2)));
        aRes.push_back(cMMCom("GenImgTM",GenImgTM_main ,"Generate fake Img name/time couple from GPS .xml file"));
        aRes.push_back(cMMCom("EsSim",EsSim_main ,"EsSim"));
        aRes.push_back(cMMCom("ProcessThmImgs",ProcessThmImgs_main,"Tool to process Thermique acquisition of IGN"));
        aRes.push_back(cMMCom("ConvertTiePPs2MM",ConvertTiePPs2MM_main,"ConvertTiePPs2MM"));
        aRes.push_back(cMMCom("DistHB",CPP_DistHistoBinaire,"Dist Binarie Code Histo of Images"));

        aRes.push_back(cMMCom("ConvHomolVSFM2MM",ConvHomolVSFM2MM_main,"Convert Tie Points from Visual SFM format (.sift & .mat) to MicMac format"));
        aRes.push_back(cMMCom("ConvTiePointPix4DMM",ConvTiePointPix4DMM_main ,"Convert tie point Pix4D Bingo to MicMac"));

        aRes.push_back(cMMCom("OrthoDirectFromDenseCloud",OrthoDirectFromDenseCloud_main ,"Ortho rectification directly from ply point cloud"));
        aRes.push_back(cMMCom("TiepGraphByCamDist",TiepGraphByCamDist_main ,"Generate Image pairs for tie points search by distant constraint"));


        aRes.push_back(cMMCom("AC_CQ",CPP_AutoCorr_CensusQuant,"Auto correl for Census Quant"));

        aRes.push_back(cMMCom("SAT4GEO_Pairs",GraphHomSat_main,"Calculate overlapping image pairs (case satellite)"));
        aRes.push_back(cMMCom("SAT4GEO_CreateEpip",CPP_AppliCreateEpi_main,"Calculate the epipolar geometry (case satellite)"));
        aRes.push_back(cMMCom("SAT4GEO_MM1P",CPP_AppliMM1P_main,"Do dense image matching in epipolar geometry (case satellite)"));
        aRes.push_back(cMMCom("SAT4GEO_EpiRPC",CPP_AppliRecalRPC_main,"Recalculate RPC for epipolar geometry images (case satellite)"));
        aRes.push_back(cMMCom("SAT4GEO_Fuse",CPP_AppliFusion_main,"Fusion of individual depth maps (case satellite)"));
		aRes.push_back(cMMCom("TransGeom", CPP_TransformGeom_main, "Transform geometry of depth map to eGeomMNTFaisceauIm1ZTerrain_Px1D"));


   }

    cCmpMMCom CmpMMCom;
    std::sort(aRes.begin(),aRes.end(),CmpMMCom);

   return AddLib(aRes,"TestLib");


}

int SampleLibElise_main(int argc, char ** argv)
{
	return GenMain(argc, argv, TestLibAvailableCommands());
}

//SateLib declarations
extern int RecalRPC_main(int argc, char ** argv);
extern int CropRPC_main(int argc, char ** argv);
extern int Grid2RPC_main(int argc, char ** argv);
extern int RPC_main(int argc, char ** argv);
extern int NewRefineModel_main(int argc, char **argv);
extern int RefineModel_main(int argc, char **argv);
extern int RefineJitter_main(int argc, char **argv);
extern int ApplyParralaxCor_main(int argc, char **argv);
extern int Dimap2Grid_main(int argc, char **argv);
extern int DimapUseRPC_main(int argc, char **argv);
extern int DigitalGlobe2Grid_main(int argc, char **argv);
extern int Aster2Grid_main(int argc, char **argv);
extern int AsterDestrip_main(int argc, char **argv);
extern int SATtoBundle_main(int argc, char ** argv);
extern int SATvalid_main(int argc, char ** argv);
extern int SATTrajectory_main(int argc, char ** argv);
extern int SatEmpriseSol_main(int argc, char ** argv);
extern int SatBBox_main(int argc, char ** argv);
extern int SatPosition_main(int argc, char ** argv);
extern int CalcBsurH_main(int argc, char ** argv);
extern int CalcBsurHGrille_main(int argc, char ** argv);
extern int CPP_SATDef2D_main(int argc, char ** argv);
extern int CPP_TestRPCDirectGen(int argc, char ** argv);
extern int CPP_TestRPCBackProj(int argc, char ** argv);
extern int CPP_TestSystematicResiduals(int argc, char ** argv);
extern int DoTile_main(int argc, char ** argv);
extern int ASTERGT2MM_main(int argc, char ** argv);
extern int ASTERGT_strip_2_MM_main(int argc, char ** argv);
extern int ASTERProjAngle_main(int argc, char ** argv);
extern int ASTERProjAngle2OtherBand_main(int argc, char ** argv);

const std::vector<cMMCom> & SateLibAvailableCommands()
{
	static std::vector<cMMCom> aRes;
	if (aRes.size()) return aRes;

	aRes.push_back(cMMCom("RecalRPC", RecalRPC_main, "Recalculate the adjusted RPCs back to geodetic coordinate system"));
	aRes.push_back(cMMCom("CropRPC", CropRPC_main, "Recalculate the RPCs for an image crop"));
	aRes.push_back(cMMCom("Grid2RPC", Grid2RPC_main, "Calculate RPCs from the GRIDs"));
	aRes.push_back(cMMCom("RPC", RPC_main, "test functions for upcoming RPC functions"));
	aRes.push_back(cMMCom("Dimap2Grid", Dimap2Grid_main, "Create a Grid file from a Dimap (SPOT or Pleiades) "));
	aRes.push_back(cMMCom("DimapUseRPC", DimapUseRPC_main, "Use Direct (image to ground) or Inverse (ground to image) RPC from Dimap file "));
	aRes.push_back(cMMCom("DigitalGlobe2Grid", DigitalGlobe2Grid_main, "Create a Grid file from a DigitalGlobe RPB file (WorldView/Geoeye/IKONOS...) "));
	aRes.push_back(cMMCom("Aster2Grid", Aster2Grid_main, "Creates a Grid file from the meta-data of an Aster Images"));
	aRes.push_back(cMMCom("ASTERGT2MM", ASTERGT2MM_main, "Convert ASTER geoTiff format to MicMac Xml, also destrip images"));
	aRes.push_back(cMMCom("ASTERStrip2MM", ASTERGT_strip_2_MM_main, "Convert a strip of ASTER geoTiff format to MicMac Xml, also destrip images"));
	aRes.push_back(cMMCom("ASTERProjAngle", ASTERProjAngle_main, "Compute the orbit angle for each point in DEM"));
	aRes.push_back(cMMCom("ASTERProjAngle2OtherBand", ASTERProjAngle2OtherBand_main, "Compute the orbit angle for each point in another band"));
	aRes.push_back(cMMCom("ApplyParralaxCor", ApplyParralaxCor_main, "Apply parralax correction from MMTestOrient to an image"));
	aRes.push_back(cMMCom("RefineModel", RefineModel_main, "Refine an approximate model "));
	aRes.push_back(cMMCom("Refine", NewRefineModel_main, "Refine an approximate model "));
	aRes.push_back(cMMCom("RefineJitter", RefineJitter_main, "/!\\ V0.01 Highly experimental /!\\ Refine a grid with Affine + jitter model based on SIFT obs"));
	aRes.push_back(cMMCom("AsterDestrip", AsterDestrip_main, "Destrip Aster Images "));
	aRes.push_back(cMMCom("SATtoBundle", SATtoBundle_main, "Export a satellite image to a grid of bundles"));
	aRes.push_back(cMMCom("SATValid", SATvalid_main, "Validate the prj function by either retrieving the line of optical centers or the provided GCPs"));
    aRes.push_back(cMMCom("SatFootprint", SatEmpriseSol_main, "Satellite foortprints in ply"));
    aRes.push_back(cMMCom("SatBBox", SatBBox_main, "Get satellite's footprint (in txt) BBox (from GRID)"));
    aRes.push_back(cMMCom("SatTrajectory", SATTrajectory_main, "Satellite trajectories in ply"));
    aRes.push_back(cMMCom("SatPosition", SatPosition_main, "Satellite position"));
    aRes.push_back(cMMCom("BsurH", CalcBsurH_main, "Calculate the b/h ratio for a pattern of images"));
	aRes.push_back(cMMCom("BsurHGRI", CalcBsurHGrille_main, "Calculate the b/h ratio for a pattern of images"));
	aRes.push_back(cMMCom("SATD2D", CPP_SATDef2D_main, "Visualize 2D deformation fields of a pushbroom image"));
	aRes.push_back(cMMCom("TestRPC", CPP_TestRPCDirectGen, "Test the calculation of direct RPCs"));
	aRes.push_back(cMMCom("TestRPCBackprj", CPP_TestRPCBackProj, "Backproject a point to images"));
	aRes.push_back(cMMCom("TestRPCSystRes", CPP_TestSystematicResiduals, "Print mean residuals for a stereo pair"));
	aRes.push_back(cMMCom("ImageTiling", DoTile_main, "Tile an image pair to selected size"));
	cCmpMMCom CmpMMCom;
	std::sort(aRes.begin(), aRes.end(), CmpMMCom);

	return AddLib(aRes, "SateLib");
}

int SateLib_main(int argc, char ** argv)
{
	return GenMain(argc, argv, SateLibAvailableCommands());
}


//===============================================
// SimuLib declarations
//===============================================

int CPP_AddNoiseImage(int, char **);
int CPP_SimulDep(int, char **);

const std::vector<cMMCom> & SimuLibAvailableCommands()
{
	static std::vector<cMMCom> aRes;
	if (aRes.size()) return aRes;

	aRes.push_back(cMMCom("AddNoise", CPP_AddNoiseImage, "Add noise to images"));
	aRes.push_back(cMMCom("SimulDep", CPP_SimulDep, "Run N Matching to average noise"));

	return AddLib(aRes, "SimuLib");
}





int SimuLib_Main(int argc, char ** argv)
{
	return GenMain(argc, argv, SimuLibAvailableCommands());
}
//================= XLib =======================

extern int XeresTest_Main(int, char**);
extern int XeresTieP_Main(int, char**);
extern int XeresMergeTieP_Main(int, char**);
extern int XeresHomMatch_main(int, char**);
extern int XeresReNameInit_main(int, char**);
extern int XeresCalibMain_main(int, char**);

const std::vector<cMMCom> & XLibAvailableCommands()
{
	static std::vector<cMMCom> aRes;

	if (aRes.empty())
	{
		aRes.push_back(cMMCom("Test", XeresTest_Main, "test Xeres"));
		aRes.push_back(cMMCom("TieP", XeresTieP_Main, "Xeres tie points"));
		aRes.push_back(cMMCom("MergeTieP", XeresMergeTieP_Main, "Xeres : merge tie points"));
		aRes.push_back(cMMCom("MatchGr", XeresHomMatch_main, "Xeres : generate graph for mathcing"));
		aRes.push_back(cMMCom("ReName0", XeresReNameInit_main, "Xeres : Rename image for Xeres convention"));
		aRes.push_back(cMMCom("Calib", XeresCalibMain_main, "Xeres : Pipeline for calibration images (corners like)"));
	}

	cCmpMMCom CmpMMCom;
	std::sort(aRes.begin(), aRes.end(), CmpMMCom);

	return AddLib(aRes, "XLib");
}

int XLib_Main(int argc, char ** argv)
{
	return GenMain(argc, argv, XLibAvailableCommands());
}

//=====================================

int GenMain(int argc, char ** argv, const std::vector<cMMCom> & aVComs)
{
	if ((argc == 1) || ((argc == 2) && (std::string(argv[1]) == "-help")))
	{
		BanniereMM3D();

		std::cout << "mm3d : Allowed commands \n";
		for (unsigned int aKC = 0; aKC<aVComs.size(); aKC++)
		{
			std::cout << " " << aVComs[aKC].mName << "\t" << aVComs[aKC].mComment << "\n";
		}
		return EXIT_SUCCESS;
	}

	if ((argc >= 2) && (argv[1][0] == 'v') && (argv[1] != std::string("vic")))
	{
		ELISE_ASSERT(ELISE_QT > 0, std::string("Qt not installed, " + std::string(argv[1]) + " not available").c_str());

		std::string cmds[] = { std::string("vMICMAC"), std::string("vApero"), std::string("vAnn"), std::string("vCalibFinale"),
			std::string("vCalibInit"), std::string("vMergeDepthMap"), std::string("vPastis"),
			std::string("vPointeInitPolyg"), std::string("vPorto"), std::string("vRechCibleDRad"),
			std::string("vRechCibleInit"), std::string("vReduc2MM"), std::string("vReducHom"),
			std::string("vSaisiePts"), std::string("vScriptCalib"), std::string("vSift"),
			std::string("vSysCoordPolyn"), std::string("vTestChantier"), std::string("vvic")
		};
		std::vector <std::string> vCmds(cmds, cmds + 19);
		if (std::find(vCmds.begin(), vCmds.end(), argv[1]) != vCmds.end())
		{
			ELISE_ASSERT(false, (argv[1] + std::string(" not available")).c_str());
		}

		MMVisualMode = true;
		MMRunVisualMode = MMRunVisualModeQt;
		argv[1]++;
	}
    
	// MPD : deplace sinon core dump qd argc==1
	// Pour l'analyse de la ligne de commande, on ne peut pas desactiver le bloquage de l'exe via l'option ExitOnBrkp
	// puisqu le XML n'a pas encore ete analyse, on change donc provisoirement le comportement par defaut
	// bool aValInit_TheExitOnBrkp=TheExitOnBrkp;
	// TheExitOnBrkp=true;
	MMD_InitArgcArgv(argc, argv);
#if ELISE_QT
	initQtLibraryPath();
#endif
	// TheExitOnBrkp=true;

	// On reactive le blocage par defaut
	// TheExitOnBrkp=aValInit_TheExitOnBrkp;

	std::string aCom = argv[1];
	// std::string aLowCom = current_program_subcommand();
	std::string aLowCom = StrToLower(aCom);  // MPD modif, sinon suggestions de marche pas en TestLib

	std::vector<cSuggest *> mSugg;

	//  std::cout << "JJJJJ " << aLowCom << " " << aCom  << " " << StrToLower(aCom) << "\n";

	cSuggest *PatMach = new cSuggest("Pattern Match", aLowCom);
	cSuggest *PrefMach = new cSuggest("Prefix Match", aLowCom + ".*");
	cSuggest *SubMach = new cSuggest("Subex Match", ".*" + aLowCom + ".*");
	mSugg.push_back(PatMach);
	mSugg.push_back(PrefMach);
	mSugg.push_back(SubMach);

	const cMMCom *toExecute = NULL;
	for (unsigned int aKC = 0; aKC<aVComs.size(); aKC++)
	{
		if (StrToLower(aVComs[aKC].mName) == StrToLower(aCom))
		{
			toExecute = &aVComs[aKC];
			break;
		}
		for (int aKS = 0; aKS<int(mSugg.size()); aKS++)
		{
			mSugg[aKS]->Test(aVComs[aKC]);
		}
	}

	// use suggestion if there is only one and no exact match has been found

	if (toExecute != NULL)
	{
		cArgLogCom aLog = toExecute->mLog;
		bool DoLog = (aLog.mNumArgDir >0) && (aLog.mNumArgDir<argc);
		string outDirectory;
		if (DoLog)
		{
			outDirectory = (isUsingSeparateDirectories() ? MMLogDirectory() : DirOfFile(argv[aLog.mNumArgDir]) + aLog.mDirSup);
			LogIn(argc, argv, outDirectory, aLog.mNumArgDir);
		}

		int aRes = toExecute->mCommand(argc - 1, argv + 1);
		if (DoLog) LogOut(aRes, outDirectory);

		delete PatMach;
		delete PrefMach;
		delete SubMach;

		if (Chol16Byte) std::cout << "WARN : 16 BYTE ACCURACY FOR LEAST SQUARE\n";

		return aRes;
	}

	for (unsigned int aKS = 0; aKS<mSugg.size(); aKS++)
	{
		if (!mSugg[aKS]->mRes.empty())
		{
			std::cout << "Suggest by " << mSugg[aKS]->mName << "\n";
			for (unsigned int aKC = 0; aKC<mSugg[aKS]->mRes.size(); aKC++)
			{
				std::cout << "    " << mSugg[aKS]->mRes[aKC].mName << "\n";
			}
			delete PatMach;
			delete PrefMach;
			delete SubMach;
			return EXIT_FAILURE;
		}
	}



	std::cout << "For command = " << argv[1] << "\n";
	ELISE_ASSERT(false, "Unknown command in mm3d");

	delete PatMach;
	delete PrefMach;
	delete SubMach;
	return  EXIT_FAILURE;
}

bool J4M()  //  indicate if we are in Jean Michael Muler Mic Mac ....
{
	return false;
}

int main(int argc, char ** argv)
{
if (0)
{
   for (int ak=0 ; ak<argc ; ak++)
    std::cout << "MMM [" << argv[ak] << "]\n";
}
	//  Genere un warning si la ligne de commande contient des caratere non ASCII, car ceux ci
	// peuvent être invisible et genere des erreurs peu comprehensibles

	{
		bool NonAsciiGot = false;
		for (int aKA = 0; aKA<argc; aKA++)
		{
			char * anArg = argv[aKA];

			for (char * aC = anArg; *aC; aC++)
			{
				if (!isascii(*aC))
				{
					if (NonAsciiGot)
					{
					}
					else
					{
						NonAsciiGot = true;
						std::cout << "WARN Non Asccii on [" << anArg << "] at pos " << aC - anArg << " Num=" << int(*(U_INT1 *)aC) << "\n";
						getchar();
					}
				}
			}
		}
	}

	// ===================
	ElTimer aT0;
	bool showDuration = false;
	if ((strcmp(argv[0], "mm3d") == 0) && J4M()) //show nothing if called by makefile  
												 // MPD @  Jean Michael, mets a jour la fonction J4M en te basant sur MPD_MM , ER_MM .....
	{
		showDuration = true;
		std::cout << "Command: ";
		for (int aK = 0; aK<argc; aK++)
		{
			std::cout << argv[aK] << " ";
		}
		std::cout << std::endl;
	}

	// transforme --AA en AA , pour la completion sur les options
	for (int aK = 0; aK<argc; aK++)
	{
		if ((argv[aK][0] == '-') && (argv[aK][1] == '-'))
			argv[aK] += 2;
	}

	int ret = GenMain(argc, argv, getAvailableCommands());

	if (showDuration) //show nothing if called by makefile
	{
		std::cout << "\nTotal duration: " << aT0.uval() << " s" << std::endl;
	}
	return ret;
}

/*
Gen Bundle => By Pattern
MMTestOrient => Export Final

Tapioca 200000
*/


void CatCom(std::vector<cMMCom> & aRes, const std::vector<cMMCom> & ToAdd)
{
	std::copy(ToAdd.begin(), ToAdd.end(), std::back_inserter(aRes));
}

const std::vector<cMMCom> & AllCom()
{
	static std::vector<cMMCom> aRes;
	if (aRes.empty())
	{
		CatCom(aRes, getAvailableCommands());
		CatCom(aRes, TestLibAvailableCommands());
		CatCom(aRes, SateLibAvailableCommands());
		CatCom(aRes, SimuLibAvailableCommands());
		CatCom(aRes, XLibAvailableCommands());
	}
	return aRes;
}

typedef std::map<std::string, cMMCom*> tDicMMCom;

tDicMMCom & DicAllCom()
{
	static tDicMMCom aRes;
	if (aRes.empty())
	{
		const std::vector<cMMCom> & aV = AllCom();
		for (int aK = 0; aK<int(aV.size()); aK++)
		{
			if (DicBoolFind(aRes, aV[aK].mName))
			{
				std::cout << "For name = " << aV[aK].mName << "\n";
				ELISE_ASSERT(false, "Conflict in MicMac Command");
			}
			aRes[aV[aK].mName] = new cMMCom(aV[aK]);
		}
	}
	return aRes;
}





void ShowAllCom()
{
	const std::vector<cMMCom> &  AllC = AllCom();
	for (int aK = 0; aK<int(AllC.size()); aK++)
	{
		std::cout << AllC[aK].mName;
		const std::string & aLib = AllC[aK].mLib;
		if (aLib != "")
			std::cout << " in " << aLib;
		std::cout << "\n";
	}
}

// A mettre en global
template <class Type> std::list<Type> List1El(const Type & aVal) { return std::list<Type>(1, aVal); }





//  ======== FEATURE =================

#define  KFEATURE 5
std::string FeatStdSubStr(const std::string &aStr) { return aStr.substr(KFEATURE, std::string::npos); }
std::string FeatStr(const eCmdMM_Feature  &aFeat) { return FeatStdSubStr(eToString(aFeat)); }

std::list<std::string>  ListMMGetFeature(const  cXml_Specif1MMCmd & aSpec)
{
	// return List1El(FeatStdSubStr(eToString(aSpec.MainFeature())));
	return List1El(FeatStr(aSpec.MainFeature()));
}
std::list<std::string>  ListMMAllFeature(const  cXml_Specif1MMCmd & aSpec)
{
	std::list<std::string> aRes(1, FeatStr(aSpec.MainFeature()));
	for (auto iT = aSpec.OtherFeature().begin(); iT != aSpec.OtherFeature().end(); iT++)
		aRes.push_back(FeatStr(*iT));

	return aRes;
}

//  ======== DATA =================

#define  KDATA    6
std::string DataStdSubStr(const std::string &aStr) { return aStr.substr(KDATA, std::string::npos); }
std::string DataStr(const eCmdMM_DataType  &aFeat) { return DataStdSubStr(eToString(aFeat)); }

std::list<std::string>  ListMMGetInput(const  cXml_Specif1MMCmd & aSpec)
{
	return List1El(DataStr(aSpec.MainInput()));
}
std::list<std::string>  ListMMAllInput(const  cXml_Specif1MMCmd & aSpec)
{
	std::list<std::string> aRes(1, DataStr(aSpec.MainInput()));
	for (auto iT = aSpec.OtherInput().begin(); iT != aSpec.OtherInput().end(); iT++)
		aRes.push_back(DataStr(*iT));

	return aRes;
}

std::list<std::string>  ListMMGetOutput(const  cXml_Specif1MMCmd & aSpec)
{
	return List1El(DataStr(aSpec.MainOutput()));
}
std::list<std::string>  ListMMAllOutput(const  cXml_Specif1MMCmd & aSpec)
{
	std::list<std::string> aRes(1, DataStr(aSpec.MainOutput()));
	for (auto iT = aSpec.OtherOutput().begin(); iT != aSpec.OtherOutput().end(); iT++)
		aRes.push_back(DataStr(*iT));

	return aRes;
}

//  ======== NAME  =================

std::list<std::string>   ListMMGetName(const  cXml_Specif1MMCmd & aSpec)
{
	return List1El(aSpec.Name());
}

// ====================================

template <class Type> void FilterOnListName(cXml_SpecifAllMMCmd & aLSpec, std::string & aName, Type aCalc, bool UseRealPat)
{
	if (EAMIsInit(&aName))
	{
		std::string aNameAuto = aName;
		if (!UseRealPat)
			aNameAuto = ".*" + aName + ".*";
		std::list<cXml_Specif1MMCmd> aRes;
		cElRegex anAuto(aNameAuto, 10, REG_EXTENDED, false);
		for (auto aL = aLSpec.OneCmd().begin(); aL != aLSpec.OneCmd().end(); aL++)
		{
			std::list<std::string> aLName = aCalc(*aL);
			bool OneMatch = false;
			for (auto itN = aLName.begin(); (itN != aLName.end()) && (!OneMatch); itN++)
			{
				OneMatch = anAuto.Match(*itN);
			}
			if (OneMatch)
			{
				aRes.push_back(*aL);
			}
		}
		aLSpec.OneCmd() = aRes;
	}
}




void ActionHelpOnHelp(int argc, char ** argv)
{
	bool Help;
	eCmdMM_Feature aFeature;
	std::string aStr = "-help";

	std::cout << "=========== For Feature  ===================\n";
	StdReadEnum(Help, aFeature, aStr, eCmf_NbVals, true, KFEATURE);

	eCmdMM_DataType aDataType;
	std::cout << "=========== For Data Type ===================\n";
	StdReadEnum(Help, aDataType, aStr, eCmDt_NbVals, true, KDATA);

}

class cAppli_MMHelp
{
public:
	std::string mSubName;
	std::string mSubMainFeature;
	std::string mSubAnyFeatures;
	std::string mSubMainInput;
	std::string mSubAnyInput;
	std::string mSubMainOutput;
	std::string mSubAnyOutput;
	std::string mShowAllCom;

	bool mUseRealPatt;
	cXml_SpecifAllMMCmd mLMMC;
	std::map<std::string, cXml_Specif1MMCmd *> mDicXmlC;
	std::string mFileXmlCmd;


	cAppli_MMHelp(int argc, char ** argv) :
		mUseRealPatt(false),
		mFileXmlCmd(Basic_XML_MM_File("HelpMMCmd.xml"))
	{
		// std::cout << "Fffffffff " << FeatStr(eCmf_Interf) << " " << DataStr(eCmDt_CloudXML) << "\n";
		TheActionOnHelp = ActionHelpOnHelp;

		ElInitArgMain
		(
			argc, argv,
			LArgMain(),  // << EAMC(aModele,"Calibration model",eSAM_None,ListOfVal(eTT_NbVals,"eTT_"))
			LArgMain() << EAM(mSubName, "Name", true, "substring for Name")

			<< EAM(mSubMainFeature, "Feature", true, "substring for main feature")
			<< EAM(mSubMainInput, "Input", true, "substring for main input")
			<< EAM(mSubMainOutput, "Output", true, "substring for main input")

			<< EAM(mSubAnyFeatures, "AllFeature", true, "substring for any feature")
			<< EAM(mSubAnyInput, "AllInput", true, "substring for any input")
			<< EAM(mSubAnyOutput, "AllOutput", true, "substring for any output")
			//==================================
			<< EAM(mShowAllCom, "ShowAllCom", true, "TUNING : file to show All Com")
			<< EAM(mFileXmlCmd, "FileXmlCmd", true, "TUNING : file to get Xml specif of Cmd")
		);

		// mLMMC = StdGetFromPCP(Basic_XML_MM_File("HelpMMCmd.xml"),Xml_SpecifAllMMCmd);
		mLMMC = StdGetFromPCP(mFileXmlCmd, Xml_SpecifAllMMCmd);

		tDicMMCom & aDicC = DicAllCom();
		// On verifie que ttes les commandes XML correspondent a du C++ et on initialise la lib/group si necessaire
		// la lib/group est initialisee  dans AllCom et donc DicAllCom
		for (auto aL = mLMMC.OneCmd().begin(); aL != mLMMC.OneCmd().end(); aL++)
		{
			cMMCom* aCom = aDicC[aL->Name()];
			if (aCom == 0)
			{
				std::cout << "For name=" << aL->Name() << "\n";
				ELISE_ASSERT(false, "is not a micmac command");
			}
			eCmdMM_Group aGrp = eCmGrp_mm3d;
			if (!aL->Group().IsInit())
			{
				// Com->mLib
				if (aCom->mLib != "")
				{
					aGrp = Str2eCmdMM_Group("eCmGrp_" + aCom->mLib);
				}
				aL->Group().SetVal(aGrp);
			}
			mDicXmlC[aL->Name()] = &(*aL);
		}


		// Sans doute provisoire genere toute les commande en indiquant si elle sont XML-documentee
		if (EAMIsInit(&mShowAllCom))
		{
			FILE * aFSAC = FopenNN(mShowAllCom, "w", "cAppli_MMHelp ShowAllCom");
			for (auto itCom = aDicC.begin(); itCom != aDicC.end(); itCom++)
			{
				std::string aName = itCom->second->mName;
				std::string aOc = DicBoolFind(mDicXmlC, aName) ? "Done1111" : "Done0000";
				fprintf(aFSAC, "%s: %s in [%s]\n", aOc.c_str(), aName.c_str(), itCom->second->mLib.c_str());
			}
			fclose(aFSAC);

			return;
		}

		FilterOnListName(mLMMC, mSubName, ListMMGetName, mUseRealPatt);

		FilterOnListName(mLMMC, mSubMainFeature, ListMMGetFeature, mUseRealPatt);
		FilterOnListName(mLMMC, mSubMainInput, ListMMGetInput, mUseRealPatt);
		FilterOnListName(mLMMC, mSubMainOutput, ListMMGetOutput, mUseRealPatt);

		FilterOnListName(mLMMC, mSubAnyFeatures, ListMMAllFeature, mUseRealPatt);
		FilterOnListName(mLMMC, mSubAnyInput, ListMMAllInput, mUseRealPatt);
		FilterOnListName(mLMMC, mSubAnyOutput, ListMMAllOutput, mUseRealPatt);

		for (auto aL = mLMMC.OneCmd().begin(); aL != mLMMC.OneCmd().end(); aL++)
		{
			std::cout << aL->Name();
			if (aL->Option().IsInit())
				std::cout << " " << aL->Option().Val();
			std::cout << " SubLib=[" << eToString(aL->Group().Val()).substr(7, std::string::npos) << "]";
			std::cout << "\n";
		}
	}
};


int CPP_MMHelp(int argc, char ** argv)
{
	cAppli_MMHelp(argc, argv);
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
