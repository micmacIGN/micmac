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

#if ELISE_QT
#include "general/visual_mainwindow.h"
#endif

// bin/Tapioca MulScale "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" 300 -1 ExpTxt=1
// bin/Tapioca All  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1  ExpTxt=1
// bin/Tapioca Line  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1   3 ExpTxt=1
// bin/Tapioca File  "../micmac_data/ExempleDoc/Boudha/MesCouples.xml" -1  ExpTxt=1

#define DEF_OFSET -12349876

// cf. CPP_Pastis.cpp
extern const std::string PASTIS_MATCH_ARGUMENT_NAME;
extern const std::string PASTIS_DETECT_ARGUMENT_NAME;
extern bool process_pastis_tool_string( string &io_tool, string &o_args );
extern const std::string PASTIS_IGNORE_MAX_NAME;
extern const std::string PASTIS_IGNORE_MIN_NAME;
extern const std::string PASTIS_IGNORE_UNKNOWN_NAME;

extern int TiepGeoref_main(int argc,char **argv);

bool ExpTxt=0;
int	ByP=-1;
string g_toolsOptions; // contains arguments to pass to Pastis concerning detecting and matching tools
std::string aDir,aPat,aPatOri;
std::string aPat2="";
std::string aFullDir;
int aFullRes = -1;
cInterfChantierNameManipulateur * anICNM =0;
std::string BinPastis;
std::string MkFT;
std::string PostFix;
std::string TheType = "XXXX";
list<string> aFileList;				// all filenames matching input pattern, computed by DoDevelop
vector<string> aKeypointsFileArray; // keypoints filenames associated to files in aFileList and a specified resolution, computed by DoDetectKeypoints
bool ignoreMin = false,
     ignoreMax = false,
     ignoreUnknown = false;

std::string StrMkT() { return (ByP ? (" \"MkF=" + MkFT +"\" ") : "") ; }

#define aNbType 6
std::string  Type[aNbType] = {"MulScale","All","Line","File","Graph","Georef"};

/*
void StdAdapt2Crochet(std::string & aStr)
{
    if (aStr=="") return;

    if ((aStr.find('@')!=(std::string::npos)) && (TheType != Type[3]))
    {
         aStr = "[["+aPat+"]]";
    }
}
*/
void StdAdapt2Crochet(std::string & aStr)
{
    if (TheType != Type[3])
    {
        GlobStdAdapt2Crochet(aStr);
    }
}


std::string RelAllIm()
{
    if (aPat2=="")
        return QUOTE(std::string("NKS-Rel-AllCpleOfPattern@")+ aPat) + std::string(" ");

    return  QUOTE(std::string("NKS-Rel-AllCpleOf2Pat@")+ aPat +"@"+aPat2) + std::string(" ");
}



std::string NKS()
{
    return
            std::string(" NKS=NKS-Assoc-CplIm2Hom@")
            + std::string(PostFix)
            + std::string("@")
            + std::string(ExpTxt? "txt" :  "dat") ;
}

void DoMkT()
{
	if (ByP) launchMake( MkFT, "all", ByP, ""/*"-s"*/, /*Tapioca stops on make failure?*/false );
}


extern std::string TheGlobSFS ;

void DoDevelopp(int aSz1,int aSz2)
{
    aFileList = anICNM->StdGetListOfFile(aPatOri,1);

    cEl_GPAO  aGPAO;
    //string post;
    string taskName;
    int iImage = 0;
    for (std::list<std::string>::const_iterator iT= aFileList.begin() ; iT!=aFileList.end() ; iT++, iImage++)
    {

        std::string  aNOri = anICNM->Dir()+*iT;
        //std::string  aNTif = NameFileStd(aNOri,1,false,true,false);

        //std::string aCom = MMBin() + "PastDevlop " + aNOri + " Sz1=" +ToString(aSz1) + " Sz2="+ToString(aSz2);
        std::string aCom = MM3dBinFile_quotes("PastDevlop") + " " + protect_spaces(aNOri) + " Sz1=" +ToString(aSz1) + " Sz2="+ToString(aSz2);

        if (TheGlobSFS!="") aCom = aCom+ " " + TheGlobSFS;
        std::cout<<aCom<<std::endl;

        taskName = string( "T" ) + ToString( iImage ) + "_";
        aGPAO.GetOrCreate( taskName, aCom ); // always call PastDevlop (in case asked resolution changed)
        aGPAO.TaskOfName("all").AddDep( taskName );
    }

    aGPAO.GenerateMakeFile(MkFT);
    DoMkT();
}


void getPastisGrayscaleFilename
     (
          const std::string & aParamDir,
          const string &i_baseName,
          int i_resolution,
          string &o_grayscaleFilename
     )
{
    // SFS
    if ( i_resolution<=0 )
    {
        // o_grayscaleFilename = NameFileStd( aParamDir+i_baseName, 1, false, true, false );
        o_grayscaleFilename = PastisNameFileStd( aParamDir+i_baseName);
        return;
    }

    // Tiff_Im aFileInit = Tiff_Im::StdConvGen( aParamDir+i_baseName, 1, false );
    Tiff_Im aFileInit = PastisTif(aParamDir+i_baseName);
    Pt2di 	imageSize = aFileInit.sz();

    double scaleFactor = double( i_resolution ) / double( ElMax( imageSize.x, imageSize.y ) );
    double round_ = 10;
    int    round_scaleFactor = round_ni( ( 1/scaleFactor )*round_ );

    o_grayscaleFilename = ( isUsingSeparateDirectories()?MMOutputDirectory():aParamDir ) + "Pastis" + ELISE_CAR_DIR + std::string( "Resol" ) + ToString( round_scaleFactor )
            + std::string("_Teta0_") + StdPrefixGen( i_baseName ) + ".tif";
}


void getPastisGrayscaleFilename( const string &i_baseName, int i_resolution, string &o_grayscaleFilename )
{
    getPastisGrayscaleFilename(aDir,i_baseName,i_resolution,o_grayscaleFilename);
}


void InitDetectingTool( std::string & detectingTool )
{
    if (0 && MPD_MM() && ( !EAMIsInit(&detectingTool) ) )
    {
         detectingTool = "mm3d:Digeo";
    }
    else if ( ( !EAMIsInit(&detectingTool) ) && MMUserEnv().TiePDetect().IsInit() )
        detectingTool = MMUserEnv().TiePDetect().Val();
}

void InitMatchingTool( std::string& matchingTool )
{
    if ( ( !EAMIsInit(&matchingTool) ) && MMUserEnv().TiePMatch().IsInit() )
        matchingTool = MMUserEnv().TiePMatch().Val();
}

// check a tool to be used by Pastis and add it to g_toolsOptions if it succeed
// i_toolType is "Detect" or "Match" for now, it must be handle by Pastis as an argument
void check_pastis_tool( string &io_tool, const string &i_toolType )
{
    if ( io_tool.length()==0 ) return;

    string extractedArguments;
    if ( !process_pastis_tool_string( io_tool, extractedArguments ) ){
        cerr << "Tapioca: ERROR: specified string \"" << io_tool << "\" for \"" << i_toolType << "\" tool is invalid (format is : tool[:arguments] )" << endl;
        ElEXIT( EXIT_FAILURE ,"check_pastis_tool");
    }
    else{
        const ExternalToolItem &item = g_externalToolHandler.get( io_tool );
        if ( !item.isCallable() ){
            cerr << "Tapioca: ERROR: specified tool \"" << io_tool << "\" is needed by \"" << i_toolType << "\" but " << item.errorMessage() << endl;
            ElEXIT( EXIT_FAILURE ,"check_pastis_tool");
        }

        if ( extractedArguments.length()!=0 ) io_tool.append( string(":") + extractedArguments );
        if ( io_tool.find(' ')!=string::npos ) io_tool = string("\"")+io_tool+"\"";
        if ( g_toolsOptions.length()!=0 ) g_toolsOptions.append( string(" ") );
        g_toolsOptions.append( i_toolType+'='+io_tool );
    }
}

void check_detect_and_match_tools( string &detectingTool, string &matchingTool, bool ignoreMax, bool ignoreMin, bool ignoreUnknown, string &ignoreMinMaxStr )
{
    g_toolsOptions.clear();

    InitDetectingTool( detectingTool );
    check_pastis_tool( detectingTool, PASTIS_DETECT_ARGUMENT_NAME );
    cout << "--- using detecting tool : [" << detectingTool << ']' << endl;

    InitMatchingTool( matchingTool );
    check_pastis_tool( matchingTool, PASTIS_MATCH_ARGUMENT_NAME );
    cout << "--- using matching tool : [" << matchingTool << ']' << endl;

    if ( ignoreMax||ignoreMin||ignoreUnknown ){
        if ( detectingTool.find( TheStrSiftPP )!=string::npos ){
            cerr << "WARNING: the detecting tool [" << TheStrSiftPP << "] is not compatible with NoMax, NoMin, NoUnknown options. [mm3d Digeo] will be used instead." << endl;
            detectingTool = "mm3d:Digeo";
        }
        if ( matchingTool.find( TheStrAnnPP )!=string::npos ){
            cerr << "WARNING: the matching tool [" << TheStrAnnPP << "] is not compatible with NoMax, NoMin, NoUnknown options. [mm3d Ann] will be used instead." << endl;
            matchingTool = "mm3d:Ann";
        }
    }

    ignoreMinMaxStr.clear();
    const string space(" ");
    if ( ignoreMax ) ignoreMinMaxStr += space + PASTIS_IGNORE_MAX_NAME + "=1";
    if ( ignoreMin ) ignoreMinMaxStr += space + PASTIS_IGNORE_MIN_NAME + "=1";
    if ( ignoreUnknown ) ignoreMinMaxStr += space + PASTIS_IGNORE_UNKNOWN_NAME + "=1";
}


int MultiEch(int argc,char ** argv, const std::string &aArg="")
{
    int aSsRes = 300;
    int aNbMinPt=2;
    int DoLowRes = 1;
    string detectingTool, matchingTool;
    string ignoreMinMaxStr;
    double ann_closeness_ratio=SIFT_ANN_DEFAULT_CLOSENESS_RATIO;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullDir,"Full Name (Dir+Pat)", eSAM_IsPatFile)
                            << EAMC(aSsRes,"Size of Low Resolution Images")
                            << EAMC(aFullRes,"Size of High Resolution Images"),
                LArgMain()  << EAM(ExpTxt,"ExpTxt",true, "Export files in text format (Def=false means binary)", eSAM_IsBool)
                << EAM(ByP,"ByP",true,"By process")
                << EAM(PostFix,"PostFix",false, "Add postfix in directory")
                << EAM(aNbMinPt,"NbMinPt",true,"Minimum number of points")
                << EAM(DoLowRes,"DLR",true,"Do Low Resolution")
                << EAM(aPat2,"Pat2",true, "Second pattern", eSAM_IsPatFile)

                << EAM(detectingTool,PASTIS_DETECT_ARGUMENT_NAME.c_str(),false)
                << EAM(matchingTool,PASTIS_MATCH_ARGUMENT_NAME.c_str(),false)

                << EAM(ignoreMax,PASTIS_IGNORE_MAX_NAME.c_str(),true)
                << EAM(ignoreMin,PASTIS_IGNORE_MIN_NAME.c_str(),true)
                << EAM(ignoreUnknown,PASTIS_IGNORE_UNKNOWN_NAME.c_str(),true)
                << EAM(ann_closeness_ratio,"Ratio",true,"ANN closeness ration (default="+ToString(ann_closeness_ratio)+"), lower is more exigeant)"),
                aArg
                );

    if (!MMVisualMode)
    {
        check_detect_and_match_tools( detectingTool, matchingTool, ignoreMax, ignoreMin, ignoreUnknown, ignoreMinMaxStr );

        if (aFullRes != -1)
        {
            std::cout << "Ss-RES = " << aSsRes << " ; Full-Res=" << aFullRes << "\n";
            ELISE_ASSERT(aFullRes>aSsRes,"High Res < Low Res, Probably 2 swap !!");
        }

        StdAdapt2Crochet(aPat2);
        DoDevelopp(aSsRes,aFullRes);

        if (DoLowRes)
        {
            std::string aSsR =
                    BinPastis
                    +  aDir + std::string(" ")
                    +  RelAllIm()     //   +  QUOTE(std::string("NKS-Rel-AllCpleOfPattern@")+ aPat) + std::string(" ")
                    +  ToString(aSsRes) + std::string(" ")
                    +  std::string(" NKS=NKS-Assoc-CplIm2Hom@_SRes@dat")
                    +  StrMkT()
                    +  std::string("NbMinPtsExp=2 ")
                    +  std::string("SsRes=1 ")
                    +  std::string("ForceByDico=1 ")
                    +  g_toolsOptions
                    /*+  ignoreMinMaxStr*/; // using only min or max in low resolution may not produce enough point


            if (TheGlobSFS!="")
                    aSsR += " isSFS=true";

            System(aSsR,true);
            DoMkT();
        }


        std::string aSFR =  BinPastis
                +  aDir + std::string(" ")
                + QUOTE(std::string("NKS-Rel-SsECh@")+ aPat+ std::string("@")+ToString(aNbMinPt)) + std::string(" ")
                +  ToString(aFullRes) + std::string(" ")
                +  StrMkT()
                +  std::string("NbMinPtsExp=2 ")
                +  std::string("ForceByDico=1 ")
                +  g_toolsOptions
                +  ignoreMinMaxStr + ' '
                +  " ratio=" + ToString(ann_closeness_ratio);

        if (TheGlobSFS!="")
                aSFR += " isSFS=true";

        aSFR += " " + NKS();

        System(aSFR,true);
        DoMkT();
    }

    return 0;
}






int All(int argc,char ** argv, const std::string &aArg="")
{
    string detectingTool, matchingTool, ignoreMinMaxStr;
    double ann_closeness_ratio=SIFT_ANN_DEFAULT_CLOSENESS_RATIO;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullDir,"Full Name (Dir+Pat)", eSAM_IsPatFile)
                            << EAMC(aFullRes,"Size of image"),
                LArgMain()  << EAM(ExpTxt,"ExpTxt",true,"Export files in text format (Def=false means binary)", eSAM_IsBool)
                << EAM(PostFix,"PostFix",false, "Add postfix in directory")
                << EAM(ByP,"ByP",true,"By process")
                << EAM(aPat2,"Pat2",true,"Second pattern", eSAM_IsPatFile)

                << EAM(detectingTool,PASTIS_DETECT_ARGUMENT_NAME.c_str(),false)
                << EAM(matchingTool,PASTIS_MATCH_ARGUMENT_NAME.c_str(),false)

                << EAM(ignoreMax,PASTIS_IGNORE_MAX_NAME.c_str(),true)
                << EAM(ignoreMin,PASTIS_IGNORE_MIN_NAME.c_str(),true)
                << EAM(ignoreUnknown,PASTIS_IGNORE_UNKNOWN_NAME.c_str(),true)
                << EAM(ann_closeness_ratio,"Ratio",true,"ANN closeness ration (default="+ToString(ann_closeness_ratio)+"), lower is more exigeant)"),
                aArg
                );

    if (!MMVisualMode)
    {
        check_detect_and_match_tools( detectingTool, matchingTool, ignoreMax, ignoreMin, ignoreUnknown, ignoreMinMaxStr );

        StdAdapt2Crochet(aPat2);
        DoDevelopp(-1,aFullRes);

        std::string aSFR =  BinPastis
                +  aDir + std::string(" ")
                +  RelAllIm()     //   +  QUOTE(std::string("NKS-Rel-AllCpleOfPattern@")+ aPat) + std::string(" ")
                +  ToString(aFullRes) + std::string(" ")
                +  StrMkT()
                +  std::string("NbMinPtsExp=2 ")
                +  std::string("ForceByDico=1 ")
                +  g_toolsOptions
                +  ignoreMinMaxStr + ' '
                +  " ratio=" + ToString(ann_closeness_ratio) ;

        if (TheGlobSFS!="")
                aSFR += " isSFS=true";

        aSFR += " " + NKS();
        std::cout<<aSFR<<std::endl<<std::endl;
        System(aSFR,true);

        DoMkT();
    }


    return 0;
}

// Variante de tapioca adaptee aux lignes resserees type video


int Line(int argc,char ** argv, const std::string &aArg="")
{
    int  aNbAdj = 5;
    bool ForceAdj = false;
    int isCirc = 0;
    string detectingTool, matchingTool, ignoreMinMaxStr;
    std::vector<int> aLineJump;
    double ann_closeness_ratio=SIFT_ANN_DEFAULT_CLOSENESS_RATIO;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullDir,"Full Name (Dir+Pat)", eSAM_IsPatFile)
                            << EAMC(aFullRes,"Image size")
                            << EAMC(aNbAdj,"Number of adjacent images to look for"),
                LArgMain()  << EAM(ExpTxt,"ExpTxt",true,"Export files in text format (Def=false means binary)", eSAM_IsBool)
                << EAM(aLineJump,"Jump",false,"Densification by jump ")
                << EAM(PostFix,"PostFix",false,"Add postfix in directory")
                << EAM(ByP,"ByP",true,"By process")
                << EAM(isCirc,"Circ",true,"In line mode if it's a loop (begin ~ end)")
                << EAM(ForceAdj,"ForceAdSupResol",true,"to force computation even when Resol < Adj")

                << EAM(detectingTool,PASTIS_DETECT_ARGUMENT_NAME.c_str(),false)
                << EAM(matchingTool,PASTIS_MATCH_ARGUMENT_NAME.c_str(),false)

                << EAM(ignoreMax,PASTIS_IGNORE_MAX_NAME.c_str(),true)
                << EAM(ignoreMin,PASTIS_IGNORE_MIN_NAME.c_str(),true)
                << EAM(ignoreUnknown,PASTIS_IGNORE_UNKNOWN_NAME.c_str(),true)
                << EAM(ann_closeness_ratio,"Ratio",true,"ANN closeness ration (default="+ToString(ann_closeness_ratio)+"), lower is more exigeant)"),
                aArg
                );

    if (!MMVisualMode)
    {
        check_detect_and_match_tools( detectingTool, matchingTool, ignoreMax, ignoreMin, ignoreUnknown, ignoreMinMaxStr );

        if ((aFullRes < aNbAdj) && (!ForceAdj) && (aFullRes>0))
        {
            std::cout << "Resol=" << aFullRes  << " NbAdjacence=" << aNbAdj << "\n";
            ELISE_ASSERT
                    (
                        false,
                        "Probable inversion of Resol and Adjacence (use ForceAdSupResol if that's what you mean)"
                        );

        }


        DoDevelopp(-1,aFullRes);

        std::string aRel = isCirc ? "NKS-Rel-ChantierCirculaire" : "NKS-Rel-ChantierLineaire";
        std::string aGenExt = std::string("@") +  aPat+ std::string("@")+ToString(aNbAdj);
        if (EAMIsInit(&aLineJump))
        {
               ELISE_ASSERT(aLineJump.size()<=2,"aLineJump")
               if (aLineJump.size()==0) aLineJump.push_back(2);
               if (aLineJump.size()==1) aLineJump.push_back(ElSquare(aLineJump.back()));

               aRel = "NKS-Rel-LinSampled" + aGenExt + "@" + ToString(isCirc) + "@" +ToString(aLineJump[0])+ "@" + ToString(aLineJump[1]);
        }
        else
        {
           aRel = aRel + aGenExt;
        }

        std::string aSFR =  BinPastis
                +  aDir + std::string(" ")
                // +  QUOTE(std::string(aRel + "@")+ aPat+ std::string("@")+ToString(aNbAdj)) + std::string(" ")
                +  QUOTE(aRel) + std::string(" ")
                +  ToString(aFullRes) + std::string(" ")
                +  StrMkT()
                +  std::string("NbMinPtsExp=2 ")
                +  std::string("ForceByDico=1 ")
                +  g_toolsOptions
                +  ignoreMinMaxStr + ' '
                +  " ratio=" + ToString(ann_closeness_ratio);
        if (TheGlobSFS!="")
                aSFR += " isSFS=true";

        aSFR += " " + NKS();

        std::cout << aSFR << "\n";
        System(aSFR,true);
        DoMkT();
    }

    return 0;
}

int Georef(int argc,char ** argv, const std::string &aArg="")
{
    TiepGeoref_main(argc,argv);

    return 0;
}

int File(int argc,char ** argv, const std::string &aArg="")
{
    string detectingTool, matchingTool, ignoreMinMaxStr;
    double ann_closeness_ratio=SIFT_ANN_DEFAULT_CLOSENESS_RATIO;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullDir,"XML-File of pair", eSAM_IsExistFile)
                            << EAMC(aFullRes,"Resolution",eSAM_None),
                LArgMain()  << EAM(ExpTxt,"ExpTxt",true, "Export files in text format (Def=false means binary)", eSAM_IsBool)
                << EAM(PostFix,"PostFix",false,"Add postfix in directory")
                << EAM(ByP,"ByP",true,"By process")

                << EAM(detectingTool,PASTIS_DETECT_ARGUMENT_NAME.c_str(),false)
                << EAM(matchingTool,PASTIS_MATCH_ARGUMENT_NAME.c_str(),false)

                << EAM(ignoreMax,PASTIS_IGNORE_MAX_NAME.c_str(),true)
                << EAM(ignoreMin,PASTIS_IGNORE_MIN_NAME.c_str(),true)
                << EAM(ignoreUnknown,PASTIS_IGNORE_UNKNOWN_NAME.c_str(),true)
                << EAM(ann_closeness_ratio,"Ratio",true,"ANN closeness ration (default="+ToString(ann_closeness_ratio)+"), lower is more exigeant)"),
                aArg
                );

    if (!MMVisualMode)
    {
        check_detect_and_match_tools( detectingTool, matchingTool, ignoreMax, ignoreMin, ignoreUnknown, ignoreMinMaxStr );

        std::string aSFR =  BinPastis
                +  aDir + std::string(" ")
                +  QUOTE(std::string("NKS-Rel-ByFile@")+  aPat) + std::string(" ")
                +  ToString(aFullRes) + std::string(" ")
                +  StrMkT()
                +  std::string("NbMinPtsExp=2 ")
                +  std::string("ForceByDico=1 ")
                +  g_toolsOptions
                +  ignoreMinMaxStr + ' '
                +  " ratio=" + ToString(ann_closeness_ratio);
        if (TheGlobSFS!="")
                aSFR += " isSFS=true";

        aSFR += " " + NKS();


        std::cout << aSFR << "\n";
        System(aSFR,true);
        DoMkT();
    }
    /*
*/

    return 0;
}

void getKeypointFilename( const string &i_basename, int i_resolution, string &o_keypointsName )
{
    o_keypointsName = ( isUsingSeparateDirectories()?MMOutputDirectory():aDir ) + anICNM->Assoc1To2( "eModeLeBrisPP-Pastis-PtInt", i_basename, ToString( i_resolution ), true) ;
}

// create a makefile to compute keypoints for all images Using Pastis' filenames format
void DoDetectKeypoints( string i_detectingTool, int i_resolution )
{
    string detectingToolArguments;
    process_pastis_tool_string( i_detectingTool, detectingToolArguments );

    string pastisGrayscaleFilename,
            keypointsFilename,
            //grayscaleDirectory, grayscaleBasename,
            command;

    if ( !ELISE_fp::MkDirSvp( ( isUsingSeparateDirectories()?MMOutputDirectory():aDir )+"Pastis" ) )
    {
        cerr << "ERROR: creation of directory [" << aDir+"Pastis" << "] failed" << endl;
        ElEXIT( EXIT_FAILURE,std::string("Creating dir ") + aDir+"Pastis" );
    }

    cEl_GPAO  aGPAO;
    size_t nbFiles = aFileList.size(),
            iImage = 0;
    aKeypointsFileArray.resize( nbFiles );
    for ( std::list<std::string>::const_iterator iT=aFileList.begin(); iT!=aFileList.end(); iT++, iImage++ ) // aFileList has been computed by DoDevelopp
    {
        getPastisGrayscaleFilename( *iT, i_resolution, pastisGrayscaleFilename );
        //SplitDirAndFile( grayscaleDirectory, grayscaleBasename, pastisGrayscaleFilename );
        getKeypointFilename( *iT, i_resolution, aKeypointsFileArray[iImage] );
        keypointsFilename = aKeypointsFileArray[iImage];

        command = g_externalToolHandler.get( i_detectingTool ).callName() + ' ' + detectingToolArguments + ' ' +
                pastisGrayscaleFilename + " -o " + keypointsFilename;

        aGPAO.GetOrCreate( keypointsFilename, command );
        aGPAO.TaskOfName("all").AddDep( keypointsFilename );
    }

    aGPAO.GenerateMakeFile( MkFT );
    DoMkT();
}

void print_graph( const vector<vector<int> > &i_graph )
{
    size_t nbFiles = i_graph.size(),
           i, j;
    for ( j=0; j<nbFiles; j++ )
    {
        for ( i=0; i<nbFiles; i++ )
            cout << i_graph[j][i] << '\t';
        cout << endl;
    }
}

void writeBinaryGraphToXML( const string &i_filename, const vector<vector<int> > &i_graph )
{
    // convert images' filenames list into an array
    size_t nbFiles = aFileList.size();
    vector<string> filenames( nbFiles );
    copy( aFileList.begin(), aFileList.end(), filenames.begin() );

    ofstream f( i_filename.c_str() );
    f << "<?xml version=\"1.0\" ?>" << endl;
    f << "<SauvegardeNamedRel>" << endl;
    size_t i, j;
    for ( j=1; j<nbFiles; j++ )
        for ( i=0; i<j; i++ )
        {
            if ( i_graph[j][i]!=0 )
            {
                f << "\t<Cple>" << filenames[j] << ' ' << filenames[i] << "</Cple>" << endl;
                //f << "\t<Cple>" << filenames[i] << ' ' << filenames[j] << "</Cple>" << endl;
            }
        }
    f << "</SauvegardeNamedRel>" << endl;
}

// i_graph[i][j] represent the connexion between images aFileList[i] and aFileList[j]
// i_graph[i][j] = number of points of i whose nearest neighbour is in j
// a couple of images is output if i_graph[i][j]+i_graph[j][i]>i_threshold
size_t normalizeGraph( vector<vector<int> > &i_graph, int i_threshold  )
{
    size_t n = i_graph.size(),
            i, j;
    size_t count = 0;
    for ( j=1; j<n; j++ )
        for ( i=0; i<j; i++ )
        {
            int sum = i_graph[i][j]+i_graph[j][i];
            if ( sum>=i_threshold )
            {
                i_graph[j][i] = sum;
                count++;
            }
            else
                i_graph[j][i] = 0;
            i_graph[i][j] = 0;
        }
    return count;
}

void setLabel( vector<vector<int> > &i_graph, vector<int> &i_labels, size_t i_index, int i_label )
{
    if ( i_labels[i_index]==-1 )
    {
        size_t n = i_graph.size(),
                i;
        i_labels[i_index] = i_label;
        for ( i=0; i<i_index; i++ )
            if ( i_graph[i_index][i]!=0 ) setLabel( i_graph, i_labels, i, i_label );
        for ( i=i_index+1; i<n; i++ )
            if ( i_graph[i][i_index]!=0 ) setLabel( i_graph, i_labels, i, i_label );
    }
}

// delete points with a scale lesser than i_minScale of greater than i_maxScale
void delete_out_of_bound_scales( vector<DigeoPoint> &io_points, REAL i_minScale, REAL i_maxScale )
{
    vector<DigeoPoint> points( io_points.size() );
    size_t nbKeptPoints = 0;
    for ( size_t iPoint=0; iPoint<io_points.size(); iPoint++ )
    {
        if ( io_points[iPoint].scale>=i_minScale &&
             io_points[iPoint].scale<=i_maxScale )
            points[nbKeptPoints++]=io_points[iPoint];
    }
    points.resize( nbKeptPoints );
    points.swap( io_points );
}

// load all keypoints from their files and construct the proximity graph
void DoConstructGraph( const string &i_outputFilename, size_t i_nbMaxPointsPerImage, REAL i_minScale, REAL i_maxScale, int i_nbRequiredMatches, bool i_printGraph )
{
    size_t nbImages = aFileList.size();
    string keypointsFilename;
    vector<vector<DigeoPoint> > keypoints_per_image( nbImages );
    vector<DigeoPoint> all_keypoints; // a big vector with all keypoints of all images
    vector<int> all_image_indices; // contains the index of the image from which the keypoint is from
    size_t iImage = 0,
            nbTotalKeypoints = 0,
            addedPoints;

    // read all keypoints files
    cout << "--------------------> read all keypoints files" << endl;
    for ( std::list<std::string>::const_iterator iT=aFileList.begin(); iT!=aFileList.end(); iT++, iImage++ ) // aFileList has been computed by DoDevelopp
    {
        keypointsFilename = aKeypointsFileArray[iImage];

        if ( !DigeoPoint::readDigeoFile( keypointsFilename, false/*do no use multiple angles*/, keypoints_per_image[iImage] ) ){
            cerr << "WARNING: unable to read keypoints in [" << keypointsFilename << "], image [" << *iT << "] will be ignored" << endl;
            continue;
        }

        cout << keypointsFilename << endl;
        cout << "\t- " << keypoints_per_image[iImage].size() << " keypoints" << endl;
        if ( i_minScale!=std::numeric_limits<REAL>::min() || i_maxScale!=std::numeric_limits<REAL>::max() )
        {
            delete_out_of_bound_scales( keypoints_per_image[iImage], i_minScale, i_maxScale );
            cout << "\t- " << keypoints_per_image[iImage].size() << " inside scale bounds" << endl;
        }

        if ( keypoints_per_image[iImage].size()>=i_nbMaxPointsPerImage )
        {
            DigeoPoint *data = &( keypoints_per_image[iImage][0] );
            size_t nbPoints = keypoints_per_image[iImage].size();
            std::copy( data+nbPoints-i_nbMaxPointsPerImage, data+nbPoints, data );
            keypoints_per_image[iImage].resize( i_nbMaxPointsPerImage );
            addedPoints = i_nbMaxPointsPerImage;
        }
        else
            addedPoints = keypoints_per_image[iImage].size();

        nbTotalKeypoints += addedPoints;
        cout << "\t- " << keypoints_per_image[iImage].size() << " added" << endl;
    }

    if ( nbTotalKeypoints==0 )
    {
        cerr << "ERROR: no keypoint found, output file will not be generated" << endl;
        return;
    }

    cout << "total number of points = " << nbTotalKeypoints << endl;
    // merge all keypoints vectors
    size_t nbPoints;
    all_keypoints.resize( nbTotalKeypoints );
    all_image_indices.resize( nbTotalKeypoints );
    vector<vector<DigeoPoint> >::const_iterator itSrc = keypoints_per_image.begin();
    const DigeoPoint *pSrc;
    DigeoPoint *pDst = &( all_keypoints[0] );
    int *itIndex = &( all_image_indices[0] );
    const int nbImages_int = (int)nbImages;
    for ( int i = 0; i < nbImages_int; i++, itSrc++ )
    {
        nbPoints = itSrc->size();
        if ( nbPoints==0 ) continue;

        pSrc = &( ( *itSrc )[0] );
        memcpy( static_cast<void*>(pDst), pSrc, nbPoints*sizeof( DigeoPoint ) );
        pDst += nbPoints;
        while (nbPoints--) *itIndex++ = i;
    }

    // create a connectivity matrix
    vector<vector<int> > graph( nbImages );
    for ( iImage=0; iImage<nbImages; iImage++ )
        graph[iImage].resize( nbImages, 0 );

    AnnArray annArray( all_keypoints, SIFT_ANN_DESC_SEARCH );
    AnnSearcher search;
    search.setNbNeighbours( 2 );
    search.setErrorBound( 0. );
    search.setMaxVisitedPoints( SIFT_ANN_DEFAULT_MAX_PRI_POINTS );
    search.createTree( annArray );
    DigeoPoint *query = &( all_keypoints[0] );
    const ANNidx *neighbours = search.getNeighboursIndices();
    size_t iImageQuery, iImageNeighbour,
            nbBadNeighbours = 0,
            iQuery;
    for ( iQuery=0; iQuery<nbTotalKeypoints; iQuery++ )
    {
        search.search( query->descriptor(0) );
        iImageQuery 	= all_image_indices[iQuery];
        iImageNeighbour = all_image_indices[neighbours[1]];

        if ( iImageQuery==iImageNeighbour )
            nbBadNeighbours++;
        else
            graph[iImageQuery][iImageNeighbour]++;

        query++;
    }
    annClose(); // done with ANN

    //print_graph( graph );

    // stats
    cout << nbImages << " images" << endl;
    cout << nbBadNeighbours << '/' << nbTotalKeypoints << " rejected points (neighbours from the same image)" << endl;
    size_t nbChecks = normalizeGraph( graph, i_nbRequiredMatches );
    cout << nbChecks << " / " << ( nbImages*(nbImages-1) )/2 << endl;

    if ( i_printGraph ) print_graph( graph );

    // compute number of connected components
    vector<int> labels( nbImages, -1 );
    int currentLabel = 0;
    for ( size_t iStartingElement=0; iStartingElement<nbImages; iStartingElement++ )
        if ( labels[iStartingElement]==-1 ) setLabel( graph, labels, iStartingElement, currentLabel++ );
    cout << currentLabel << " connected component" << (currentLabel>1?'s':'\0') << endl;

    writeBinaryGraphToXML( i_outputFilename, graph );
}


// this option is to construct a proximity graph from points of interests
// it generates an XML file to process with the "File" option
int Graph_(int argc,char ** argv, const std::string &aArg="")
{
    int nbThreads = NbProcSys(); // default is the number of cores of the system
    int maxDimensionResize = -1;
    int nbMaxPoints = 200;
    REAL minScaleThreshold = std::numeric_limits<REAL>::min(),
            maxScaleThreshold = std::numeric_limits<REAL>::max();
    int nbRequiredMatches = 1;
    string defaultOutputName = "tapioca_connectivity_graph.xml"; // default XML filename for the graph
    string outputFile = defaultOutputName;
    string detectingTool;
    bool printGraph = false;

    // aDir is "chantier" directory
    // aPat is images' pattern
    // aFullPat is the original directory+pattern string

    ElInitArgMain
            (
                argc,argv,

                LArgMain()  << EAMC(aFullDir,"Full images' pattern (directory+pattern)", eSAM_IsPatFile)
                << EAMC(maxDimensionResize,"Processing size of image (for the greater dimension)", eSAM_None),

                LArgMain()  << EAM(nbThreads, "ByP", true, "By process")
                << EAM(detectingTool, PASTIS_DETECT_ARGUMENT_NAME.c_str(), true, "executable used to detect keypoints")
                << EAM(nbMaxPoints, "MaxPoint", true, "number of points used per image to construct the graph (default 200)")
                << EAM(minScaleThreshold, "MinScale", true, "if specified, points with a lesser scale are ignored")
                << EAM(maxScaleThreshold, "MaxScale", true, "if specified, points with a greater scale are ignored")
                << EAM(nbRequiredMatches, "NbRequired", true, "number of matches to create a connexion between two images (default 1)")
                << EAM(outputFile, "Out", true, "name of the produced XML file")
                << EAM(printGraph, "PrintGraph", true, "print result graph in standard output"),
                aArg
                );

    if (!MMVisualMode)
    {
        // if no output filename is given, use the default one in "chantier" directory
        if ( !EAMIsInit(&outputFile) || (outputFile == defaultOutputName)) //second condition for MMVisualMode
        {
            outputFile = aDir+outputFile;
            cout << "no output filename specified Using default: " << outputFile << endl;
        }


        // retrieve points of interest detecting program
        g_toolsOptions.clear();
        InitDetectingTool( detectingTool );
        if ( detectingTool.length()==0 ) detectingTool=TheStrSiftPP;
        check_pastis_tool( detectingTool, PASTIS_DETECT_ARGUMENT_NAME );

        cout << "chantierDirectory  = " << aDir << endl;
        cout << "pattern            = " << aPat << endl;
        cout << "maxDimensionResize = " << maxDimensionResize << endl;
        cout << "------" << endl;
        cout << "nbThreads          = " << nbThreads << endl;
        cout << "nbMaxPoints        = " << nbMaxPoints << endl;
        cout << "minScaleThreshold  = " << minScaleThreshold << endl;
        cout << "maxScaleThreshold  = " << maxScaleThreshold << endl;
        cout << "outputFile         = " << outputFile << endl;
        cout << "detectingTool      = " << detectingTool << endl;
        cout << "g_toolsOptions     = " << g_toolsOptions << endl;

        // convert images into TIFF and resize them if needed (maxDimensionResize!=-1)
        DoDevelopp( -1,maxDimensionResize );

        // create a makefile to detect key points for all images
        DoDetectKeypoints( detectingTool, maxDimensionResize );

        cout << "--------------------> DoDetectKeypoints" << endl;

        DoConstructGraph( outputFile, nbMaxPoints, minScaleThreshold, maxScaleThreshold, nbRequiredMatches, printGraph );
        /*
    check_detect_and_match_tools( detectingTool, matchingTool );

    if ((aFullRes < aNbAdj) && (!ForceAdj) && (aFullRes>0))
    {
        std::cout << "Resol=" << aFullRes  << " NbAdjacence=" << aNbAdj << "\n";
        ELISE_ASSERT
        (
             false,
             "Probable inversion of Resol and Adjacence (use ForceAdSupResol if that's what you mean)"
        );

    }


    DoDevelopp(-1,aFullRes);

   std::string aRel = isCirc ? "NKS-Rel-ChantierCirculaire" : "NKS-Rel-ChantierLineaire";

    std::string aSFR =  BinPastis
                     +  aDir + std::string(" ")
                     +  QUOTE(std::string(aRel + "@")+ aPat+ std::string("@")+ToString(aNbAdj)) + std::string(" ")
                     +  ToString(aFullRes) + std::string(" ")
                     +  StrMkT()
                     +  std::string("NbMinPtsExp=2 ")
                     +  std::string("ForceByDico=1 ")
                     +  g_toolsOptions + ' '
                     +  NKS();


    std::cout << aSFR << "\n";
    System(aSFR,true);
    DoMkT();
*/
        return EXIT_SUCCESS;
    }
    else
        return EXIT_SUCCESS;
}

void Del_MkTapioca(string MkFT)
{
    //Delete MkTapioca
    if (!MkFT.empty())
    {
        std::string cmdDLMkTapioca;
    #if (ELISE_unix || ELISE_Cygwin || ELISE_MacOs)
        cmdDLMkTapioca = "rm " + MkFT;
    #endif
    #if (ELISE_windows)
        replace(MkFT.begin(), MkFT.end(), '/', '\\');
        cmdDLMkTapioca = "del /Q " + MkFT;
    #endif
        system_call(cmdDLMkTapioca.c_str());
    }
}

int Tapioca_main(int argc,char ** argv)
{
#if ELISE_QT
    if (MMVisualMode)
    {
        QApplication app(argc, argv);

        QStringList items;

        for (int aK=0; aK < aNbType; ++aK)
            items << QString((Type[aK]).c_str());

        setStyleSheet(app);

        int  defaultItem = 0;

        if(argc > 1)
            defaultItem = items.indexOf(QString(argv[1]));

        bool ok = false;
        QString item = QInputDialog::getItem(NULL, app.applicationName(),
                                             QString ("Strategy"), items, defaultItem, false, &ok);

        if (ok && !item.isEmpty())
            TheType = item.toStdString();
        else
            return EXIT_FAILURE;
    }
    else{
       ELISE_ASSERT(argc >= 2,"Not enough arg");
       TheType = argv[1];
    }
#else
    ELISE_ASSERT(argc >= 2,"Not enough arg");
    TheType = argv[1];
#endif

    MMD_InitArgcArgv(argc,argv);

    int ARGC0 = argc;

    //  APRES AVOIR SAUVEGARDER L'ARGUMENT DE TYPE ON LE SUPPRIME
    if (argc>=2)
    {
        argv[1] = argv[0];
        argv++; argc--;
    }

    if (argc>=3)
    {
        aFullDir = argv[1];

#if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
#endif
        SplitDirAndFile(aDir,aPat,aFullDir);


        if ( isUsingSeparateDirectories() )
            ELISE_fp::MkDirSvp( MMTemporaryDirectory() );
        else
            ELISE_fp::MkDirSvp( aDir + "Tmp-MM-Dir/");


    aPatOri = aPat;

    StdAdapt2Crochet(aPat);
    /*
    if ((aPat.find('@')!=(std::string::npos)) && (TheType != Type[3]))
    {
         aPat = "[["+aPat+"]]";
    }
*/

    if ( isUsingSeparateDirectories() )
       MkFT= MMTemporaryDirectory()+"MkTapioca";
    else
       MkFT= aDir  + "MkTapioca";
    // MkFT= MMDir() + "MkTapioca";
    //BinPastis = MM3dBinFile("Pastis");
    BinPastis = MM3dBinFile_quotes("Pastis");

    ByP= MMNbProc();

    cTplValGesInit<std::string>  aTplFCND;
    anICNM = cInterfChantierNameManipulateur::StdAlloc(argc,argv,aDir,aTplFCND);

    MakeXmlXifInfo(aFullDir,anICNM);

    }

    if (TheType == Type[0])
    {
        int aRes = MultiEch(argc, argv, TheType);
        Del_MkTapioca(MkFT);
        BanniereMM3D();
        return aRes;
    }
    else if (TheType == Type[1])
    {
        int aRes = All(argc, argv, TheType);
        Del_MkTapioca(MkFT);
        BanniereMM3D();
        return aRes;
    }
    else if (TheType == Type[2])
    {
        int aRes = Line(argc, argv, TheType);
        Del_MkTapioca(MkFT);
        BanniereMM3D();
        return aRes;
    }
    else if (TheType == Type[3])
    {
        int aRes = File(argc, argv, TheType);
        Del_MkTapioca(MkFT);
        BanniereMM3D();
        return aRes;
    }
    else if (TheType == Type[4])
    {
        int aRes = Graph_(argc, argv, TheType);
        Del_MkTapioca(MkFT);
        BanniereMM3D();
        return aRes;
    }
    else if (TheType == Type[5])
    {
        int aRes = Georef(argc, argv, TheType);
        Del_MkTapioca(MkFT);
        BanniereMM3D();
        return aRes;
    }


    bool Error = (ARGC0>=2 ) && (TheType!= std::string("-help"));
    if (Error)
    {
        std::cout << "TAPIOCA: ERROR: unknown command : " << TheType << endl;
    }
    std::cout << "Allowed commands are : \n";
    for (int aK=0 ; aK<aNbType ; aK++)
        std::cout << "\t" << Type[aK] << "\n";
    if (!Error)
    {
        std::cout << "for details : \n";
        std::cout << "\t Tapioca MulScale -help\n";
        std::cout << "\t Tapioca All -help\n";
        std::cout << "\t...\n";
    }

    return EXIT_FAILURE;
}


/************************************************************************/
/*                                                                      */
/*             Nouvelle commande, compatible vTools                      */
/*                                                                      */
/************************************************************************/


        //=============== cArgMainTieP  ===================

class cArgMainTieP
{
     public  :
          int Exe();
     protected :
          cArgMainTieP(int argc,char ** argv,const std::string & aNameCom);

          std::string  CommandMand();
          std::string  CommandOpt();

          std::string        mNameCom;
          std::string        mFullDir;
          bool               mExpTxt;
          int                mByP;
          std::string        mPostFix;
          std::string        mDetectingTool;
          bool               mCirc;

          LArgMain CommonMandatory();
          
          // LArgMain CommonOptionnel();
          // LArgMain CommonOptionnel();
          LArgMain mComObl;
          LArgMain mComOpt;

          std::string mComGlob;
};




cArgMainTieP::cArgMainTieP(int argc,char ** argv,const std::string & aNameCom) :
    mNameCom (aNameCom),
    mExpTxt  (false),
    mByP     (-1),
    mPostFix (""),
    mCirc    (false)
{
    mComObl  << EAMC(mFullDir,"Full Name (Dir+Pat)", eSAM_IsPatFile)
    ;
    mComOpt  << EAM(mExpTxt,"ExpTxt",true, "Export files in text format (Def=false means binary)", eSAM_IsBool)
             << EAM(mByP,"ByP",true,"By process")
             << EAM(mPostFix,"PostFix",false, "Add postfix in directory")
             << EAM(mDetectingTool,PASTIS_DETECT_ARGUMENT_NAME.c_str(),false)
             << EAM(mCirc,"Circ",true,"Circular in mode Line")
    ;
}

int cArgMainTieP::Exe()
{
   if (!MMVisualMode)
   {
      return System(mComGlob);
   }
   return EXIT_SUCCESS;
}

LArgMain cArgMainTieP::CommonMandatory()
{
    return LArgMain()  << EAMC(aFullDir,"Full Name (Dir+Pat)", eSAM_IsPatFile);
}
std::string cArgMainTieP::CommandMand()
{
    return MM3dBinFile_quotes("Tapioca") + " " + mNameCom + " "+ QUOTE(mFullDir);
}


/*
                       << EAM(matchingTool,PASTIS_MATCH_ARGUMENT_NAME.c_str(),false)
                       << EAM(ignoreMax,PASTIS_IGNORE_MAX_NAME.c_str(),true)
                       << EAM(ignoreMin,PASTIS_IGNORE_MIN_NAME.c_str(),true)
                       << EAM(ignoreUnknown,PASTIS_IGNORE_UNKNOWN_NAME.c_str(),true),
int MultiEch(int argc,char ** argv, const std::string &aArg="")
*/

std::string cArgMainTieP::CommandOpt()
{
    std::string aRes = " ";

    if (EAMIsInit(&mExpTxt))        aRes += " ExpTxt="  + ToString(mExpTxt);
    if (EAMIsInit(&mCirc))        aRes += " Circ="  + ToString(mCirc);
    if (EAMIsInit(&mByP))           aRes += " ByP="     + ToString(mByP);
    if (EAMIsInit(&mPostFix))       aRes += " PostFix=" + mPostFix;
    if (EAMIsInit(&mDetectingTool)) aRes += " " + PASTIS_DETECT_ARGUMENT_NAME + "=" + mDetectingTool;

    return aRes + " ";
}

        //=============== cArgMainTiePMS  ===================

class cArgMainTiePMS : public cArgMainTieP
{
      public :
          cArgMainTiePMS(int argc,char ** argv);
      private :
          int mSsRes;
          int mFullRes;
};

cArgMainTiePMS::cArgMainTiePMS(int argc,char ** argv) :
   cArgMainTieP(argc,argv,"MulScale"),
   mSsRes     (500),
   mFullRes   (2000)
{
    ElInitArgMain
    (
        argc,argv,
        mComObl << EAMC(mSsRes,"Size of Low Resolution Images")
                << EAMC(mFullRes,"Size of High Resolution Images"),
        mComOpt
    );
    mComGlob =   CommandMand() +  " " + ToString(mSsRes) +  " " + ToString(mFullRes) + CommandOpt() ;
}

int TiePMS_main(int argc,char ** argv)
{
    cArgMainTiePMS anArg(argc,argv);
    return anArg.Exe();
}


        //=============== cArgMainTiePAll  ===================

class cArgMainTiePAll : public cArgMainTieP
{
      public :
          cArgMainTiePAll(int argc,char ** argv);
      private :
          int mFullRes;
};

cArgMainTiePAll::cArgMainTiePAll(int argc,char ** argv) :
   cArgMainTieP(argc,argv,"All"),
   mFullRes   (2000)
{
    ElInitArgMain
    (
        argc,argv,
        mComObl << EAMC(mFullRes,"Size of image"),
        mComOpt
    );
    mComGlob =   CommandMand() + " " +   ToString(mFullRes) + CommandOpt() ;
}

int TiePAll_main(int argc,char ** argv)
{
    cArgMainTiePAll anArg(argc,argv);
    return anArg.Exe();
}

        //=============== cArgMainTiePLine  ===================

class cArgMainTiePLine : public cArgMainTieP
{
      public :
          cArgMainTiePLine(int argc,char ** argv);
      private :
          int mFullRes;
          int mNbAdj;
};

cArgMainTiePLine::cArgMainTiePLine(int argc,char ** argv) :
   cArgMainTieP(argc,argv,"Line"),
   mFullRes   (2000),
   mNbAdj     (4)
{
    ElInitArgMain
    (
        argc,argv,
        mComObl << EAMC(mFullRes,"Image size")
                << EAMC(mNbAdj,"Number of adjacent images to look for"),
        mComOpt
    );
    mComGlob =   CommandMand() +  " " + ToString(mFullRes) +  " " + ToString(mNbAdj) + CommandOpt() ;
}

int TiePLine_main(int argc,char ** argv)
{
    cArgMainTiePLine anArg(argc,argv);
    return anArg.Exe();
}



/***********************************************************************************/

class cAppliMakeFileHom
{
    public :
         cAppliMakeFileHom(int argc,char **argv,int SzExecute);
    private :
         void Add(std::string aS1,std::string aS2);
         std::set<std::pair<std::string,std::string> > mRes;
         bool                                          mSym;
};







/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilite au code source et des droits de copie,
   de modification et de redistribution accordes par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitee.  Pour les m�mes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement,  a  l'utilisation,  a  la modification et/ou au
   developpement et a  la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a  des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a  charger  et  tester  l'adequation  du
   logiciel a  leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
