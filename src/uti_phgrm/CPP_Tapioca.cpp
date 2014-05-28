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

#if (ELISE_QT_VERSION >= 4)
#ifdef Int
#undef Int
#endif

#include <QApplication>
#include <QInputDialog>

#include "general/visual_mainwindow.h"
#endif

// bin/Tapioca MulScale "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" 300 -1 ExpTxt=1
// bin/Tapioca All  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1  ExpTxt=1
// bin/Tapioca Line  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1   3 ExpTxt=1
// bin/Tapioca File  "../micmac_data/ExempleDoc/Boudha/MesCouples.xml" -1  ExpTxt=1

#define DEF_OFSET -12349876
// cf. CPP_Pastis.cpp
extern const string PASTIS_MATCH_ARGUMENT_NAME;
extern const string PASTIS_DETECT_ARGUMENT_NAME;
extern bool process_pastis_tool_string( string &io_tool, string &o_args );

int ExpTxt=0;
int	ByP=-1;
string g_toolsOptions; // contains arguments to pass to Pastis concerning detecting and matching tools
std::string aDir,aPat,aPatOri;
std::string aPat2="";
std::string aFullDir;
int aFullRes;
cInterfChantierNameManipulateur * anICNM =0;
std::string BinPastis;
std::string MkFT;
std::string PostFix;
std::string TheType = "XXXX";
list<string> aFileList;				// all filenames matching input pattern, computed by DoDevelop
vector<string> aKeypointsFileArray; // keypoints filenames associated to files in aFileList and a specified resolution, computed by DoDetectKeypoints

std::string StrMkT() { return (ByP ? (" \"MkF=" + MkFT +"\" ") : "") ; }

#define aNbType 5
std::string  Type[aNbType] = {"MulScale","All","Line","File","Graph"};

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
    if (ByP)
    {
        //std::string aSMkSr = string("\"")+g_externalToolHandler.get( "make" ).callName()+"\" all -f \"" + MkFT + string("\" -j")+ToString(ByP)/*+" -s"*/;
        //System(aSMkSr,true);
        launchMake( MkFT, "all", ByP, ""/*"-s"*/, /*Tapioca stops on make failure?*/false );
    }
}


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

        taskName = string( "T" ) + ToString( iImage ) + "_";
        aGPAO.GetOrCreate( taskName, aCom ); // always call PastDevlop (in case asked resolution changed)
        aGPAO.TaskOfName("all").AddDep( taskName );
    }

    aGPAO.GenerateMakeFile(MkFT);

    DoMkT();
}


void getPastisGrayscaleFilename( const string &i_baseName, int i_resolution, string &o_grayscaleFilename )
{
    if ( i_resolution<=0 )
    {
        o_grayscaleFilename = NameFileStd( aDir+i_baseName, 1, false, true, false );
        return;
    }

    Tiff_Im aFileInit = Tiff_Im::StdConvGen( aDir+i_baseName, 1, false );
    Pt2di 	imageSize = aFileInit.sz();

    double scaleFactor = double( i_resolution ) / double( ElMax( imageSize.x, imageSize.y ) );
    double round_ = 10;
    int    round_scaleFactor = round_ni( ( 1/scaleFactor )*round_ );

    o_grayscaleFilename = aDir + "Pastis" + ELISE_CAR_DIR + std::string( "Resol" ) + ToString( round_scaleFactor )
            + std::string("_Teta0_") + StdPrefixGen( i_baseName ) + ".tif";
}

void InitDetectingTool( std::string & detectingTool )
{
    if ( ( !EAMIsInit(&detectingTool) ) && MMUserEnv().TiePDetect().IsInit() )
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
    else
    {
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

void check_detect_and_match_tools( string &detectingTool, string &matchingTool )
{
    g_toolsOptions.clear();

    InitDetectingTool( detectingTool );
    check_pastis_tool( detectingTool, PASTIS_DETECT_ARGUMENT_NAME );

    InitMatchingTool( matchingTool );
    check_pastis_tool( matchingTool, PASTIS_MATCH_ARGUMENT_NAME );
}


int MultiEch(int argc,char ** argv, const std::string &aArg="")
{
    int aSsRes = 300;
    int aNbMinPt=2;
    int DoLowRes = 1;
    string detectingTool, matchingTool;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullDir,"Full Name (Dir+Pat)", eSAM_IsPatFile)
                            << EAMC(aSsRes,"Size of Low Resolution Images")
                            << EAMC(aFullRes,"Size of High Resolution Images"),
                LArgMain()  << EAM(ExpTxt,"ExpTxt",true)
                << EAM(ByP,"ByP",true)
                << EAM(PostFix,"PostFix",false)
                << EAM(aNbMinPt,"NbMinPt",true)
                << EAM(DoLowRes,"DLR",true,"Do Low Resolution")
                << EAM(aPat2,"Pat2",true)
                << EAM(detectingTool,PASTIS_DETECT_ARGUMENT_NAME.c_str(),false)
                << EAM(matchingTool,PASTIS_MATCH_ARGUMENT_NAME.c_str(),false),
                aArg
                );

    if (!MMVisualMode)
    {
        check_detect_and_match_tools( detectingTool, matchingTool );

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
                    +  g_toolsOptions;

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
                +  g_toolsOptions + ' '
                +  NKS();

        System(aSFR,true);
        DoMkT();
    }

    return 0;
}

int All(int argc,char ** argv, const std::string &aArg="")
{
    string detectingTool, matchingTool;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullDir,"Full Name (Dir+Pat)", eSAM_IsPatFile)
                            << EAMC(aFullRes,"Size of image"),
                LArgMain()  << EAM(ExpTxt,"ExpTxt",true)
                << EAM(PostFix,"PostFix",false)
                << EAM(ByP,"ByP",true)
                << EAM(aPat2,"Pat2",true)
                << EAM(detectingTool,PASTIS_DETECT_ARGUMENT_NAME.c_str(),false)
                << EAM(matchingTool,PASTIS_MATCH_ARGUMENT_NAME.c_str(),false),
                aArg
                );

    if (!MMVisualMode)
    {
        check_detect_and_match_tools( detectingTool, matchingTool );

        StdAdapt2Crochet(aPat2);
        DoDevelopp(-1,aFullRes);

        std::string aSFR =  BinPastis
                +  aDir + std::string(" ")
                +  RelAllIm()     //   +  QUOTE(std::string("NKS-Rel-AllCpleOfPattern@")+ aPat) + std::string(" ")
                +  ToString(aFullRes) + std::string(" ")
                +  StrMkT()
                +  std::string("NbMinPtsExp=2 ")
                +  std::string("ForceByDico=1 ")
                +  g_toolsOptions + ' '
                +  NKS();


        System(aSFR,true);

        DoMkT();
    }

    return 0;
}

int Line(int argc,char ** argv, const std::string &aArg="")
{
    int  aNbAdj = 5;
    bool ForceAdj = false;
    int isCirc = 0;
    string detectingTool, matchingTool;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullDir,"Full Name (Dir+Pat)", eSAM_IsPatFile)
                            << EAMC(aFullRes,"Image size")
                            << EAMC(aNbAdj,"Number of adjacent images to look for"),
                LArgMain()  << EAM(ExpTxt,"ExpTxt",true,"Export Pts in text format")
                << EAM(PostFix,"PostFix",false,"Add post fix in directory")
                << EAM(ByP,"ByP",true,"By process")
                << EAM(isCirc,"Circ",true,"In line mode if it's a loop (begin ~ end)")
                << EAM(ForceAdj,"ForceAdSupResol",true,"to force computation even when Resol < Adj")
                << EAM(detectingTool,PASTIS_DETECT_ARGUMENT_NAME.c_str(),false)
                << EAM(matchingTool,PASTIS_MATCH_ARGUMENT_NAME.c_str(),false),
                aArg
                );

    if (!MMVisualMode)
    {
        check_detect_and_match_tools( detectingTool, matchingTool );

        if ((aFullRes < aNbAdj) && (!ForceAdj) && (aFullRes>0))
        {
            std::cout << "Resol=" << aFullRes  << " NbAdjacence=" << aNbAdj << "\n";
            ELISE_ASSERT
                    (
                        false,
                        "Probable inversion of Resol and Adjacence (use ForceAdSupResol is that's what you mean)"
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
    }

    return 0;
}



int File(int argc,char ** argv, const std::string &aArg="")
{
    string detectingTool, matchingTool;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullDir,"XML-File of pair", eSAM_IsExistFile)
                            << EAMC(aFullRes,"Resolution",eSAM_None),
                LArgMain()  << EAM(ExpTxt,"ExpTxt",true)
                << EAM(PostFix,"PostFix",false)
                << EAM(ByP,"ByP",true)
                << EAM(detectingTool,PASTIS_DETECT_ARGUMENT_NAME.c_str(),false)
                << EAM(matchingTool,PASTIS_MATCH_ARGUMENT_NAME.c_str(),false),
                aArg
                );

    if (!MMVisualMode)
    {
        check_detect_and_match_tools( detectingTool, matchingTool );

        std::string aSFR =  BinPastis
                +  aDir + std::string(" ")
                +  QUOTE(std::string("NKS-Rel-ByFile@")+  aPat) + std::string(" ")
                +  ToString(aFullRes) + std::string(" ")
                +  StrMkT()
                +  std::string("NbMinPtsExp=2 ")
                +  std::string("ForceByDico=1 ")
                +  g_toolsOptions + ' '
                +  NKS();


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
    /*
     o_keypointsName = aDir+"Pastis/LBPp"+i_basename+".dat";
     */

    o_keypointsName = aDir + anICNM->Assoc1To2( "eModeLeBrisPP-Pastis-PtInt", i_basename, ToString( i_resolution ), true) ;
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

    if ( !ELISE_fp::MkDirSvp( aDir+"Pastis" ) )
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
    size_t 	nbFiles = i_graph.size(),
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
// i_graph[i][j] = number of points of in i whose nearest neighbour is in j
// a couple of images is output if i_graph[i][j]+i_graph[j][i]>i_threshold
size_t normalizeGraph( vector<vector<int> > &i_graph, int i_threshold  )
{
    size_t n = i_graph.size(),
            i, j;
    size_t count = 0;
    for ( j=1; j<n; j++ )
        for ( i=0; i<j; i++ )
        {
            if ( i_graph[i][j]+i_graph[j][i]>=i_threshold )
            {
                i_graph[j][i] = 1;
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

// delete points with a scale less than i_threshold
void delete_out_of_bound_scales( vector<SiftPoint> &io_points, REAL i_minScale, REAL i_maxScale )
{
    vector<SiftPoint> points( io_points.size() );
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
void DoConstructGraph( const string &i_outputFilename, size_t i_nbMaxPointsPerImage, REAL i_minScale, REAL i_maxScale, int i_nbRequiredMatches )
{
    size_t nbImages = aFileList.size();
    string keypointsFilename;
    vector<vector<SiftPoint> > keypoints_per_image( nbImages );
    vector<SiftPoint> 		   all_keypoints; 		// a big vector with all keypoints of all images
    vector<int> 	  		   all_image_indices;	// contains the index of the image from which the keypoint is from
    size_t iImage = 0,
            nbTotalKeypoints = 0,
            addedPoints;

    // read all keypoints files
    cout << "--------------------> read all keypoints files" << endl;
    for ( std::list<std::string>::const_iterator iT=aFileList.begin(); iT!=aFileList.end(); iT++, iImage++ ) // aFileList has been computed by DoDevelopp
    {
        keypointsFilename = aKeypointsFileArray[iImage];

        if ( !read_siftPoint_list( keypointsFilename, keypoints_per_image[iImage] ) ){
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
            SiftPoint *data = &( keypoints_per_image[iImage][0] );
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
    vector<vector<SiftPoint> >::const_iterator itSrc = keypoints_per_image.begin();
    const SiftPoint *pSrc;
    SiftPoint *pDst = &( all_keypoints[0] );
    int *itIndex = &( all_image_indices[0] );
    for ( iImage=0; iImage<nbImages; iImage++, itSrc++ )
    {
        nbPoints = itSrc->size();
        if ( nbPoints==0 ) continue;

        pSrc = &( ( *itSrc )[0] );
        memcpy( pDst, pSrc, nbPoints*sizeof( SiftPoint ) );
        pDst += nbPoints;
        while ( nbPoints-- ) *itIndex++=iImage;
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
    SiftPoint *query = &( all_keypoints[0] );
    const ANNidx *neighbours = search.getNeighboursIndices();
    size_t iImageQuery, iImageNeighbour,
            nbBadNeighbours = 0,
            iQuery;
    for ( iQuery=0; iQuery<nbTotalKeypoints; iQuery++ )
    {
        search.search( query->descriptor );
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

    //print_graph( graph );

    vector<int> labels( nbImages, -1 );
    int currentLabel = 0;
    for ( size_t iStartingElement=0; iStartingElement<nbImages; iStartingElement++ )
        if ( labels[iStartingElement]==-1 )	setLabel( graph, labels, iStartingElement, currentLabel++ );
    cout << currentLabel << " connected component" << endl;

    writeBinaryGraphToXML( i_outputFilename, graph );
}


// this option is to construct a proximity graph from points of interests
// it generates an XML file to process with the "File" option
int Graph_(int argc,char ** argv)
{
    int nbThreads = NbProcSys(); // default is the number of cores of the system
    int maxDimensionResize = -1;
    int nbMaxPoints = 200;
    REAL minScaleThreshold = std::numeric_limits<REAL>::min(),
            maxScaleThreshold = std::numeric_limits<REAL>::max();
    int nbRequiredMatches = 1;
    string outputFile = "tapioca_connectivity_graph.xml"; // default XML filename for the graph
    string detectingTool, detectingToolArguments;

    // aDir is "chantier" directory
    // aPat is images' pattern
    // aFullPat is the original directory+pattern string

    ElInitArgMain
            (
                argc,argv,

                LArgMain()  << EAMC(aFullDir,"Full images' pattern (directory+pattern)", eSAM_IsPatFile)
                << EAMC(maxDimensionResize,"processing size of image  (for the greater dimension)", eSAM_None),

                LArgMain()  << EAM(nbThreads, "ByP", true, "By processe")
                << EAM(detectingTool, PASTIS_DETECT_ARGUMENT_NAME.c_str(), true, "executable used to detect keypoints")
                << EAM(nbMaxPoints, "MaxPoint", true, "number of points used per image to construct the graph (default 200)")
                << EAM(minScaleThreshold, "MinScale", true, "if specified, points with a lesser scale are ignored")
                << EAM(maxScaleThreshold, "MaxScale", true, "if specified, points with a greater scale are ignored")
                << EAM(nbRequiredMatches, "NbRequired", true, "number of matches to create a connexion between two images (default 1)")
                << EAM(outputFile, "Out", true, "name of the produced XML file")
                );

    // if no output filename is given, use the default one in "chantier" directory
    if ( !EAMIsInit(&outputFile) )
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

    DoConstructGraph( outputFile, nbMaxPoints, minScaleThreshold, maxScaleThreshold, nbRequiredMatches );

    /*
    check_detect_and_match_tools( detectingTool, matchingTool );

    if ((aFullRes < aNbAdj) && (!ForceAdj) && (aFullRes>0))
    {
        std::cout << "Resol=" << aFullRes  << " NbAdjacence=" << aNbAdj << "\n";
        ELISE_ASSERT
        (
             false,
             "Probable inversion of Resol and Adjacence (use ForceAdSupResol is that's what you mean)"
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

int Tapioca_main(int argc,char ** argv)
{
#if(ELISE_QT_VERSION >= 4)
    if (MMVisualMode)
    {
        QStringList items;

        for (int aK=0; aK < aNbType; ++aK)
            items << QString((Type[aK]).c_str());

        QApplication app(argc, argv);

        setStyleSheet(app);

        bool ok = false;
        QString item = QInputDialog::getItem(NULL, app.applicationName(),
                                             QString ("Strategy"), items, 0, false, &ok);

        if (ok && !item.isEmpty())
            TheType = item.toStdString();
        else
            return EXIT_FAILURE;
    }
    else
        TheType = argv[1];
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

#if(ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
#endif
        SplitDirAndFile(aDir,aPat,aFullDir);


    aPatOri = aPat;

    StdAdapt2Crochet(aPat);
    /*
    if ((aPat.find('@')!=(std::string::npos)) && (TheType != Type[3]))
    {
         aPat = "[["+aPat+"]]";
    }
*/

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
        int aRes = MultiEch(argc,argv,TheType);
        BanniereMM3D();
        return aRes;
    }
    else if (TheType == Type[1])
    {
        int aRes = All(argc,argv,TheType);
        BanniereMM3D();
        return aRes;
    }
    else if (TheType == Type[2])
    {
        int aRes = Line(argc,argv,TheType);
        BanniereMM3D();
        return aRes;
    }
    else if (TheType == Type[3])
    {
        int aRes = File(argc,argv,TheType);
        BanniereMM3D();
        return aRes;
    }
    else if (TheType == Type[4])
    {
        int aRes = Graph_(argc,argv);
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
