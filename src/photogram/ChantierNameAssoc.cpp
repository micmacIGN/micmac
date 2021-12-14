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

#include "general/CMake_defines.h"
#if ELISE_QT
    #ifdef Int
        #undef Int
    #endif
    #include "QCoreApplication"
    #include "QStringList"
    #include "QDir"
#endif

#include "StdAfx.h"


bool SplitIn2ArroundEqSvp
     (
             const std::string  &  a2Stplit,
             char            aCar,
             std::string  &  aBefore,
             std::string  &  aAfter
     );


std::string GetNameWithoutPerc(const std::string & aName)
{
   std::string aBefore,anAfter;
   bool OkSplit = SplitIn2ArroundEqSvp(aName,'%',aBefore,anAfter);

   if (OkSplit && ELISE_fp::IsDirectory(aBefore)) return anAfter;

   return "";
}


bool NameIsNKS(const std::string & aPat)
{
    return (aPat[0]=='N') && (aPat[1]=='K') && (aPat[2]=='S') && (aPat[3]=='-');
}

bool NameIsNKSAssoc(const std::string & aPat)
{
    return        NameIsNKS (aPat)
              && (aPat[4]=='A') 
              && (aPat[5]=='s') 
              && (aPat[6]=='s') 
              && (aPat[7]=='o') 
              && (aPat[8]=='c') 
              && (aPat[9]=='-') 
           ;
}

bool NameIsNKSSet(const std::string & aPat)
{
    return        NameIsNKS(aPat) 
              && (aPat[4]=='S') 
              && (aPat[5]=='e') 
              && (aPat[6]=='t') 
              && (aPat[7]=='-') 
           ;
}







extern void NewSplit( const std::string  &  a2Stplit,std::string & aK0,std::vector<std::string>  & aSup);


//   GESTION DE LA NATURE DES CLES


bool TransFormArgKey ( cByAdjacence & aAdj , bool AMMNoArg, const std::vector<std::string> & aVParam);
template <class Type> bool TransFormArgKey ( TypeSubst<Type> & aIS , bool AMMNoArg, const std::vector<std::string> & aVParam );
bool TransFormArgKey(cFiltreByRelSsEch & aF , bool AMMNoArg, const std::vector<std::string> & aVParam);



bool TransFormArgKey
    (
    std::string & aName ,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aDirExt
    )
{
    if ( aName.find('#')==std::string::npos)
        return false;

    if (aDirExt.size()==0)
    {
        if (AMMNoArg)
            return true;
        std::cout << "FOR STR=" << aName << "\n";
        ELISE_ASSERT(false,"TransFormArgKey No arg with #");
    }

    std::string aRes;


    const char * aN = aName.c_str();

    while (*aN)
    {
        if (*aN=='#')
        {
            aN++;
            int aK = *aN-'0';
            if ((aK<1) || (aK>=9) || (aK> int(aDirExt.size())))
            {
                std::cout << "FOR STR=" << aName << " K=" << aK  << " NbArg = " << aDirExt.size() << "\n";
                ELISE_ASSERT(false,"TransFormArgKey cannot substituate");
            }
            aN++;
            aRes += aDirExt[aK-1];
        }
        else
        {
            aRes += *aN;
            aN++;
        }
    }

    aName = aRes;

    return true;
}

bool TransFormArgKey
    (
    cTplValGesInit<std::string> & aName ,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aDirExt
    )
{
    if (aName.IsInit())
        return TransFormArgKey(aName.Val(),AMMNoArg,aDirExt);
    return false;
}


template<class TypeCtn> bool ContainerTransFormArgKey
    (
    TypeCtn & aV,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aDirExt
    )
{
    bool aRes = false;
    for (typename TypeCtn::iterator it=aV.begin(); it!=aV.end() ; it++)
    {
        bool aTr = TransFormArgKey(*it,AMMNoArg,aDirExt);
        aRes = aRes || aTr;
    }
    return aRes;
}


bool TransFormArgKey
    (
    cNameFilter & aSND,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aDirExt
    )
{
    bool aRes = false;

    if (ContainerTransFormArgKey(aSND.FocMm(),AMMNoArg,aDirExt))
        aRes = true;

    if (TransFormArgKey(aSND.Min(),AMMNoArg,aDirExt))
        aRes = true;
    if (TransFormArgKey(aSND.Max(),AMMNoArg,aDirExt))
        aRes = true;

    for
        (
        std::list<cKeyExistingFile>::iterator itK=aSND.KeyExistingFile().begin();
    itK!=aSND.KeyExistingFile().end();
    itK++
        )
    {
        ContainerTransFormArgKey(itK->KeyAssoc(),AMMNoArg,aDirExt);
    }

    return aRes;
}

bool TransFormArgKey
    (
    cSetNameDescriptor & aSND,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aDirExt
    )
{
    bool aRes = TransFormArgKey(aSND.SubDir(),AMMNoArg,aDirExt);

    if (ContainerTransFormArgKey(aSND.PatternAccepteur(),AMMNoArg,aDirExt))
        aRes = true;

    if (ContainerTransFormArgKey(aSND.PatternRefuteur(),AMMNoArg,aDirExt))
        aRes = true;

    if (ContainerTransFormArgKey(aSND.NamesFileLON(),AMMNoArg,aDirExt))
        aRes = true;

    if (aSND.Filter().IsInit())
    {
        TransFormArgKey(aSND.Filter().Val(),AMMNoArg,aDirExt);
    }

    return aRes;
}

bool TransFormArgKey
    (
    cBasicAssocNameToName & aBANA,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aDirExt
    )
{
    bool aRes =  TransFormArgKey(aBANA.PatternTransform(),AMMNoArg,aDirExt);

    if ( TransFormArgKey(aBANA.PatternSelector(),AMMNoArg,aDirExt))
        aRes = true;
    if ( ContainerTransFormArgKey(aBANA.CalcName(),AMMNoArg,aDirExt))
        aRes = true;

    return aRes;
}

bool TransFormArgKey
    (
    cAssocNameToName & anANA,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aDirExt
    )
{
    bool aRes = TransFormArgKey(anANA.Direct(),AMMNoArg,aDirExt);

    if (anANA.Inverse().IsInit())
    {
        if (TransFormArgKey(anANA.Inverse().Val(),AMMNoArg,aDirExt))
            aRes = true;
    }
    return aRes;
}

bool TransFormArgKey
    (
    cKeyedNamesAssociations & aKNA,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aDirExt
    )
{
    bool aRes = TransFormArgKey(aKNA.SubDirAutoMake(),AMMNoArg,aDirExt);
    for
        (
        std::list<cAssocNameToName>::iterator itANN=aKNA.Calcs().begin();
    itANN!=aKNA.Calcs().end();
    itANN++
        )
    {
        if (TransFormArgKey(*itANN,AMMNoArg,aDirExt))
            aRes = true;
    }
    for (int aK=0 ; aK<int(aDirExt.size()) ; aK++)
    {
        aKNA.Key() = aKNA.Key() +"@" + aDirExt[aK];
    }

    return aRes;
}

//   =================  POUR LES RELATIONS ==================

template <class Type> bool TransFormArgKey
    (
    TypeSubst<Type> & aIS ,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aVParam
    )
{
    bool aRes =  aIS.Subst(AMMNoArg,aVParam);
    return aRes;
}

bool TransFormArgKey
    (
    cFiltreByRelSsEch & aF ,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aVParam
    )
{
    bool aRes = false;

    if (TransFormArgKey(aF.KeySet(),AMMNoArg,aVParam))
        aRes = true;

    if (TransFormArgKey(aF.KeyAssocCple(),AMMNoArg,aVParam))
        aRes = true;

    if (TransFormArgKey(aF.SeuilBasNbPts(),AMMNoArg,aVParam))
        aRes = true;

    if (TransFormArgKey(aF.SeuilHautNbPts(),AMMNoArg,aVParam))
        aRes = true;

    if (TransFormArgKey(aF.NbMinCple(),AMMNoArg,aVParam))
        aRes = true;
    return aRes;
}

bool TransFormArgKey
    (
    cFiltreDeRelationOrient & aF ,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aVParam
    )
{
    bool aRes = false;

    if (aF.FiltreByRelSsEch().IsInit())
    {
        if (TransFormArgKey(aF.FiltreByRelSsEch().Val(),AMMNoArg,aVParam))
            aRes = true;
    }
    return aRes;
}


bool TransFormArgKey
    (
    cByAdjacence & aAdj ,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aVParam
    )
{
    bool aRes = false;
    if (ContainerTransFormArgKey(aAdj.KeySets(),AMMNoArg,aVParam))
        aRes = true;

    if (aAdj.DeltaMax().IsInit())
    {
        if (TransFormArgKey(aAdj.DeltaMax().Val(),AMMNoArg,aVParam))
            aRes = true;
    }

    if (aAdj.DeltaMin().IsInit())
    {
        if (TransFormArgKey(aAdj.DeltaMin().Val(),AMMNoArg,aVParam))
            aRes = true;
    }

    if (aAdj.Sampling().IsInit())
    {
        if (TransFormArgKey(aAdj.Sampling().Val(),AMMNoArg,aVParam))
            aRes = true;
    }

    if (aAdj.Circ().IsInit())
    {
        if (TransFormArgKey(aAdj.Circ().Val(),AMMNoArg,aVParam))
            aRes = true;
    }



    if (aAdj.Filtre().IsInit())
    {
        if (TransFormArgKey(aAdj.Filtre().Val(),AMMNoArg,aVParam))
            aRes = true;
    }

    return aRes;
}
/*
*/



bool TransFormArgKey
    (
    cNameRelDescriptor & aNRD ,
    bool AMMNoArg,  // Accept mismatch si DirExt vide
    const std::vector<std::string> & aVParam
    )
{
    bool aRes = false;

    if (ContainerTransFormArgKey(aNRD.ByAdjacence(),AMMNoArg,aVParam))
        aRes = true;

    if (ContainerTransFormArgKey(aNRD.NameFileIn(),AMMNoArg,aVParam))
        aRes = true;

    return aRes;
}

static const std::string aFileTmp = "MicMacInstall.txt";

#if (!ELISE_windows)
// execute command i_base_cmd and get its standard output in a string
bool ElGetStrSys( const std::string & i_base_cmd, std::string &o_result )
{
    o_result = "";

    FILE *f = popen_call( i_base_cmd.c_str(), "r" );
    if ( f==NULL ) return false;

    // read popen's output
    char buffer[501];
    size_t nbRead;
    while ( feof( f )==0 )
    {
        nbRead = fread( buffer, 1, 500, f );
        if ( nbRead>0 )
        {
            buffer[nbRead-1] = '\0';
            o_result.append( string( buffer ) );
        }
    }
    pclose( f );

    return true;
}
#endif

#if ELISE_Darwin
    #include <mach-o/dyld.h>
#endif

static std::string ArgvMMDir;
static std::string CurrentProgramFullName;
static std::string CurrentProgramSubcommand = "unknown";
std::string MM3DFixeByMMVII ="";

void MMD_InitArgcArgv(int argc,char ** argv,int aNbMin)
{
    static bool First=true;
    if (!First) return;
    First = false;

    AnalyseContextCom(argc,argv);
    MemoArg(argc,argv);

    if (((aNbMin >=0) && (argc < aNbMin)) && (!MMVisualMode))
    {
        if (argc>0)
            std::cout << "For command " << argv[0] << "\n";
        else
            std::cout << "For unknown command \n";
        std::cout << "Got " << argc << " args for " << aNbMin << "required \n";
    }

    if ((ArgvMMDir=="") && (argc!=0))
    {
        MemoArg(argc,argv);
#if ELISE_windows
        TCHAR FilePath[MAX_PATH] = { 0 };
        GetModuleFileName(NULL,FilePath, MAX_PATH );
        CurrentProgramFullName = string( FilePath );
        std::string sFile;

        replace( CurrentProgramFullName.begin(), CurrentProgramFullName.end(), '\\', '/' );
        SplitDirAndFile(ArgvMMDir,sFile,CurrentProgramFullName);

        ArgvMMDir.resize( ArgvMMDir.length()-1 );
        SplitDirAndFile(ArgvMMDir,sFile,ArgvMMDir);
#else
        std::string aFullArg0;
        // try to get executable full path Using /proc filesystem
        // not compatible with all unix
        #if ELISE_Darwin
        //TODO use ctPath
            uint32_t size = 0;
            char *buf = NULL;
            if ( _NSGetExecutablePath(buf, &size)==-1 ){
                buf = new char[size];
                _NSGetExecutablePath( buf, &size);
                aFullArg0.assign(buf);
                if ( strlen(argv[0])>2 && argv[0][0]=='.' && argv[0][1]=='/' )
                    aFullArg0 = aFullArg0.erase( aFullArg0.find("./"), 2 );
                delete [] buf;
            }
        #else
            char buf[1024];
            ssize_t len;
            if ( ( len= readlink( "/proc/self/exe", buf, sizeof(buf)-1 ) ) != -1 )
            {
                buf[len] = '\0'; // make sure the string is null terminated, some implementation of readlink may not do it
                aFullArg0 = buf;
            }
        #endif
        else
        {
            // if the /proc filesystem is not available, try Using the "which" command
            bool whichSucceed = ElGetStrSys( "which "+ std::string( argv[0] ), aFullArg0 );

            // modif Greg: il y a un probleme sous MacOS, on perd le 'd' de mm3d
            // remove the which's ending '\n'
            if (aFullArg0[aFullArg0.size()-1] == '\n')
                aFullArg0.resize( aFullArg0.size()-1 );

            // if which failed then we're doomed
            ELISE_ASSERT( whichSucceed, "MMD_InitArgcArgv : unable to retrieve binaries directory" );
        }
        if (MM3DFixeByMMVII !="")
        {
           aFullArg0 = MM3DFixeByMMVII ;
        }

        std::string aPatProg = "([0-9]|[a-z]|[A-Z]|_)+";
        cElRegex  anAutomProg(aPatProg,10);
        if (anAutomProg.Match(aFullArg0))
             ArgvMMDir = std::string("..")+ELISE_CAR_DIR;
        else
        {
            cElRegex  anAutomProg(std::string("(.*)bin")+ELISE_CAR_DIR+aPatProg,10);
            ArgvMMDir = MatchAndReplace(anAutomProg,aFullArg0,"$1");
            if (ArgvMMDir=="") ArgvMMDir=std::string(".")+ELISE_CAR_DIR;
        }
        CurrentProgramFullName = aFullArg0;
#endif

        if ( argc>1 )
            CurrentProgramSubcommand = StrToLower( argv[1] );
    }
}




int CalcNbProcSys()
{
#if ELISE_windows
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );
    return sysinfo.dwNumberOfProcessors;
#else
    // return GetValStrSys<int>("cat /proc/cpuinfo | grep processor  | wc -l");
    return sysconf (_SC_NPROCESSORS_CONF);
#endif
}

extern int TheNbProcCom;
int NbProcSys()
{
    if (TheNbProcCom>0) return TheNbProcCom;
    static int aRes = CalcNbProcSys();
    if ( MMUserEnv().NbMaxProc().IsInit() ) ElSetMin( aRes, MMUserEnv().NbMaxProc().Val() );

    return aRes;
}




std::string Basic_XML_User_File(const std::string & aName)
{
   return MMDir() + "include"+ELISE_CAR_DIR+"XML_User"+ELISE_CAR_DIR+ aName;
}
std::string XML_User_Or_MicMac(const std::string & aName)
{
  std::string aRes = Basic_XML_User_File(aName);
  if ( ELISE_fp::exist_file(aRes))
     return aRes;
  return Basic_XML_MM_File(aName);
}
const cMMUserEnvironment & MMUserEnv()
{
    static cMMUserEnvironment * aRes = 0;
    if (aRes ==0)
    {
        std::string aName = XML_User_Or_MicMac("MM-Environment.xml");

        if ( !ELISE_fp::exist_file(aName) )
             aRes = new cMMUserEnvironment;
        else
        {
             cMMUserEnvironment aMME =  StdGetObjFromFile<cMMUserEnvironment>
                                   (
                                       aName,
                                       StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                       "MMUserEnvironment",
                                       "MMUserEnvironment"
                                   );
             aRes = new cMMUserEnvironment(aMME);
        }
    }
    return *aRes;
}





std::string MM3DStr = "mm3d";

//   Binaire a "l'ancienne"  MMDir() + std::string("bin" ELISE_STR_DIR  COMMANDE)
std::string MMBinFile(const std::string & aCom)
{
  return  MMDir() + std::string("bin" +  std::string(ELISE_STR_DIR) +  aCom  + " ");
}
//   Nouveau par mm3d   MMDir() + std::string("bin" ELISE_STR_DIR "mm3d"  COMMANDE)
std::string MM3dBinFile(const std::string & aCom)
{
  return  MMDir() + std::string("bin" +  std::string(ELISE_STR_DIR) + MM3DStr +  " "+ aCom  + " ");
}
//   Nouveau par mm3d   MMDir() + std::string("bin" ELISE_STR_DIR "mm3d"  COMMANDE)
std::string MM3dBinFile_quotes(const std::string & aCom)
{
  return string("\"") + MMDir() + std::string("bin" +  std::string(ELISE_STR_DIR) + MM3DStr +  "\" "+ aCom  + " ");
}
//   MMDir() + std::string("include" ELISE_STR_DIR "XML_MicMac" ELISE_STR_DIR "Apero-Cloud.xml ")
std::string Basic_XML_MM_File(const std::string & aFile)
{
   return   MMDir() + std::string("include" +std::string(ELISE_STR_DIR) + "XML_MicMac" + std::string(ELISE_STR_DIR) + aFile);
}
std::string Specif_XML_MM_File(const std::string & aFile)
{
    return   aFile;
}

std::string XML_MM_File(const std::string & aFile)
{

   return   Basic_XML_MM_File(aFile) + " ";
}








    static bool DebugPCP = false;


    /*******************************************************/
    /*                                                     */
    /*             cInterfNameCalculator                   */
    /*                                                     */
    /*******************************************************/

    cInterfNameCalculator::cInterfNameCalculator(cInterfChantierNameManipulateur * aICNM) :
    mICNM (aICNM)
    {
    }
    cInterfNameCalculator::~cInterfNameCalculator() { }

    const  cInterfNameCalculator::tNuplet   cInterfNameCalculator::theNotDef;
    const cInterfNameCalculator::tNuplet  & cInterfNameCalculator::NotDef() { return theNotDef; }
    bool cInterfNameCalculator::IsDefined(const tNuplet & aNuplet)
    {
        return ! aNuplet.empty();
    }

    cInterfNameCalculator::tNuplet cInterfNameCalculator::Inverse(const tNuplet &)
    {
        return NotDef();
    }

    cInterfNameCalculator * cInterfNameCalculator::StdCalcFromXML
        (
        cInterfChantierNameManipulateur * aICNM,
        const cAssocNameToName & aN2N
        )
    {
        if (aN2N.Inverse().IsInit())
            return new cInv_AutomNC(aICNM,aN2N.Direct(),aN2N.Inverse().Val());
        if (aN2N.AutoInverseBySym().Val())
        {
            cBasicAssocNameToName aInv = aN2N.Direct();
            ELISE_ASSERT(aInv.CalcName().size()==1,"AutoInverseBySym cInterfNameCalculator::StdCalcFromXML");
            ElSwap(aInv.PatternTransform(),aInv.CalcName()[0]);
            return new cInv_AutomNC(aICNM,aN2N.Direct(),aInv);
        }
        return new cNI_AutomNC(aICNM,aN2N.Direct());
    }

    cMultiNC * cInterfNameCalculator::StdCalcFromXML
        (
        const cKeyedNamesAssociations & aKNA,
        cInterfChantierNameManipulateur * aICNM,
        const std::string & aSubDir,
        bool                aSubDirRec,
        const std::list<cAssocNameToName> & aLA
        )
    {
        std::vector<cInterfNameCalculator *> aVNC;
        for
            (
            std::list<cAssocNameToName>::const_iterator itA=aLA.begin();
        itA != aLA.end();
        itA++
            )
        {
            aVNC.push_back(StdCalcFromXML(aICNM,*itA));
        }

        return new cMultiNC(aICNM,aKNA,aICNM->Dir(),aSubDir,aSubDirRec,aVNC);
    }

    cInterfChantierNameManipulateur *    cInterfNameCalculator::ICNM()
    {
        return 0;
    }

    cKeyedNamesAssociations * cInterfNameCalculator::KNA()
    {
        return 0;
    }


    /*******************************************************/
    /*                                                     */
    /*             cInterfChantierNC                       */
    /*                                                     */
    /*******************************************************/

    cInterfChantierNC::cInterfChantierNC() { }
    cInterfChantierNC::~cInterfChantierNC() { }

    /*
    cDicoChantierNC *  cInterfChantierNC::StdCalcFromXML(const std::list<cAssocNameToName> & aList)
    {
    for
    (
    std::list<cAssocNameToName>::const_iterator itL=aList.begin();
    itL!=aList.end();
    itL++
    )
    {
    }
    }
    */

    void cInterfChantierNC::VerifSol(const tNuplet & aSol,const tKey & aKey,const tNuplet& aNuple)
    {
        if (! cInterfNameCalculator::IsDefined(aSol))
        {
            std::cout << "For Key=" << aKey << "\n";
            std::cout << "For Word :\n";
            for (int aK=0 ; aK<int(aNuple.size()) ; aK++)
                std::cout << "     [" << aNuple[aK] << "]\n";
            ELISE_ASSERT(false,"Cannot compute association");
        }
    }

    cInterfChantierNC::tNuplet  cInterfChantierNC::DefinedDirect(const tKey & aKey,const tNuplet& aNuple)
    {
        tNuplet aRes = Direct(aKey,aNuple);
        VerifSol(aRes,aKey,aNuple);
        return aRes;
    }


    cInterfChantierNC::tNuplet  cInterfChantierNC::DefinedInverse(const tKey & aKey,const tNuplet& aNuple)
    {
        tNuplet aRes = Inverse(aKey,aNuple);
        VerifSol(aRes,aKey,aNuple);
        return aRes;
    }

    cDicoChantierNC *  cInterfChantierNC::StdCalcFromXML
        (
        cInterfChantierNameManipulateur * aICNM,
        const std::list<cKeyedNamesAssociations> & aLKA
        )
    {
        cDicoChantierNC * aRes = new cDicoChantierNC;
        for
            (
            std::list<cKeyedNamesAssociations>::const_iterator itKA=aLKA.begin();
        itKA !=aLKA.end();
        itKA++
            )
        {
            if (! itKA->IsParametrized().Val())
            {
                cKeyedNamesAssociations  aKey = *itKA;
                std::vector<std::string> aVS;
                if (TransFormArgKey(aKey,true,aVS))
                {
                    std::cout << "\n   === FOR KEY " << itKA->Key() << "\n";
                    ELISE_ASSERT(false,"Set has # and is not Parametrized");
                }
            }
            aRes->Add(aICNM,*itKA);
        }

        return aRes;
    }



    // tNuplet  DefinedInverse(const tKey &,const tNuplet&);



    /*******************************************************/
    /*                                                     */
    /*                    cNI_AutomNC                      */
    /*                                                     */
    /*******************************************************/


    cNI_AutomNC::cNI_AutomNC
        (
        cInterfChantierNameManipulateur * aICNM,
        const cBasicAssocNameToName & aN2N
        )  :
    cInterfNameCalculator(aICNM),
        mICNM           (aICNM),
        mAutomTransfo  (aN2N.PatternTransform(),10),
        mAutomSel      (aN2N.PatternSelector().ValWithDef(aN2N.PatternTransform()),10),
        mNames2Replace (aN2N.CalcName()),
        mSep           (aN2N.Separateur().Val()),
        mFilter        (aN2N.Filter()),
        mDBNT          (aN2N.NameTransfo())
    {
    }

    cInterfNameCalculator::tNuplet cNI_AutomNC::Direct(const tNuplet & aNuplet)
    {
        std::string  aWord = aNuplet[0];
        for (int aK=1 ; aK<int(aNuplet.size()) ; aK++)
        {
            aWord = aWord + mSep + aNuplet[aK];
        }
        aWord = mICNM->DBNameTransfo(aWord,mDBNT);


        if (DebugPCP)
        {
            std::string aSel =  mAutomSel.NameExpr();
            std::string aTra =  mAutomTransfo.NameExpr();
            std::cout << "TEST, SEL = "  << aSel << "\n";
            std::cout << "      TRA = "  << aTra << "\n";
        }

        if ((! mAutomSel.Match(aWord)) || (!NameFilter(mICNM,mFilter,aWord)))
            return NotDef();



        tNuplet aResult;
        for (int aK=0 ; aK<int(mNames2Replace.size()) ; aK++)
            aResult.push_back(MatchAndReplace(mAutomTransfo,aWord,mNames2Replace[aK]));


        return aResult;
    }

    /*******************************************************/
    /*                                                     */
    /*                    cNI_AutomNC                      */
    /*                                                     */
    /*******************************************************/

    cInv_AutomNC::cInv_AutomNC
        (
        cInterfChantierNameManipulateur * aICNM,
        const cBasicAssocNameToName  & aAutDir,
        const cBasicAssocNameToName &  aAutInv,
        cInv_AutomNC * aInv
        )  :
    cNI_AutomNC  (aICNM,aAutDir),
        mInv( NULL )
    {
        mInv = ((aInv==0) ? new cInv_AutomNC(aICNM,aAutInv,aAutDir,this) : aInv);
    }

    cInterfNameCalculator::tNuplet cInv_AutomNC::Inverse(const tNuplet & aNuple)
    {
        return mInv->Direct(aNuple);
    }


    /*******************************************************/
    /*                                                     */
    /*                    cMultiNC                         */
    /*                                                     */
    /*******************************************************/

    cInterfNameCalculator::tNuplet cMultiNC::Direct(const tNuplet & aNuplet)
    {
        AutoMakeSubDir();
        for (int aK=0 ; aK<int(mVNC.size()) ; aK++)
        {
            cInterfNameCalculator::tNuplet aRes = mVNC[aK]->Direct(aNuplet);
            if (IsDefined(aRes))
            {
                AutoMakeSubDirRec(aRes);
                return aRes;
            }
        }
        return NotDef();
    }

    void cMultiNC::AutoMakeSubDirRec(const tNuplet & aNuplet)
    {
        if (mSubDirRec)
            ELISE_fp::MkDirRec( ( isUsingSeparateDirectories()?MMOutputDirectory():mDir )+aNuplet[0] );
    }

    cInterfNameCalculator::tNuplet cMultiNC::Inverse(const tNuplet & aNuplet)
    {
        for (int aK=0 ; aK<int(mVNC.size()) ; aK++)
        {
            cInterfNameCalculator::tNuplet aRes = mVNC[aK]->Inverse(aNuplet);
            if (IsDefined(aRes))
                return aRes;
        }
        return NotDef();
    }


    void cMultiNC::AutoMakeSubDir()
    {
       if (mSubDir!="") ELISE_fp::MkDir( ( isUsingSeparateDirectories()?MMOutputDirectory():mDir )+mSubDir );
    }

    cMultiNC::cMultiNC
        (
        cInterfChantierNameManipulateur * aICNM,
        const cKeyedNamesAssociations & aKNA,
        const std::string& aDir,
        const std::string& aSubDir,
        bool              aSubDirRec,
        const std::vector<cInterfNameCalculator *> &  aVNC
        ) :
    cInterfNameCalculator(aICNM),
        mICNM    (aICNM),
        mKNA     (aKNA),
        mVNC     (aVNC),
        mDir     (aDir),
        mSubDir  (aSubDir),
        mSubDirRec (aSubDirRec)
    {
        // std::cout << "mSubDirRec " << mSubDirRec <<  " " << this<<  "\n";
    }

    cInterfChantierNameManipulateur * cMultiNC::ICNM()
    {
        return mICNM;
    }

    cKeyedNamesAssociations * cMultiNC::KNA()
    {
        return &mKNA;
    }


    /*******************************************************/
    /*                                                     */
    /*                    cInterfChantierSetNC             */
    /*                                                     */
    /*******************************************************/

    cInterfChantierSetNC::~cInterfChantierSetNC()
    {
    }


    cDicoSetNC * cInterfChantierSetNC::StdCalcFromXML
        (
        cInterfChantierNameManipulateur * aICNM,
        const std::list<cKeyedSetsOfNames> & aLKSN
        )
    {
        cDicoSetNC * aRes = new cDicoSetNC();
        for
            (
            std::list<cKeyedSetsOfNames>::const_iterator itKSN = aLKSN.begin();
        itKSN != aLKSN.end();
        itKSN++
            )
        {
            const cSetNameDescriptor & aS0 = itKSN->Sets();
            if (! itKSN->IsParametrized().Val())
            {
                cSetNameDescriptor  aSet = aS0;
                std::vector<std::string> aVS;
                if (TransFormArgKey(aSet,true,aVS))
                {
                    std::cout << "\n   === FOR KEY " << itKSN->Key() << "\n";
                    ELISE_ASSERT(false,"Set has # and is not Parametrized");
                }
            }

            std::string key = itKSN->Key();
            cSetName* name = new cSetName(aICNM,aS0);

            aRes->Add(key,name);
        }

        return aRes;
    }



    /*******************************************************/
    /*                                                     */
    /*                    cSetName                         */
    /*                                                     */
    /*******************************************************/

    /*
    cSetName::~cSetName()
    {
    }
    */

    /*******************************************************/
    /*                                                     */
    /*                    cSetNameByAutom                  */
    /*                                                     */
    /*******************************************************/

    cSetName::cSetName
        (
        cInterfChantierNameManipulateur * aICNM,
        const cSetNameDescriptor & aSND
        ) :
    mExtIsCalc (false),
        mDefIsCalc (false),
        mICNM   (aICNM),
        mDir    (aICNM->Dir()),
        mSND    (aSND)
    {
    }

    cInterfChantierNameManipulateur * cSetName::ICNM()
    {
        return mICNM;
    }


const cSetNameDescriptor & cSetName::SND() const
{
    return mSND;
}

void  cSetName::AddListName(cLStrOrRegEx & aLorReg,const std::list<std::string> & aLName,cInterfChantierNameManipulateur *anICNM)
{
    for (std::list<std::string>::const_iterator itS=aLName.begin(); itS!=aLName.end() ; itS++)
        aLorReg.AddName(*itS,anICNM);
}

cLStrOrRegEx::cLStrOrRegEx()
{
}


bool cLStrOrRegEx::AuMoinsUnMatch(const std::string & aName)
{
    return ::AuMoinsUnMatch(mAutom,aName) || DicBoolFind(mSet,aName);
}

void  cLStrOrRegEx::AddName(const std::string & aName,cInterfChantierNameManipulateur *anICNM)
{
   if (NameIsNKSSet(aName))
   {
      const cInterfChantierNameManipulateur::tSet * aSetIm = anICNM->Get(aName);
      for (int aK=0 ; aK<int(aSetIm->size()) ; aK++)
      {
          mSet.insert((*aSetIm)[aK]);
      }
   }
   else
   {
      mAutom.push_back(new cElRegex(aName,10));
   }
}

    void cSetName::CompileDef()
    {
        if (!mDefIsCalc)
        {
            mDefIsCalc = true;
            AddListName(mLR,mSND.PatternRefuteur(),mICNM);
            AddListName(mLA,mSND.PatternAccepteur(),mICNM);
            // mLR = CompilePats(mSND.PatternRefuteur());
            // mLA = CompilePats(mSND.PatternAccepteur());
        }
    }

    /*
    if (Ok && mSND.Min().IsInit()   && (*itN< mSND.Min().Val()))
    Ok = false;
    if (Ok && mSND.Max().IsInit()   && (*itN> mSND.Max().Val()))
    Ok = false;
    */

    bool cSetName::SetBasicIsIn(const std::string & aName)
    {
        CompileDef();

        return       mLA.AuMoinsUnMatch(aName)
            && (! mLR.AuMoinsUnMatch(aName))
            && (NameFilter(mICNM,mSND.Filter(),aName))
            ;
    }

    bool cSetName::IsSetIn(const std::string & aName)
    {
        return      SetBasicIsIn(aName)
            && (  (!(mSND.Min().IsInit()) ) || (!(aName<mSND.Min().Val())))
            && (  (!(mSND.Max().IsInit()) ) || (!(aName>mSND.Max().Val())));
    }


void cSetName::InternalAddList(const std::list<std::string> & aLN)
{
    for
    (
        std::list<std::string>::const_iterator itN=aLN.begin();
        itN!=aLN.end();
        itN++
    )
    {
         std::string aName = mSND.SubDir().Val() +  *itN; // 
         bool Ok = ! mLR.AuMoinsUnMatch(aName);

         if (Ok && mSND.Min().IsInit()   && (*itN< mSND.Min().Val()))
            Ok = false;
         if (Ok && mSND.Max().IsInit()   && (*itN> mSND.Max().Val()))
            Ok = false;

         if (Ok && (!NameFilter(mSND.SubDir().Val(),mICNM,mSND.Filter(),*itN)))
            Ok = false;

         if (Ok)
         {
             mRes.push_back(aName);
         }
     }
}


const cInterfChantierSetNC::tSet  * cSetName::Get()
{
        CompileDef();
        if (!mExtIsCalc)
        {
            mExtIsCalc = true;

            for
            (
                   std::list<std::string>::const_iterator itL=mSND.NamesFileLON().begin();
                   itL!=mSND.NamesFileLON().end();
                   itL++
            )
            {
                cListOfName aLON = StdGetFromPCP(mDir+*itL,ListOfName);
                InternalAddList(aLON.Name());
            }



            for
            (
                   std::list<std::string>::const_iterator itA=mSND.PatternAccepteur().begin();
                   itA!=mSND.PatternAccepteur().end();
                   itA++
            )
            {
                std::list<std::string> aLN;
                if (NameIsNKSSet(*itA))
                {
                     const cInterfChantierNameManipulateur::tSet * mSetIm = mICNM->Get(*itA);
                     std::copy(mSetIm->begin(),mSetIm->end(),back_inserter(aLN));
                }
                else
                {

                    aLN = RegexListFileMatch
                          (
                                  (mSND.AddDirCur().Val()?mDir:"") + mSND.SubDir().Val(),
                                  *itA,
                                  mSND.NivSubDir().Val(),
                                  mSND.NameCompl().Val()
                          );
                }
                InternalAddList(aLN);
            }

            for
            (
                std::list<std::string>::const_iterator itA=mSND.Name().begin();
                itA!=mSND.Name().end();
                itA++
            )
            {
                mRes.push_back(*itA);
            }
            std::sort(mRes.begin(),mRes.end());
            mRes.erase(std::unique(mRes.begin(),mRes.end()),mRes.end());
        }

        return &mRes;
}

    /*******************************************************/
    /*                                                     */
    /*                    cDicoSetNC                       */
    /*                                                     */
    /*******************************************************/

    cDicoSetNC::cDicoSetNC  ()
    {
    }

    void cDicoSetNC::assign(const tKey & aKey,cSetName * aSet)
    {
        if ( isUsingSeparateDirectories() )
        {
            if ( aKey.find("NKS-Set-Orient")!=string::npos )
            {
                aSet->setDir( MMOutputDirectory() );
            }
        }
        mDico[aKey] = aSet;
    }

    void cDicoSetNC::Add(const tKey & aKey,cSetName * aSet)
    {
        if (mDico[aKey]!=0)
        {
            std::cout << "For key =" << aKey <<"\n";
            ELISE_ASSERT(false,"Key non unique dans cDicoSetNC::Add");
        }
        mDico[aKey] = aSet;
    }

    cSetName *  cDicoSetNC::GetSet(const tKey & aKey)
    {
        // Si le nom existe dans le dico, on le retourne
        {
            //std::cout << "FOUND0 " << aKey<<"\n";
            std::map<tKey,cSetName *>::iterator anIt = mDico.find(aKey);
            if (anIt!=mDico.end())
                return anIt->second  ;
        }

        std::vector<std::string>  aVParams;
        std::string aKeySsArb;
        // SplitInNArroundCar(aKey,'@',aKeySsArb,aVParams);
        NewSplit(aKey,aKeySsArb,aVParams);
        //  pas d'arrowbasc : fin
        if ( aVParams.size()==0)
        {
            return GetSet("NKS-Set-OfPattern@"+aKey);
            //std::cout << "NO @ " << aKey<<"\n";
            return 0;
        }
        // Si le nom sans arrobasque n'existe pas fin
        std::map<tKey,cSetName *>::iterator anIt = mDico.find(aKeySsArb);
        if (anIt==mDico.end())
        {
            return 0;
        }


        // Si il existe sans @, on le construit avec
        cSetNameDescriptor & aSND = * new cSetNameDescriptor(anIt->second->SND());
        TransFormArgKey(aSND,false,aVParams);
        cSetName *aSet = new cSetName(anIt->second->ICNM(),aSND);
        assign( aKey, aSet );

        return aSet;
    }

    const cInterfChantierSetNC::tSet *  cDicoSetNC::Get(const tKey & aKey)
    {
        cSetName *  aSet = GetSet(aKey);

        return  aSet ? aSet->Get() : 0;
    }

    bool cDicoSetNC::SetHasKey(const tKey & aKey) const
    {
        return  mDico.find(aKey) != mDico.end();
    }

    const bool  * cDicoSetNC::SetIsIn(const tKey & aKey,const std::string & aName)
    {
        static const bool SFalse = false;
        static const bool STrue  = true;

        std::map<tKey,cSetName *>::iterator anIt = mDico.find(aKey);

        if  (anIt==mDico.end())
            return 0;

        return anIt->second->SetBasicIsIn(aName) ? & STrue : & SFalse;
    }


    /*******************************************************/
    /*                                                     */
    /*                    cDicoChantierNC                  */
    /*                                                     */
    /*******************************************************/

    void cDicoChantierNC::PrivateAdd(const tKey & aKey,cInterfNameCalculator * aNC)
    {
        if (mINC_Dico[aKey]!=0)
        {
            std::cout << "For key =" << aKey <<"\n";
            ELISE_ASSERT(false,"Key non unique dans cDicoChantierNC::Add");
        }
        mINC_Dico[aKey] = aNC;
    }

    void cDicoChantierNC::Add(cInterfChantierNameManipulateur * aICNM,const cKeyedNamesAssociations & aKNA)
    {
        PrivateAdd
            (
            aKNA.Key(),
            cInterfNameCalculator::StdCalcFromXML
            (
            aKNA,
            aICNM,
            aKNA.SubDirAutoMake().Val(),
            aKNA.SubDirAutoMakeRec().Val(),
            aKNA.Calcs()
            )
            );
    }

    std::map<cDicoChantierNC::tKey,cInterfNameCalculator *>::iterator  cDicoChantierNC::StdFindKey(const tKey & aKey)
    {
        std::map<tKey,cInterfNameCalculator *>::iterator  anIt = mINC_Dico.find(aKey);

        if (anIt!=mINC_Dico.end())
        {
            return anIt;
        }

        std::string aKeySsArb;
        std::vector<std::string> aVParams;
        //SplitInNArroundCar(aKey,'@',aKeySsArb,aVParams);
        NewSplit(aKey,aKeySsArb,aVParams);
        // S'il n'y a pas d'@
        if ( aVParams.size() == 0)
        {
            return anIt;
        }

        // Si le nom sans @ n'existe pas
        anIt = mINC_Dico.find(aKeySsArb);
        if (anIt==mINC_Dico.end())
        {
            return anIt;
        }
        cInterfNameCalculator * anINC = anIt->second;
        cKeyedNamesAssociations * aKNA0 = anINC->KNA();
        // Si on de derive pas d'une classe cree par cKeyedNamesAssociations
        if (aKNA0 ==0)
        {
            return mINC_Dico.end();
        }
        cKeyedNamesAssociations  & aNewKNA = * new cKeyedNamesAssociations(*aKNA0);
        TransFormArgKey(aNewKNA,false,aVParams);
        Add(anINC->ICNM(),aNewKNA);
        // mDico[aKey]  = new;

        return mINC_Dico.find(aKey);
    }


    cInterfChantierNC::tNuplet  cDicoChantierNC::Direct(const tKey & aKey,const tNuplet& aNuple)
    {
        // std::map<tKey,cInterfNameCalculator *>::iterator anIt = mDico.find(aKey);
        std::map<tKey,cInterfNameCalculator *>::iterator anIt = StdFindKey(aKey);

        if (anIt==mINC_Dico.end())
        {
            if (DebugPCP)
            {
                std::cout << "Cannot find autom for Key " << aKey << "\n";
            }
            return cInterfNameCalculator::NotDef();
        }
        if (DebugPCP)
        {
            std::cout << "Autom  found for Key " << aKey << "\n";
        }

        tNuplet aRes =  anIt->second->Direct(aNuple);
        VerifSol(aRes,aKey,aNuple);
        return aRes;
    }

    cInterfChantierNC::tNuplet  cDicoChantierNC::Inverse(const tKey & aKey,const tNuplet& aNuple)
    {
        //  std::map<tKey,cInterfNameCalculator *>::iterator anIt = mDico.find(aKey);
        std::map<tKey,cInterfNameCalculator *>::iterator anIt = StdFindKey(aKey);

        if (anIt==mINC_Dico.end())
            return cInterfNameCalculator::NotDef();
        tNuplet aRes =  anIt->second->Inverse(aNuple);
        VerifSol(aRes,aKey,aNuple);
        return aRes;
    }




    bool cDicoChantierNC::AssocHasKey(const tKey & aKey) const
    {
        std::map<cDicoChantierNC::tKey,cInterfNameCalculator *>::iterator anIt =
            const_cast<cDicoChantierNC*>(this)->StdFindKey(aKey);


        return (anIt != mINC_Dico.end());
        /*
        std::cout  << " AHK " << ( anIt == mINC_Dico.end() )
        << "  " << (mINC_Dico.find(aKey) != mINC_Dico.end()) << "\n";
        // const_cast<cDicoChantierNC*>(this)->StdFindKey(aKey);

        return (mINC_Dico.find(aKey) != mINC_Dico.end());
        */
    }

    /*******************************************************/
    /*                                                     */
    /*       cStdChantierMonoManipulateur                  */
    /*                                                     */
    /*******************************************************/

    cStdChantierMonoManipulateur::cStdChantierMonoManipulateur
        (
        eOriSCMN                          anOrig,
        cInterfChantierNameManipulateur * aGlob,
        const std::string & aDir,
        const cChantierDescripteur & aCD
        ) :
    cInterfChantierNameManipulateur(0,0,aDir),
        mGlob                          (aGlob),
        //mOrig                          (anOrig),
        mSets (cInterfChantierSetNC::StdCalcFromXML(aGlob,aCD.KeyedSetsOfNames())),
        mAssoc (cInterfChantierNC::StdCalcFromXML(aGlob,aCD.KeyedNamesAssociations())),
        mAncCompute (0),
        mLBatch     (aCD.BatchChantDesc()),
        mLKMatr     (aCD.KeyedMatrixStruct()),
        mLShow      (aCD.ShowChantDesc())
    {
        if (aCD.ExitOnBrkp().IsInit())
            TheExitOnBrkp = aCD.ExitOnBrkp().Val();

        for
            (
            std::list<cKeyedSetsORels>::const_iterator itK=aCD.KeyedSetsORels().begin();
        itK !=aCD.KeyedSetsORels().end();
        itK++
            )
        {
            ELISE_ASSERT(mRels[itK->Key()]==0,"Non unique key for KeyedSetsORels");
            mRels[itK->Key()]= new cStdChantierRel(*aGlob,itK->Sets());
        }

        for
            (
            std::list<cClassEquivDescripteur>::const_iterator itK=aCD.KeyedClassEquiv().begin();
        itK !=aCD.KeyedClassEquiv().end();
        itK++
            )
        {
            ELISE_ASSERT(mEquivs[itK->KeyClass()]==0,"Non unique key for ClassEquivDescripteur");
            mEquivs[itK->KeyClass()]= new cStrRelEquiv(*aGlob,*itK);
        }





        for
            (
            std::list<cAPrioriImage>::const_iterator itA = aCD.APrioriImage().begin();
        itA != aCD.APrioriImage().end();
        itA++
            )
        {
            cContenuAPrioriImage * aCAPI = new cContenuAPrioriImage(itA->ContenuAPrioriImage());
            mVecAP.push_back(aCAPI);
            aCAPI->ElInt_CaPImAddedSet().SetVal(itA->KeyedAddedSet().ValWithDef(""));
            aCAPI->ElInt_CaPImMyKey().SetVal(itA->Key().Val());
            for
                (
                std::list<std::string>::const_iterator itN = itA->Names().begin();
            itN != itA->Names().end();
            itN++
                )
            {
                AddDicAP(*itN,itA->Key().Val(),aCAPI);
                /*
                std::string aNamePair = *itN + "@" + itA->Key().Val();
                mDicAP[aNamePair] = aCAPI;
                */
            }
        }

        {
            if (aCD.BaseDatas().IsInit())
            {
                AddBase(aCD.BaseDatas().Val());
            }
            for
                (
                std::list<std::string>::const_iterator itF=aCD.FilesDatas().begin();
            itF!=aCD.FilesDatas().end();
            itF++
                )
            {
                AddBase
                    (
                    StdGetObjFromFile<cBaseDataCD>
                    (
                    aDir+*itF,
                    StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                    "BaseDataCD",
                    "BaseDataCD"
                    )
                    );
            }

            if (aCD.KeySuprAbs2Rel().IsInit())
            {
                if (aGlob)
                {
                    aGlob->SetKeySuprAbs2Rel(new std::string(aCD.KeySuprAbs2Rel().Val()));
                }
            }

            if (aCD.MakeDataBase().IsInit())
            {
                ELISE_ASSERT
                    (
                    anOrig==eMMLCD_SCMN,
                    "MakeDataBase , only in MicMac-LocalChantierDescripteur.xml"
                    );
                if (aGlob)
                {
                    aGlob->SetMkDB(aCD.MakeDataBase().Val());
                }
            }
        }
    }


    cSetName *  cStdChantierMonoManipulateur::GetSet(const tKey & aKey)
    {
        return mSets->GetSet(aKey);
    }

    template <class Type>  Type *  cMapIdName2Val<Type>::Get(const std::string& anIdBase,const std::string& anIdVal)
    {

        typename tAllBases::iterator   itB = mDatas.find(anIdBase);
        if (itB==mDatas.end())
        {
            return 0;
        }

        typename tOneBase::iterator   itR = itB->second.find(anIdVal);
        if (itR==itB->second.end())
        {
            return 0;
        }
        return &(itR->second);
    }

    template <class Type>  const Type *  cMapIdName2Val<Type>::Get(const std::string& anIdBase,const std::string& anIdVal) const
    {
        return const_cast<cMapIdName2Val<Type> *>(this)->Get(anIdBase,anIdVal);
    }

    template <class Type>  void cMapIdName2Val<Type>::Add
        (
        const std::string& anIdBase,
        const std::string& anIdVal,
        const Type & aVal
        )
    {
        tOneBase & aBase = mDatas[anIdBase];
        if (DicBoolFind(aBase,anIdVal))
        {
            std::cout << "====== For Base " << anIdBase << " For Entry " << anIdVal << "\n";
            ELISE_ASSERT(false,"Multiple definition, in cMapIdName2Val<Type>::Add");
        }
        aBase[anIdVal] = aVal;
    }


    template <class tBase,class tEntriesXML>
    void Add2Base(tBase & aBase,const std::string & aNameBase,const std::list<tEntriesXML>& aLE)
    {
        for
            (
            typename std::list<tEntriesXML>::const_iterator itE=aLE.begin();
        itE!=aLE.end();
        itE++
            )
        {
            aBase.Add(aNameBase,itE->Key(),itE->Val());
        }
    }

    // UNE MACRO, Trop complique de templatiser, desole ....

#define MACROADD2BASE(tEntry)\
    for\
    (\
    std::list<cBases##tEntry>::const_iterator it= aBDC.Bases##tEntry().begin();\
    it != aBDC.Bases##tEntry().end();\
    it++\
    )\
    {\
    Add2Base(mBases##tEntry,it->NameBase(),it->tEntry##Entries());\
    }


    void cStdChantierMonoManipulateur::AddBase(const cBaseDataCD& aBDC)
    {
        MACROADD2BASE(Pt3dr)
            MACROADD2BASE(Scal)

    }

#define MACROADDGETBASE(tType,tEntry)\
    const tType  * cStdChantierMonoManipulateur::SvpGet##tEntry(const std::string& anIdBase,const std::string& anIdVal) const\
    {\
    return mBases##tEntry.Get(anIdBase,anIdVal);\
        }\
        const tType  * cStdChantierMultiManipulateur::SvpGet##tEntry(const std::string& anIdBase,const std::string& anIdVal) const\
    {\
    for (int aK = (int)(mVM.size() - 1) ; aK>=0 ; aK--)\
    {\
    const tType * aRes = mVM[aK]->SvpGet##tEntry(anIdBase,anIdVal);\
    if (aRes)  \
    return aRes;\
        }\
        return 0;\
        }\
        const tType  & cInterfChantierNameManipulateur::Get##tEntry(const std::string& anIdBase,const std::string& anIdVal)\
    {\
    const tType * aRes=SvpGet##tEntry(anIdBase,anIdVal);\
    if (aRes==0)\
    {\
    std::cout << "Base " << anIdBase  << " Entry= " << anIdVal << "\n";\
    ELISE_ASSERT(false,"Cannot find requested entry cInterfChantierNameManipulateur::GetXXX");\
        }\
        return *aRes;\
        }


    MACROADDGETBASE(Pt3dr,Pt3dr)
        MACROADDGETBASE(double,Scal)

        /*
        voir exemple  dans :

        micmac/applis/XML-Pattron/Muru/VolRadialeNord-2/

        Dico-RTL-BiaisGPS.xml   definit un fichier de bases de donnees


        charge par  :
        <FilesDatas>  Dico-RTL-BiaisGPS.xml </FilesDatas>
        dans MicMac-LocalChantierDescripteur.xml


        Utilise dans  Apero-2-Compense-BiaisGPS.xml avec <CalcOffsetCentre>
        Voir le code Pt3dr cTypeEnglob_Centre::CreateFromXML

        */





        cTplValGesInit<cResBoxMatr> cStdChantierMonoManipulateur::GetBoxOfMatr
        (
        const tKey& aKey,
        const std::string& aName
        )
    {
        cTplValGesInit<cResBoxMatr> aRes;
        for
            (
            std::list<cKeyedMatrixStruct>::const_iterator itM=mLKMatr.begin();
        itM!=mLKMatr.end();
        itM++
            )
        {
            if (itM->Key() == aKey)
            {
                const cImMatrixStructuration & aMatr = itM->Matrix();
                const tSet * aV = Get(aMatr.KeySet());
                const std::string * aV0 = &((*aV)[0]);
                const std::string * aVN = aV0+aV->size();
                const std::string * aLB = lower_bound(aV0,aVN,aName);

                if ((aLB==aVN) || (*aLB!=aName))
                {
                    std::cout << "For Key " << aKey
                        <<  " Cannot get " << aName
                        <<  " in " << aMatr.KeySet()<< "\n";
                    ELISE_ASSERT(false,"GetBoxOfMatr");
                }

                cResBoxMatr aMR;

                Pt2di aPer = aMatr.Period();
                int aIP = aPer.x * aPer.y;

                int aK = (int)(aLB - aV0);
                aMR.mNumMatr = aK/aIP;
                aMR.mNumGlob = aK;
                aK = aK % aIP;

                if ( aMatr.XVarieFirst())
                    aMR.mId = Pt2di(aK%aPer.x,aK/aPer.x);
                else
                    aMR.mId = Pt2di(aK/aPer.y,aK%aPer.y);

                if (! aMatr.XCroissants())
                    aMR.mId.x = aPer.x-1-aMR.mId.x;
                if (! aMatr.YCroissants())
                    aMR.mId.y = aPer.y-1-aMR.mId.y;

                aMR.mBox =
                    Box2dr
                    (
                    Pt2dr(double(aMR.mId.x)/aPer.x,double(aMR.mId.y)/aPer.y),
                    Pt2dr(double(aMR.mId.x+1)/aPer.x,double(aMR.mId.y+1)/aPer.y)
                    );

                aRes.SetVal(aMR);
                return aRes;
            }
        }

        return aRes;
    }


    const cBatchChantDesc *
        cStdChantierMonoManipulateur::BatchDesc(const tKey & aKey) const
    {
        for
            (
            std::list<cBatchChantDesc>::const_iterator itB=mLBatch.begin();
        itB!=mLBatch.end();
        itB++
            )
        {
            if (itB->Key() == aKey)
                return & (*itB);
        }

        return 0;
    }


    const cShowChantDesc * cStdChantierMonoManipulateur::ShowChant(const tKey & aFile) const
    {
        for
            (
            std::list<cShowChantDesc>::const_iterator itS=mLShow.begin();
        itS!=mLShow.end();
        itS++
            )
        {
            if (itS->File() == aFile)
                return & (*itS);
        }

        return 0;
    }

    void  cStdChantierMonoManipulateur::AddDicAP
        (
        const std::string & aName,
        const std::string & aKey,
        cContenuAPrioriImage * aCAPI
        )
    {
        std::string aNamePair = aName + "@" + aKey;
        cContenuAPrioriImage *& aRef = mDicAP[aNamePair];
        if (aRef==0)
        {
            mDicAP[aNamePair] = aCAPI;
        }
        else if (aRef != aCAPI)
        {
            std::cout << "For " << aNamePair << "\n";
            ELISE_ASSERT
                (
                false,
                "Associations multiples incoherentes de ContenuAPrioriImage"
                );
        }
    }

    cContenuAPrioriImage * cStdChantierMonoManipulateur::GetAPriori
        (
        const std::string & aName,
        const std::string & aKey,
        cInterfChantierNameManipulateur * ancetre
        )
    {
        Compute(ancetre);
        std::string aNamePair = aName+ "@" + aKey;
        std::map<std::string,cContenuAPrioriImage *>::iterator it = mDicAP.find(aNamePair);

        if (it!=mDicAP.end())
            return it->second;
        return 0;
    }

    bool  cStdChantierMonoManipulateur::RelHasKey(const tKey & aKey)
    {

        if ( mRels.find(aKey) != mRels.end()) return true;

        std::vector<std::string>  aVParams;
        std::string aKeySsArb;
        // SplitInNArroundCar(aKey,'@',aKeySsArb,aVParams);
        NewSplit(aKey,aKeySsArb,aVParams);
        if (aVParams.size())
        {
            std::map<tKey,cStdChantierRel *>::const_iterator iter = mRels.find(aKeySsArb);
            if (iter == mRels.end())
                return false;
            cStdChantierRel *  aSCR = iter->second;
            cNameRelDescriptor & aNRD = *(new cNameRelDescriptor(aSCR->NRD()));

            // std::cout << aKey  << "+++" << aVParams[0] << " " << aVParams[1]<< " " << aVParams.size() << "\n";
            TransFormArgKey(aNRD,false,aVParams);
            // std::cout << "lllllllllllll\n";
            cStdChantierRel * aNewSCR =  new cStdChantierRel(*mGlob,aNRD);
            mRels[aKey] = aNewSCR;
            return true;
        }
        /// XXXX   BBBBBB
        return false;
    }


std::string cInterfChantierNameManipulateur::StdKeyOrient(const tKey & aKeyOri)
{
   if (AssocHasKey(aKeyOri)) return aKeyOri;

   std::string aKey = aKeyOri;
   if (aKey.c_str()[0] != '-') aKey = "-" + aKey ;
   return "NKS-Assoc-Im2Orient@" + aKey;
}


std::string StdNameGBOrient(const std::string & anOri,const std::string & aName,bool AddMinus)
{
   return "Ori" + std::string(AddMinus?"-":"")+ anOri +"/GB-Orientation-" + aName  + ".xml";
}
std::string StdNameCSOrient(const std::string & anOri,const std::string & aName,bool AddMinus)
{
   return "Ori" + std::string(AddMinus?"-":"")+ anOri +"/Orientation-" + aName  + ".xml";
}




std::string  cInterfChantierNameManipulateur::StdNameHomol
             (
                  const std::string & anExt,
                  const std::string & aI1,
                  const std::string & aI2
             )
{
    return Assoc1To2(std::string("NKS-Assoc-CplIm2Hom@")+anExt+"@dat",aI1,aI2,true);
}

ElPackHomologue cInterfChantierNameManipulateur::StdPackHomol
                (
                    const std::string & anExt,
                    const std::string & aI1,
                    const std::string & aI2
                )
{
   return ElPackHomologue::FromFile(StdNameHomol(anExt,aI1,aI2));
}



std::string cInterfChantierNameManipulateur::NameOriStenope(const tKey & aKeyOri,const std::string & aNameIm)
{
           std::string aKey = StdKeyOrient(aKeyOri);
           return Assoc1To1(aKey,aNameIm,true);
}
 


    std::vector<std::string>  cInterfChantierNameManipulateur::GetSetOfRel(const tKey & aKey,const std::string & aStr0,bool Sym)
    {
        std::set<std::string> aRes;

        const tRel * aRel = GetRel(aKey);

        for (int aK=0 ; aK<int(aRel->size()) ;aK++)
        {
            const std::string & aStr1 = (*aRel)[aK].N1();
            const std::string & aStr2 = (*aRel)[aK].N2();

            if (aStr1 == aStr0)
            {
                aRes.insert(aStr2);
            }
            if (Sym && (aStr2==aStr0))
            {
                aRes.insert(aStr1);
            }
        }

        return std::vector<std::string>(aRes.begin(),aRes.end());;
    }



    const cInterfChantierNameManipulateur::tRel *
        cStdChantierMonoManipulateur::GetRel(const tKey & aKey)
    {
        ELISE_ASSERT(RelHasKey(aKey),"No Key in  cStdChantierMonoManipulateur::GetRel");
        return mRels[aKey]->GetRel();
    }

    cStrRelEquiv *  cStdChantierMonoManipulateur::GetEquiv(const tKey & aKey)
    {
        std::map<tKey,cStrRelEquiv *>::const_iterator it = mEquivs.find(aKey);
        return (it== mEquivs.end()) ? 0 : it->second;
    }


    const bool  *  cStdChantierMonoManipulateur::SetIsIn(const tKey & aKey,const std::string & aName)
    {
        return mSets->SetIsIn(aKey,aName);
    }


    const cStdChantierMonoManipulateur::tSet *  cStdChantierMonoManipulateur::Get(const tKey & aKey)
    {
        return mSets->Get(aKey);
    }

    cStdChantierMonoManipulateur::tNuplet
        cStdChantierMonoManipulateur::Direct(const tKey & aKey,const tNuplet& aNuple)
    {
        return mAssoc->Direct(aKey,aNuple);
    }

    cStdChantierMonoManipulateur::tNuplet
        cStdChantierMonoManipulateur::Inverse(const tKey & aKey,const tNuplet& aNuple)
    {
        return mAssoc->Inverse(aKey,aNuple);
    }

    std::map<std::string,cStdChantierMonoManipulateur *> cStdChantierMonoManipulateur::theDicAlloc;

    cStdChantierMonoManipulateur *
        cStdChantierMonoManipulateur::StdGetFromFile
        (
        eOriSCMN                          anOrig,
        cInterfChantierNameManipulateur * aGlob,
        const std::string & aDir,
        const std::string & aFileXML,
        bool                AddCamDB
        )
    {
        //if (theDicAlloc[aFileXML] == 0)
        {
            cChantierDescripteur  aCD  =
                StdGetObjFromFile<cChantierDescripteur>
                (
                aFileXML,
                StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                "ChantierDescripteur",
                "ChantierDescripteur",
                false,
                &(aGlob->ArgTree())
                );

            if (AddCamDB && aCD.LocCamDataBase().IsInit())
            {
                DC_Add(aCD.LocCamDataBase().Val());
            }
            return new cStdChantierMonoManipulateur(anOrig,aGlob,aDir,aCD);
            //theDicAlloc[aFileXML]  = new cStdChantierMonoManipulateur(aGlob,aDir,aCD);
        }


        // return theDicAlloc[aFileXML] ;
    }

    bool cStdChantierMonoManipulateur::AssocHasKey(const tKey & aKey) const
    {
        return mAssoc->AssocHasKey(aKey);
    }

    bool cStdChantierMonoManipulateur::SetHasKey(const tKey & aKey) const
    {
        return mSets->SetHasKey(aKey);
    }


    void cStdChantierMonoManipulateur::Compute(cInterfChantierNameManipulateur * ancetre)
    {
        if (mAncCompute != 0)
        {
            ELISE_ASSERT(mAncCompute==ancetre,"Incoherent ancester in cStdChantierMonoManipulateur::Compute");
            return;
        }
        mAncCompute = ancetre;

        for ( int aKAp=0 ; aKAp<int(mVecAP.size()); aKAp++)
        {
            cContenuAPrioriImage & aCAPI = *(mVecAP[aKAp]);
            const std::string & aKeySet = aCAPI.ElInt_CaPImAddedSet().Val();
            if (aKeySet!="")
            {
                const tSet * aSet = ancetre->Get(aKeySet);
                for (int aKs=0 ; aKs<int(aSet->size()) ; aKs++)
                {
                    AddDicAP
                        (
                        (*aSet)[aKs],
                        aCAPI.ElInt_CaPImMyKey().Val(),
                        &aCAPI
                        );
                }
            }
        }
    }

    /*******************************************************/
    /*                                                     */
    /*        cStdChantierMultiManipulateur                */
    /*                                                     */
    /*******************************************************/

    cStdChantierMultiManipulateur::cStdChantierMultiManipulateur(int argc,char ** argv,const std::string & aDir) :
    cInterfChantierNameManipulateur(argc,argv,aDir)
    {
    }


    void cStdChantierMultiManipulateur::Add(cInterfChantierNameManipulateur * aCNM)
    {
        mVM.push_back(aCNM);
    }

    const char cInterfChantierNameManipulateur::theCharModifDico = '+';
    const char cInterfChantierNameManipulateur::theCharSymbOptGlob = '@';  // Surtout pas '-', interfere avec ELDcraw !!

    const cInterfChantierNameManipulateur::tSet *  cStdChantierMultiManipulateur::Get(const tKey & aKey)
    {
        for (int aK = (int)(mVM.size() - 1) ; aK>=0 ; aK--)
        {
            const tSet * aSet = mVM[aK]->Get(aKey);
            if (aSet!=0)
            {
                if (!MPD_MM()) 
                {
                    std::cout<<"\""<<aKey<<"\": "<<aSet->size()<<" matches."<<std::endl;
                }

                return aSet;
            }
        }
        {
              std::string aName = GetNameWithoutPerc(aKey);
              if (aName!="") return Get(aName);
/*
              std::string aDir,aName;
              SplitDirAndFile(aDir,aName,aKey);
              if (aName.size() < aKey.size())
                 return Get(aName);
*/
        }
        std::cout << "For Key = " << aKey << "\n";
        ELISE_ASSERT(false,"Cannot get keyed set");
        return 0;
    }

    const cBatchChantDesc *
        cStdChantierMultiManipulateur::BatchDesc(const tKey & aKey) const
    {
        for (int aK = (int)(mVM.size() - 1) ; aK>=0 ; aK--)
        {
            const cBatchChantDesc  * aL = mVM[aK]->BatchDesc(aKey);
            if (aL!=0)
                return aL;
        }
        std::cout << "For Key = " << aKey << "\n";
        ELISE_ASSERT(false,"Cannot get keyed BatchDesc");
        return 0;
    }

    const cShowChantDesc *
        cStdChantierMultiManipulateur::ShowChant(const tKey & aKey) const
    {
        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            const cShowChantDesc  * aL = mVM[aK]->ShowChant(aKey);
            if (aL!=0)
                return aL;
        }
        std::cout << "For Key = " << aKey << "\n";
        ELISE_ASSERT(false,"Cannot get keyed BatchDesc");
        return 0;
    }


    cSetName *  cStdChantierMultiManipulateur::GetSet(const tKey & aKey)
    {
        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            cSetName * aRes =  mVM[aK]->GetSet(aKey);
            if (aRes) return aRes;
        }
        return 0;

    }




    cContenuAPrioriImage * cStdChantierMultiManipulateur::GetAPriori
        (
        const std::string & aName,
        const std::string & aKey,
        cInterfChantierNameManipulateur  * ancetre
        )
    {
        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            cContenuAPrioriImage  * aRes = mVM[aK]->GetAPriori(aName,aKey,ancetre);
            if (aRes != 0)
                return aRes;
        }
        return 0;
    }

    cStrRelEquiv *  cStdChantierMultiManipulateur::GetEquiv(const tKey & aKey)
    {
        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            cStrRelEquiv * aRes = mVM[aK]->GetEquiv(aKey);
            if (aRes)
                return aRes;
        }
        std::cout << "For Key = " << aKey << "\n";
        ELISE_ASSERT(false,"Cannot get keyed Equiv");
        return 0;
    }


    const cInterfChantierNameManipulateur::tRel *
        cStdChantierMultiManipulateur::GetRel(const tKey & aKey)
    {
        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            if ( mVM[aK]->RelHasKey(aKey))
                return mVM[aK]->GetRel(aKey);
        }
        std::cout << "For Key = " << aKey << "\n";
        ELISE_ASSERT(false,"Cannot get keyed rel");
        return 0;
    }

    void cInterfChantierNameManipulateur::SetMapCmp
        (
        const std::list<cCmdMappeur> & aLCM,
        int argc,
        char ** argv
        )
    {
        for
            (
            std::list<cCmdMappeur>::const_iterator itC=aLCM.begin();
        itC!=aLCM.end();
        itC++
            )
        {
            SetMapCmp(*itC,argc,argv);
        }
    }

    void cInterfChantierNameManipulateur::SetMapCmp
        (
        const cCmdMappeur & aCM,
        int argc,
        char ** argv
        )
    {
        if (!aCM.ActivateCmdMap())
            return;

        std::string aCom0 = string("\"")+current_program_fullname()+"\" "+current_program_subcommand()+" ";
        for (int aKA=1 ; aKA<argc ; aKA++)
        {
            aCom0 = aCom0 +  std::string(argv[aKA])+ " ";
        }
        aCom0 = aCom0 + "ActivateCmdMap=false ";

        std::vector<std::string> aVFiles;
        std::vector<std::string> aVMapFile;
        std::string aKeyAssoc;
        if (aCM.CM_One().IsInit())
        {
            aVMapFile.push_back("");
            aVFiles.push_back("");
        }
        else if (aCM.CM_Set().IsInit())
        {
            const cCM_Set & aCS = aCM.CM_Set().Val();
            const std::vector<std::string> * aSet= Get(aCS.KeySet());
            for (int aKS=int(aSet->size())-1 ; aKS>=0 ;  aKS--)
            {
                std::string aVal = (*aSet)[aKS];
                aVFiles.push_back(aVal);

                if (aCS.KeyAssoc().IsInit())
                {
                    aKeyAssoc = aCM.KeyAssoc().Val();
                    aVal = Assoc1To1(aCM.KeyAssoc().Val(),aVal,true);
                }

                aVMapFile.push_back
                    (
                    std::string(" +")
                    + aCM.NameVarMap()
                    + std::string("=")
                    + aVal
                    );
            }
        }
        else
        {
            ELISE_ASSERT(false,"Unknown mode in CmdMappeur");
        }


        std::vector<std::string> aVAux;
        std::vector<std::string> aVMapAux;
        for
            (
            std::list<cCMVA>::const_iterator itCV=aCM.CMVA().begin();
        itCV!=aCM.CMVA().end();
        itCV++
            )
        {
            std::string  aSAux="";
            for
                (
                std::list<cCpleString>::const_iterator itCpl=itCV->NV().begin();
            itCpl!=itCV->NV().end();
            itCpl++
                )
            {
                if (aSAux=="")
                {
                    aVAux.push_back(itCpl->N2());
                }
                aSAux=aSAux+std::string(" +")+itCpl->N1()+std::string("=")+itCpl->N2();
            }
            aVMapAux.push_back(aSAux);
        }
        if ( aVMapAux.empty())
        {
            aVMapAux.push_back("");
            aVAux.push_back("");
        }


        cEl_GPAO * aGPAO=0;
        if (aCM.ByMkF().IsInit())
            aGPAO = new cEl_GPAO;


        for (int aKF=0; aKF<int(aVMapFile.size()) ; aKF++)
        {
            std::vector<std::string> aStrRel;
            if (aCM.CmdMapRel().IsInit())
            {
                const cCmdMapRel & aCMR = aCM.CmdMapRel().Val();
                std::vector<std::string> aVArc = TheGlob->GetSetOfRel(aCMR.KeyRel(),aVFiles[aKF]);

                for (int aKA=0 ; aKA<int(aVArc.size()) ; aKA++)
                {
                    std::string aN2 = aVArc[aKA];
                    if (aKeyAssoc!="")
                        aN2 =  Assoc1To1(aKeyAssoc,aN2,true); ;
                    aStrRel.push_back(" +"+ aCMR.NameArc() + "=" +aN2);
                }
            }
            else
            {
                aStrRel.push_back("");
            }
            for (int aKR=0; aKR<int(aStrRel.size()) ; aKR++)
            {
                for (int aKV=0; aKV<int(aVMapAux.size()) ; aKV++)
                {
                    std::string aCom= aCom0+aVMapFile[aKF]+aVMapAux[aKV] + aStrRel[aKR];
                    std::cout << aCom << "\n";

                    if (aGPAO)
                    {
                        std::string aTarget =  std::string("T_")
                            +  ToString(aKF)
                            +  std::string("_")
                            +  ToString(aKV);
                        if (aCM.KeyTargetMkF().IsInit())
                        {
                            aTarget = Dir()+ Assoc1To2
                                (
                                aCM.KeyTargetMkF().Val(),
                                aVFiles[aKF],
                                aVAux[aKV],
                                true
                                );

                        }
                        aGPAO->GetOrCreate(aTarget,aCom);
                        aGPAO->TaskOfName("all").AddDep(aTarget);
                    }
                    else
                    {
                        int OK= system_call(aCom.c_str());
                        ELISE_ASSERT(OK==0,"Pb in ss process By Cmd-Map");
                        std::cout << "RESULT = " << OK << "\n";
                    }
                }
            }
        }
        if (aGPAO)
            aGPAO->GenerateMakeFile(aCM.ByMkF().Val());
    }




    cTplValGesInit<cResBoxMatr> cStdChantierMultiManipulateur::GetBoxOfMatr
        (
        const tKey& aKey,
        const std::string& aName
        )
    {
        cTplValGesInit<cResBoxMatr> aRes;

        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            aRes = mVM[aK]->GetBoxOfMatr(aKey,aName);
            if (aRes.IsInit())
                return aRes;
        }
        std::cout << "For Key = " << aKey  << " Name " << aName << "\n";
        ELISE_ASSERT(false,"Cannot get GetBoxOfMatr");
        return aRes;
    }




    bool  cStdChantierMultiManipulateur::AssocHasKey(const tKey & aKey) const
    {
        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            if ( mVM[aK]->AssocHasKey(aKey))
                return true;
        }
        return false;
    }

    bool  cStdChantierMultiManipulateur::SetHasKey(const tKey & aKey) const
    {

        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            if ( mVM[aK]->SetHasKey(aKey))
                return true;
        }

        std::string aKeySsArb,aNameSubDir;
        SplitIn2ArroundCar(aKey,'@',aKeySsArb,aNameSubDir,true);

        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            if ( mVM[aK]->SetHasKey(aKeySsArb))
                return true;
        }
        return false;
    }

    bool  cStdChantierMultiManipulateur::RelHasKey(const tKey & aKey)
    {
        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            if ( mVM[aK]->RelHasKey(aKey))
                return true;
        }
        return false;
    }




    const bool  *  cStdChantierMultiManipulateur::SetIsIn(const tKey & aKey,const std::string & aName)
    {
        for (int aK = (int)(mVM.size() - 1) ; aK>=0 ; aK--)
        {
            const bool * aRes = mVM[aK]->SetIsIn(aKey,aName);
            if (aRes!=0)
                return aRes;
        }
        std::cout << "For Key = " << aKey << "\n";
        ELISE_ASSERT(false,"Cannot get keyed set");
        return 0;
    }

    cInterfChantierNameManipulateur::tNuplet
        cStdChantierMultiManipulateur::Direct(const tKey & aKey,const tNuplet& aNuple)
    {
        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            tNuplet  aRes = mVM[aK]->Direct(aKey,aNuple);
            if (cInterfNameCalculator::IsDefined(aRes))
                return aRes;
        }
        if (DebugPCP)
        {
            return cInterfNameCalculator::NotDef();
        }
        else
        {
            DebugPCP= true;
            Direct( aKey,aNuple);
        }

        for (int aKn=0 ; aKn<int(aNuple.size()) ; aKn++)
            std::cout << "  NAME :   " << aNuple[aKn] << "\n";
        std::cout << "For Key = " << aKey << "\n";
        ELISE_ASSERT(false,"Cannot get keyed association");
        return cInterfNameCalculator::NotDef();
    }

    cInterfChantierNameManipulateur::tNuplet
        cStdChantierMultiManipulateur::Inverse(const tKey & aKey,const tNuplet& aNuple)
    {
        for (int aK = (int)(mVM.size() - 1); aK>=0 ; aK--)
        {
            tNuplet  aRes = mVM[aK]->Inverse(aKey,aNuple);
            if (cInterfNameCalculator::IsDefined(aRes))
                return aRes;
        }
        std::cout << "For Key = " << aKey << "\n";
        ELISE_ASSERT(false,"Cannot get keyed association");
        return cInterfNameCalculator::NotDef();
    }

    cSetName *  cInterfChantierNameManipulateur::KeyOrPatSelector(const std::string & aKoP)
    {
        cSetName * aSet = GetSet(aKoP);
        if (aSet) return aSet;
        aSet = GetSet("NKS-Set-OfPattern@"+aKoP);
        if (aSet) return aSet;
        ELISE_ASSERT(false,"Incoherence ine cInterfChantierNameManipulateur::KeyOrPatSelector");
        return 0;
    }


    cSetName *  cInterfChantierNameManipulateur::KeyOrPatSelector(const cTplValGesInit<std::string> & aStr)
    {
        return KeyOrPatSelector(aStr.ValWithDef(".*"));
    }




    /*******************************************************/
    /*                                                     */
    /*        cInterfChantierNameManipulateur              */
    /*                                                     */
    /*******************************************************/

    const std::string cInterfChantierNameManipulateur::theNameGlob
        = "DefautChantierDescripteur.xml";


    const std::string cInterfChantierNameManipulateur::theNameLoc
        = "MicMac-LocalChantierDescripteur.xml";

    const std::string cInterfChantierNameManipulateur::theNamePrelim
        = "MicMac-Prelim-LCD.xml";

    std::string MMDir(){ return ArgvMMDir; }

    std::string current_program_fullname()   { return CurrentProgramFullName;}
    std::string current_program_subcommand() { return CurrentProgramSubcommand;}

    bool MPD_MM()
    {
        static bool aRes = MMUserEnv().UserName().ValWithDef("") == "MPD";
        return aRes;
    }
    bool ERupnik_MM()
    {
        static bool aRes = MMUserEnv().UserName().ValWithDef("") == "ERupnik";
        return aRes;
    }


bool DebugConvCal() {return false;}

#if ELISE_QT
    string MMQtLibraryPath()
    {
        #if defined(__APPLE__) || defined(__MACH__)
            return MMDir() + "Frameworks";
		#elif ELISE_windows
			return MMBin();
        #endif
        return string();
    }

    // there is alway one path in the list to avoid multiple library loading
    void setQtLibraryPath(const string &i_path)
	{
        QString path( i_path.c_str() );
        if ( !QDir(path).exists() ) cerr << "WARNING: setQtLibraryPath(" << i_path << "): path does not exist" << endl;

        QCoreApplication::setLibraryPaths( QStringList(path) );
        // Sometimes the setLibraryPaths change the decimal-point character according the local OS config
        // to be sure that atof("2.5") is always 2.5 it's necessary to force setLocale
        setlocale(LC_NUMERIC, "C");
    }

    // if default path does not exist, replace it by deployment path
    // used by mm3d and SaisieQT
    void initQtLibraryPath()
    {
        // set to install plugins directory if it exists
        const string installPlugins = QT_INSTALL_PLUGINS;
        if ( !installPlugins.empty() && QDir( QString(installPlugins.c_str())).exists() )
        {
            setQtLibraryPath(installPlugins);
            return;
        }

        // set to deployment path if it exists
        const string deploymentPath = MMQtLibraryPath();

        if ( !deploymentPath.empty() && QDir( QString(deploymentPath.c_str())).exists() )
        {
            setQtLibraryPath(deploymentPath);
            return;
        }

        // keep the first existing path to avoid multiple library loading
        QStringList paths = QCoreApplication::libraryPaths();
        for ( int i=0; i<paths.size(); i++ )
        {
            if ( QDir( paths.at(i) ).exists() )
            {
                setQtLibraryPath( paths.at(i).toStdString() );
                return;
            }
        }

        cerr << "WARNING: initQtLibraryPath: no valid path found" << endl;
    }
#endif // ELISE_QT

    std::string MMBin() { return MMDir()+"bin"+ELISE_CAR_DIR; }

    int MMNbProc(){	return NbProcSys();	}

    cInterfChantierNameManipulateur::cInterfChantierNameManipulateur
        (
        int argc, char ** argv,
        const std::string & aDir
        ) :
    mDir   (aDir),
        mArgTr (aDir,true,true),
        mStrKeySuprAbs2Rel (0),
        mDB                (0),
        mMkDB              (0)
    {

        mArgTr.SetDico("MMDir",MMDir(),true);
        mArgTr.SetDico("MMNbProc",ToString(MMNbProc()),true);
        mArgTr.SetDico("MMCmdRmFile",SYS_RM,true);
        mArgTr.SetDico("MPD_MM",ToString(MPD_MM()),true);


        for (int aK=0 ;aK< argc ; aK++)
        {
            const char * aStr = argv[aK];
            if (aStr[0]==theCharModifDico)
            {
                aStr++;
                std::string aKey,aVal;
                SplitIn2ArroundEq(aStr,aKey,aVal);
                mArgTr.SetDico(aKey,aVal,true);
            }
        }

    }

    std::list<std::string> cInterfChantierNameManipulateur::GetListImByDelta
        (
        const cListImByDelta & aLIBD,
        const std::string & aN1
        )
    {
        std::vector<std::string> aInput;
        aInput.push_back(aN1);
        tNuplet aSplit=  Direct(aLIBD.KeySplitName(),aInput) ;

        ELISE_ASSERT(aSplit.size()==3,"Bad split size in Key/ListImByDelta");

        std::string aPref = aSplit[0];
        std::string aStrNum1 = aSplit[1];
        int aNbDig = (int)aStrNum1.size();
        int aVMax = round_ni(pow(10.0,aNbDig));
        int aNum1;
        FromString(aNum1,aStrNum1);
        std::string aPost = aSplit[2];


        std::list<std::string> aRes;
        for
            (
            std::list<int>::const_iterator itI=aLIBD.Delta().begin();
        itI!=aLIBD.Delta().end();
        itI++
            )
        {
            int aNum2 = aNum1 + *itI;
            if ((aNum2 >=0) && (aNum2<aVMax))
            {
                std::string aStrNum2 = ToStringNBD(aNum2,aNbDig);

                aRes.push_back(aPref+aStrNum2+aPost);
            }

        }

        return aRes;
    }


    void cInterfChantierNameManipulateur::StdTransfoNameFile(std::string & aName)
    {
        if (! mStrKeySuprAbs2Rel) return;

        if (ELISE_fp::exist_file(aName))  return;

        aName = mDir + Assoc1To1(*mStrKeySuprAbs2Rel,aName,true);
    }


    void cInterfChantierNameManipulateur::SetMkDB(const cMakeDataBase & aMkDB)
    {
        mMkDB = new cMakeDataBase(aMkDB);
    }


    void cInterfChantierNameManipulateur::SetKeySuprAbs2Rel(std::string * aStr)
    {
        mStrKeySuprAbs2Rel = aStr;
    }

    cArgCreatXLMTree &  cInterfChantierNameManipulateur::ArgTree()
    {
        return mArgTr;
    }

    bool  cInterfChantierNameManipulateur::IsFile(const std::string & aName)
    {
        bool aHK = AssocHasKey(aName);
        bool aIF = ELISE_fp::exist_file(mDir+aName);

        // std::cout << aHK << " " << aIF << "\n";

        if (aIF && (! aHK))
            return true;

        if ((!aIF) && aHK)
            return false;

        std::cout << "FOR NAME = [" << aName << "]\n";
        if (aIF &&  aHK)
        {
            ELISE_ASSERT(false," AMBIGUITE sur cInterfChantierNameManipulateur::IsFile ");
        }
        ELISE_ASSERT(false," Ni File ni Key,  cInterfChantierNameManipulateur::IsFile ");
        return true;
    }

    void cInterfChantierNameManipulateur::CD_Add(cChantierDescripteur *)
    {
    }

    std::string   cInterfChantierNameManipulateur::StdCorrect
        (
        const std::string & aKeyOrFile,
        const std::string & anEntry ,
        bool                Direct
        )
    {
        return IsFile(aKeyOrFile) ?
aKeyOrFile         :
        Assoc1To1(aKeyOrFile,anEntry,Direct);
    }

    std::string   cInterfChantierNameManipulateur::StdCorrect2
        (
        const std::string & aKeyOrFile,
        const std::string & anEntry1 ,
        const std::string & anEntry2 ,
        bool                Direct
        )
    {
        return IsFile(aKeyOrFile) ?
aKeyOrFile         :
        Assoc1To2(aKeyOrFile,anEntry1,anEntry2,Direct);
    }




std::list<std::string> cInterfChantierNameManipulateur::StdGetListOfFile
        (
           const std::string & aKeyOrPat,
           int aProf,
           bool ErrorWhenEmpty
        )
{
        if (SetHasKey(aKeyOrPat))
        {
            const  std::vector<std::string> * aV = Get(aKeyOrPat);
            return std::list<std::string>(aV->begin(),aV->end());
        }
        std::list<std::string> aRes =  RegexListFileMatch(mDir,aKeyOrPat,aProf,false);
        if (aRes.empty())
        {
            std::string aName = GetNameWithoutPerc(aKeyOrPat);
            if (aName!="") return StdGetListOfFile(aName,aProf);
           // GetNameWithoutPerc
/*
            std::string aDir,aName;
            SplitDirAndFile(aDir,aName,aKeyOrPat);
            if (aName.size() < aKey.size())
                 return StdGetListOfFile
*/
            // Si la directory a ete ajoutee deux fois ...
            if (ELISE_fp::exist_file(mDir + NameWithoutDir(aKeyOrPat)))
            {
                aRes.push_back(NameWithoutDir(aKeyOrPat));
                return aRes;
            }
            if (ELISE_fp::exist_file(aKeyOrPat))
            {
                aRes.push_back(aKeyOrPat);
                return aRes;
            }
            if (ErrorWhenEmpty)
            {
               std::cout << "For Key-Or-Pat=" << aKeyOrPat << " Dir= " << mDir << "\n";
               ELISE_ASSERT(false,"Empty list for StdGetListOfFile (one of the input file name is wrong)");
            }
        }
        return aRes;
}




    cInterfChantierNameManipulateur::~cInterfChantierNameManipulateur()
    {
    }


    cInterfChantierNameManipulateur * cInterfChantierNameManipulateur::TheGlob = 0;

    cInterfChantierNameManipulateur * cInterfChantierNameManipulateur::Glob()
    {
        return TheGlob;
    }
    cInterfChantierNameManipulateur *
        cInterfChantierNameManipulateur::StdAlloc
        (
        int argc, char ** argv,
        const std::string & aDir,
        const cTplValGesInit<std::string> aName,
        cChantierDescripteur * aCDisc,
        bool                   DoMkDB
        )
    {
        cStdChantierMultiManipulateur * aRes = new cStdChantierMultiManipulateur(argc,argv,aDir);

        std::string aPrelim = aDir+theNamePrelim;
        if (ELISE_fp::exist_file(aPrelim))
            aRes->Add(cStdChantierMonoManipulateur::StdGetFromFile(eUnknownSCMN,aRes,aDir,aPrelim,false));


        std::string aFullNG = StdGetFileXMLSpec(theNameGlob);
        aRes->Add(cStdChantierMonoManipulateur::StdGetFromFile(eDefSCMN,aRes,aDir,aFullNG,false));

        std::string aLoc = aDir+theNameLoc;

        if (ELISE_fp::exist_file(aLoc))
            aRes->Add(cStdChantierMonoManipulateur::StdGetFromFile(eMMLCD_SCMN,aRes,aDir,aLoc,true));

        if (aName.IsInit())
        {
            aRes->Add(cStdChantierMonoManipulateur::StdGetFromFile(eUnknownSCMN,aRes,aDir,aDir+aName.Val(),false));
        }

        if (aCDisc)
        {
            aRes->Add(new cStdChantierMonoManipulateur(eUnknownSCMN,aRes,aDir,*aCDisc));
        }

        if (DoMkDB)
            aRes->MkDataBase();

        TheGlob = aRes;
        return aRes;
    }

cInterfChantierNameManipulateur* cInterfChantierNameManipulateur::BasicAlloc(const std::string & aDir)
{
   static std::map<std::string,cInterfChantierNameManipulateur *> TheMap;
   cInterfChantierNameManipulateur * aRes = TheMap[aDir];
   if (! aRes)
   {
      cTplValGesInit<std::string> aNoName;
      aRes =  StdAlloc(0,0,aDir,aNoName);
      TheMap[aDir] = aRes;
   }

   return aRes;
}

    void cStdChantierMultiManipulateur::CD_Add(cChantierDescripteur * aCDisc)
    {
        if (aCDisc)
            Add(new cStdChantierMonoManipulateur(eUnknownSCMN,this,Dir(),*aCDisc));
    }


    std::string  cInterfChantierNameManipulateur::Assoc1To1
        (const tKey & aKey,const std::string & aName,bool isDirect)
    {
        std::vector<std::string> aInput;
        aInput.push_back(aName);

        tNuplet aRes= isDirect ? Direct(aKey,aInput)  : Inverse(aKey,aInput);

        ELISE_ASSERT(aRes.size()==1,"Multiple res in Assoc1To1");

        return aRes[0];
    }

    std::string  cInterfChantierNameManipulateur::Assoc1To1
        (const std::list<tKey> & aLKey,const std::string & aName,bool isDirect)
    {
        std::string aRes = aName;
        for
            (
            std::list<tKey>::const_iterator itK=aLKey.begin();
        itK!=aLKey.end();
        itK++
            )
        {
            aRes = Assoc1To1(*itK,aRes,isDirect);
        }

        return aRes;
    }


    std::string  cInterfChantierNameManipulateur::Assoc1ToN
        (const tKey & aKey,const tSet & aVNames,bool isDirect)
    {

        tNuplet aRes= isDirect ? Direct(aKey,aVNames)  : Inverse(aKey,aVNames);
        ELISE_ASSERT(aRes.size()==1,"Multiple res in Assoc1ToN");

        return aRes[0];
    }

    const  std::string &  cInterfChantierNameManipulateur::Dir() const
    {
        return mDir;
    }

      void cInterfChantierNameManipulateur::setDir( const std::string &i_directory ){ mDir=i_directory; }



    std::string  cInterfChantierNameManipulateur::Assoc1To2
        (
        const tKey & aKey,
        const std::string & aName1,
        const std::string & aName2,
        bool isDirect
        )
    {
        std::vector<std::string> aInput;
        aInput.push_back(aName1);
        aInput.push_back(aName2);

        tNuplet aRes= isDirect ? Direct(aKey,aInput)  : Inverse(aKey,aInput);

        ELISE_ASSERT(aRes.size()==1,"Multiple res in Assoc1To2");

        return aRes[0];
    }

std::string cInterfChantierNameManipulateur::NameImEpip(const std::string & anOri,const std::string & aIm1,const std::string & aIm2)
{
   return Assoc1To3("NKS-Assoc-NameImEpip@tif",anOri,aIm1,aIm2,true);
}

std::string cInterfChantierNameManipulateur::NameOrientEpipGen(const std::string & anOri,const std::string & aIm1,const std::string & aIm2)
{
  return Assoc1To2 ( "NKS-Assoc-CplIm2OriGenEpi@"+anOri+"@txt", aIm1,aIm2,true);

}

std::string cInterfChantierNameManipulateur::NameAppuiEpip(const std::string & anOri,const std::string & aIm1,const std::string & aIm2)
{
   return Assoc1To3("NKS-Assoc-NameAppuiEpip",anOri,aIm1,aIm2,true);
}

    std::string  cInterfChantierNameManipulateur::Assoc1To3
        (
        const tKey & aKey,
        const std::string & aName1,
        const std::string & aName2,
        const std::string & aName3,
        bool isDirect
        )
    {
        std::vector<std::string> aInput;
        aInput.push_back(aName1);
        aInput.push_back(aName2);
        aInput.push_back(aName3);

        tNuplet aRes= isDirect ? Direct(aKey,aInput)  : Inverse(aKey,aInput);

        ELISE_ASSERT(aRes.size()==1,"Multiple res in Assoc1To3");

        return aRes[0];
    }





    std::pair<std::string,std::string> cInterfChantierNameManipulateur::Assoc2To1
        (
        const tKey & aKey,
        const std::string & aName,
        bool isDirect
        )
    {
        std::vector<std::string> aInput;
        aInput.push_back(aName);

        tNuplet aRes= isDirect ? Direct(aKey,aInput)  : Inverse(aKey,aInput);

        ELISE_ASSERT(aRes.size()==2,"Wrong res number in Assoc2To1");

        return std::pair<std::string,std::string>(aRes[0],aRes[1]);

    }


    /*******************************************************/
    /*                                                     */
    /*                        cStdChantierRel              */
    /*                                                     */
    /*******************************************************/




    cStdChantierRel::cStdChantierRel
        (
        cInterfChantierNameManipulateur &aICNM,
        const cNameRelDescriptor & aNRD
        ) :
    mICNM   (aICNM),
        mNRD    (aNRD),
        mIsComp (false),
        mIsReflexif  (aNRD.Reflexif().Val())
    {
    }

    const cNameRelDescriptor &  cStdChantierRel::NRD() const
    {
        return mNRD;
    }


    //===============  cComputeFiltreRelOr    ==================


    cComputeFiltreRelOr::cComputeFiltreRelOr
        (
        const cTplValGesInit<cFiltreDeRelationOrient> & aTplF,
        cInterfChantierNameManipulateur &               anICNM
        ) :
    mPF     (aTplF.PtrVal()),
        mICNM    (anICNM),
        mFSsEch (0)
    {
        if (mPF && (mPF->FiltreByRelSsEch().IsInit()))
        {
            mFSsEch = new cComputeFiltreRelSsEch(mICNM,mPF->FiltreByRelSsEch().Val());
        }
    }

    cComputeFiltreRelOr::~cComputeFiltreRelOr()
    {
        delete mFSsEch;
    }

    bool cComputeFiltreRelOr::OKSsEch(const std::string & aNA,const std::string & aNB) const
    {
        if (! mFSsEch)
            return true;

        return mFSsEch->OkCple(aNA,aNB);
    }

    bool cComputeFiltreRelOr::OKMatrix(const std::string & aNA,const std::string & aNB) const
    {
        if (! mPF->FiltreAdjMatrix().IsInit() )
            return true;

        const std::string & aKM = mPF->FiltreAdjMatrix().Val();
        cResBoxMatr aRA = mICNM.GetBoxOfMatr(aKM,aNA).Val();
        cResBoxMatr aRB = mICNM.GetBoxOfMatr(aKM,aNB).Val();

        Pt2di aMaxEc = mPF->EcartFiltreMatr().Val();
        Pt2di anEc = aRA.mId-aRB.mId;

        return (ElAbs(anEc.x)<=aMaxEc.x) &&  (ElAbs(anEc.x)<=aMaxEc.x) ;
    }

    bool cComputeFiltreRelOr::OKEmprise(const std::string & aNA,const std::string & aNB) const
    {
        if (!mPF->FiltreEmprise().IsInit())
            return true;

        const cFiltreEmprise & aFE = mPF->FiltreEmprise().Val();
        const std::string & aKO = aFE.KeyOri();
        std::string aNameOriA = mICNM.Dir() + mICNM.Assoc1To1(aKO,aNA,true);
        std::string aNameOriB = mICNM.Dir() + mICNM.Assoc1To1(aKO,aNB,true);

        ElCamera *  aCamA = Cam_Gen_From_File(aNameOriA,aFE.Tag().Val(),aFE.MemoFile().Val(),true,&mICNM);
        ElCamera *  aCamB = Cam_Gen_From_File(aNameOriB,aFE.Tag().Val(),aFE.MemoFile().Val(),true,&mICNM);

        return (aCamA->RatioInterSol(*aCamB)> aFE.RatioMin());
    }

    bool cComputeFiltreRelOr::OKEquiv(const std::string & aNA,const std::string & aNB) const
    {
        if (! mPF->KeyEquiv().IsInit())
            return true;

        const std::string & aKey = mPF->KeyEquiv().Val();
        return mICNM.Assoc1To1(aKey,aNA,true) ==  mICNM.Assoc1To1(aKey,aNB,true);
    }

    bool cComputeFiltreRelOr::OK_CFOR(const std::string & aNA,const std::string & aNB) const
    {
        return    (mPF==0)
            || (
            OKEquiv(aNA,aNB)
            &&  OKEmprise(aNA,aNB)
            &&  OKMatrix(aNA,aNB)
            &&  OKSsEch(aNA,aNB)
            );
    }

    //===============  cStdChantierRel    ==================

    void cStdChantierRel::AddAllCpleKeySet
        (
        const std::string & aKEY1,
        const std::string & aKEY2,
        cComputeFiltreRelOr & aCFO,
        // const cTplValGesInit<cFiltreDeRelationOrient> & aFiltre,
        // cComputeFiltreRelSsEch * & aFSsEch,
        bool aSym
        )
    {
        int aBIG = round_ni(1e8);
        AddAllCpleKeySet(aKEY1,aKEY2,-aBIG,aBIG,aCFO,aSym,false);
    }

    // static



void cStdChantierRel::AddAllCpleKeySet
        (
        const std::string & aKeyA,
        const std::string & aKeyB,
        int aDeltaMin,
        int aDeltaMax,
        cComputeFiltreRelOr& aCFO,
        // const cTplValGesInit<cFiltreDeRelationOrient> & aTplF,
        // cComputeFiltreRelSsEch *  & aFSsEch,
        bool aSym,
        bool IsCirc,
        int aSampling
        )
{
       if (aSampling<=0) return;

        const std::vector<std::string> * aSetA= mICNM.Get(aKeyA);
        const std::vector<std::string> * aSetB= mICNM.Get(aKeyB);
        bool SameSet = (aSetA==aSetB);

        std::vector<std::string> aSampleA,aSampleB;
        if (aSampling!=1)
        {
            for (int aK=0 ; aK<int(aSetA->size()) ; aK+=aSampling)
                aSampleA.push_back((*aSetA)[aK]);
            aSetA= & aSampleA;
            if (SameSet)
            {
                aSetB= & aSampleA;
            }
            else
            {
                for (int aK=0 ; aK<int(aSetB->size()) ; aK+=aSampling)
                    aSampleB.push_back((*aSetB)[aK]);
                aSetB= & aSampleB;
            }
        }

        if (IsCirc)
        {
            if (! SameSet)
            {
                std::cout << "SETS : " << aKeyA << " & " << aKeyB << "\n";
                ELISE_ASSERT
                    (
                    false,
                    "Set dif in circ cStdChantierRel::AddAllCpleKeySet"
                    );
            }
        }

        /*
        const cFiltreDeRelationOrient * aPF =  aTplF.PtrVal();
        if (aPF && (aPF->FiltreByRelSsEch().IsInit()) && (aFSsEch==0))
        {
        aFSsEch = new cComputeFiltreRelSsEch(mICNM,aPF->FiltreByRelSsEch().Val());
        }
        */

        for (int aKA=0 ;  aKA<int(aSetA->size()) ; aKA++)
        {
            // std::cout << "zzzzzzzzzz  " << aKA << " " << (aSetA->size()) << "\n";

            std::vector<int> aVKB;
            if ( IsCirc)
            {
                int aKB0 = aKA+aDeltaMin;
                int aKB1 = aKA+aDeltaMax;

                /*
                ELISE_ASSERT
                (
                (aKB1-aKB0>=0) & (aKB1-aKB0<= int(aSetA->size())),
                "Circ incoherent in cStdChantierRel::AddAllCpleKeySet"
                );
                */

                for (int  aKB= aKB0 ;aKB<=aKB1 ; aKB++)
                    aVKB.push_back(mod(aKB, (int)aSetB->size()));
            }
            else
            {
                int aKB0 = ElMax(0,aKA+aDeltaMin);
                int aKB1 = ElMin(aKA+aDeltaMax,int(aSetB->size())-1);
                for (int  aKB= aKB0 ;aKB<=aKB1 ; aKB++)
                    aVKB.push_back(aKB);
            }



            const std::string & aNA = (*aSetA)[aKA];


            for (int aKKB= 0 ;aKKB<int(aVKB.size()) ; aKKB++)
            {
                int aKB = aVKB[aKKB];

                if  ((aKA != aKB) || (! SameSet))
                {
                    const std::string & aNB = (*aSetB)[aKB];
                    if (aCFO.OK_CFOR(aNA,aNB))
                    {
                        Add(cCpleString(aNA,aNB));
                        if ((! SameSet) &&(aSym))
                        {
                            Add(cCpleString(aNB,aNA));
                        }
                    }
                }
            }
        }
}


    void cStdChantierRel::Add(const cCpleString & aCpl)
    {
        if ((!mIsReflexif)  && (aCpl.N1()==aCpl.N2()))
            return;
        if (mExcl.find(aCpl) == mExcl.end())
            mIncl.push_back(aCpl);
    }

    void cStdChantierRel::Add
        (
        const std::string& aPre,
        const cCpleString & aCple,
        const std::string& aPost
        )
    {
        if ((aPre=="") && (aPost==""))
            Add(aCple);
        else
            Add(aCple.AddPrePost(aPre,aPost));

    }

    bool cStdChantierRel::AddAFile(std::string  aName,bool must_exist)
    {
        aName =  mICNM.Dir() + aName;
        if (ELISE_fp::exist_file(aName))
        {
            cSauvegardeNamedRel aSNR = StdGetObjFromFile<cSauvegardeNamedRel>
                (
                aName,
                StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                "SauvegardeNamedRel",
                "SauvegardeNamedRel"
                );
            for
                (
                std::vector<cCpleString>::const_iterator itC=aSNR.Cple().begin();
            itC!=aSNR.Cple().end();
            itC++
                )
            {
                Add(*itC);
            }
            return true;
        }

        if (must_exist)
        {
            std::cout << "For file name " << aName << "\n";
            ELISE_ASSERT(false,"Required file for relation do not exist");
        }

        return false;
    }



    void cStdChantierRel::Compute()
    {
        if (mIsComp)
            return;


        std::string  aNameSauv = mNRD.NameFileSauvegarde().ValWithDef("");
        if (aNameSauv!="")
        {
            if (AddAFile(aNameSauv,false))
            {
                return;
            }
        }

        for
            (
            std::list<std::string>::const_iterator itS=mNRD.NameFileIn().begin();
        itS!=mNRD.NameFileIn().end();
        itS++
            )
        {
            AddAFile(*itS,true);
        }
        /*
        if (aNameSauv!="")
        {
        aNameSauv =  mICNM.Dir() + aNameSauv;
        if (ELISE_fp::exist_file(aNameSauv))
        {
        cSauvegardeNamedRel aSNR = StdGetObjFromFile<cSauvegardeNamedRel>
        (
        aNameSauv,
        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
        "SauvegardeNamedRel",
        "SauvegardeNamedRel"
        );
        for
        (
        std::vector<cCpleString>::const_iterator itC=aSNR.Cple().begin();
        itC!=aSNR.Cple().end();
        itC++
        )
        Add(*itC);
        return;
        }
        }
        */

        mIsComp = true;
        mExcl.insert(mNRD.CplesExcl().begin(),mNRD.CplesExcl().end());

        for
            (
            std::list<cRelByGrapheExpl>::const_iterator  itGE= mNRD.RelByGrapheExpl().begin();
        itGE != mNRD.RelByGrapheExpl().end();
        itGE++
            )
        {
            std::string aPref = itGE->Prefix2Name().ValWithDef("");
            std::string aPost = itGE->Postfix2Name().ValWithDef("");

            for
                (
                std::list<cCpleString>::const_iterator itC = itGE->Cples().begin();
            itC != itGE->Cples().end();
            itC++
                )
            {
                Add(aPref,*itC,aPost);
            }

            for
                (
                std::list<std::vector<std::string> >::const_iterator itV = itGE->CpleSymWithFirt().begin();
            itV != itGE->CpleSymWithFirt().end();
            itV++
                )
            {
                const std::vector<std::string> & aVS = *itV;
                for (int aK=1 ; aK<int(aVS.size()) ; aK++)
                {
                    Add(aPref,cCpleString(aVS[0],aVS[aK]),aPost);
                    Add(aPref,cCpleString(aVS[aK],aVS[0]),aPost);
                }
            }

            for
                (
                std::list<std::vector<std::string> >::const_iterator itV = itGE->ProdCartesien().begin();
            itV != itGE->ProdCartesien().end();
            itV++
                )
            {
                const std::vector<std::string> & aVS = *itV;
                for (int aK1=0 ; aK1<int(aVS.size()) ; aK1++)
                {
                    for (int aK2=0 ; aK2<int(aVS.size()) ; aK2++)
                    {
                        if (aK1!=aK2)
                            Add(aPref,cCpleString(aVS[aK1],aVS[aK2]),aPost);
                    }
                }
            }




            for
                (
                std::list<cGrByDelta>::const_iterator itGD = itGE->GrByDelta().begin();
            itGD != itGE->GrByDelta().end();
            itGD++
                )
            {
                const std::vector<std::string> * aSet= mICNM.Get(itGD->KeySet());
                for
                    (
                    std::list<cOneSpecDelta>::const_iterator  itO=itGD->OneSpecDelta().begin();
                itO!=itGD->OneSpecDelta().end();
                itO++
                    )
                {

                    const std::vector<std::string> & aVS = itO->Soms();
                    const std::vector<int> & aVI = itO->Delta();
                    for (int aKS=0 ; aKS<int(aVS.size()) ; aKS++)
                    {
                        std::string aName1 = aPref+aVS[aKS]+aPost;
                        // std::cout << "--- Name1[" << aKS << "]=" << aName1 << "\n";
                        int anInd1 = IndFind(*aSet,aName1);
                        if (anInd1<0)
                        {
                            std::cout << "For name " << aName1 << "\n";
                            std::cout << "For set " << itGD->KeySet() << "\n";
                            ELISE_ASSERT(false,"Cannot find ind cStdChantierRel::Compute");
                        }
                        for (int aKI=0 ; aKI<int(aVI.size()) ; aKI++)
                        {
                            int anInd2 = anInd1+ aVI[aKI];
                            if ((anInd2>=0) && (anInd2<int(aSet->size())))
                            {
                                const std::string & aName2 = (*aSet)[anInd2];
                                Add(cCpleString(aName1,aName2));
                                // std::cout << "   " << aName1 << " " << aName2 << "\n";
                            }
                        }
                    }
                }
            }

        }

        for
            (
            std::list<cByGroupesDImages>::iterator itG = mNRD.ByGroupesDImages().begin();
        itG != mNRD.ByGroupesDImages().end();
        itG++
            )
        {
            cComputeFiltreRelOr aCFO(itG->Filtre(),mICNM);
            // cComputeFiltreRelSsEch * aFSsEch = 0;
            for
                (
                std::list<cCpleString>::iterator itC = itG->CplesKey().begin();
            itC != itG->CplesKey().end();
            itC++
                )
            {
                AddAllCpleKeySet(itC->N1(),itC->N2(),aCFO,itG->Sym().Val());
                if (itG->Reflexif().Val())
                {
                    AddAllCpleKeySet(itC->N1(),itC->N1(),aCFO,itG->Sym().Val());
                    AddAllCpleKeySet(itC->N2(),itC->N2(),aCFO,itG->Sym().Val());
                }
            }

            for
                (
                std::list<cByAdjDeGroupes>::iterator itA = itG->ByAdjDeGroupes().begin();
            itA != itG->ByAdjDeGroupes().end();
            itA++
                )
            {
                const std::vector<std::string> & aVN = itA->KeySets();
                int aDMin = itA->DeltaMin();
                int aDMax = itA->DeltaMax();

                for (int aKA=0 ; aKA<int(aVN.size()) ; aKA++)
                {
                    int aK0B = ElMax(0,aKA+aDMin);
                    int aK1B = ElMin(int(aVN.size()-1),aKA+aDMax);
                    for (int aKB = aK0B ; aKB<=aK1B ; aKB++)
                    {
                        AddAllCpleKeySet(aVN[aKA],aVN[aKB],aCFO,itG->Sym().Val());
                    }
                }
            }
        }

        for
            (
            std::list<cByAdjacence>::iterator itA = mNRD.ByAdjacence().begin();
        itA != mNRD.ByAdjacence().end();
        itA++
            )
        {
            string oldDirectory = mICNM.Dir();
            if ( isUsingSeparateDirectories() ) mICNM.setDir( MMOutputDirectory() );
            cComputeFiltreRelOr aCFO(itA->Filtre(),mICNM);

            // cComputeFiltreRelSsEch * aFSsEch = 0;
            int aNbSet = (int)itA->KeySets().size();
            int aDefDeltaMin = (aNbSet==2) ? -1000000 : 0;
            int aDeltaMin = itA->DeltaMin().ValWithDef(IntSubst(aDefDeltaMin)).Val();
            int aDeltaMax = itA->DeltaMax().Val().Val();

            const std::string & aKeyA= itA->KeySets()[0];
            const std::string & aKeyB= itA->KeySets().back();
            int aSampling = 1;
            if (itA->Sampling().IsInit())
               aSampling =itA->Sampling().Val().Val();

            if ( isUsingSeparateDirectories() ) mICNM.setDir( oldDirectory );
               AddAllCpleKeySet
               (
                    aKeyA, aKeyB,
                    aDeltaMin, aDeltaMax,
                    aCFO,
                    itA->Sym().Val(),
                    itA->Circ().Val().Val(),
                    aSampling
               );
        }

        ComputeFiltrageSpatial();


        std::sort(mIncl.begin(),mIncl.end());
        mIncl.erase(std::unique(mIncl.begin(),mIncl.end()),mIncl.end());


        if (aNameSauv!="")
        {
            cSauvegardeNamedRel aSNR;
            aSNR.Cple() = mIncl;
            MakeFileXML(aSNR,aNameSauv);
        }
    }

    std::vector<cCpleString> * cStdChantierRel::GetRel()
    {
        Compute();
        return &mIncl;
    }



    /*******************************************************/
    /*                                                     */
    /*                        cCompileCAPI                 */
    /*                                                     */
    /*******************************************************/

cCompileCAPI::cCompileCAPI()
{
}

extern std::string TheGlobSFS;

std::string PastisNameFileStd(const std::string & aFullNameOri)
{
   std::string aDir,aNameSsDir;
   SplitDirAndFile(aDir,aNameSsDir,aFullNameOri);
   cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

   std::string aNameSift = aICNM->Assoc1To1("NKS-Assoc-SFS",aNameSsDir,true);

   if  (TheGlobSFS!="") 
       aNameSift = "SFS";

   if (aNameSift=="NONE") 
   {
      return  NameFileStd( aFullNameOri, 1, false, true, false );
   }
      
   if (aNameSift=="SFS") 
      aNameSift =  std::string("Tmp-MM-Dir")+ELISE_CAR_DIR + aNameSsDir + "_sfs.tif";

   aNameSift =  aDir + aNameSift;
 
   if (!  ELISE_fp::exist_file(aNameSift))
   {
      std::string aCom =  MMBin() +  MM3DStr + " TestLib PrepSift  " + aFullNameOri  + " NameOut="+ aNameSift;
      System(aCom);
   }

   return aNameSift;
}

Tiff_Im StdTiffFromName(const std::string & aFullNameOri);

Tiff_Im PastisTif(const std::string &  aNameOri)
{
    // SFS
    // return  Tiff_Im::StdConvGen(aName,1,false);
    std::string aName = PastisNameFileStd(aNameOri);
    return  StdTiffFromName(aName);
    // return  Tiff_Im(aName.c_str());
}

    void DoSimplePastisSsResol(const std::string & aFullName,int aResol,bool forceTMP)
    {
        std::string aDir,aName;
        SplitDirAndFile(aDir,aName,aFullName);

        Tiff_Im aTF = PastisTif(aFullName); // Tiff_Im::StdConvGen(aFullName,1,false);

        Pt2di aSz = aTF.sz();

        if (forceTMP&&(aResol<=0)) aResol=aSz.x;
        if (aResol <=0) return;

        double aScale = double(aResol) / double(ElMax(aSz.x,aSz.y));
        double  Arrondi = 10;
        int iScale = round_ni((1/aScale) * Arrondi);

        aScale = Arrondi/iScale;

        string outDirectory = ( isUsingSeparateDirectories()?MMOutputDirectory():aDir );
        std::string aNameFinal = outDirectory
                             + std::string("Pastis")+ELISE_CAR_DIR
            + std::string("Resol") + ToString(iScale)
            + std::string("_Teta0")
            + std::string("_")
            + StdPrefixGen(aName)+".tif";

        if (! ELISE_fp::exist_file(aNameFinal))
        {
            ELISE_fp::MkDirRec( outDirectory+"Pastis"+ELISE_CAR_DIR );
            Pt2di aSzF = round_down(Pt2dr(aSz)*aScale);
            Tiff_Im aNewF
                (
                aNameFinal.c_str(),
                aSzF,
                aTF.type_el(),
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero
                );
            ELISE_COPY
                (
                aNewF.all_pts(),
                StdFoncChScale
                (
                aTF.in(0),
                Pt2dr(0,0),
                Pt2dr(1/aScale,1/aScale)
                ),
                aNewF.out()
                );
        }

    }




    cCompileCAPI::cCompileCAPI
        (cInterfChantierNameManipulateur & aIMCN,
        const cContenuAPrioriImage & aCAPI,
        const std::string &aDir,
        const std::string & aName,
        const std::string & aName2,
        bool forceTMP
        ) :
    mScale     (aCAPI.Scale().Val()),
        mTeta      (aCAPI.Teta().Val()),
        mBox       (Pt2dr(0,0),Pt2dr(0,0)),
        mRTr_I2R   (Pt2dr(0,0),Pt2dr(1,0)),
        mRTr_R2I   (Pt2dr(0,0),Pt2dr(1,0))
    {        
        std::string aNameInit = aDir+aName;
        if ( isUsingSeparateDirectories() && !ELISE_fp::exist_file(aNameInit) ) aNameInit = MMInputDirectory()+aName;
        mFullNameFinal = aNameInit;

        Tiff_Im aFileInit = PastisTif(aNameInit);
        mSzIm = aFileInit.sz();

        bool aWithBox = false;
        std::string aStrBox ="";

        if (aCAPI.BoiteEnglob().IsInit())
        {
            mBox = aCAPI.BoiteEnglob().Val();
            aWithBox = true;
        }
        else
        {
            mBox = Box2di(Pt2di(0,0),mSzIm);
        }
        mBox = Inf(mBox,Box2di(Pt2di(0,0),mSzIm));


        if (aCAPI.MatrixSplitBox().IsInit())
        {
            aWithBox = true;
            const cMatrixSplitBox & aMSB = aCAPI.MatrixSplitBox().Val();
            cResBoxMatr aRBM= aIMCN.GetBoxOfMatr(aMSB.KeyMatr(),aName2).Val();

            Box2dr aBS = aRBM.mBox;
            double aRab = aMSB.Rab().Val();
            Pt2dr aPRab(aRab,aRab);

            Pt2di aP0 = round_ni(mBox.FromCoordBar(aBS._p0)-aPRab);
            Pt2di aP1 = round_ni(mBox.FromCoordBar(aBS._p1)+aPRab);

            mBox = Inf(mBox,Box2di(aP0,aP1));

        }

        if (aWithBox)
        {
            aStrBox =  std::string("Box")
                + std::string("_") + ToString(mBox._p0.x)
                + std::string("_") + ToString(mBox._p0.y)
                + std::string("_") + ToString(mBox._p1.x)
                + std::string("_") + ToString(mBox._p1.y);
        }


        Pt2di aSzInit = mBox._p1 - mBox._p0;

        if ((mScale !=1.0) || (mTeta!=0.0) ||  aWithBox || forceTMP)
        {
            double  Arrondi = 10;
            int iScale = round_ni(mScale * Arrondi);
            mScale = iScale /Arrondi;

            int  iTeta = round_ni(mTeta);
            mTeta = iTeta;
            string pastisDirectory = ( isUsingSeparateDirectories()?MMOutputDirectory():aDir )+"Pastis"+ELISE_CAR_DIR;
            ELISE_fp::MkDir(pastisDirectory);
            mFullNameFinal = pastisDirectory
                + aStrBox
                + std::string("Resol") + ToString(iScale)
                + std::string("_Teta") + ToString(iTeta)
                + std::string("_")
                + StdPrefixGen(aName)+".tif";

            mRotI2R = Pt2dr::FromPolar(1.0,mTeta*PI/180.0);
            if (euclid(mRotI2R,Pt2dr(round_ni(mRotI2R))) <1e-2)
                mRotI2R = Pt2dr(round_ni(mRotI2R));

            Box2dr aBox0(Pt2dr(0,0),Pt2dr(aSzInit));
            Pt2dr Coins[4];
            aBox0.Corners(Coins);

            Pt2dr aP0 = Coins[0] * mRotI2R;
            Pt2dr aP1 = Coins[0] * mRotI2R;


            for (int aK=0 ; aK< 4 ; aK++)
            {
                aP0.SetInf(Coins[aK] * mRotI2R);
                aP1.SetSup(Coins[aK] * mRotI2R);
            }

            mRTr_I2R = ElSimilitude(-aP0,mRotI2R);
            mRTr_R2I = mRTr_I2R.inv();
            Pt2di aSzTurn = round_ni(aP1-aP0);

            if (! ELISE_fp::exist_file(mFullNameFinal))
            {

// std::cout << "Aaaaaaaa " << aSzInit << aSzTurn << " Scale " << mScale << "Teta " << mTeta   << "\n";

                Fonc_Num fRes =  trans(aFileInit.in(0),mBox._p0);
                Pt2di aSzRes = aSzInit;

                if (mTeta !=0)
                {
                    Im2D_REAL4 aImInit(aSzInit.x,aSzInit.y);
                    ELISE_COPY
                    (
                        aImInit.all_pts(),
                        trans(aFileInit.in(0),mBox._p0),
                        aImInit.out()
                    );

                    Im2D_REAL4 aImTurn(aSzTurn.x,aSzTurn.y);
                    TIm2D<REAL4,REAL8> aTT(aImTurn);

                    Pt2di aP;
                    TIm2D<REAL4,REAL8> aTI(aImInit);
                    for (aP.x=0 ; aP.x<aSzTurn.x; aP.x++)
                    {
                        for (aP.y=0 ; aP.y<aSzTurn.y; aP.y++)
                        {
                            aTT.oset(aP,aTI.getr(mRTr_R2I(Pt2dr(aP)),0.0));
                        }
                    }
                    fRes = aImTurn.in(0);
                    aSzRes = aImTurn.sz();
                }

                // std::cout << "SCALE = " << mScale << "\n";
                if (mScale !=1.0)
                {
                    Pt2di aSzS =  round_ni(Pt2dr(aSzRes)/mScale);
// std::cout << "Bbbbbbb sssssss " << aSzS << "\n";
                    Im2D_REAL4  aImScale(aSzS.x,aSzS.y);

                    ELISE_COPY
                    (
                        aImScale.all_pts(),
                        StdFoncChScale
                        (
                        fRes,
                        Pt2dr(0,0),
                        Pt2dr(mScale,mScale)
                        ),
                        aImScale.out()
                    );

                    fRes = aImScale.in(0);
                    aSzRes = aImScale.sz();
                }
                Tiff_Im aNewF
                    (
                    mFullNameFinal.c_str(),
                    aSzRes,
                    aFileInit.type_el(),
                    Tiff_Im::No_Compr,
                    Tiff_Im::BlackIsZero
                    );
                ELISE_COPY
                    (
                    aNewF.all_pts(),
                    Tronque(aNewF.type_el(),fRes),
                    aNewF.out()
                    );
            }
        }

        mSim_R2I =    ElSimilitude(Pt2dr(mBox._p0),Pt2dr(1,0))
            * mRTr_R2I
            * ElSimilitude(Pt2dr(0,0),Pt2dr(mScale,0));
    }

    Pt2dr cCompileCAPI::Rectif2Init(const Pt2dr & aP)
    {
        return mSim_R2I(aP);
    }

    const std::string & cCompileCAPI::NameRectif() const
    {


        return mFullNameFinal;
    }


    cContenuAPrioriImage  cInterfChantierNameManipulateur::APrioriWithDef(const std::string & aName,const std::string & aKey)
    {
        cContenuAPrioriImage aRes;

        cContenuAPrioriImage * aPtr = GetAPriori(aName,aKey,this);
        if (aPtr)
        {
            aRes =  *aPtr;
            if (aRes.KeyAutoAdaptScale().IsInit())
            {
                std::string aStrF = Assoc1To1(aRes.KeyAutoAdaptScale().Val(),aName,1);
                double aFoc;
                bool aOK = FromString(aFoc,aStrF);
                ELISE_ASSERT(aOK,"Cannot Convert string to float in cInterfChantierNameManipulateur::APrioriWithDef");
                aRes.Scale().SetVal(aRes.Scale().Val()*aFoc);
            }
        }
        else
        {

            if (aKey!="DefKey")
            {
                std::cout << "For Key = " << aKey  << " Im = " << aName << "\n";
                ELISE_ASSERT(false,"cInterfChantierNameManipulateur::APrioriWithDef");
            }
            // cContenuAPrioriImage aRes;
            aRes.Scale().SetVal(1.0);
            aRes.Teta().SetVal(0.0);
            aRes.PdsMaxAdaptScale().SetVal(0.5);
            aRes.BoiteEnglob().SetNoInit();
        }

        return aRes;
    }

    void Adjust(cContenuAPrioriImage & aC1,cContenuAPrioriImage & aC2)
    {

        /*
        // std::cout << "In === "<< aC1.Scale().Val() << " " << aC2.Scale().Val () << "\n";
        if (aC1.Scale().Val() < aC2.Scale().Val())
        {
        // aC2.Scale().Val() /= aC1.Scale().Val();
        // aC1.Scale().Val() = 1.0;
        aC1.Scale().Val() /= aC2.Scale().Val();
        aC2.Scale().Val() = 1.0;
        }
        //std::cout << "Out  === "<< aC1.Scale().Val() << " " << aC2.Scale().Val() << "\n\n";
        */

        double aPds = (aC1.PdsMaxAdaptScale().Val() + aC2.PdsMaxAdaptScale().Val()) / 2.0;

        double aSMax = ElMax(aC1.Scale().Val(),aC2.Scale().Val());
        double aSMin = ElMin(aC1.Scale().Val(),aC2.Scale().Val());

        double aPLog = aPds *log(aSMax) + (1-aPds)*log(aSMin);
        double aScN = exp(aPLog);

        aC1.Scale().Val() /=  aScN;
        aC2.Scale().Val() /=  aScN;
    }


    double cInterfChantierNameManipulateur::BiggestDim
        (
        const cContenuAPrioriImage & aPriori,
        const std::string & aNameIm
        )
    {
        double aRes=0;
        if (aPriori.BoiteEnglob().IsInit())
        {
            aRes = dist8(aPriori.BoiteEnglob().Val().sz());
        }
        else
        {
            aRes = dist8(PastisTif(mDir+aNameIm).sz()); 
        }

        return aRes;
        // return aRes / aPriori.Scale().Val();
    }

    std::pair<cCompileCAPI,cCompileCAPI> cInterfChantierNameManipulateur::APrioriAppar
        (
        const std::string & aN1,
        const std::string & aN2,
        const std::string & aKEY1,
        const std::string & aKEY2,
        double aSzMax,
        bool forceTMP
        )
    {
        cContenuAPrioriImage aC1 = APrioriWithDef(aN1,aKEY1);
        cContenuAPrioriImage aC2 = APrioriWithDef(aN2,aKEY2);

        // Ajustement des echelles : on ne touche pas a la moins resolue
        Adjust(aC1,aC2);
        Adjust(aC2,aC1);

        bool aTest =  false; // (aC1.Scale().Val() != aC2.Scale().Val());
        if (aTest )
        {
            std::cout << aN1 << " " << aN2 << "\n";
            std::cout << "S1 " << aC1.Scale().Val()  << " S2 " << aC2.Scale().Val() << "\n";
        }

        if (aSzMax >0)
        {
            double aD1 = BiggestDim(aC1,aN1);
            double aD2 = BiggestDim(aC2,aN2);
            double aMult = aSzMax/ElMin(aD1,aD2);
            //std::cout << "DDMM " << aD1 << " " << aD2  << " " << aMult << "\n";
            aC1.Scale().Val() /= aMult;
            aC2.Scale().Val() /= aMult;
            if (aSzMax<=1.0)
            {
                aC1.Scale().Val() = 1/aSzMax;
                aC2.Scale().Val() = 1/aSzMax;
            }
            if (aTest )
            {
                std::cout<< "D1 "  << aD1 << " D2 " << aD2 << "\n";
                std::cout<< " Mult" << aMult << "\n";
                std::cout << "SZMAX ::  S1 " << aC1.Scale().Val()  << " S2 " << aC2.Scale().Val() << "\n";
                getchar();
            }
        }

        //std::cout << "oooo " << aC1.Scale().Val()  << " " <<  aC2.Scale().Val() << "\n";

        std::pair<cCompileCAPI,cCompileCAPI> aRes;

        string outDirectory = ( isUsingSeparateDirectories()?MMOutputDirectory():mDir );
        aRes.first  = cCompileCAPI(*this,aC1,outDirectory,aN1,aN2,forceTMP);
        aRes.second = cCompileCAPI(*this,aC2,outDirectory,aN2,aN1,forceTMP);

        return  aRes;
    }

    /******************************************************************/
    /*                                                                */
    /*                  MakeStdOrient ....                            */
    /*                                                                */
    /******************************************************************/

    cResulMSO::cResulMSO() :
    mIsKeyOri (false),
        mCam      (0),
        mNuage    (0),
        mCapt3d   (0)
    {
    }

    ElCamera * & cResulMSO::Cam()           {return mCam;}
    cElNuage3DMaille * & cResulMSO::Nuage() {return mNuage;}
    bool   & cResulMSO::IsKeyOri()          {return mIsKeyOri;}
    cBasicGeomCap3D * & cResulMSO::Capt3d()      {return mCapt3d;}


bool  cInterfChantierNameManipulateur::TestStdOrient
    (
        const std::string & aManquant,
        const std::string & aPrefix,
        std::string & anOri,
                bool  AddNKS
    )
{
        // std::cout << "ttTEST " << anOri << "\n";
        if (anOri.find(aPrefix) != 0)
            return false;

        string inputDirectory = ( isUsingSeparateDirectories()?MMOutputDirectory():mDir );
        std::string aDir = inputDirectory + aManquant + anOri + ELISE_CAR_DIR;
        std::list<std::string> aL = RegexListFileMatch(aDir,".*(GB-Orientation-|Orientation-|AutoCal).*\\.(xml|XML)",2,false);
        // std::list<std::string> aL = RegexListFileMatch(mDir,aManquant + anOri+ "(Orientation-|AutoCal).*\\.xml",2);

        // std::cout << "3-ttTEST " <<  aDir  << " " << aL.size() << "\n";
        if (aL.empty())
            return false;


        anOri = anOri.substr(aPrefix.size(),std::string::npos);
                if (AddNKS)
        anOri =  "NKS-Assoc-Im2Orient@-" + anOri;
        return true;
}


bool cInterfChantierNameManipulateur::CorrecNameOrient(std::string & aNameOri,bool SVP)
{
    if (aNameOri=="NONE") return true;

    int aL = (int)strlen(aNameOri.c_str());
    if (aL && (aNameOri[aL-1]==ELISE_CAR_DIR))
    {
        aNameOri = aNameOri.substr(0,aL-1);
    }

    if  (TestStdOrient("Ori-","",aNameOri,false))
        return true;

    if  (TestStdOrient("","Ori-",aNameOri,false))
        return true;

    if  (TestStdOrient("Ori","-",aNameOri,true))
        return true;

    if (!SVP)
    {
        std::cout << "############## For Value " << aNameOri << " ############ \n";
        ELISE_ASSERT(false,"Ori name is not a valid existing directory");
    }

     return false;
}



cResulMSO cInterfChantierNameManipulateur::MakeStdOrient
          (
               std::string & anOri,
               bool AccepNone,
               std::string * aNameIm,
               bool          SVP
          )
{
       std::string anOriInit = anOri;

        cResulMSO  aResult;
        if (AccepNone && (anOri=="NONE"))
            return aResult;

        if (AssocHasKey(anOri))
        {
            if (aNameIm)
            {
                anOri = Assoc1To1(anOri,*aNameIm,true);
                if ((anOri=="NONE") && AccepNone)
                {
                    return aResult;
                }
            }
            else
            {
                return aResult;
            }
        }


        if (IsPostfixed(anOri) && (StdPostfix(anOri)=="xml") && ELISE_fp::exist_file(mDir+anOri))
        {
            cElXMLTree aTreeGlob(mDir+anOri,0);
            cElXMLTree * aTreeNuage = aTreeGlob.Get("XML_ParamNuage3DMaille");
            if (aTreeNuage)
            {
                aResult.Nuage() = cElNuage3DMaille::FromFileIm(mDir+anOri);
                aResult.Capt3d() = aResult.Nuage();
                return aResult;
            }

            cElXMLTree * aTreeCam = aTreeGlob.Get("OrientationConique");
            if (aTreeCam)
            {
                aResult.Cam() = Cam_Gen_From_File(mDir+anOri,"OrientationConique",this);
                aResult.Capt3d() = aResult.Cam();
                return aResult;
            }
        }


        const char * aC = anOri.c_str();
        int aL = (int)strlen(aC);
        if ((aL!=0) &&  (aC[aL-1] == ELISE_CAR_DIR))
        {
            anOri = anOri.substr(0,aL-1);
        }


        if (
            TestStdOrient("Ori-","",anOri,true)
            || TestStdOrient("","Ori-",anOri,true)
            || TestStdOrient("Ori","-",anOri,true)
            )
        {
            aResult.IsKeyOri() = true;
            return aResult;
        }


       if (aNameIm)
       {
            // On recoit NKS-Assoc-Im2Orient@-BundleCorrec-Deg2
            static cElRegex aSuprNKS("NKS-Assoc-Im2Orient@-(.*)",10);
            if (aSuprNKS.Match(anOriInit))
            {
                 std::string anOri = aSuprNKS.KIemeExprPar(1);
                 cBasicGeomCap3D * aBGC = StdCamGenerikOfNames(anOri,*aNameIm,SVP);
                 
                 aResult.Capt3d() = aBGC;
                 return aResult;
            }
       }

        std::cout << "For Key = " << anOri << "\n";
        ELISE_ASSERT
            (
            false,
            "names does not appear to be a valid orientation key"
            );

        return aResult;
}

/*
bool FromString(std::vector<std::string> &aRes ,const std::string & aStr)
{
    stringstream aStream(aStr);
    ElStdRead(aStream,aRes,ElGramArgMain::StdGram);
    return true;
}
*/


std::vector<std::string> cInterfChantierNameManipulateur::StdGetVecStr(const std::string & aStr)
{
      std::vector<std::string>  aRes;

      if ((aStr[0]=='[') && (aStr[aStr.size()-1]==']'))
      {
           stringstream aStream(aStr);
           ElStdRead(aStream,aRes,ElGramArgMain::StdGram);
      }
      else if (IsPostfixed(aStr) && ELISE_fp::exist_file(mDir+aStr))
      {
           if (StdPostfix(aStr)=="txt")
              return VecStrFromFile(mDir+aStr);
           else if (StdPostfix(aStr)=="xml")
           {
              cElXMLTree aTree (aStr);
              cElXMLTree  * aSub = aTree.GetUnique("DicoAppuisFlottant");
              if (aSub)
              {
                 cDicoAppuisFlottant  aDAF = StdGetFromPCP(aStr,DicoAppuisFlottant);
                 for (std::list<cOneAppuisDAF>::const_iterator   itOAF=aDAF.OneAppuisDAF().begin(); itOAF!=aDAF.OneAppuisDAF().end(); itOAF++)
                 {
                    aRes.push_back(itOAF->NamePt());
                 }
                 return aRes;
              }

           }
      }
      else
      {
           aRes.push_back(aStr);
      }


      return aRes;
}


void StdCorrecNameHomol(std::string & aNameH,const std::string & aDir)
{

    int aL = (int)strlen(aNameH.c_str());
    if (aL && (aNameH[aL-1]==ELISE_CAR_DIR))
    {
        aNameH = aNameH.substr(0,aL-1);
    }

    if ((strlen(aNameH.c_str())>=5) && (aNameH.substr(0,5)==std::string("Homol")))
       aNameH = aNameH.substr(5,std::string::npos);

    std::string aTest =  ( isUsingSeparateDirectories()?MMOutputDirectory():aDir ) + "Homol"+aNameH+ ELISE_CAR_DIR;

    if (!ELISE_fp::IsDirectory(aTest))
    {
         std::cout << "For Name homol= " << aNameH << "\n";
         ELISE_ASSERT(false,"Name is not a correct homologue prefix");
    }
}


bool StdCorrecNameOrient(std::string & aNameOri,const std::string & aDir,bool SVP)
{
    if (aNameOri =="") return true;
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    return anICNM->CorrecNameOrient(aNameOri,SVP);
}

#define NBExtTifMas 1
bool  TestStdMasq
    (
        const std::string & aManquant,
        const std::string & aDir,
        const std::string & aPat,
        std::string & aMasq
    )
{
    std::string aVExt[NBExtTifMas] = {"tif"}; // ,"tiff","TIF","TIFF"};
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::list<std::string> aL = anICNM->StdGetListOfFile(aPat,1);

    for (std::list<std::string>::const_iterator itS=aL.begin(); itS!=aL.end() ; itS++)
    {
        for (int aKExt=0 ; aKExt<NBExtTifMas; aKExt++)
        {
            std::string aName =  aDir + StdPrefix(*itS) + aManquant + aMasq + "." + aVExt[aKExt];
            if ( ELISE_fp::exist_file(aName))
            {
               aMasq = aManquant+aMasq;
               return true;
            }
        }
    }


    return false;
}


void   CorrecNameMasq
    (
        const std::string & aDir,
        const std::string & aPat,
        std::string & aMasq
    )
{
   if (TestStdMasq("",aDir,aPat,aMasq)) return;
   if (TestStdMasq("_",aDir,aPat,aMasq)) return;
   if (TestStdMasq("_Masq",aDir,aPat,aMasq)) return;

   std::cout << "############## For Value " << aMasq << " ############ \n";
   ELISE_ASSERT(false,"Key is not a valid masq extension");
}

static void __check_directory( const cTplValGesInit<std::string> &i_XMLDirectory, string &o_directory )
{
    if ( !i_XMLDirectory.IsInit() ){ o_directory.clear(); return; }

    o_directory = i_XMLDirectory.Val();
    if ( o_directory.length()==0 ) return;
    char &lastChar = o_directory[o_directory.length()-1];
    if ( lastChar=='\\' ) lastChar='/';
    else if ( lastChar!='/' ) o_directory.push_back('/');
    ELISE_fp::MkDir(o_directory);
}

std::string MMOutputDirectory()
{
    static string res;
    static bool isInit = false;
    if ( isInit ) return res;

    isInit = true;
    __check_directory( MMUserEnv().OutputDirectory(), res );
    return res;
}

std::string MMLogDirectory()
{
    static string res;
    static bool isInit = false;
    if ( isInit ) return res;

    isInit = true;
    __check_directory( MMUserEnv().LogDirectory(), res );
    return res;
}

std::string MMTemporaryDirectory()
{
    static string res;
    static bool isInit = false;
    if ( isInit ) return res;
    isInit = true;
    res = MMOutputDirectory()+temporarySubdirectory;
    ELISE_fp::MkDirSvp(res);
    return res;
}

static string _inputDirectory;
static bool _isInputDirectorySet = false;

void setInputDirectory( const std::string &i_directory )
{
    _inputDirectory = i_directory;
    _isInputDirectorySet = true;
}

bool isInputDirectorySet(){ return _isInputDirectorySet; }

std::string MMInputDirectory()
{
   ELISE_ASSERT( _isInputDirectorySet, ( current_program_subcommand()+": MMInputDirectory is used but not set" ).c_str() );
   return _inputDirectory;
}

/*
TestStdMasq("",aDir,aPat,aMasq);
TestStdMasq("Masq",aDir,aPat,aMasq);
TestStdMasq("_Masq",aDir,aPat,aMasq);
*/

/* Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,  a l'utilisation,  a la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007/*/
