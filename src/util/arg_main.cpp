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
#include <iterator>
//#include <process.h>


#if ElMemberTpl

#define SzBuf 2000
std::map<void *,std::string>  MapValuesEAM;
std::set<void *>  AllAddrEAM;

class cFileDebug
{
    public :
        ~cFileDebug()
        {
            Close(0);
        }
        void Close(int aVal)
        {
           if (mFile)
           {
              fprintf(mFile," RES=%d\n",aVal);
              fprintf(mFile,"End}\n");
              fclose(mFile);
           }
           mFile = 0;
        }
        void Open(const std::string & aName)
        {
            // ELISE_ASSERT(mFile==0,"Multiple Open in File Debug");
            if (mFile==0)
            {
                mFile = fopen(aName.c_str(),"a+");
                fprintf(mFile,"{Begin\n");
            }
        }
        FILE * File() {return mFile;}

        static cFileDebug  TheOne;

    private :
       cFileDebug() :
           mFile (0)
       {
       }

       FILE * mFile;
};
cFileDebug  cFileDebug::TheOne;
FILE * TheFileDebug()
{
   return cFileDebug::TheOne.File();
}
void OpenFileDebug(const std::string & aName)
{
   cFileDebug::TheOne.Open(aName);
}










std::string TheStringMemoArgOptGlob = "";

static const int TheNbKeyACM=9;
int TheNbProcCom = -1;
std::string TheSpecMess="";
const char *  TheKeyACM[TheNbKeyACM] ={"ExitOnBrkp","ExitOnNan","MajickFile","EnBoucle","NbMaxProc","SpecMessage","SFS","ExitOnWarn","GoonOnWarn"};

std::string TheGlobSFS="";


void AnalyseContextCom(int argc,char ** argv)
{
   static bool First = true;
   if (!First) return;
   First = false;


   for (int aK=0 ; aK<argc ; aK++)
   {
       if (argv[aK][0] == cInterfChantierNameManipulateur::theCharSymbOptGlob)
       {
           bool ForAction = false;
           std::string ArgK = argv[aK]+1;

           std::string aBeforEq,aAfterEq;
           SplitIn2ArroundCar(ArgK,'=', aBeforEq,aAfterEq,true);
           if (ArgK ==TheKeyACM[0])
           {
               TheExitOnBrkp = true;
               ForAction = true;
           }
           else if (ArgK==TheKeyACM[1])
           {
               TheExitOnNan = true;
               ForAction = true;
           }
           else if (ArgK==TheKeyACM[2])
           {
               TheMajickFile=true;
               ForAction = true;
           }
           else if (ArgK==TheKeyACM[3])
           {
               TheNbIterProcess=100000000;
               ForAction = true;
           }
           else if (aBeforEq==TheKeyACM[4])
           {
               bool Ok=FromString(TheNbProcCom,aAfterEq);
               ELISE_ASSERT(Ok,"Cannot read value in @NbProc");
               ForAction = true;
           }
           else if (aBeforEq==TheKeyACM[5])
           {
               TheSpecMess= aAfterEq;
               ForAction = true;
           }
           else if (aBeforEq==TheKeyACM[6])
           {
               TheGlobSFS=   argv[aK];
               ForAction = true;
           }
           else if (aBeforEq==TheKeyACM[7])
           {
               TheExitOnWarn = true;
               ForAction = true;
           }
           else if (aBeforEq==TheKeyACM[8])
           {
               TheGoonOnWarn = true;
               ForAction = true;
           }
/*
*/



           if (ForAction)
           {
               TheStringMemoArgOptGlob += " " + std::string(argv[aK]);
           }
           else
           {
                std::cout << "   Warning, unknown special arg  " << argv[aK] << "\n";
                for (int aK=0 ; aK< TheNbKeyACM ; aK++)
                    std::cout << "   Opt[" << aK << "]= "<<  cInterfChantierNameManipulateur::theCharSymbOptGlob << TheKeyACM[aK] << "\n";
           }
       }
   }
}

static char buf[SzBuf];

std::list<std::string>  TheEmptyListEnum;

bool MMVisualMode = false;

std::string MakeStrFromArgcARgvWithSubst(int  argc,char** argv,int aKSubst,std::string aSubst, bool aProtect,bool StrongProtect)
{
	if (aProtect) aSubst = PATTERN_QUOTE(aSubst);
 
	std::string aRes;
	for (int aK=0 ; aK<argc ; aK++)
	{
		if (aK == aKSubst)
			aRes = aRes + aSubst + " ";
		else
		{
			const string str(argv[aK]);
			aRes = aRes + (aProtect ? (StrongProtect ? QUOTE(str) : PATTERN_QUOTE(str)) : str) + " ";
		}
	}

	return aRes;
}
std::string MakeStrFromArgcARgvWithSubst(int  argc,char** argv,int aKSubst,std::string aSubst, bool aProtect)
{
   return MakeStrFromArgcARgvWithSubst(argc,argv,aKSubst,aSubst,aProtect,true);
}

std::string MakeStrFromArgcARgv( int argc,char **argv, bool aProtect )
{
     return MakeStrFromArgcARgvWithSubst(argc, argv, -1, "", aProtect);
}

std::string SubstArgcArvGlob(int aKSubst,std::string aSubst, bool aProtect)
{
     return MakeStrFromArgcARgvWithSubst(MemoArgc, MemoArgv, aKSubst, aSubst, aProtect);
}

int MemoArgc=-1;
char ** MemoArgv=0;
std::string GlobArcArgv;
static std::vector<std::string>  GlobMessErrContext;
void AddMessErrContext(const std::string & aMes)
{
   GlobMessErrContext.push_back(aMes);
}

int mm_getpid()
{
#if ELISE_windows
    return _getpid();
#else
    return getpid();
#endif
}

const std::string & mm_getstrpid()
{
    static std::string aRes = ToString(mm_getpid());
    return aRes;
}

void MemoArg(int argc,char** argv)
{
    AnalyseContextCom(argc,argv);
    static bool First  = true;
    if (! First) return;

    First = false;
    MMD_InitArgcArgv(argc,argv);
    MemoArgc = argc;
    MemoArgv = argv;
    // GlobArcArgv = MakeStrFromArgcARgv(argc,argv,true);
    // MODIF MPD car sinon les pattern reconnu par le shell ne sont pas quote 
     GlobArcArgv = MakeStrFromArgcARgvWithSubst(argc, argv, -1, "", true,true);
}

void ShowArgs()
{
    static bool Done=false;
    if (MemoArgv && (!Done))
    {
        Done=true;
        std::cout << "ARG MAIN RECEIVED \n";
        for (int aK=0 ; aK< MemoArgc ; aK++)
            std::cout << "  " << MemoArgv[aK] << "\n";
        std::cout << "\n";
    }
}


std::istream & VStrElStdRead
    (
    std::istream &is,
    ElSTDNS vector<std::string> & vec,
    const ElGramArgMain & Gram
    )
{

    vec.clear();

    if (Gram.mCharBeginTab != -1)
    {
        if (is.get() != Gram.mCharBeginTab)
        {
            ELISE_ASSERT(false,"istream >> vector<Type>");
        }
    }

    int c;
    vec.push_back("");
    while ((c = is.get()) !=   Gram.mCharEndTab)
    {
        ELISE_ASSERT (c!=-1,"Unexpected End Of String in ElStdRead(vector<Type> &)");
        if (c==',')
            vec.push_back("");
        else
            vec.back() += (char) c;
    }
    ELISE_ASSERT (is.get()==-1,"Expected End Of String param vector (did you forget a space between two parameters?)");

    if ((vec.size()==1) && (vec[0]==""))
        vec.pop_back();
    return is;
}



ElGramArgMain::ElGramArgMain(char CharEq,int CharBeginTab,char CharEndTab,bool aAnyEqual) :
mCharEq         (CharEq),
    mCharBeginTab   (CharBeginTab),
    mCharEndTab     (CharEndTab),
    mAnyEqual       (aAnyEqual)
{
}

bool ElGramArgMain::AnyEqual() const
{
    return mAnyEqual;
}

const ElGramArgMain ElGramArgMain::SPHGram(' ',-1,'\n',false);
const ElGramArgMain ElGramArgMain::THOMGram('=',-1,'\n',false);
const ElGramArgMain ElGramArgMain::HDRGram(' ',-1,'\n',true);

const ElGramArgMain ElGramArgMain::StdGram('=','[',']',false);


istream & operator >> (istream & is,ElSTDNS string & st)
{

    is >> buf;
    st = ElSTDNS string(buf);
    return is;
}


template <class Type>  std::istream & operator >>
    (std::istream &is,ElSTDNS vector<Type> & vec)
{
    return ElStdRead(is,vec,ElGramArgMain::StdGram);
}
template std::istream & operator >> (std::istream &is,ElSTDNS vector<INT> & vec);





#define DEFINE_theEmptyLvalADM(Type,Name)\
    template <> const  ElSTDNS list<Type > ElArgMain<Type >::theEmptyLvalADM;\
    template <> const char * str_type(Type *) { return Name; }


DEFINE_theEmptyLvalADM(Box2di,"Box2di");
DEFINE_theEmptyLvalADM(Box2dr,"Box2dr");
DEFINE_theEmptyLvalADM(bool,"bool");
DEFINE_theEmptyLvalADM(INT,"INT");
DEFINE_theEmptyLvalADM(REAL,"REAL");
DEFINE_theEmptyLvalADM(Pt2di,"Pt2di");
DEFINE_theEmptyLvalADM(Pt2dr,"Pt2dr");
DEFINE_theEmptyLvalADM(Pt3dr,"Pt3dr");
DEFINE_theEmptyLvalADM(Pt3di,"Pt3di");
DEFINE_theEmptyLvalADM(ElSTDNS string,"string");
DEFINE_theEmptyLvalADM(U_INT1,"u_int1");
DEFINE_theEmptyLvalADM(INT1,"int1");
DEFINE_theEmptyLvalADM(char,"char");
DEFINE_theEmptyLvalADM(ElSTDNS vector<Pt2dr>,"vector<Pt2dr>");
DEFINE_theEmptyLvalADM(ElSTDNS vector<int>,"vector<int>");
DEFINE_theEmptyLvalADM(ElSTDNS vector<double>,"vector<double>");
DEFINE_theEmptyLvalADM(ElSTDNS vector < ElSTDNS vector < int > >,"vector<vector<int> >");
DEFINE_theEmptyLvalADM(ElSTDNS vector<std::string>,"vector<std::string>");

template <> eArgMainBaseType ElArgMain<Box2di>::type() const { return AMBT_Box2di; }
template <> eArgMainBaseType ElArgMain<Box2dr>::type() const { return AMBT_Box2dr; }
template <> eArgMainBaseType ElArgMain<bool>::type()   const { return AMBT_bool;   }
template <> eArgMainBaseType ElArgMain<INT>::type()    const { return AMBT_INT;    }
template <> eArgMainBaseType ElArgMain<REAL>::type()   const { return AMBT_REAL;   }
template <> eArgMainBaseType ElArgMain<Pt2di>::type()  const { return AMBT_Pt2di;  }
template <> eArgMainBaseType ElArgMain<Pt2dr>::type()  const { return AMBT_Pt2dr;  }
template <> eArgMainBaseType ElArgMain<Pt3dr>::type()  const { return AMBT_Pt3dr;  }
template <> eArgMainBaseType ElArgMain<Pt3di>::type()  const { return AMBT_Pt3di;  }
template <> eArgMainBaseType ElArgMain<string>::type() const { return AMBT_string; }
template <> eArgMainBaseType ElArgMain<U_INT1>::type() const { return AMBT_U_INT1; }
template <> eArgMainBaseType ElArgMain<INT1>::type()   const { return AMBT_INT1;   }
template <> eArgMainBaseType ElArgMain<char>::type()   const { return AMBT_char;   }
template <> eArgMainBaseType ElArgMain<vector<Pt2dr> >::type()        const { return AMBT_vector_Pt2dr;  }
template <> eArgMainBaseType ElArgMain<vector<int> >::type()          const { return AMBT_vector_int;    }
template <> eArgMainBaseType ElArgMain<vector<double> >::type()       const { return AMBT_vector_double; }
template <> eArgMainBaseType ElArgMain<vector<vector<int> > >::type() const { return AMBT_vvector_int;   }
template <> eArgMainBaseType ElArgMain<vector<std::string> >::type()  const { return AMBT_vector_string; }


std::vector<cMMSpecArg>  LArgMain::ExportMMSpec(bool isOpt) const
{
   std::vector<cMMSpecArg> aRes;
   int aNum=0;
   for (std::list<GenElArgMain*>::const_iterator itA=_larg.begin(); itA!=_larg.end() ; itA++, aNum++)
   {
       aRes.push_back(cMMSpecArg(*itA,aNum,isOpt));
   }

   return aRes;
}




INT LArgMain::Init
    (
    int argc,
    char ** argv
    ) const
{
    if (argc<(INT)_larg.size())
    {
        std::ostringstream oss0, oss1;
        oss0 << argc;
        oss1 << (INT)_larg.size();

        Tjs_El_User.ElAssert
            (
            argc>=(INT)_larg.size(),
            EEM0 << "Not enough Arg in LArgMain::Init"
                 << " Got : " << oss0.str().c_str() << " expecting " << oss1.str().c_str()
            );
    }

    INT k=0;
    for
        (
        ElSTDNS list<GenElArgMain *>::const_iterator it = _larg.begin();
    it != _larg.end();
    it++
        )
    {
        MapValuesEAM[(*it)->AddrArg()] = argv[k];
        (*it)->InitEAM(argv[k],ElGramArgMain::StdGram);
        k++;
    }
    return k;
}


int LArgMain::Size() const {return (int)_larg.size();}



bool  LArgMain::OneInitIfMatchEq
    (
    char * s,
    const ElGramArgMain & Gram,
    bool  anAcceptUnknown
    )  const
{
    for
        (
        ElSTDNS list<GenElArgMain *>::const_iterator it = _larg.begin();
    it != _larg.end();
    it++
        )
        if ((*it)->InitIfMatchEq(s,Gram))
            return true;

    if (anAcceptUnknown || s[0]=='+'  ||  s[0]==cInterfChantierNameManipulateur::theCharSymbOptGlob)
        return false;


    Tjs_El_User.ElAssert
        (
        false,
        EEM0 << "LArgMain , Don't understand :["
        << s <<"]"
        );
    return false;
}



void LArgMain::InitIfMatchEq
    (
    std::vector<char *> * aRes,
    int argc,
    char ** argv,
    const ElGramArgMain & Gram,
    bool VerifInit,
    bool AccUnK
    ) const
{


    for (INT k=0; k<argc ; k++)
    {
        if ( (! OneInitIfMatchEq(argv[k],Gram,AccUnK)) && (aRes!=0))
            aRes->push_back(argv[k]);
    }

   // if (VerifInit)
        //VerifInitialize();

}

void LArgMain::VerifInitialize() const
{
    for
        (
        ElSTDNS list<GenElArgMain *>::const_iterator it = _larg.begin();
    it != _larg.end();
    it++
        )

    {
        Tjs_El_User.ElAssert
            (
            (*it)->IsInit(),
            EEM0 << "LArgMain, unitialized var :" << (*it)->name()
            );
    }
}


LArgMain::LArgMain() {}

void LArgMain::show(bool isNamed) const
{
    for
        (
        ElSTDNS list<GenElArgMain *>::const_iterator it = _larg.begin();
    it!= _larg.end();
    it++
        )
        (*it)->show(isNamed);
}



LArgMain::~LArgMain()
{
    for
        (
        ElSTDNS list<GenElArgMain *>::iterator it = _larg.begin();
    it!= _larg.end();
    it++
        )
        delete (*it);
}

LArgMain & LArgMain::operator << (const LArgMain  & v)
{
   for (list<GenElArgMain *>::const_iterator it=v._larg.begin() ; it!=v._larg.end() ; it++)
       _larg.push_back((*it)->dup());

   return *this;
}


std::list<std::string> ModifListe(const std::list<std::string> & aList,const char * aNameType)
{
   if (std::string(aNameType)==std::string("bool"))
   {
       std::list<std::string> aL;
       aL.push_back("true");
       aL.push_back("false");

       return aL;
   }
   return aList;
}



GenElArgMain::GenElArgMain(const char * Name,bool IsInit,eSpecArgMain aSpec,const std::list<std::string> & aLEnumVal) :
        _name  (Name),
    _is_init (IsInit),
        mSpec    (aSpec),
        mLEnum   (aLEnumVal)
{
}

eSpecArgMain GenElArgMain::Spec() const
{
   return mSpec;
}

const std::list<std::string> &  GenElArgMain::ListEnum() const
{
   return mLEnum;
}



bool GenElArgMain::IsActif() const
{
    return _name!=theChaineInactif;
}

const char *  GenElArgMain::ActifStr(bool isAct)
{
    return isAct ? theChaineActif.c_str() : theChaineInactif.c_str();
}

const std::string  GenElArgMain::theChaineInactif = "#%$(|[])??";
const std::string  GenElArgMain::theChaineActif="OK";





bool GenElArgMain::InitIfMatchEq(const ElSTDNS string &s,const ElGramArgMain & Gram)  const
{
    const char * N = _name.c_str();
    const char * EQ = s.c_str();

    // std::cout << "HHHHHHHH "  << EQ << " " << AddrArg() << "\n";
    

    while ((*N==* EQ) && *N )
    {
        N++;
        EQ++;
    }
    if (*N==0 && (*EQ == Gram.mCharEq))
    {
        if (Gram.AnyEqual())
        {
            while(EQ[1] == Gram.mCharEq)
                EQ++;
        }
        InitEAM(ElSTDNS string(EQ+1),Gram);
        MapValuesEAM[AddrArg()] = s;
        return true;
    }
    return false;
}


bool GenElArgMain::IsInit() const
{
    return _is_init;
}

const char * GenElArgMain::name() const
{
    return _name.c_str();
}

// typedef void (*tActionOnHelp)(int argc,char ** argv);

void DefActionOnHelp(int argc,char ** argv)
{
}

tActionOnHelp TheActionOnHelp = DefActionOnHelp;



std::vector<char *>  	ElInitArgMain
    (
         int argc,char ** argv,
         const LArgMain & LGlob,
         const LArgMain & L1,
         const std::string & aFirstArg,
         bool  VerifInit,
         bool  AccUnK,
         int   aNbArgGlobGlob
    )
{
    setlocale(LC_ALL,"C");
    MMD_InitArgcArgv( argc,argv);

    std::vector<char *> aRes;

    if (MMVisualMode)
    {
        if(argc > 1)
        {
            MMVisualMode = false;
            ElInitArgMain(argc,argv,LGlob,L1,aFirstArg,VerifInit,AccUnK,aNbArgGlobGlob);
            MMVisualMode = true;
        }

        std::vector<cMMSpecArg> aVAM = LGlob.ExportMMSpec();
        std::vector<cMMSpecArg> aVAO = L1.ExportMMSpec(true);
        MMRunVisualMode(argc,argv,aVAM,aVAO,aFirstArg);
        return aRes;
    }
    setlocale(LC_ALL,"C");

    aRes.push_back(argv[0]);

    argc--;
    argv++;

    bool Help = false;

    AnalyseContextCom(argc,argv);
 //std::cout << "ARGCCCC " << argc << " " <<  LGlob.Size() << endl;
    if ((argc==0) && ( LGlob.Size() !=0)) Help = true;
    for (int aK=0 ; aK<argc ; aK++)
    {
        if (std::string(argv[aK])=="-help")
        {
            Help = true;
        }
    }

    // if ((argc >=1) && (ElSTDNS string(argv[0]) == "-help"))
    if (Help)
    {
        TheActionOnHelp(argc,argv);
        cout << "*****************************\n";
        cout << "*  Help for Elise Arg main  *\n";
        cout << "*****************************\n";
        cout << "Mandatory unnamed args : \n";
        LGlob.show(false);
        cout << "Named args : \n";
        L1.show(true);
        StdEXIT(-1);
    }

    INT k = LGlob.Init(argc,argv);

    if (aNbArgGlobGlob !=-1)
    {
        ELISE_ASSERT(k<=aNbArgGlobGlob," ElInitArgMain ArgGlob");
        ELISE_ASSERT(aNbArgGlobGlob<=argc," Nb Arg Glob Pas assez important");
        for(;k<aNbArgGlobGlob ; k++)
            aRes.push_back(argv[k]);
    }
    L1.InitIfMatchEq(&aRes,argc-k,argv+k,ElGramArgMain::StdGram,VerifInit,AccUnK);
    setlocale(LC_ALL,"C");


    return aRes;
}

void  	ElInitArgMain
    (
    const std::string & aNameI,
    const LArgMain & LGlob,
    const LArgMain & L1,
    const std::string & aFirstArg
    )
{
    static std::vector<char *> VNames;
    VNames.clear();

    static std::vector<char> aName;
    aName.clear();

    static std::string aBid = "Bid ";

    std::copy(aBid.begin(),aBid.end(),std::back_inserter(aName));
    std::copy(aNameI.begin(),aNameI.end(),std::back_inserter(aName));
    aName.push_back(0);
    char * pC = &aName[0];


    bool LastVide = true;
    for (char * Cur = pC+1; *Cur  ; Cur++)
    {
        if ((*Cur==' ')&&(Cur[-1]!= 0)&&(Cur[-1]!=' '))
        {
            *Cur = 0;
            VNames.push_back(pC);
            pC = Cur+1;
            LastVide = true;
        }
        else if (*Cur !=' ')
            LastVide = false;
    }
    if (! LastVide)
        VNames.push_back(pC);

    ElInitArgMain((int) VNames.size(),&(VNames[0]),LGlob,L1, aFirstArg);
}

INT GenFileInitArgs
    (
    const ElGramArgMain & aGram,
    INT   aCarCommentaire,
    INT   FirstLineUsed,
    INT   FirstCarInLine,
    const ElSTDNS string & NameFile,
    const LArgMain & L1 ,
    INT   StopCar,
    bool  anAcceptUnknown
    )
{
    FILE * fp = ElFopen(NameFile.c_str(),"r");
    if (fp==0)
    {
        cout << "CANT OPEN" << NameFile << "\n";
        ELISE_ASSERT(fp!=0,"SphInitArgs, cannot open FILE");
    }

    INT aLine = 0;
    bool cont = true;
    size_t buflen=0;
    while (cont && fgets(buf,SzBuf,fp))
    {
        buflen = strlen(buf); //remove all endline chars
        while ( (buflen>0) && ((buf[buflen-1]=='\r')||(buf[buflen-1]=='\n')) )
        {
            buf[buflen-1]= '\0';
            buflen--;
        }
        
        if (*buf == StopCar)
            cont = false;
        else
        {
            if (*buf != aCarCommentaire)
            {
                if (aLine >= FirstLineUsed)
                {
                    for (char * c=buf; *c ; c++)
                        if(*c== '\t') *c=' ';
                    L1.OneInitIfMatchEq(buf+FirstCarInLine,aGram,anAcceptUnknown);
                }
            }
            aLine++;
        }
    }

    //L1.VerifInitialize();
    INT res = ftell(fp);
    ElFclose(fp);

    return res;
}

void SphInitArgs
    (
    const ElSTDNS string & NameFile,
    const LArgMain & L1
    )
{
    GenFileInitArgs
        (
        ElGramArgMain::SPHGram,
        -1,
        0,
        0,
        NameFile,
        L1,
        EOF,
        false
        );
}

void StdInitArgsFromFile
    (
    const ElSTDNS string & NameFile,
    const LArgMain & L1
    )
{
    GenFileInitArgs
        (
        ElGramArgMain::StdGram,
        '#',
        0,
        0,
        NameFile,
        L1,
        EOF,
        false
        );
}

void HdrInitArgsFromFile
    (
    const ElSTDNS string & NameFile,
    const LArgMain & L1
    )
{
    GenFileInitArgs
        (
        ElGramArgMain::HDRGram,
        '!',      // aCarCommentaire
        0,        // FirstLineUsed
        0,        // FirstCarInLine
        NameFile, // NameFile
        L1,       // LArgMain L1
        EOF,      // StopCar
        true
        );
}


INT ThomInitArgs
    (
    const ElSTDNS string & NameFile,
    const LArgMain & L1
    )
{
    return GenFileInitArgs
        (
        ElGramArgMain::THOMGram,
        'C',//-1,
        1,
        2,
        NameFile,
        L1,
        '*',
        // false
        true  // Maintenant on accepte et ignore les tag thom inconnus
        );
}

bool IsThomFile (const std::string & aName)
{
    FILE * fp = ElFopen(aName.c_str(),"r");
    if (! fp)
        return false;
    char * astr = fgets(buf,SzBuf,fp);
    if (! astr)
    {
        ElFclose(fp);
        return false;
    }
    bool res = (strcmp(buf,"C ENTETE\n")==0);
    ElFclose(fp);
    return res;
}
#endif

std::string StrFromArgMain(const std::string & aStr)
{
    int aL = (int)aStr.length();
    aL--;
    while ((aL>=0) && isspace(aStr.at(aL)))
        aL--;
    if (aStr.at(aL)) aL++;
    std::string aRes = aStr.substr(0,aL);

    // std::cout << "[" <<aStr <<"]->[" << aRes << "]\n";

    return aRes;

}

/*******************************************************/
/*                                                     */
/*             cAppliBatch                             */
/*                                                     */
/*******************************************************/

int TopSystem(const std::string & aCom)
{
   return ::System(aCom,false,true,true);
}

int System(const std::string & aComOri,bool aSVP,bool AddOptGlob,bool UseTheNbIterProcess)
{
    std::string aCom = aComOri;
    int aNbIter = UseTheNbIterProcess  ? TheNbIterProcess  : 1;
    if (aNbIter > 1) aSVP = true;


    // Pour que les args de contextes soient automatiquement passes aux process fils
    if (AddOptGlob)
    {
       aCom += TheStringMemoArgOptGlob ;
    }

    // std::cout << "Sys:"<< aCom << "\n";

    int aRes = 0;
    for (int aKIter = 0 ;  aKIter < aNbIter ; aKIter++)
    {
        if (aKIter>0) SleepProcess(1); // Pour pouvoir plus facilement arreter par ^C

    #if (ELISE_windows)
        if ( aCom.size()!=0 && aCom[0]=='\"' )
            aRes = system_call( ( string("\"")+aCom+"\"" ).c_str() );
        else
            aRes = system_call( aCom.c_str() );
    #else
// std::cout << "SYS " << __FILE__ << __LINE__ << " " << aCom << "\n";
		#ifdef __TRACE_SYSTEM__
        aRes = system_call(aCom.c_str());
      #else
        aRes = system(aCom.c_str());
      #endif
// std::cout << "SYS " << __FILE__ << __LINE__ << " " << aRes << " " << aCom << "\n";
    #endif
    if ((aRes != 0) && (!aSVP))
    {
            // Modif MPD : sur de gros chantier avec un max de  MicMac en paral, il faut savoir quelle commande a plante
            // sans avoir a inspecter un terminal sature

/*
            std::string aFileName = Dir2Write() + GetUnikId() + ".txt";
            FILE * aFP = fopen(aFileName.c_str(),"a+");
            if (aFP)
            {
                fprintf(aFP,"Failed in command\n");
                fprintf(aFP,"%s\n",aCom.c_str());
                fprintf(aFP,"PID=%d\n",getpid());
                fclose(aFP);
            }
*/


        std::cout  << "FAIL IN : \n";
            std::cout << aCom << "\n";
            ElEXIT(-1,(std::string("System-call :") + aCom));
    }
     }


     return aRes;
}

void ElExit(int aLine,const char * aFile,int aCode,const std::string & aMessage)
{
   cFileDebug::TheOne.Close(aCode);

   if (aCode==0)
      StdEXIT(0);

    std::string aFileName = ( isUsingSeparateDirectories() ?
                              MMTemporaryDirectory() + "MM-Error-"+ GetUnikId() + ".txt" :
                              Dir2Write() + "MM-Error-"+ GetUnikId() + ".txt" );

   FILE * aFP = fopen(aFileName.c_str(),"a+");
   if (aFP)
   {
      fprintf(aFP,"Exit with code %d \n",aCode);
      fprintf(aFP,"Generated from line %d  of file %s \n",aLine,aFile);
      fprintf(aFP,"PID=%d\n",mm_getpid());
      if (aMessage!="")
         fprintf(aFP,"Context=[%s]\n",aMessage.c_str());

      for (int aK=0 ; aK<(int)GlobMessErrContext.size() ; aK++)
         fprintf(aFP,"GMEC:%s\n",GlobMessErrContext[aK].c_str()),

      fprintf(aFP,"MM3D-Command=[%s]\n",GlobArcArgv.c_str());
   }

// std::cout << "ELExit " << __FILE__ << __LINE__ << " " << aCode << " " << GlobArcArgv << "\n";
   StdEXIT(aCode);
}



int cAppliBatch::System(const char * aFile,const std::string & aCom,bool aSVP)
{
    if ((aFile!=0) && ByMKf() && (!mIsRelancedByThis))
    {
        mGPAO.GetOrCreate(aFile,aCom);
        return 0;
    }
    //std::cout << aCom << "\n";

    if (mModeExe== eExeDoNothing)
    {
        return 0;
    }
    else if ((mModeExe== eExeDoSys) || (mModeExe==eExeDoIfFileDontExist))
    {

        if (
            (mModeExe==eExeDoIfFileDontExist)
            && (aFile!=0)
            &&  (ELISE_fp::exist_file(aFile))
            )
        {
            return 0;
        }
        return ::System(aCom,aSVP);


    }
    else
    {
        ELISE_ASSERT(false,"Mode Exe non suporte");
        return -1;
    }
    return -1;
}

int cAppliBatch::System(const std::string & aCom,bool aSVP)
{
    return System((char*)0,aCom,aSVP);
}

void cAppliBatch::AddPatSauv(const std::string & aPat)
{
    mPatSauv.push_back(aPat);
}


void cAppliBatch::DoPurge()
{
    // std::cout << mNivPurge << " " << eNoPurge << " " << ePurgeTmp << "\n" ;

    if (mPostFixWorkDir=="")
        return;

    if (mNivPurge==eNoPurge)
        return;

    if (mNivPurge == ePurgeTmp)
    {
        ELISE_fp::MkDir(mDirSauv);
        for (int aK=0 ; aK<int(mPatSauv.size()) ; aK++)
        {
            std::string aCom =     std::string(SYS_MV)+' '
                +  mDirTmp+mPatSauv[aK] + std::string(" ")
                +  mDirSauv;
            System(aCom,true);
        }
    }

    ELISE_fp::PurgeDir( mDirTmp );
}

std::string cAppliBatch::ComCommune() const
{
    std::string aRes = std::string(" @WorkDir=") + mDirChantier
        + std::string(" ") + mArgAdd + std::string(" ");
    if (mNbFile >= 1)
        aRes = aRes +  std::string(" @Im1=") + mCurF1;
    if (mNbFile >= 2)
        aRes = aRes +  std::string(" @Im2=") + mCurF2;

    return aRes;
}

const std::string & cAppliBatch::CurF1() const {return mCurF1;}
const std::string & cAppliBatch::CurF2() const {return mCurF2;}
cInterfChantierNameManipulateur * cAppliBatch::ICNM() {return mICNM;}
cInterfChantierNameManipulateur * cAppliBatch::ICNM() const {return mICNM;}
const std::string & cAppliBatch::DirChantier() const {return mDirChantier;}
const std::string & cAppliBatch::DirTmp() const {return mDirTmp;}
const std::string & cAppliBatch::DirSauv() const {return mDirSauv;}

const std::string &  cAppliBatch::CurF(int aK) const
{
    ELISE_ASSERT((aK>=0)&&(aK<mNbFile),"cAppliBatch::CurF");
    if (aK==0) return mCurF1;
    if (aK==1) return mCurF2;
    ELISE_ASSERT(false,"cAppliBatch::CurF");
    return mCurF1;
}

void  cAppliBatch::UseLFile(const std::list<string> & aLFile1)
{
    for
        (
        std::list<std::string>::const_iterator itF1 = aLFile1.begin() ;
    itF1 != aLFile1.end() ;
    itF1++
        )
    {
        mCurF1 = *itF1;
        if (mFileByPat && (mNbFile==2))
                {
            mCurF2 = mICNM->Assoc1To1(mPatF2,mCurF1,true);
                }
        DoOne();
    }
}


void cAppliBatch::DoAll()
{
    ELISE_ASSERT
        (
        mNbFile<3,
        "cAppliBatch::DoAll() En Mode Pat, Multiple file non traite  "
        );

    if (mByNameFile)
    {
        //std::cout << "BY FILE \n";
        mCurF1 = mPatF1;
        if (mNbFile>=2)
            mCurF2 = mPatF2;
        DoOne();
        return;
    }

    if (mFileByPat)
    {
        //std::cout << "BY PATTERN \n";
        if (mNbFile==2)
        {
            UseLFile(mListFile1ByPat);
            /*
            std::cout << "PatF2=" <<mPatF2 << "\n";
            ELISE_ASSERT
            (
            false,
            "cAppliBatch::DoAll() En Mode Pat, Multiple file non traite  "
            );
            */
        }
        else if(mNbFile==1)
        {
            UseLFile(mListFile1ByPat);
        }
        return;
    }

    //std::cout << "BY DICO \n";
    if (mNbFile==1)
    {
        UseLFile(mICNM->StdGetListOfFile(mPatF1));
        return;
    }

    if (mNbFile==2)
    {
        const std::vector<cCpleString> * aVC = mICNM->GetRel(mPatF1);
        for (int aK=0 ; aK<int(aVC->size()) ; aK++)
        {
            int aKC = (mReverse ? int(aVC->size()) - 1 - aK : aK);
            mCurF1 = (*aVC)[aKC].N1();
            mCurF2 = (*aVC)[aKC].N2();
            DoOne();
        }
    }
}

bool cAppliBatch::NivPurgeIsInit()
{
    return mNivPurgeIsInit;
}
void cAppliBatch::SetNivPurge(eNivPurge  aNiv)
{
    mNivPurge = aNiv;
}

bool cAppliBatch::NivExeIsInit()
{
    return mExeIsInit;
}
void cAppliBatch::SetNivExe(eModeExecution aMode)
{
    mModeExe = aMode;
}


cAppliBatch::eModeExecution cAppliBatch::ModeExe() const
{
    return mModeExe;
}

std::string cAppliBatch::protectFilename( const string &i_filename ) const
{
    return (ByMKf()?protect_spaces(i_filename):string("\"")+i_filename+"\"");
}

std::string cAppliBatch::ComForRelance()
{
    std::string aRes = mDebCom;

    if (mNbFile>=1)
        aRes = aRes + " " + protectFilename( mCurF1 );

    if (mNbFile>=2)
        aRes = aRes + " " + protectFilename( mCurF2 );

    aRes = aRes + mEndCom + " IsRelancedByThis=1";

    return aRes;
}

void cAppliBatch::DoOne()
{
    mVCurF.clear();
    if (mNbFile>=1)
        mVCurF.push_back(mCurF1);
    if (mNbFile>=2)
        mVCurF.push_back(mCurF2);

    if (mDOIDE)
    {
        if (mNbFile==2)
        {
            std::string aNF =   mDirChantier
                + mICNM->Assoc1ToN(mKeyDOIDE,mVCurF,true);
            if(ELISE_fp::exist_file(aNF))
            {
                return;
            }
        }
    }


    Exec();
    DoPurge();

//    if (! ByMKf())
//    {
//        std::cout << "\n";
//        std::cout << "       -----------------------------\n";
//        std::cout << "\n";
//    }
}

cEl_GPAO &  cAppliBatch::GPAO ()
{
    return mGPAO;
}

bool  cAppliBatch::ByMKf() const
{
    return mMKf != "";
}

const std::string & cAppliBatch::MKf() const
{
    ELISE_ASSERT(ByMKf(),"cAppliBatch::MKf");
    return mMKf;
}

bool  cAppliBatch::IsRelancedByThis() const
{
    return mIsRelancedByThis != 0;
}


int cAppliBatch::ARGC()
{
    return (int)mArgsNC.size();
}

char ** cAppliBatch::ARGV()
{
    return &(mArgsNC[0]);
}

cAppliBatch::~cAppliBatch()
{
    if (ByMKf() && (!mIsRelancedByThis))
    {
        mGPAO.GenerateMakeFile(mMKf,mModeAddMkf);
    }
}

cAppliBatch::cAppliBatch
    (
    int argc,
    char **  argv,
    int aNbArgGlob,
    int   aNbFile,
    const std::string & aPostFixWorkDir,
    const std::string & aKeyDOIDE,
        bool  aForceDico
    )  :
mICNM           (0),
    mPostFixWorkDir (aPostFixWorkDir),
    mNbFile         (aNbFile),
    mByNameFile     (false),
    mFileByPat      (false),
    mByDico         (false),
    mReverse        (0),
    mDOIDE          (0),
    mKeyDOIDE       (aKeyDOIDE),
    mMKf            (""),
   mModeAddMkf       (false),
    mIsRelancedByThis (0)
{
    bool A1IsFile=false;

    MMD_InitArgcArgv( argc, argv );

    if ((argc >2) && (aNbFile>0))
    {
        if ( ELISE_fp::exist_file(argv[1]))
        {
            A1IsFile = true;
            mByNameFile = true;
            ELISE_ASSERT(aNbFile==1," cAppliBatch::cAppliBatch  Incoherenc");
        }
        else
        {
            // check last caracter of argv[1] (which is chantier directory at this point)
            // if it's not a '\' or '/', we need to add one before concatenation with argv[2]
            char lastChar = argv[1][strlen(argv[1])-1];
            if ( (lastChar!='\\') && (lastChar!='/') )
            {
                if ( lastChar==' ' )
                    // it may append that 'make' replaces an ending '\' by a ' '
                    argv[1][strlen(argv[1])-1] = ELISE_CAR_DIR;
                else
                {
                    string newArg1 = std::string( argv[1] ).append(1, ELISE_CAR_DIR);
                    // we don't 'delete []' argv[1] for we didn't 'new' it, it would cause some desallocation problems
                    // TODO: we should use a better solution later
                    argv[1] = new char[newArg1.length()+1];
                    strcpy( argv[1], newArg1.c_str() );
                }
            }

            if ( ELISE_fp::exist_file(std::string(argv[1])+std::string(argv[2])))
            {
                mByNameFile = true;
            }
            else
            {
                std::string aST2(argv[2]);

                if (aForceDico || aST2.find("[[")!=std::string::npos)
                {
                    mByDico = true;
                }
                else
                {

                    mListFile1ByPat = RegexListFileMatch
                        (
                        std::string(argv[1]),
                        std::string(argv[2]),
                        1,
                        false
                        );

                    if ( mListFile1ByPat.empty())
                    {
                        mByDico = true;
                    }
                    else
                    {
                        mFileByPat = true;
                    }
                }
            }
        }
    }



    if (mByDico && (aNbFile==2))
        aNbArgGlob--;

    int aExe = -1;
    int aNivPurge = -1;

    std::string aICNM="";
    mPatF2="";


    mArgsNC = ElInitArgMain
        (
        argc,argv,
        LArgMain()  << EAM(mDirChantier)
        << EAM(mPatF1,GenElArgMain::ActifStr((aNbFile>=1) && (!A1IsFile)))
        << EAM(mPatF2,GenElArgMain::ActifStr((!mByDico) && (aNbFile>=2))),
        LArgMain()  <<  EAM(aExe,"Exe",true)
        <<  EAM(aICNM,"IC",true)
        <<  EAM(aNivPurge,"Purge",true)
        <<  EAM(mArgAdd,"Add",true)
        <<  EAM(mReverse,"Rev",true)
        <<  EAM(mDOIDE,"DOIDE",true)
        <<  EAM(mMKf,"MkF",true)
        <<  EAM(mModeAddMkf,"MkAdd",true)
        <<  EAM(mIsRelancedByThis,"IsRelancedByThis",true),
        "",
        true,
        true,
        aNbArgGlob
        );

    setInputDirectory( mDirChantier );

#if (ELISE_windows)
    replace( mDirChantier.begin(), mDirChantier.end(),'\\', '/' );
#endif

    if (!mByNameFile)
        MakeFileDirCompl(mDirChantier);
    if (A1IsFile)
    {
        // retrieve chantier directory from first file's path
        std::string aFulName = mDirChantier;
        SplitDirAndFile(mDirChantier,mPatF1,aFulName);
    }

    mDebCom = protectFilename(current_program_fullname())+" "+current_program_subcommand()+" "+protectFilename(mDirChantier)+" ";

    mEndCom = "";
    for (int aK=3+(mPatF2!="") ; aK<argc ; aK++)
        mEndCom = mEndCom + " " + protectFilename(argv[aK]);

    mExeIsInit = (aExe!=-1);
    mModeExe =  mExeIsInit ? (eModeExecution) aExe :eExeDoNothing;
    mNivPurgeIsInit = (aNivPurge !=-1);

    if (! mNivPurgeIsInit)
    {
        if (mByNameFile)
            mNivPurge = eNoPurge;
        else
            mNivPurge = ePurgeTmp;
    }
    else
    {
        mNivPurge = (eNivPurge) aNivPurge;
    }

    //std::cout <<  "DIIIRCC=" << mDirChantier << "\n"; getchar();


   mDirSauv = mDirChantier + "Sauv-" + mPostFixWorkDir + ELISE_CAR_DIR;
   mDirTmp  = mDirChantier + "Tmp-" + mPostFixWorkDir + ELISE_CAR_DIR;

    cTplValGesInit<std::string>  aTplICNM;
    if (aICNM != "")
        aTplICNM.SetVal(aICNM);

    mICNM = cInterfChantierNameManipulateur::StdAlloc(argc,argv,mDirChantier,aTplICNM);


    // std::cout <<  "Purge " << mNivPurge << "\n"; getchar();

}

/*
const std::string & cAppliBatch::ThisBin() const
{
    return mThisBin;
}
*/




bool EAMIsInit(void * anAdr)
{
    return (AllAddrEAM.find(anAdr)) != AllAddrEAM.end();
}

std::string StrInitOfEAM(void * anAdr) {return MapValuesEAM[anAdr];}

// protect spaces with backslashes (for use with 'make')
string protect_spaces( const string &i_str )
{
    string out_str = i_str;
    size_t lastPos = 0;
    while ( true )
    {
        lastPos = out_str.find( ' ', lastPos );
        if ( lastPos==string::npos ) return out_str;
        out_str.insert(lastPos,1,'\\');
        lastPos += 2;
    }
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
