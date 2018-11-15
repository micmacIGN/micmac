#ifndef  _cMMVII_Appli_H_
#define  _cMMVII_Appli_H_

namespace MMVII
{

/** \file cMMVII_Appli.h
    \brief Contains definition of mother class of all applicarion

*/

class cSpecMMVII_Appli;
class cMMVII_Ap_NameManip;
class cMMVII_Appli;
// class cSetName;
class cColStrAObl;
class cColStrAOpt;
typedef std::pair<std::string,std::string> t2S;



//  Some typedef to facilitate type declaration
typedef std::unique_ptr<cMMVII_Appli>   tMMVII_UnikPApli;
typedef tMMVII_UnikPApli (* tMMVII_AppliAllocator)(int argc, char ** argv,const cSpecMMVII_Appli &);

/* ============================================ */
/*                                              */
/*        PROJECT HANDLINS  CLASSES             */
/*        WILL PROBABLY BE A SEPARATE FILE      */
/*                                              */
/* ============================================ */

/// Class for creting/storing set of files

/** In MM, many (almost all ?) command require a set of file, generally image, as
    one of their main parameters.
 
    The class cSetName allow to create a set of name from a pattern or an existing-xml file.
    The command cAppli_EditSet allow to create sets with  with boolean expression.
*/



     // ========================== cSpecMMVII_Appli ==================

/// Class for specification of a command

/** The specification of a command contains :
     - a name to retrieve it
     - a basic commentary
     - an allocator of type "tMMVII_AppliAllocator" , because all must create a application deriving of cMMVII_Appli
       but the class are declared in separate cpp file (dont want to export all application) 
    - a vector of Feature 
    - 2 vector of Data type, for input and ouput 
*/

class cSpecMMVII_Appli
{
     public :
       typedef std::vector<eApF>   tVaF;  ///< Features
       typedef std::vector<eApDT>  tVaDT; ///< Data types

       cSpecMMVII_Appli
       (
           const std::string & aName,
           tMMVII_AppliAllocator,          
           const std::string & aComment,
               // Features, Input, Output =>  main first, more generally sorted by pertinence
           const tVaF     & aFeatures, 
           const tVaDT    & aInputs,   
           const tVaDT    & aOutputs  ,
           const std::string & aNameFile 
       );

       void Check(); ///< Check that specification if ok (at least vectors non empty)
       static const std::vector<cSpecMMVII_Appli*> & VecAll(); ///< vectors of all specifs
       static cSpecMMVII_Appli* SpecOfName(const std::string & aName,bool SVP); ///< Get spec; non case sensitive search

       const std::string &    Name() const; ///< Accessor
       tMMVII_AppliAllocator  Alloc() const; ///< Accessor
       const std::string &    Comment() const; ///< Accessor
       const std::string &    NameFile() const; ///< Accessor
    private :
   // Data
       std::string           mName;       ///< User name
       tMMVII_AppliAllocator mAlloc;      ///< Allocator
       std::string           mComment;    ///< Comment on what the command is suposed to do
       tVaF                  mVFeatures;  ///< Features, at leat one
       tVaDT                 mVInputs;    //
       tVaDT                 mVOutputs;
       std::string           mNameFile;   ///< C++ file where it is defined, may be usefull for devlopers ?

};

/// Class to store Mandatory args for recursive call
/**
    When MMVII calls MMVII, we try to it as structured as possible (not
    only a string). So  that eventually we can parse & analyze
    parameters.

    Not to confuse  with cCollecSpecArg2007 ;
           cCollecSpecArg2007 => used for init , memorize
           cColStrAObl => reset as soon as they are used by ExeCallMMVII
     operator <<  ares defined on both classes ...
*/

class cColStrAObl
{
    public :
        typedef std::vector<std::string> tCont;
        cColStrAObl &  operator << (const std::string &);
        const  tCont & V() const;
        void clear();
        cColStrAObl(); ///< Necessary as X(const X&) is declared (but not defined=>delete)
    private : 
        cColStrAObl(const cColStrAObl&) = delete;
        tCont mV;
};

/// Class to store Optionnal args for recursive call
/**
    Equivalent of cColStrAObl, Use pair of strings Name/Value
    idem not to conduse with cCollecSpecArg2007
*/
class cColStrAOpt
{
    public :
        typedef std::vector<t2S> tCont;
        cColStrAOpt &  operator << (const t2S &);
        const  tCont & V() const;
        void clear();
        cColStrAOpt(); ///< Necessary as X(const X&) is declared (but not defined=>delete)
    private :
        cColStrAOpt(const cColStrAOpt&) = delete;
        tCont mV;
};


// Ces separation en classe cMMVII_Ap_NameManip etc ... a pour unique but 
// de classer les fonctionnalite et rendre plus lisible (?) une classe qui 
// risque de devenir la classe mamouth ...
 
     // ========================== cMMVII_Ap_NameManip  ==================

/**
       Contain string manipulation, for now lowe level of split string
*/
class cMMVII_Ap_NameManip
{
    public  :
        // Meme fonction, parfois + pratique return value, sinon + economique par ref
        void SplitString(std::vector<std::string > & aRes,const std::string & aStr,const std::string & aSpace);
        std::vector<std::string >  SplitString(const std::string & aStr,const std::string & aSpace);

        cMMVII_Ap_NameManip();
        ~cMMVII_Ap_NameManip();

    protected :
      
        cCarLookUpTable *                       mCurLut; /// Lut use for Split , recycled each time
        cGestObjetEmpruntable<cCarLookUpTable>  mGoClut; /// Memry ressource to allocate cCarLookUpTable
    private :
        // Avance jusqu'au premier char !=0 et Lut[cahr] !=0
        const char * SkipLut(const char *,int aVal);
        void GetCurLut();
        void RendreCurLut();
};


     // ========================== cMMVII_Ap_NameManip  ==================

/**
    Manage CPU related information on Applis
*/
class cMMVII_Ap_CPU
{
    public  :
        cMMVII_Ap_CPU ();
        typedef std::chrono::system_clock::time_point tTime;
    protected :
         tTime       mT0 ;       ///< More or less creation time
         int         mPid;       ///< Processus id
         int         mNbProcSys; ///< Number of processor on the system
};

     // ========================== cMMVII_Appli  ==================

cMultipleOfs& StdOut(); /// Call the ostream of cMMVII_Appli if exist (else std::cout)
cMultipleOfs& HelpOut();
cMultipleOfs& ErrOut();


/// Mother class of all appli

/** Any application of MMVII must inherit of cMMVII_Appli.
    
    It must exist one and  only one application in one process. This
   application can be reached by method TheAppli().

   The object is first constructed, then it action is done with the
   method Exe(); this separation is necessary because some time we will need
   to call virtual method in exe.


   The constructor of inheriting class, should (1) call cMMVII_Appli(argc,argv)
   (2) call  InitParam for parsing the command line. This separatiion is nessary
   because InitParam use ressource of cMMVII_Appli.
 
*/


class cMMVII_Appli : public cMMVII_Ap_NameManip,
                     public cMMVII_Ap_CPU
{
    public :
        /// According to StdOut param can be std::cout, a File, both or none
        cMultipleOfs & StdOut();
        cMultipleOfs & HelpOut();
        cMultipleOfs & ErrOut();

        int  ExeCallMMVII(const cSpecMMVII_Appli & aCom,const cColStrAObl&,const cColStrAOpt&); ///< MMVII call itself
        std::string  StrCallMMVII(const cSpecMMVII_Appli & aCom,const cColStrAObl&,const cColStrAOpt&); ///< MMVII call itself
        cColStrAObl& StrObl();
        cColStrAOpt& StrOpt();
 
        static bool   ExistAppli();         ///< Return if the appli exist, no error
        static cMMVII_Appli & TheAppli();   ///< Return the unique appli, error if not
        virtual int Exe() = 0;              ///< Do the "real" job
        bool ModeHelp() const;              ///< If we are in help mode, don't execute
        virtual ~cMMVII_Appli();            ///< Always virtual Dstrctr for "big" classes
        bool    IsInit(void *);             ///< indicate for each variable if it was initiazed by argc/argv
        static void SignalInputFormat(int); ///< indicate that a xml file was read in the given version
        static bool        OutV2Format() ;  ///<  Do we write in V2 Format

        void InitParam();

        const std::string & TmpDirTestMMVII()   const;   ///< where to put binary file for bench, Export for global bench funtion
        const std::string & InputDirTestMMVII() const;   ///<  where are input files for bench   , Export for global bench funtion

        static int  SeedRandom();  ///< SeedRand if Appli init, else default


    protected :
        /// Constructor, essenntially memorize command line and specifs
        cMMVII_Appli(int,char **,const cSpecMMVII_Appli &);
        /// Second step of construction, parse the command line and initialize values

        const tNameSet &                         MainSet0() const;         ///< MainSet(0) , 
        const tNameSet &                         MainSet1() const;         ///< MainSet(1) , 
        const tNameSet &                         MainSet(int aK) const;    ///< MainSets[aK] , check range !=0 before
        void                                     CheckRangeMainSet(int) const;  ///< Check range in [0,NbMaxMainSets[

        virtual cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) = 0;  ///< A command specifies its mandatory args
        virtual cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) = 0;  ///< A command specifies its optional args
        void InitOutFromIn(std::string &aFileOut,const std::string& aFileIn); ///< If out is not init set In, else DirProj+Out

        void                                      Warning(const std::string & aMes,eTyW,int line,const std::string & File);

    private :
        cMMVII_Appli(const cMMVII_Appli&) = delete ; ///< New C++11 feature , forbid copy 
        cMMVII_Appli & operator = (const cMMVII_Appli&) = delete ; ///< New C++11 feature , forbid copy 

        void                                      GenerateHelp(); ///< In Help mode print the help
        void                                      InitProject();  ///< Create Dir (an other ressources) that may be used by all processe

        static cMMVII_Appli *                     msTheAppli;     ///< Unique application
        static bool                               msInDstructor;  ///< Some caution must be taken once destruction has begun
        static const int                          msDefSeedRand;  ///<  Default value for Seed random generator
        void                                      AssertInitParam() const; ///< Check Init was called
    protected :
        virtual int                               DefSeedRand();  ///< Clas can redefine instead of msDefSeedRand, value <=0 mean init from time:w
        cMemState                                 mMemStateBegin; ///< To check memory management
        int                                       mArgc;          ///< memo argc
        char **                                   mArgv;          ///< memo argv
        const cSpecMMVII_Appli &                  mSpecs;         ///< The basic specs

        std::string                               mDirBinMMVII;   ///< where is the binary
        std::string                               mTopDirMMVII;   ///< directory  mother of src/ bin/ ...

        std::string                               mFullBin;       ///< full name of binarie =argv[0]
        std::string                               mDirMMVII;      ///< directory of binary
        std::string                               mBinMMVII;      ///< name of Binary (MMVII ?)
        std::string                               mDirMicMacv1;   ///< Dir where is located MicMac V1
        std::string                               mDirMicMacv2;   ///< Dir where is located MicMac V2
        std::string                               mDirProject;    ///< Directory of the project (./ if no way to find it)
        std::string                               mDirTestMMVII;  ///< Directory for read/write bench files
        std::string                               mTmpDirTestMMVII;  ///< Tmp files (not versionned)
        std::string                               mInputDirTestMMVII;  ///< Input files (versionned on git)
        bool                                      mModeHelp;      ///< Is help present on parameter
        bool                                      mDoGlobHelp;    ///< Include common parameter in Help
        bool                                      mDoInternalHelp;///< Include internal parameter in Help
        std::string                               mPatHelp;       ///< Possible filter on name of optionnal param shown
        bool                                      mShowAll;       ///< Tuning, show computation details
        int                                       mLevelCall;     ///< as MM call it self, level of call
        bool                                      mDoInitProj;    ///< Init : Create folders of project, def (true<=> LevCall==1)
        cExtSet<void *>                           mSetInit;       ///< Adresses of all initialized variables
        bool                                      mInitParamDone; ///< To Check Post Init was not forgotten
        cColStrAObl                               mColStrAObl;    ///< To use << for passing multiple string
        cColStrAOpt                               mColStrAOpt;    ///< To use << for passing multiple pair
    private :
        cCollecSpecArg2007                        mArgObl;        ///< Mandatory args
        cCollecSpecArg2007                        mArgFac;        ///< Optional args
        static const int                          NbMaxMainSets=3; ///< seems sufficient, Do not hesitate to increase if one command requires more
        std::vector<tNameSet>                     mVMainSets;  ///< For a many commands probably
        std::string                               mIntervFilterMS[NbMaxMainSets];  ///< Filterings interval

        // Variable for setting num of mm version for output
        int                                       mNumOutPut;  ///< specified by user
        bool                                      mOutPutV1;   ///< computed from mNumOutPut
        bool                                      mOutPutV2;   ///< computed from mNumOutPut
        bool                                      mHasInputV1; ///< Is there any input in V1 format ?
        bool                                      mHasInputV2; ///< Is there any input in V2 format ?
        // For controling output
        std::unique_ptr<cMMVII_Ofs>               mFileStdOut;  ///< Redirection of std output
        cMultipleOfs                              mStdCout;     ///< Standard Ouput (File,Console, both or none)
        std::string                               mParamStdOut; ///< Users value
        int                                       mSeedRand;    ///< Seed for random generator
};

};
#endif  //  _cMMVII_Appli_H_
