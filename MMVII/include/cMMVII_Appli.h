#ifndef  _cMMVII_Appli_H_
#define  _cMMVII_Appli_H_

#include "MMVII_util.h"
#include "MMVII_Stringifier.h"
#include "MMVII_Bench.h"


namespace MMVII
{

/** \file cMMVII_Appli.h
    \brief Contains definition of mother class of all applicarion

*/

class cMMVII_Ap_CPU;
class cSpecMMVII_Appli;
class cMMVII_Ap_NameManip;
class cMMVII_Appli;
// class cSetName;
class cColStrAObl;
class cColStrAOpt;
typedef std::pair<std::string,std::string> t2S;
class cAppliBenchAnswer; // With this class, an Appli indicate how it deal with Bench





//  Some typedef to facilitate type declaration
typedef std::unique_ptr<cMMVII_Appli>   tMMVII_UnikPApli;
typedef tMMVII_UnikPApli (* tMMVII_AppliAllocator)(const std::vector<std::string> & aVArgcv,const cSpecMMVII_Appli &);

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
       int AllocExecuteDestruct(const std::vector<std::string> &) const;

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
/*
       static std::vector<cSpecMMVII_Appli*> VecAll(const std::string &); ///< vectors of all specifs
       static std::vector<cSpecMMVII_Appli*> & SortedVecAll(); ///< vectors of all specifs
       static std::vector<cSpecMMVII_Appli*> & SortedVecAll(); ///< vectors of all specifs
*/

       static cSpecMMVII_Appli* SpecOfName(const std::string & aName,bool SVP); ///< Get spec; non case sensitive search

       const std::string &    Name() const; ///< Accessor
       tMMVII_AppliAllocator  Alloc() const; ///< Accessor
       const std::string &    Comment() const; ///< Accessor
       const std::string &    NameFile() const; ///< Accessor

       // bool HasDataTypeIn(const eApDT & aType) const;
       // bool HasDataTypeOut(const eApDT & aType) const;

       // Display command line args
       static void ShowCmdArgs(void);

    private :
       static std::vector<cSpecMMVII_Appli*> TheVecAll;
       static std::vector<cSpecMMVII_Appli*> & InternVecAll(); ///< vectors of all specifs
   // Data
       std::string           mName;       ///< User name
       tMMVII_AppliAllocator mAlloc;      ///< Allocator
       std::string           mComment;    ///< Comment on what the command is suposed to do
       tVaF                  mVFeatures;  ///< Features, at leat one
       tVaDT                 mVInputs;    ///<  Vector Input Data Type
       tVaDT                 mVOutputs;   ///<  Vector Output Data Type
       std::string           mNameFile;   ///< C++ file where it is defined, may be usefull for devlopers ?

   // Args in the first call of an AllocExecuteDestruct(args) are stored in TheCmdArgs
       static std::vector<std::string> TheCmdArgs;
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

class cExplicitCopy
{
};

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
        static const cColStrAOpt Empty; ///< Default paramater
        cColStrAOpt(cExplicitCopy,const cColStrAOpt&) ;
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

/**  Classes for computing segmentation of times
 *      it maintain a map Name->Time that is updated
 *     
 *     Each time an cAutoTimerSegm is created on a cTimerSegm, the name is
 *     changed (so accumulation is done on another name), when cAutoTimerSegm is
 *     destroyed, the current state is destroyed
 */

class cAutoTimerSegm;
typedef std::string tIndTS;
typedef std::map<tIndTS,double> tTableIndTS;
class cTimerSegm
{
   public :
        
       friend class cAutoTimerSegm;

       cTimerSegm(cMMVII_Ap_CPU *);
       void  SetIndex(const tIndTS &);
       const tTableIndTS &  Times() const;
       void Show();
       ~cTimerSegm();
   private :
       tTableIndTS          mTimers;
       tIndTS               mLastIndex;
       cMMVII_Ap_CPU *      mAppli;
       double               mCurBeginTime;
};


cTimerSegm & GlobAppTS();

class cAutoTimerSegm
{
     public :
          cAutoTimerSegm(cTimerSegm & ,const tIndTS& anInd);  // push index in Timer while saving its state
          cAutoTimerSegm(const tIndTS& anInd);  // calls previous with GlobAppTS
          ~cAutoTimerSegm(); // restore the state of timer
     private :
	  cAutoTimerSegm(const cAutoTimerSegm&) = delete;
          cTimerSegm & mTS;  // save the global timer
          tIndTS  mSaveInd;  // save the curent index in TS to restore it at end
};

/**  Class for executing some acion at given period */
class cTimeSequencer
{
    public :
         cTimeSequencer(double aPeriod);
	 bool ItsTime2Execute();
    public :
	 double mPeriod;
	 double mLastime;
};

/**
    Manage CPU related information on Applis
*/
typedef std::chrono::system_clock::time_point tTime;
class cMMVII_Ap_CPU
{
    public  :
        cMMVII_Ap_CPU ();
        double SecFromT0() const;
        // Accessors
        std::string    StrDateBegin() const;  
        std::string    StrDateCur() const;  
        const std::string  &  StrIdTime() const;  
        cTimerSegm  &  TimeSegm();  ///<  To have a global time Segm
    protected :
         tTime         mT0 ;       ///< More or less creation time
         int           mPid;       ///< Processus id
         int           mNbProcSystem; ///< Number of processor on the system
         int           mNbProcAllowed; ///< Number of processor really allowed
         float         mMulNbInMk; ///< in a Mkfile we will allow "mNbProcSys * mMulNbInMk" task
         std::string   mStrIdTime;   ///< Make more a less a unique id  Sec + 1O-4 sec for hour 0
         cTimerSegm                                mTimeSegm;  ///<  To have a global time Segm
};

/**   When we will deal with cluster computing, it will be usefull that command can specify
   their ressource , for now this class is just a "coquille vide" arround a string
*/
class cParamCallSys
{
    public :
       cParamCallSys(const cSpecMMVII_Appli & aCom2007,bool InArgSep);
       void AddArgs(const std::string &);

       int Execute() const;
       const std::string & Com() const ; ///< Accessor
    private :
       const cSpecMMVII_Appli * mSpec;  ///< used for allocation in ArgSep mode
       bool        mArgSep;  ///< means that args are maintend separated 4 use in arg/argv mode, else construc glob com
       std::string mCom; ///< used with ! mArgSep
       int         mArgc;  ///< classical arg count
       std::vector<std::string> mArgv; ///< used with mArgSep
};


     // ========================== cMMVII_Appli  ==================

cMultipleOfs& StdOut(); /// Call the ostream of cMMVII_Appli if exist (else std::cout)
cMultipleOfs& HelpOut();
cMultipleOfs& ErrOut();


/// Mother class of all appli

/** Any application of MMVII must inherit of cMMVII_Appli.
    
    It must exist one and  only one application in one process. This
   application can be reached by method CurrentAppli().

   The object is first constructed, then it action is done with the
   method Exe(); this separation is necessary because some time we will need
   to call virtual method in exe.


   The constructor of inheriting class, should (1) call cMMVII_Appli(argc,argv)
   (2) call  InitParam for parsing the command line. This separatiion is necessary
   because InitParam use ressource of cMMVII_Appli.
 
*/

typedef const char * tConstCharPtr;


class cMMVII_Appli : public cMMVII_Ap_NameManip,
                     public cMMVII_Ap_CPU
{
    public :

        typedef std::vector<eSharedPO>  tVSPO;
        /// Temporary; will add later a "real" warning mechanism, for now track existing
        void MMVII_WARNING(const std::string &);

        /// According to StdOut param can be std::cout, a File, both or none
        cMultipleOfs & StdOut();
        cMultipleOfs & HelpOut();
        cMultipleOfs & ErrOut();

        /// External call sys : use GlobalSysCall + register the command in log files
        int ExtSysCall(const std::string & aCom, bool SVP);


        static bool WithWarnings();
        /// MMVII call itself
        int   ExeCallMMVII(const cSpecMMVII_Appli & aCom,const cColStrAObl&,const cColStrAOpt&,bool ByLineCom=true); 
        void  ExeMultiAutoRecallMMVII
                                (  const std::string & aNameOpt  ,  //!  Name of parameter to substitue, if mandatory "0", "1" ....
                                   const std::vector<std::string> &  LVals, //! List of value for each process
                                   const cColStrAOpt &  aLSubst = cColStrAOpt::Empty,
                                   eTyModeRecall = eTyModeRecall::eTMR_Parall
                                 ); ///< MMVII reccall the same command itself


        int ExeComSerial(const std::list<cParamCallSys> &);    ///< 1 after 1
        int ExeComParal(const std::list<cParamCallSys> &,bool Silence=false);     ///< soon paral with Make for now (other to see ...)
        int ExeComParal(const std::list<std::string> & aLCom,bool Silence=false); ///< in paral for any command; cut in pack and call ExeOnePackComParal
        int ExeOnePackComParal(const std::list<std::string> & aLCom,bool Silence=false); ///< really run in paral for any command



        cColStrAObl& StrObl();
        cColStrAOpt& StrOpt();
        void InitColFromVInit(); ///< Put in StrObl and StrOpt value from initial parameter
 
        static bool   ExistAppli();         ///< Return if the appli exist, no error
        static cMMVII_Appli & CurrentAppli();   ///< Return the unique appli, error if not
        virtual int Exe() = 0;              ///< Do the "real" job
        virtual int ExeOnParsedBox(); ///< Action to exec for each box, When the appli parse a big file , def error


        virtual std::vector<std::string>  Samples() const; ///< For help, gives samples of "good" use
        bool ModeHelp() const;              ///< If we are in help mode, don't execute
        virtual ~cMMVII_Appli();            ///< Always virtual Dstrctr for "big" classes
        bool    IsInit(const void *);       ///< indicate for each variable if it was initiazed by argc/argv
        bool    IsInSpecObl(const void *);  ///< indicate for each variable if it was in an arg opt list (used with cPhotogrammetricProject)
        bool    IsInSpecFac(const void *);  ///< indicate for each variable if it was in an arg obl list (used with cPhotogrammetricProject)
        bool    IsInSpec(const void *);     ///< IsInSpecObl  or IsInSpecFac

	void    SetVarInit(void * aPtr);

	//  Print the effective value of all params
	//  In some case, init can be complicated, with many default case
	void  ShowAllParams() ;

        template <typename T> inline void SetIfNotInit(T & aVar,const T & aValue)
        {
            if (! IsInit(&aVar))
	    {
               aVar = aValue;
	       SetVarInit(&aVar);  //MPD :add 27/02/23 , seems logical, hope no side effect ?
	    }
        }
        static void SignalInputFormat(int); ///< indicate that a xml file was read in the given version
        static bool        OutV2Format() ;  ///<  Do we write in V2 Format

        void InitParam();  ///< Parse the parameter list
        void SetNot4Exe(); ///< Indicate that the appli was not fully initialized

        int NbProcAllowed() const; ///< Accessor to nb of process allowed for the appli
        const std::string & DirProject() const;     ///<  Accessor to directoy of project
        static const std::string & TopDirMMVII();   ///<  main directory of MMVII , upon include,src ..
        static const std::string & TmpDirTestMMVII();     ///< where to put binary file for bench, Export for global bench funtion
        static const std::string & InputDirTestMMVII();   ///<  where are input files for bench   , Export for global bench funtion
        static const std::string & DirMicMacv1();         ///<  Main directory of micmac V1

        ///  Name of folder specific to the command
        std::string  DirTmpOfCmd(eModeCreateDir=eModeCreateDir::CreateIfNew) const;   
        ///  Name of folder specific to the process
        std::string  DirTmpOfProcess(eModeCreateDir=eModeCreateDir::ErrorIfExist) const; 

        //  ===========  Name for Caracteristique points files  , Tile -1,-1 mean no tile
  
          /// Name to generates images for inspection
        std::string NamePCarImage(const std::string & aNameIm,eTyPyrTieP,const std::string & aSpecific,const cPt2di & aTile) const;
          /// Name to generates PCar , most general
        std::string NamePCar(const std::string & aNameIm,eModeOutPCar,eTyPyrTieP,bool InPut,bool IsMax,const cPt2di & aTile) const;
          /// Name to generates PCar , current : V2Bin , Input, no tile
        std::string StdNamePCarIn(const std::string & aNameIm,eTyPyrTieP,bool IsMax) const;

        static int  SeedRandom();  ///< SeedRand if Appli init, else default

        int   LevelCall() const;     ///< Accessor to mLevelCall

        virtual cAppliBenchAnswer BenchAnswer() const; ///< Has it a bench, default : no
        virtual int  ExecuteBench(cParamExeBench &) ; ///< Execute bench, higher lev, higher test, Default Error, Appli is not benchable
        std::string  CommandOfMain() const; ///< Glob command by aggregation of ArgcArgv

        static void AddObj2DelAtEnd(cObj2DelAtEnd *);

        static void InitMMVIIDirs(const std::string& aMMVIIDir);

        static const std::string & DirRessourcesMMVII();       ///< Location of all ressources
    protected :

        /// Constructor, essenntially memorize command line and specifs
        cMMVII_Appli(const std::vector<std::string> & aVArgcv, const cSpecMMVII_Appli &,tVSPO=EmptyVSPO);
        /// Second step of construction, parse the command line and initialize values

        const tNameSet &                         MainSet0() const;         ///< MainSet(0) , 
        const tNameSet &                         MainSet1() const;         ///< MainSet(1) , 
        const tNameSet &                         MainSet(int aK) const;    ///< MainSets[aK] , check range !=0 before
        void                                     CheckRangeMainSet(int) const;  ///< Check range in [0,NbMaxMainSets[
        std::vector<std::string>                 VectMainSet(int aK) const; ///< interface to MainSet
        virtual bool            AcceptEmptySet(int aK) const; ///< Generally if the set is empty, it's an error

        virtual cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) = 0;  ///< A command specifies its mandatory args
        virtual cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) = 0;  ///< A command specifies its optional args
        void InitOutFromIn(std::string &aFileOut,const std::string& aFileIn); ///< If out is not init set In, else DirProj+Out

        void                                      Warning(const std::string & aMes,eTyW,int line,const std::string & File);

        bool RunMultiSet(int aKParam,int aKSet,bool MkFSilence=false);  /// If VectMainSet > 1 => Call itsef in // , result indicates if was executed
        int  ResultMultiSet() const; /// Iff RunMultiSet was execute
        tPtrArg2007 AOptBench();  ///< to add in args mode if command can execute in bench mode

        static const std::string & FullBin();            ///< Protected accessor to full pathname of MMVII executable
        static const std::string & DirTestMMVII();       ///< Protected accessor to dir to read/write test bench

    private :
        cMMVII_Appli(const cMMVII_Appli&) = delete ; ///< New C++11 feature , forbid copy 
        cMMVII_Appli & operator = (const cMMVII_Appli&) = delete ; ///< New C++11 feature , forbid copy 
        // Subst  (aNameOpt,aVal)
        // aNameOpt :  si existe substitue, si "+" ajoute a mandatory, si "3"  => sub 3 mandatory, si MMVII_NONE
        cParamCallSys  StrCallMMVII ( const cSpecMMVII_Appli & aCom, const cColStrAObl&, const cColStrAOpt&,
                                      bool Separate, // Separate argv for call inside
                                      const cColStrAOpt &  aLSubst  = cColStrAOpt::Empty); ///< MMVII call itself
        std::list<cParamCallSys>  ListStrCallMMVII
                                (  const cSpecMMVII_Appli & aCom,const cColStrAObl&,const cColStrAOpt&,
                                   const std::string & aNameOpt  , const std::vector<std::string> &  LVals,
                                   bool Separate
                                   //  const cColStrAOpt &  aLSubst = cColStrAOpt::Empty
                                 ); ///< MMVII call itself

        std::list<cParamCallSys>  ListStrAutoRecallMMVII
                                (  const std::string & aNameOpt  , const std::vector<std::string> &  LVals,
                                   bool Separate,
                                   const cColStrAOpt &  aLSubst = cColStrAOpt::Empty
                                 ); ///< MMVII reccall the same command itself

        void                                      GenerateHelp(); ///< In Help mode print the help
        void PrintAdditionnalComments(tPtrArg2007 anArg); ///< Print the optional comm in mode glob help

        void                                      InitProject();  ///< Create Dir (an other ressources) that may be used by all processe
        void                                      LogCommandIn(const std::string&,bool Main);  ///< Log command begin
        void                                      LogCommandOut(const std::string&,bool Main); ///< Log command end
        std::string                               NameFileLog(bool Finished) const; ///< File 4 log each process
        static std::vector<cMMVII_Appli *>        TheStackAppli;     ///< Unique application
        static int                                TheNbCallInsideP;  ///< Number of Appli created in the same process
        static bool                               msInDstructor;  ///< Some caution must be taken once destruction has begun
        static const int                          msDefSeedRand;  ///<  Default value for Seed random generator
        static bool                               msWithWarning;  ///<   do we print warnings
        void                                      AssertInitParam() const; ///< Check Init was called
    protected :
        virtual int                               DefSeedRand();  ///< Clas can redefine instead of msDefSeedRand, value <=0 mean init from time:w
        cMemState                                 mMemStateBegin; ///< To check memory management


        std::vector<std::string>                  mArgv;      ///< copy of local copy ArgArgv to be safe
        int                                       mArgc;          ///< memo argc
        const cSpecMMVII_Appli &                  mSpecs;         ///< The basic specs
        bool                                      mForExe; ///< To distinguish not fully initialized in X::~X()

        std::string                               mDirProject;    ///< Directory of the project (./ if no way to find it)
        std::string                               mFileLogTop;    ///< File for login the top command
        bool                                      mModeHelp;      ///< Is help present on parameter
        bool                                      mDoGlobHelp;    ///< Include common parameter in Help
        bool                                      mDoInternalHelp;///< Include internal parameter in Help
        std::string                               mPatHelp;       ///< Possible filter on name of optionnal param shown
        bool                                      mShowAll;       ///< Tuning, show computation details
        int                                       mLevelCall;     ///< as MM call it self, level of call
        cExtSet<const void *>                     mSetInit;       ///< Adresses of all initialized variables
        cExtSet<const void *>                     mSetVarsSpecObl; ///< Adresses var in specif, obligatory
        cExtSet<const void *>                     mSetVarsSpecFac; ///< Adresses var in specif, faculative 
        bool                                      mInitParamDone; ///< To Check Post Init was not forgotten
        cColStrAObl                               mColStrAObl;    ///< To use << for passing multiple string
        cColStrAOpt                               mColStrAOpt;    ///< To use << for passing multiple pair
    private :
        cCollecSpecArg2007                        mArgObl;        ///< Mandatory args
        cCollecSpecArg2007                        mArgFac;        ///< Optional args
        static const int                          NbMaxMainSets=3; ///< seems sufficient, Do not hesitate to increase if one command requires more
        std::vector<tNameSet>                     mVMainSets;  ///< For a many commands probably
        int                                       mResulMultiS;///< Save Result of Mutlti Set Recall in //
        bool                                      mRMSWasUsed; ///< Indicate if MultiCall was used

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
        // Control position/hierachy of call
        int                                       mNumCallInsideP; ///< Numero of Appli in the process of creation
        bool                                      mMainAppliInsideP; ///< Is the main/firsy Appli inside the process
        bool                                      mMainProcess; ///< Is the current process
        bool                                      mGlobalMainAppli; ///< Both main process && main appli inside P
        std::string                               mPrefixNameAppli;  ///< String Id of process
        std::string                               mPrefixGMA;        ///< Sting Id of Global Main Appli
        std::string                               mDirProjGMA;        ///< Dir Project Main Appli
     
        static std::string                        mDirBinMMVII;   ///< where is the binary
        static std::string                        mTopDirMMVII;   ///< directory  mother of src/ bin/ ...

        static std::string                        mFullBin;       ///< full name of binarie =argv[0]
        static std::string                        mDirMMVII;      ///< directory of binary
        static std::string                        mDirMicMacv1;   ///< Dir where is located MicMac V1
        static std::string                        mDirMicMacv2;   ///< Dir where is located MicMac V2
        static std::string                        mDirTestMMVII;  ///< Directory for read/write bench files
        static std::string                        mTmpDirTestMMVII;  ///< Tmp files (not versionned)
        static std::string                        mInputDirTestMMVII;  ///< Input files (versionned on git)
        static std::string                        mDirRessourcesMMVII;  ///< Directory for read/write bench files


    protected :
     // ###########"  SHARED OPTIMIZED PARAMETER #####################
        bool   HasSharedSPO(eSharedPO) const;  ///< Is this type of parameter activated
        static const tVSPO    EmptyVSPO;       ///< Defaut Vector  shared optional parameter
        tVSPO                 mVSPO;           ///< Vector of shared optional parameter , use for arg spec
        //  ====  TieP Stuff: param, name ... ============

        /// General form to be called by PrefixPCarOut and PrefixPCarIn
        std::string PrefixPCar(const std::string & aNameIm,const std::string & aPref) const;
        ///  The prefix for PCar when they are writen by the appli
        std::string PrefixPCarOut(const std::string & aNameIm) const;
        ///  The prefix for PCar when they are read by the appli
        std::string PrefixPCarIn(const std::string & aNameIm) const;

        std::string NamePCarGen(const std::string & aNameIm,eModeOutPCar,eTyPyrTieP,bool InPut,
                                const std::string & aSpecific,const cPt2di & aTile) const;

        std::string                               mCarPPrefOut;  ///< Prefix for output Carac point ...
        std::string                               mCarPPrefIn;   ///< Prefix for input  Carac point ...
        std::string                               mTiePPrefOut;  ///< Prefix for output Tie Points ...
        std::string                               mTiePPrefIn;   ///< Prefix for inout  Tie Points ...

        static std::vector<cObj2DelAtEnd *>       mVectObj2DelAtEnd; ///< for object which deletion is delegated to appli
        bool                                      mIsInBenchMode;   ///< is the command executed for bench (will probably make specific test)
};


bool    IsInit(const void *);  ///< Call IsInit on the appli def

// To build application outside of the main MMVII executable

// To build a completly separate application
// Must be called in main() before any use of MMVII items
void InitStandAloneAppli(const char* aAppName, const char *aComment="");

// To build a MMVII like application (arguments parsing)
// a cSpecMMVII_Appli must be defined ...
int InitStandAloneAppli(const cSpecMMVII_Appli & aSpec, int argc, char*argv[]);

};
#endif  //  _cMMVII_Appli_H_
