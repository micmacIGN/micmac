#ifndef  _cMMVII_Appli_H_
#define  _cMMVII_Appli_H_

/** \file cMMVII_Appli.h
    \brief Contains definition of mother class of all applicarion

*/



// FR=Cette classe contiendra la specification des argumenst, type EAM ....
// EN=This class will  contain the specification of command paramater, (EAM like in micmac 1.0)

//  FR= Cette classe contient la mini docutmation (nom, commentaire, entree sortie ...)
//  EN= This class will contain the inline minimal documation (name, commentary, input-output etc ...)
class cSpecMMVII_Appli;

class cMMVII_Ap_NameManip;

// FR=La classe mere de toute les application, il en existe une et une seule par processe MMVII
// EN= Mother class of any application, must exit exaclty one by process
class cMMVII_Appli;



// EN= Some typedef to facilitate type declaration

// typedef cMMVII_Appli *  tMMVII_AppliPtr;
typedef std::unique_ptr<cMMVII_Appli>   tMMVII_UnikPApli;
typedef tMMVII_UnikPApli (* tMMVII_AppliAllocator)(int argc, char ** argv);



     // ========================== cSpecMMVII_Appli ==================
class cSpecMMVII_Appli
{
     public :
       typedef std::vector<eApF>   tVaF; 
       typedef std::vector<eApDT>  tVaDT; 

       cSpecMMVII_Appli
       (
           const std::string & aName,
           tMMVII_AppliAllocator,          
           const std::string & aComment,
               // Features, Input, Output =>  main first, more generally sorted by pertinence
           const tVaF     & aFeatures, 
           const tVaDT    & aInputs,   
           const tVaDT    & aOutputs  
       );

       void Check();
       static std::vector<cSpecMMVII_Appli*> & VecAll();

       const std::string &    Name() const;
       tMMVII_AppliAllocator  Alloc() const;
       const std::string &    Comment() const;
    private :
   // Data
       std::string           mName;
       tMMVII_AppliAllocator mAlloc;
       std::string           mComment;
       tVaF                  mVFeatures;
       tVaDT                 mVInputs;
       tVaDT                 mVOutputs;

};


// Ces separation en classe cMMVII_Ap_NameManip etc ... a pour unique but 
// de classer les fonctionnalite et rendre plus lisible (?) une classe qui 
// risque de devenir la classe mamouth ...
 
     // ========================== cMMVII_Ap_NameManip  ==================
class cMMVII_Ap_NameManip
{
    public  :
        // Meme fonction, parfois + pratique return value, sinon + economique par ref
        void SplitString(std::vector<std::string > & aRes,const std::string & aStr,const std::string & aSpace);
        std::vector<std::string >  SplitString(const std::string & aStr,const std::string & aSpace);

        cMMVII_Ap_NameManip();
        ~cMMVII_Ap_NameManip();

    protected :
      
        cCarLookUpTable *                       mCurLut;
        cGestObjetEmpruntable<cCarLookUpTable>  mGoClut;
    private :
        // Avance jusqu'au premier char !=0 et Lut[cahr] !=0
        const char * SkipLut(const char *,int aVal);
        void GetCurLut();
        void RendreCurLut();
};


     // ========================== cMMVII_Ap_NameManip  ==================
class cMMVII_Ap_CPU
{
    public  :
        cMMVII_Ap_CPU ();
    protected :
         int mPid;       // Processus id
         int mNbProcSys; // Number of processor on the system
};

     // ========================== cMMVII_Appli  ==================
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
        static cMMVII_Appli & TheAppli(); ///< Return the unique appli, error if not
        virtual int Exe() = 0;            ///< Do the "real" job
        bool ModeHelp() const;            ///< If we are in help mode, don't execute
        virtual ~cMMVII_Appli();        
        bool    IsInit(void *);           ///< indicate for each variable if it was initiazed by argc/argv

    protected :
        /// Constructor, essenntially memorize command line
        cMMVII_Appli(int,char **);
        /// Second step of construction, parse the command line and initialize values
        void InitParam(cCollecArg2007 & anArgObl, cCollecArg2007 & anArgFac);


    private :
        cMMVII_Appli(const cMMVII_Appli&) = delete ; ///< New C++11 feature , forbid copy 
        cMMVII_Appli & operator = (const cMMVII_Appli&) = delete ; ///< New C++11 feature , forbid copy 

        void                                      GenerateHelp(); /// In Help mode print the help
        void                                      InitProject();  /// Create Dir (an other ressources) that may be used by all processe

        static cMMVII_Appli *                     msTheAppli;     ///< Unique application
    protected :
        cMemState                                 mMemStateBegin; ///< To check memory management
        int                                       mArgc;          ///< memo argc
        char **                                   mArgv;          ///< memo argv
        std::string                               mFullBin;       ///< full name of binarie =argv[0]
        std::string                               mDirMMVII;      ///< directory of binary
        std::string                               mBinMMVII;      ///< name of Binary (MMVII ?)
        std::string                               mDirMicMacv1;   ///< Dir where is located MicMac V1
        std::string                               mDirMicMacv2;   ///< Dir where is located MicMac V2
        std::string                               mDirProject;    ///< Directory of the project (./ if no way to find it)
        std::string                               mDirTestMMVII;  ///< Directory for read/write bench files
        bool                                      mModeHelp;      ///< Is help present on parameter
        bool                                      mDoGlobHelp;    ///< Include common parameter in Help
        bool                                      mDoInternalHelp;///< Include internal parameter in Help
        bool                                      mShowAll;       ///< Tuning, show computation details
        int                                       mLevelCall;     ///< as MM call it self, level of call
        cCollecArg2007                            mArgObl;        ///< Mandatory args
        cCollecArg2007                            mArgFac;        ///< Optional args
        bool                                      mDoInitProj;    ///< Init : Create folders of project, def (true<=> LevCall==1)
        cInterfSet<void *>*                       mSetInit;       ///< Adresses of all initialized variables
};

#endif  //  _cMMVII_Appli_H_
