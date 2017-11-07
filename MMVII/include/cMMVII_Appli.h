#ifndef  _cMMVII_Appli_H_
#define  _cMMVII_Appli_H_

// FR=Cette classe contiendra la specification des argumenst, type EAM ....
// EN=This class will  contain the specification of command paramater, (EAM like in micmac 1.0)
class cArgMMVII_Appli;

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

     // ========================== cArgMMVII_Appli  ==================



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
class cMMVII_Appli : public cMMVII_Ap_NameManip,
                     public cMMVII_Ap_CPU
{
    public :
        static cMMVII_Appli & TheAppli();
        virtual int Exe() = 0;
        bool ModeHelp() const;
        virtual ~cMMVII_Appli();

    protected :
        void InitParam(cCollecArg2007 & anArgObl, cCollecArg2007 & anArgFac);
        cMMVII_Appli(int,char **);


    private :
        cMMVII_Appli(const cMMVII_Appli&) = delete ; // New C++11 feature , forbid copy 
        cMMVII_Appli & operator = (const cMMVII_Appli&) = delete ; // New C++11 feature , forbid copy 

        void                                      GenerateHelp();

        static cMMVII_Appli *                     msTheAppli;
        static void                               InitMemoryState();
    protected :
        int                                       mArgc;
        char **                                   mArgv;
        std::string                               mFullBin;
        std::string                               mDirMMVII;
        std::string                               mBinMMVII;
        std::string                               mDirMicMacv1;
        std::string                               mDirProject;
        bool                                      mModeHelp;
        int                                       mLevelCall;
        cCollecArg2007                            mArgObl; ///< Mandatory args
        cCollecArg2007                            mArgFac; ///< Optional args
        cMemState                                 mMemStateBegin; ///< Initialise juste avant mArgObl/mArgFac
      
};

cMMVII_Appli * BenchAppli(int,char **);

#endif  //  _cMMVII_Appli_H_



