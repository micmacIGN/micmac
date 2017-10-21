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

typedef cMMVII_Appli *  tMMVII_AppliPtr;
typedef tMMVII_AppliPtr (* tMMVII_AppliAllocator)(int argc, char ** argv);

     // ========================== cArgMMVII_Appli  ==================

class cArgMMVII_Appli
{
};


// FR= veut les mettre ici pour lisibilite, met des define pour eviter les pb de link a cause definition multiple
// EN= want them here for visibility, use define to avoir link
#define SPECMMVII_Feature     "Test Ori Match TieP" 
#define SPECMMVII_DateType    "Ori TieP Ply None Console" 


     // ========================== cSpecMMVII_Appli ==================
class cSpecMMVII_Appli
{
     public :
       cSpecMMVII_Appli
       (
           const std::string & aName,
           tMMVII_AppliAllocator,          
           const std::string & aComment,
               // Features, Input, Output =>  main first, more generally sorted by pertinence
           const std::string & aFeatures, // Must be a sublist of SPECMMVII_Feature
           const std::string & aInputs,   // Must be a sublist of SPECMMVII_DateType
           const std::string & aOutputs   // Must be a sublist of SPECMMVII_DateType
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
       std::string           mFeatures;
       std::string           mInputs;
       std::string           mOutputs;

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

     // ========================== cMMVII_Appli  ==================
class cMMVII_Appli : public cMMVII_Ap_NameManip
{
    public :
        static cMMVII_Appli & TheAppli();
        virtual int Exe() = 0;
        virtual ~cMMVII_Appli();

    protected :
        cMMVII_Appli(int,char **,cArgMMVII_Appli);

    private :
        cMMVII_Appli(const cMMVII_Appli&); // Not implemanted

        static cMMVII_Appli *                     msTheAppli;
        cMemState                                 mMemStateBegin;
};

cMMVII_Appli * BenchAppli(int,char **);

#endif  //  _cMMVII_Appli_H_



