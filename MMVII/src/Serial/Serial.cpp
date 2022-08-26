#include "include/MMVII_all.h"
#include <boost/functional/hash.hpp>


/** \file Serial.cpp
    \brief Implementation of serialisation servive

    This serialization principle is very close (at least inspired by) boost one. 
  In fact I hesitated to use boost, but could not find a way satisfying for handling optional value
  in XML (accept to read old xml file when the structure has grow with optionnale value).

     serializing a class,  consist essentially to describe serially all its data. This done 
  by  defining a function AddData that   calling seriall the AddData of its member.
  See cTestSerial0, cTestSerial1 and cTestSerial2 at the end of this file for
  examples.

*/


namespace MMVII
{

/* ========================================================= */
/*                                                           */
/*            cRawData4Serial                                */
/*                                                           */
/* ========================================================= */

cRawData4Serial::cRawData4Serial(void * aAdr,int aNbElem) :
    mAdr    (aAdr),
    mNbElem (aNbElem)
{
}

void * cRawData4Serial::Adr()    const {return mAdr   ;}
int    cRawData4Serial::NbElem() const {return mNbElem;}


/* ========================================================= */
/*                                                           */
/*            cAr2007                                        */
/*                                                           */
/* ========================================================= */

/// Base class of all archive class

/**
     Base class of all archive class;
 
     Adding a new kind of archive, essentially consist to indicate how to read/write atomic values.
    It is a bit more complicated with tagged format
*/

class cAr2007 : public cMemCheck
{
    public  :
         friend class cAuxAr2007;

         template <class Type> void TplAddDataTerm (const cAuxAr2007& anOT, Type  &    aVal)
         {
                RawAddDataTerm(aVal);
         }
         ///  Tagged File = xml Like, important for handling optionnal parameter
         bool  Tagged() const; 
         ///  May optimize the action
         bool  Input() const; 
         /// Allow to  know by advance if next optionnal value is present, usefull with Xml
         /// Default return error
         virtual int NbNextOptionnal(const std::string &);
         virtual ~cAr2007();
         virtual void Separator(); /**< Used in final but non atomic type, 
                                        for ex with Pt : in text separate x,y, in bin do nothing */
         virtual size_t HashKey() const;

    protected  :
         cAr2007(bool InPut,bool Tagged);
         int   mLevel;
         bool  mInput;
         bool  mTagged; 
     private  :

         /// By default error, to redefine in hashing class
         /// This message is send before each data is serialized, tagged file put/read their opening tag here
         virtual void RawBeginName(const cAuxAr2007& anOT);
         /// This message is send each each data is serialized, tagged file put/read their closing tag here
         virtual void RawEndName(const cAuxAr2007& anOT);


      // Final atomic type for serialization
         virtual void RawAddDataTerm(int &    anI) =  0; ///< Heriting class descrine how they serialze int
         virtual void RawAddDataTerm(size_t &    anI) =  0; ///< Heriting class descrine how they serialze int
         virtual void RawAddDataTerm(double &    anI) =  0; ///< Heriting class descrine how they serialze double
         virtual void RawAddDataTerm(std::string &    anI) =  0; ///< Heriting class descrine how they serialze string
         virtual void RawAddDataTerm(cRawData4Serial  &    aRDS) =  0; ///< Heriting class descrine how they serialze string
         virtual void RawAddDataTerm(std::map<std::string,int>&    aRDS) ;
         void RawAddDataTerm(bool &    anI) ; ///< use int to do it
         virtual void RawAddDataTerm(tU_INT2 &    anI) ; ///< use int to do it
      // Final non atomic type for serialization
};


void cAr2007::RawBeginName(const cAuxAr2007& anOT) {}
void cAr2007::RawEndName(const cAuxAr2007& anOT) {}
bool cAr2007::Tagged() const {return mTagged;}
bool cAr2007::Input() const  {return mInput;}

cAr2007::cAr2007(bool Input,bool isTagged) :
   mLevel  (0),
   mInput  (Input),
   mTagged (isTagged)
{
}

void cAr2007::RawAddDataTerm(bool &    aBool)
{
    int aI = aBool;
    RawAddDataTerm(aI);
    aBool = aI;
}
void cAr2007::RawAddDataTerm(tU_INT2 &    aUI2)
{
    int aI = aUI2;
    RawAddDataTerm(aI);
    aUI2 = aI;
}

void cAr2007::RawAddDataTerm(std::map<std::string,int>&    aRDS)
{
   for (auto i : aRDS){
       std::string aStr = i.first;
       RawAddDataTerm(aStr);
       RawAddDataTerm(i.second);
   }

}

void cAr2007::Separator()
{
}

cAr2007::~cAr2007()
{
}

void DeleteAr(cAr2007 * anAr)
{
   delete anAr;
}


size_t cAr2007::HashKey() const
{
   MMVII_INTERNAL_ASSERT_always(!mInput,"Internal error, no cAr2007::HashKey");
   return 0;
}

/// This function must has been redefined by all the input Archives
int cAr2007::NbNextOptionnal(const std::string &)
{
   MMVII_INTERNAL_ASSERT_always(!mInput,"Internal error, no cAr2007::NbNextOptionnal");
   return -1;
}

void AddData(const  cAuxAr2007 & anAux, size_t  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }
void AddData(const  cAuxAr2007 & anAux, int  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }
void AddData(const  cAuxAr2007 & anAux, double  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }
void AddData(const  cAuxAr2007 & anAux, std::string  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }
void AddData(const  cAuxAr2007 & anAux, cRawData4Serial  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }
void AddData(const  cAuxAr2007 & anAux, bool  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }
void AddData(const  cAuxAr2007 & anAux, std::map<std::string,int>  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }

template <class Type> void AddTabData(const  cAuxAr2007 & anAux, Type *  aVD,int aNbVal)
{
    // A precaution, probably it work but need to test
    MMVII_INTERNAL_ASSERT_always(aNbVal,"Not Sur AddTabData work for NbVal=0, check....");
    if (aNbVal)
       AddData(anAux,aVD[0]);
    for (int aK=1 ; aK<aNbVal ; aK++)
    {
        anAux.Ar().Separator();
        AddData(anAux,aVD[aK]);
    }
}

template <class Type,int Dim> void AddData(const  cAuxAr2007 & anAux, cPtxd<Type,Dim>  &  aPt) 
{
   AddTabData(anAux,aPt.PtRawData(),Dim);
}

template <class Type,int Dim> void AddData(const  cAuxAr2007 & anAux, cTplBox<Type,Dim>  &  aBox) 
{
   AddData(cAuxAr2007("P0",anAux),aBox.P0ByRef());
   AddData(cAuxAr2007("P1",anAux),aBox.P1ByRef());
   // Need to recreate a coherent object
// StdOut() << "AddDataAddDataBox " << aBox.P0ByRef() << " " << aBox.P1ByRef() << "\n";
   if (anAux.Input())
      aBox = cTplBox<Type,Dim>(aBox.P0(),aBox.P1());
}



#define MACRO_INSTANTIATE_AddDataPtxD(DIM)\
template  void AddData(const  cAuxAr2007 & anAux, cPtxd<tREAL8,DIM>  &  aVal) ;\
template  void AddData(const  cAuxAr2007 & anAux, cPtxd<tINT4,DIM>  &  aVal) ;\
template  void AddData(const  cAuxAr2007 & anAux, cTplBox<tINT4,DIM>  &  aVal) ;\

MACRO_INSTANTIATE_AddDataPtxD(1)
MACRO_INSTANTIATE_AddDataPtxD(2)
MACRO_INSTANTIATE_AddDataPtxD(3)

void AddData(const  cAuxAr2007 & anAux, tNamePair  &  aVal) 
{
    AddData(anAux,aVal.V1());
    anAux.Ar().Separator();
    AddData(anAux,aVal.V2());
}

size_t  HashValFromAr(cAr2007& anAr) {return anAr.HashKey();}


/* ========================================================= */
/*                                                           */
/*            cAuxAr2007                                     */
/*                                                           */
/* ========================================================= */

cAuxAr2007::cAuxAr2007 (const std::string & aName,cAr2007 & aTS2) :
    mName     (aName),
    mAr      (aTS2)
{
    mAr.RawBeginName(*this);  // Indicate an opening tag
    mAr.mLevel++;             // Incremente the call level for indentatio,
}

cAuxAr2007::cAuxAr2007 (const std::string & aName, const cAuxAr2007 & anAux) :
    cAuxAr2007(aName,anAux.mAr)
{
}

cAuxAr2007::~cAuxAr2007 ()
{
    // undo what the constructor did
    mAr.mLevel--;
    mAr.RawEndName(*this);
}


bool cAuxAr2007::Input()  const
{
   return mAr.Input();
}
bool cAuxAr2007::Tagged()   const
{
   return mAr.Tagged();
}
int  cAuxAr2007::NbNextOptionnal(const std::string & aTag)  const
{
   return mAr.NbNextOptionnal(aTag);
}

/*============================================================*/
/*                                                            */
/*          cIXml_Ar2007                                      */
/*                                                            */
/*============================================================*/

static const char * aXMLBeginCom = "<!--";
static const char * aXMLEndCom = "-->";
static const char * aXMLBeginCom2 = "<?";
static const char * aXMLEndCom2 = "?>";


/// Class for recuperating End Of File error
class cXMLEOF
{
};


/// Xml read archive
/**
    An archive for reading XML file saved by MMVII with cOXml_Ar2007
    Probably the more complicated class for cAr2007
*/

class cIXml_Ar2007 : public cAr2007
{
     public :
          cIXml_Ar2007(std::string const  & aName) : 
                cAr2007     (true,true), // Input, Tagged
                mMMIs       (aName),
                mExcepOnEOF (false)
           {
           }

           bool IsFileOfFirstTag(bool Is2007,const std::string &);
     private :
           inline std::istream  & Ifs() {return mMMIs.Ifs();}

        // Inherited from cAr2007
           /// put <Tag>
           void RawBeginName(const cAuxAr2007& anOT) override;
           /// put </Tag>
           void RawEndName(const cAuxAr2007& anOT) override;
           void RawAddDataTerm(int &    anI) override;
           void RawAddDataTerm(size_t &    anI) override;
           void RawAddDataTerm(double &    anI) override;
           void RawAddDataTerm(std::string &    anI) override;
           void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
           void RawAddDataTerm(std::map<std::string,int>&    aRDS) override;
           /// Read next tag, if its what expected return 1, restore state of file
           int NbNextOptionnal(const std::string &) override;

           void Error(const std::string & aMes);

           std::string GetNextString();

        // Utilitaire de manipulation 

           /// If found Skeep one extpected string, and indicate if it was found, 
           bool SkeepOneString(const char * aString);
           /// Skeep a comment
           bool SkeepCom();
           /// Skeep all series of space and comment
           int  SkeepWhite();
           /// Skeep one <!-- --> or <? ?>
           bool SkeepOneKindOfCom(const char * aBeg,const char * anEnd);
           /// Get one tag
           bool GetTag(bool close,const std::string & aName);

           /// Get a char, and check its not EOF, only access to mMMIs.get() in this class
           int GetNotEOF();


           cMMVII_Ifs                        mMMIs; ///< secured istream
           bool                              mExcepOnEOF; ///< Do We use exception on EOF
};

bool cIXml_Ar2007::IsFileOfFirstTag(bool Is2007,const std::string  & aName)
{
    bool aRes = false;
    mExcepOnEOF = true;
    try {
        aRes = ((!Is2007) || GetTag(false,TagMMVIISerial)) && GetTag(false,aName);
    }
    catch (cXMLEOF anE)
    {
        return false;
    }
    return aRes;
}

bool IsFileXmlOfGivenTag(bool Is2007,const std::string & aName,const std::string & aTag)
{
  if ((Postfix(aName,'.',true) != "xml") || (! ExistFile(aName)))
     return false;

  cIXml_Ar2007 aFile (aName);
  return aFile.IsFileOfFirstTag(Is2007,aTag);
}


void cIXml_Ar2007::RawAddDataTerm(size_t &    aSz) 
{
    FromS(GetNextString(),aSz);
}

void cIXml_Ar2007::RawAddDataTerm(int &    anI) 
{
    FromS(GetNextString(),anI);
}
void cIXml_Ar2007::RawAddDataTerm(double &    aD) 
{
    FromS(GetNextString(),aD);
}
void cIXml_Ar2007::RawAddDataTerm(std::string &    aS) 
{
    aS =   GetNextString();
}

void cIXml_Ar2007::RawAddDataTerm(cRawData4Serial  &    aRDS) 
{
   tU_INT1 * aPtr = static_cast<tU_INT1*>(aRDS.Adr());
   for (int aK=0 ; aK< aRDS.NbElem() ; aK++)
   {
       int aC1= FromHexaCode(GetNotEOF());
       int aC2= FromHexaCode(GetNotEOF());
       aPtr[aK] = aC1 * 16 + aC2;
   }
}

void cIXml_Ar2007::RawAddDataTerm(std::map<std::string,int>  &    aRDS)
{
   //std::map<char,int>::iterator aRDSit = aRDS.begin();
   //FromS(GetNextString(),aRDS);
   std::string aStr = aRDS.begin()->first;
   StdOut() << "eeeeeeeXML " << aStr <<  " " << aRDS.begin()->second << "\n";
   
   RawAddDataTerm(aStr);
   RawAddDataTerm(aRDS.begin()->second);



}

void  cIXml_Ar2007::RawBeginName(const cAuxAr2007& anOT) 
{
     bool GotTag =  GetTag(false,anOT.Name());
     MMVII_INTERNAL_ASSERT_always(GotTag,"cIXml_Ar2007 did not get entering tag=" +anOT.Name());
}


void  cIXml_Ar2007::RawEndName(const cAuxAr2007& anOT) 
{
     bool GotTag =  GetTag(true,anOT.Name());
     MMVII_INTERNAL_ASSERT_always(GotTag,"cIXml_Ar2007 did not get closing tag=" +anOT.Name());
}

std::string  cIXml_Ar2007::GetNextString()
{
    SkeepWhite();
    std::string aRes;
 
    int aC =  GetNotEOF();
     // Case string between " "
    if (aC=='"')
    {
        for(;;)
        {
            int aC= GetNotEOF();
            if (aC=='"')  // End of "
              return aRes;
            if (aC=='\\')  /* Maybe  \"  */
            {
                int aC2 = GetNotEOF();
                if (aC2=='"')   /*  really \"  */
                {
                   aRes+= aC2;
                }
                else   /*  no finaly just a  \somehtinh */
                {
                   Ifs().unget();
                   aRes+= aC;
                }
            }
            else
               aRes+= aC;
        }
    }


    while ((aC!='<') && (!std::isspace(aC)))
    {
       aRes += aC;
       aC =  GetNotEOF();
    }
    Ifs().unget(); // put back < or ' '  etc ..
    
    SkeepWhite();
    return aRes;
}


int cIXml_Ar2007::NbNextOptionnal(const std::string & aTag) 
{
    std::streampos  aPos = Ifs().tellg();
    bool GotTag = GetTag(false,aTag);
    Ifs().seekg(aPos);

    return GotTag ? 1 : 0;
}


bool cIXml_Ar2007::GetTag(bool aClose,const std::string & aName)
{
    SkeepWhite();
    std::string aTag = std::string(aClose ? "</" : "<") + aName + ">";
   
    return SkeepOneString(aTag.c_str());
}




void cIXml_Ar2007::Error(const std::string & aMesLoc)
{
    std::string aMesGlob =   aMesLoc + "\n" 
                           + "while processing file=" +  mMMIs.Name() + " at char " + ToS(int(Ifs().tellg()));

    MMVII_INTERNAL_ASSERT_bench(false,aMesGlob);
}

bool cIXml_Ar2007::SkeepOneString(const char * aString)
{
     int aNbC=0;
     while (*aString)
     {
         // int aC = Ifs().get();
         int aC = GetNotEOF();
         aNbC++;
         if (aC != *aString)
         {
             for (int aK=0 ; aK<aNbC ; aK++)
             {
                  Ifs().unget();
             }
             return false;
         }
         ++aString;
     }
     return true;
}

int cIXml_Ar2007::GetNotEOF()
{
   int aC = Ifs().get();
   if (aC==EOF)
   {
       if (mExcepOnEOF)
          throw cXMLEOF();
       else
          Error("Unexpected EOF");
   }
   return aC;
}

bool  cIXml_Ar2007::SkeepOneKindOfCom(const char * aBeg,const char * anEnd)
{
   if (! SkeepOneString(aBeg))
      return false;

   while (! SkeepOneString(anEnd))
   {
        GetNotEOF();
   }
   return true;
}


bool  cIXml_Ar2007::SkeepCom()
{
    return    SkeepOneKindOfCom(aXMLBeginCom,aXMLEndCom)
           || SkeepOneKindOfCom(aXMLBeginCom2,aXMLEndCom2);
}


int cIXml_Ar2007::SkeepWhite()
{
   int aC=' ';
   while (isspace(aC)|| (aC==0x0A)) // Apparement 0x0A est un retour chariot
   {
       while (SkeepCom());
       // aC = Ifs().get();
       aC = GetNotEOF();
   }
   Ifs().unget();
   return aC;
}


/*============================================================*/
/*                                                            */
/*          cOXml_Ar2007                                      */
/*                                                            */
/*============================================================*/

/// Xml write archive
/**
    An archive for reading XML file saved by MMVII with cOXml_Ar2007
    Much easier than reading ...
*/
class cOXml_Ar2007 : public cAr2007
{
     public :
          cOXml_Ar2007(const std::string & aName) ;
          inline std::ostream  & Ofs() {return mMMOs.Ofs();}
          ~cOXml_Ar2007();

     private :
         void Indent(); ///< add white correspond to xml level

         void RawBeginName(const cAuxAr2007& anOT)  override; ///< Put opening tag
         void RawEndName(const cAuxAr2007& anOT)  override;  ///< Put closing tag
         void RawAddDataTerm(int &    anI)  override;  ///< write int in text
         void RawAddDataTerm(size_t &    anI) override;
         void RawAddDataTerm(double &    anI)  override;  ///< write double in text
         void RawAddDataTerm(std::string &    anI)  override; // write string
         void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
         void Separator() override;  ///< put a ' ' between field of final non atomic type

         cMMVII_Ofs     mMMOs;  ///< secure oftsream to write values
         bool mXTerm;  ///< mXTerm is activated by RawAdds.. , it allow to put values on the same line
         bool mFirst;  ///< new line is done before <tag> or </tag>, mFirst is used to avoid at first one
};


cOXml_Ar2007::cOXml_Ar2007(const std::string & aName) : 
   cAr2007(false,true),  // Output, Tagged
   mMMOs(aName,false), 
   mXTerm (false), 
   mFirst(true) 
{
   mMMOs.Ofs().precision(15);
   // Not sure all that is usefull, however, untill now I skipp <? ?>
   mMMOs.Ofs() <<  "<?xml" 
               << " version=\"1.0\""
               << " encoding=\"ISO8859-1\"" 
               << " standalone=\"yes\"" 
               << " ?>" << std::endl;
}


cOXml_Ar2007::~cOXml_Ar2007()
{
   Ofs()  << std::endl;
}

void cOXml_Ar2007::RawAddDataTerm(size_t &    aSz) {Ofs() <<aSz; mXTerm=true;}
void cOXml_Ar2007::RawAddDataTerm(int &    anI) {Ofs() <<anI; mXTerm=true;}
void cOXml_Ar2007::RawAddDataTerm(double &  aD) {Ofs() <<aD; mXTerm=true;}
void cOXml_Ar2007::RawAddDataTerm(std::string &  anS) 
{  
    // To allow white in string, put it between ""
    Ofs() << '"' <<anS << '"'; mXTerm=true;
}


void cOXml_Ar2007::RawAddDataTerm(cRawData4Serial  &    aRDS) 
{
   tU_INT1 * aPtr = static_cast<tU_INT1*>(aRDS.Adr());
   for (int aK=0 ; aK< aRDS.NbElem() ; aK++)
   {
       int aICar = aPtr[aK];
       Ofs() << ToHexacode(aICar/16) << ToHexacode(aICar%16) ;
   }
   mXTerm=true;
}

void cOXml_Ar2007::Separator() {Ofs() << ' ';}

void cOXml_Ar2007::Indent()
{
     for (int aK=0 ; aK<mLevel ; aK++)
         Ofs()  << "   ";
}

void cOXml_Ar2007::RawBeginName(const cAuxAr2007& anOT)
{
    if (!mFirst) Ofs()  << std::endl;
    mFirst = false;
    Indent();
    Ofs()  << "<" << anOT.Name() << ">";
}

void cOXml_Ar2007::RawEndName(const cAuxAr2007& anOT)
{
    if (! mXTerm){  Ofs()  << std::endl; Indent(); }
    Ofs()  << "</" << anOT.Name() << ">";
    mXTerm = false;
}
/*============================================================*/
/*                                                            */
/*          cHashValue_Ar2007                                 */
/*                                                            */
/*============================================================*/

/// hashvalue  archive
/**
    An archive for writing hashing of a given value
*/
class  cHashValue_Ar2007 : public cAr2007
{
    public :
        cHashValue_Ar2007 (bool Ordered) :
            cAr2007   (false,false),  // Is Not Input, Tagged
            mHashKey  (0),
            mOrdered  (Ordered)
        {
        }
        void RawAddDataTerm(int &    anI)  override;
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(double &    anI)  override;
        void RawAddDataTerm(std::string &    anI)  override;
        void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
        void RawAddDataTerm(std::map<std::string,int>&   aRDS) override;
        size_t HashKey() const override {return mHashKey;}
        
    private :
         size_t     mHashKey;
         bool       mOrdered;
};

void cHashValue_Ar2007::RawAddDataTerm(cRawData4Serial  &    aRDS)
{
   tU_INT1 * aPtr = static_cast<tU_INT1*>(aRDS.Adr());
   for (int aK=0 ; aK< aRDS.NbElem() ; aK++)
   {
       int aICar = aPtr[aK];
       RawAddDataTerm(aICar);
   }
}

void cHashValue_Ar2007::RawAddDataTerm(std::map<std::string,int>&   aRDS)
{
    std::string aTest = aRDS.begin()->first;
    StdOut() << "eeeeeeeHash " << aTest <<  " " << aRDS.begin()->second << "\n";
    RawAddDataTerm(aTest);
    RawAddDataTerm(aRDS.begin()->second);
}

void cHashValue_Ar2007::RawAddDataTerm(size_t &  aSz) 
{
    if (mOrdered)
       boost::hash_combine(mHashKey, aSz);
    else
       mHashKey ^= std::hash<size_t>()(aSz);
}

void cHashValue_Ar2007::RawAddDataTerm(int &    anI) 
{
    if (mOrdered)
       boost::hash_combine(mHashKey, anI);
    else
       mHashKey ^= std::hash<int>()(anI);
}

void cHashValue_Ar2007::RawAddDataTerm(double &    aD) 
{
    if (mOrdered)
       boost::hash_combine(mHashKey, aD);
    else
       mHashKey ^= std::hash<double>()(aD);
}

void cHashValue_Ar2007::RawAddDataTerm(std::string &    anS) 
{
    if (mOrdered)
       boost::hash_combine(mHashKey, anS);
    else 
       mHashKey ^= std::hash<std::string>()(anS);
}

cAr2007* AllocArHashVal(bool Ordered) {return new cHashValue_Ar2007(Ordered);}


/*============================================================*/
/*                                                            */
/*          cOBin_Ar2007                                      */
/*                                                            */
/*============================================================*/

/// binary write archive
/**
    An archive for writing binary file 
    No much more to do than descripe dumping of atomic type
*/
class  cOBin_Ar2007 : public cAr2007
{
    public :
        cOBin_Ar2007 (const std::string & aName) :
            cAr2007(false,false),  // Is Not Input, Tagged
            mMMOs  (aName,false)
        {
        }
        void RawAddDataTerm(tU_INT2 &    anI) override ; 
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(int &    anI)  override;
        void RawAddDataTerm(double &    anI)  override;
        void RawAddDataTerm(std::string &    anI)  override;
        void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
        
    private :
         cMMVII_Ofs     mMMOs;
};

void cOBin_Ar2007::RawAddDataTerm(size_t &    aSz) { mMMOs.Write(aSz); }
void cOBin_Ar2007::RawAddDataTerm(tU_INT2 &    anI) { mMMOs.Write(anI); }
void cOBin_Ar2007::RawAddDataTerm(int &    anI) { mMMOs.Write(anI); }
void cOBin_Ar2007::RawAddDataTerm(double &    anI) { mMMOs.Write(anI); }
void cOBin_Ar2007::RawAddDataTerm(std::string &    anI) { mMMOs.Write(anI); }

void cOBin_Ar2007::RawAddDataTerm(cRawData4Serial  &    aRDS) 
{
   mMMOs.VoidWrite(aRDS.Adr(),aRDS.NbElem());
}


/*============================================================*/
/*                                                            */
/*          cIBin_Ar2007                                      */
/*                                                            */
/*============================================================*/

/// binary read archive
/**
    An archive for reading binary file saved by MMVII with cOBin_Ar2007
    No much more to do than descripe undumping of atomic type
*/
class  cIBin_Ar2007 : public cAr2007
{
    public :
        cIBin_Ar2007 (const std::string & aName) :
            cAr2007(true,false),  // Input, Tagged
            mMMIs  (aName)
        {
        }
        int NbNextOptionnal(const std::string &) override;
        void RawAddDataTerm(int &    anI)  override;
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(double &    anI)  override;
        void RawAddDataTerm(std::string &    anI)  override;
        void RawAddDataTerm(cRawData4Serial  &    anI) override;
        void RawAddDataTerm(tU_INT2 &    anI) override ; 
        
    private :
         cMMVII_Ifs     mMMIs;
};

void cIBin_Ar2007::RawAddDataTerm(size_t &    aSz) { mMMIs.Read(aSz); }
void cIBin_Ar2007::RawAddDataTerm(tU_INT2 &    anI) { mMMIs.Read(anI); }
void cIBin_Ar2007::RawAddDataTerm(int &    anI) { mMMIs.Read(anI); }
void cIBin_Ar2007::RawAddDataTerm(double &    anI) { mMMIs.Read(anI); }
void cIBin_Ar2007::RawAddDataTerm(std::string &    anI) { mMMIs.Read(anI); }

int cIBin_Ar2007::NbNextOptionnal(const std::string &) 
{
   return mMMIs.TplRead<int>();
}

void cIBin_Ar2007::RawAddDataTerm(cRawData4Serial  &    aRDS) 
{
   mMMIs.VoidRead(aRDS.Adr(),aRDS.NbElem());
}

/*============================================================*/
/*                                                            */
/*       Global scope functions        MMVII::                */
/*                                                            */
/*============================================================*/

/**
   Implementation of AllocArFromFile. The type is fixed by string postfix ,
   but we need to know if its for input or for output

   Return a unique_ptr as it used in  SaveInFile/ReadInFile and destroy after
*/

// std::unique_ptr<cAr2007 >  AllocArFromFile(const std::string & aName,bool Input)
cAr2007 *  AllocArFromFile(const std::string & aName,bool Input)
{
   std::string aPost = Postfix(aName,'.',true);
// StdOut() << "AllocArFromFile, " << aName << " => " << aPost << "\n";
   cAr2007 * aRes = nullptr;

   if (UCaseEqual(aPost,PostF_XmlFiles))
   {
       if (Input)
          aRes =  new cIXml_Ar2007(aName);
       else
          aRes =  new cOXml_Ar2007(aName);
   }
   else if (UCaseEqual(aPost,PostF_DumpFiles) || UCaseEqual(aPost,"dat"))
   {
       if (Input)
          aRes =  new cIBin_Ar2007(aName);
       else
          aRes =  new cOBin_Ar2007(aName);
   }


   MMVII_INTERNAL_ASSERT_always(aRes!=0,"Do not handle postfix for "+ aName);
   return aRes;
}


/***********************************************************/


};

