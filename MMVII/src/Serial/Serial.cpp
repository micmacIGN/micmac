#include "include/MMVII_all.h"
#include <boost/algorithm/cxx14/equal.hpp>


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
template <class Type> bool EqualCont(const Type &aV1,const Type & aV2)
{
    return  boost::algorithm::equal(aV1.begin(),aV1.end(),aV2.begin(),aV2.end());
}

/* ========================================================= */
/*                                                           */
/*            cAr2007                                        */
/*                                                           */
/* ========================================================= */

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

    protected  :
         cAr2007(bool InPut,bool Tagged);
         int   mLevel;
         bool  mInput;
         bool  mTagged; 
     private  :

         /// This message is send before each data is serialized, tagged file put/read their opening tag here
         virtual void RawBeginName(const cAuxAr2007& anOT);
         /// This message is send each each data is serialized, tagged file put/read their closing tag here
         virtual void RawEndName(const cAuxAr2007& anOT);


      // Final atomic type for serialization
         virtual void RawAddDataTerm(int &    anI) =  0; ///< Heriting class descrine how they serialze int
         virtual void RawAddDataTerm(double &    anI) =  0; ///< Heriting class descrine how they serialze double
         virtual void RawAddDataTerm(std::string &    anI) =  0; ///< Heriting class descrine how they serialze string
      // Final non atomic type for serialization
         virtual void RawAddDataTerm(cPt2dr &    aP) ; ///< Default value should be OK ok
         virtual void Separator(); /**< Used in final but non atomic type, 
                                        for ex with Pt : in text separate x,y, in bin do nothing */
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

void cAr2007::Separator()
{
}
void cAr2007::RawAddDataTerm(cPt2dr &    aP) 
{
    RawAddDataTerm(aP.x());
    Separator();
    RawAddDataTerm(aP.y());
}

cAr2007::~cAr2007()
{
}

int cAr2007::NbNextOptionnal(const std::string &)
{
   MMVII_INTERNAL_ASSERT_always(!mInput,"Internal error, no cAr2007::NbNextOptionnal");
   return -1;
}

void AddData(const  cAuxAr2007 & anAux, int  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }
void AddData(const  cAuxAr2007 & anAux, double  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }
void AddData(const  cAuxAr2007 & anAux, std::string  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }
void AddData(const  cAuxAr2007 & anAux, cPt2dr  &  aVal) {anAux.Ar().TplAddDataTerm(anAux,aVal); }




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


/// Xml read archive
/**
    An archive for reading XML file saved by MMVII with cOXml_Ar2007
    Probably the more complicated class for cAr2007
*/

class cIXml_Ar2007 : public cAr2007
{
     public :
          cIXml_Ar2007(std::string const  & aName) : 
                cAr2007  (true,true), // Input, Tagged
                mMMIs    (aName)
           {
           }

     private :
           inline std::istream  & Ifs() {return mMMIs.Ifs();}

        // Inherited from cAr2007
           /// put <Tag>
           void RawBeginName(const cAuxAr2007& anOT) override;
           /// put </Tag>
           void RawEndName(const cAuxAr2007& anOT) override;
           void RawAddDataTerm(int &    anI) override;
           void RawAddDataTerm(double &    anI) override;
           void RawAddDataTerm(std::string &    anI) override;
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
};


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
         void RawAddDataTerm(double &    anI)  override;  ///< write double in text
         void RawAddDataTerm(std::string &    anI)  override; // write string
         void Separator() override;  ///< put a ' ' between field of final non atomic type

         cMMVII_Ofs     mMMOs;  ///< secure oftsream to write values
         bool mXTerm;  ///< mXTerm is activated by RawAdds.. , it allow to put values on the same line
         bool mFirst;  ///< new line is done before <tag> or </tag>, mFirst is used to avoid at first one
};


cOXml_Ar2007::cOXml_Ar2007(const std::string & aName) : 
   cAr2007(false,true),  // Output, Tagged
   mMMOs(aName), 
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

void cOXml_Ar2007::RawAddDataTerm(int &    anI) {Ofs() <<anI; mXTerm=true;}
void cOXml_Ar2007::RawAddDataTerm(double &  aD) {Ofs() <<aD; mXTerm=true;}
void cOXml_Ar2007::RawAddDataTerm(std::string &  anS) 
{  
    Ofs() <<anS; mXTerm=true;
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
            cAr2007(false,false),  // Output, Tagged
            mMMOs  (aName)
        {
        }
        void RawAddDataTerm(int &    anI)  override;
        void RawAddDataTerm(double &    anI)  override;
        void RawAddDataTerm(std::string &    anI)  override;
        
    private :
         cMMVII_Ofs     mMMOs;
};

void cOBin_Ar2007::RawAddDataTerm(int &    anI) { mMMOs.Write(anI); }
void cOBin_Ar2007::RawAddDataTerm(double &    anI) { mMMOs.Write(anI); }
void cOBin_Ar2007::RawAddDataTerm(std::string &    anI) { mMMOs.Write(anI); }


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
            cAr2007(true,false),  // Output, Tagged
            mMMIs  (aName)
        {
        }
        int NbNextOptionnal(const std::string &) override;
        void RawAddDataTerm(int &    anI)  override;
        void RawAddDataTerm(double &    anI)  override;
        void RawAddDataTerm(std::string &    anI)  override;
        
    private :
         cMMVII_Ifs     mMMIs;
};

void cIBin_Ar2007::RawAddDataTerm(int &    anI) { mMMIs.Read(anI); }
void cIBin_Ar2007::RawAddDataTerm(double &    anI) { mMMIs.Read(anI); }
void cIBin_Ar2007::RawAddDataTerm(std::string &    anI) { mMMIs.Read(anI); }

int cIBin_Ar2007::NbNextOptionnal(const std::string &) 
{
   return mMMIs.TplRead<int>();
}


/*============================================================*/
/*                                                            */
/*       Global scope functions        MMVII::                */
/*                                                            */
/*============================================================*/

/**
   Implementation of AllocArFromFile. The type is fixed by extension,
   but we need to know if its for input or for output

   Return a unique_ptr as it used in  SaveInFile/ReadInFile and destroy after
*/

std::unique_ptr<cAr2007 >  AllocArFromFile(const std::string & aName,bool Input)
{
   std::string aPost = Postfix(aName);
   cAr2007 * aRes = nullptr;

   if (aPost=="xml")
   {
       if (Input)
          aRes =  new cIXml_Ar2007(aName);
       else
          aRes =  new cOXml_Ar2007(aName);
   }
   else if ((aPost=="dmp") || (aPost=="dat"))
   {
       if (Input)
          aRes =  new cIBin_Ar2007(aName);
       else
          aRes =  new cOBin_Ar2007(aName);
   }

   MMVII_INTERNAL_ASSERT_always(aRes!=0,"Do not handle postfix for "+ aName);
   return std::unique_ptr<cAr2007 >(aRes);
}

/*
template<class Type> void  SaveInFile(const Type & aVal,const std::string & aName)
{
    std::unique_ptr<cAr2007 > anAr = AllocArFromFile(aName,false);

    cAuxAr2007  aGLOB(cAr2007::TagMMVIISerial,*anAr);
    // cAuxAr2007  anOpen(aTag,*anAr);
    AddData(aGLOB,const_cast<Type&>(aVal));
} 

template<class Type> void  ReadFromFile(Type & aVal,const std::string & aName)
{
    std::unique_ptr<cAr2007 > anAr = AllocArFromFile(aName,true);

    cAuxAr2007  aGLOB(cAr2007::TagMMVIISerial,*anAr);
    AddData(aGLOB,aVal);
} 
*/


/***********************************************************/

/// class to illustrate basic serialization

class cTestSerial0
{
     public :
        cTestSerial0() : 
             mP1 (1,2) , 
             mP2(3,3) 
        {
        }
        bool operator ==   (const cTestSerial0 & aT0) const {return (mP1==aT0.mP1) && (mP2==aT0.mP2);}
        cPt2dr mP1;
        cPt2dr mP2;
};

void f()
{
    cPt2dr aP;
    (aP==aP);
}

///  To serialize cTestSerial0, just indicate that it is made of mP1 and mP2

void AddData(const cAuxAr2007 & anAux, cTestSerial0 &    aTS0) 
{
    AddData(cAuxAr2007("P1",anAux),aTS0.mP1);
    AddData(cAuxAr2007("P2",anAux),aTS0.mP2);
}

///  a more complex class to illustrate serializaion
/**
    This class illustrate that there is no problem to use
  recursively the  serializain: once AddData has been defined
  in cTestSerial0 it can be used in AddData
*/
class cTestSerial1
{
     public :
        cTestSerial1() : 
             mS("Hello"), 
             mP3(3.1,3.2) ,
             mLI{1,22,333},
             mO2 (cPt2dr(100,1000))
        {
        }
        bool operator ==   (const cTestSerial1 & aT1) const 
        {
            return     (mTS0==aT1.mTS0) 
                    && (mS==aT1.mS) 
                    && (mP3==aT1.mP3) 
                    && (mO1==aT1.mO1) 
                    && (mO2==aT1.mO2)
                    && EqualCont(mLI,aT1.mLI)   ;
        }
        cTestSerial0            mTS0;
        std::string             mS;
        cPt2dr                  mP3;
        std::list<int>          mLI;
        boost::optional<cPt2dr> mO1;
        boost::optional<cPt2dr> mO2;


/*
        bool operator() == (const cTestSerial1 & aS)
        {
             return (mS==aTS.
        }
        /// Check a TS is OK, as all the object are the same  just check its value
        void Check(const cTestSerial1 & aS)
*/
};


void AddData(const cAuxAr2007 & anAux, cTestSerial1 &    aTS1) 
{
    AddData(cAuxAr2007("TS0",anAux),aTS1.mTS0);
    AddData(cAuxAr2007("S",anAux),aTS1.mS);
    AddData(cAuxAr2007("P3",anAux),aTS1.mP3);
    AddData(cAuxAr2007("LI",anAux),aTS1.mLI);
    OptAddData(anAux,"O1",aTS1.mO1);
    OptAddData(anAux,"O2",aTS1.mO2);
}


///  a class to illustrate flexibility in serialization
/**  This class illusrate that the serialization protocol
  is very flexible, in this class we save the mTS0.mP1 data
  field at the same xml-level 
*/

class cTestSerial2 : public cTestSerial1
{
};

void AddData(const cAuxAr2007 & anAux, cTestSerial2 &    aTS2) 
{
    AddData(cAuxAr2007("TS0:P1",anAux),aTS2.mTS0.mP1);
    AddData(cAuxAr2007("TS0:P2",anAux),aTS2.mTS0.mP2);
    AddData(cAuxAr2007("S",anAux),aTS2.mS);
    AddData(cAuxAr2007("P3",anAux),aTS2.mP3);
    AddData(cAuxAr2007("LI",anAux),aTS2.mLI);
    OptAddData(anAux,"O1",aTS2.mO1);
    OptAddData(anAux,"O2",aTS2.mO2);
}



class cAppli_MMVII_TestSerial : public cMMVII_Appli
{
     public :
        cAppli_MMVII_TestSerial(int argc,char** argv) ;
        int Exe();
};

cAppli_MMVII_TestSerial::cAppli_MMVII_TestSerial (int argc,char **argv) :
    cMMVII_Appli
    (
        argc,
        argv,
        DirCur(),
        cArgMMVII_Appli
        (
           mArgObl,
           mArgFac
        )
    )
{
}
           
int cAppli_MMVII_TestSerial::Exe()
{
    std::string aDir= DirCur();

    SaveInFile(cTestSerial1(),aDir+"F1.xml");

    cTestSerial1 aP12;
    ReadFromFile(aP12,aDir+"F1.xml");
    // Check the value read is the same
    MMVII_INTERNAL_ASSERT_bench(aP12==cTestSerial1(),"cAppli_MMVII_TestSerial");
    {
       // Check that == return false if we change a few
       cTestSerial1 aPModif = aP12;
       aPModif.mO1 = cPt2dr(14,18);
       MMVII_INTERNAL_ASSERT_bench(!(aPModif==cTestSerial1()),"cAppli_MMVII_TestSerial");
    }
    SaveInFile(aP12,aDir+"F2.xml");

    cTestSerial1 aP23;
    ReadFromFile(aP23,aDir+"F2.xml");
    SaveInFile(aP23,aDir+"F3.dmp");


    cTestSerial1 aP34;
    ReadFromFile(aP34,aDir+"F3.dmp");
    // Check dump value are preserved
    MMVII_INTERNAL_ASSERT_bench(aP34==cTestSerial1(),"cAppli_MMVII_TestSerial");
    SaveInFile(aP34,aDir+"F4.xml");


    SaveInFile(cTestSerial2(),aDir+"F_T2.xml");
    cTestSerial2 aT2;
    // Generate an error
    if (0)
      ReadFromFile(aT2,aDir+"F2.xml");
    ReadFromFile(aT2,aDir+"F_T2.xml"); // OK , read what we wrote as usual
    // and the value is the same
    MMVII_INTERNAL_ASSERT_bench(aT2==cTestSerial1(),"cAppli_MMVII_TestSerial");
    
    ReadFromFile(aT2,aDir+"F3.dmp");   // OK also in binary, the format has no influence
    // And the value is still the same as dump is compatible at binary level
    MMVII_INTERNAL_ASSERT_bench(aT2==cTestSerial1(),"cAppli_MMVII_TestSerial");

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_MMVII_TestSerial(int argc,char ** argv)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_TestSerial(argc,argv));
}

cSpecMMVII_Appli  TheSpec_TestSerial
(
     "TestSerial",
      Alloc_MMVII_TestSerial,
      "This command execute some experiments en home made serrialization",
      "Test",
      "None",
      "Console"
);


};

