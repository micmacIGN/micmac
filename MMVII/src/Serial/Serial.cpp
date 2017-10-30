#include "include/MMVII_all.h"

// #include<list>
// #include<boost/optional.hpp>


namespace MMVII
{

/* ========================================================= */
/*                                                           */
/*            cAr2007                                        */
/*                                                           */
/* ========================================================= */

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

         virtual void RawBeginName(const cAuxAr2007& anOT);
         virtual void RawEndName(const cAuxAr2007& anOT);
         virtual void RawAddDataTerm(int &    anI) =  0;
         virtual void RawAddDataTerm(double &    anI) =  0;
         virtual void RawAddDataTerm(std::string &    anI) =  0;

         virtual void RawAddDataTerm(cPt2dr &    aP) ;
         virtual void Separator();
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
    mAr.RawBeginName(*this);
    mAr.mLevel++;
}

cAuxAr2007::cAuxAr2007 (const std::string & aName, const cAuxAr2007 & anAux) :
    cAuxAr2007(aName,anAux.mAr)
{
}

cAuxAr2007::~cAuxAr2007 ()
{
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
           void RawBeginName(const cAuxAr2007& anOT) override;
           void RawEndName(const cAuxAr2007& anOT) override;
           void RawAddDataTerm(int &    anI) override;
           void RawAddDataTerm(double &    anI) override;
           void RawAddDataTerm(std::string &    anI) override;
           // void RawAddDataTerm(cPt2dr &    aP) override;
           int NbNextOptionnal(const std::string &) override;

           void Error(const std::string & aMes);

           std::string GetNextString();

        // Utilitaire de manipulation 
           int  SkeepWhite();
           // Skeep all comment
           bool SkeepCom();
           // Skeep one <!-- --> or <? ?>
           bool SkeepOneKindOfCom(const char * aBeg,const char * anEnd);
           // Skeep one extpected string
           bool SkeepOneString(const char * aString);
           bool GetTag(bool close,const std::string & aName);
           int GetNotEOF();


           // std::unique_ptr<std::istream>     mPtrIS;
           cMMVII_Ifs                        mMMIs;
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
         int aC = Ifs().get();
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
       aC = Ifs().get();
   }
   Ifs().unget();
   return aC;
}


/*============================================================*/
/*                                                            */
/*          cOXml_Ar2007                                      */
/*                                                            */
/*============================================================*/

class cOXml_Ar2007 : public cAr2007
{
     public :
          cOXml_Ar2007(const std::string & aName) : 
                cAr2007(false,true),  // Output, Tagged
                mMMOs(aName), 
                mXTerm (false), 
                mFirst(true) 
           {
               mMMOs.Ofs().precision(15);
           }
           inline std::ostream  & Ofs() {return mMMOs.Ofs();}
           ~cOXml_Ar2007();

     private :
         void Indent();

         void RawBeginName(const cAuxAr2007& anOT)  override;
         void RawEndName(const cAuxAr2007& anOT)  override;
         void RawAddDataTerm(int &    anI)  override;
         void RawAddDataTerm(double &    anI)  override;
         void RawAddDataTerm(std::string &    anI)  override;
         void Separator() override;
         cMMVII_Ofs     mMMOs;
         bool mXTerm;  /// mXTerm is activated by RawAdds.. , it allow to put values on the same line
         bool mFirst;  /// new line is done before <tag> or </tag>, mFirst is used to avoid at first one
};

cOXml_Ar2007::~cOXml_Ar2007()
{
   Ofs()  << std::endl;
}

void cOXml_Ar2007::RawAddDataTerm(int &    anI) {Ofs() <<anI; mXTerm=true;}
void cOXml_Ar2007::RawAddDataTerm(double &  aD) {Ofs() <<aD; mXTerm=true;}
void cOXml_Ar2007::RawAddDataTerm(std::string &  anS) {Ofs() <<anS; mXTerm=true;}

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

class c2P
{
     public :
        c2P() : 
             mP1 (1,2) , 
             mP2(3,3) ,
             mS("Hello"), 
             mP3(3.1,3.2) ,
             mLI{1,22,333},
             mO2 (cPt2dr(100,1000))
        {
        }
        cPt2dr mP1;
        cPt2dr mP2;
        std::string mS;
        cPt2dr mP3;
        std::list<int> mLI;
        boost::optional<cPt2dr> mO1;
        boost::optional<cPt2dr> mO2;
};



void AddData(const cAuxAr2007 & anAux, c2P &    a2P) 
{
    AddData(cAuxAr2007("P1",anAux),a2P.mP1);
    AddData(cAuxAr2007("P2",anAux),a2P.mP2);
    AddData(cAuxAr2007("S",anAux),a2P.mS);
    AddData(cAuxAr2007("P3",anAux),a2P.mP3);
    AddData(cAuxAr2007("LI",anAux),a2P.mLI);
    OptAddData(anAux,"O1",a2P.mO1);
    OptAddData(anAux,"O2",a2P.mO2);
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
        )
    )
{
}
           
int cAppli_MMVII_TestSerial::Exe()
{

    SaveInFile(c2P(),"F1.xml");

    c2P aP12;
    ReadFromFile(aP12,"F1.xml");
    SaveInFile(aP12,"F2.xml");

    c2P aP23;
    ReadFromFile(aP23,"F2.xml");
    SaveInFile(aP23,"F3.dmp");


    c2P aP34;
    ReadFromFile(aP34,"F3.dmp");
    SaveInFile(aP34,"F4.xml");


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

