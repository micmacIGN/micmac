#include "cMMVII_Appli.h"
#include "Serial.h"

#include <boost/version.hpp>
#if BOOST_VERSION > 106700
#include <boost/container_hash/hash.hpp>
#else
#include <boost/functional/hash.hpp>
#endif

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

#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_MeasuresIm.h"
#include "MMVII_2Include_Serial_Tpl.h"

namespace MMVII
{

const std::string  StrElCont = "el";
const std::string  StrElMap = "Pair";

bool IsTagged(eTypeSerial aTypeS)
{
    return  (aTypeS==eTypeSerial::exml) || (aTypeS==eTypeSerial::exml2) || (aTypeS==eTypeSerial::ejson);
}



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


void cAr2007::AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux)
{
     AddData(cAuxAr2007("Nb",anAux),aNb);
}

void AddDataSizeCont(int& aNb,const cAuxAr2007 & anAux)
{
     anAux.Ar().AddDataSizeCont(aNb,anAux);
}


void cAr2007::AddComment(const std::string &){}


void AddComment(cAr2007 & anAr, const std::string & aString)
{
	anAr.AddComment(aString);
}
void AddSeparator(cAr2007 & anAr)
{
	anAr.Separator();
}



void cAr2007::RawBeginName(const cAuxAr2007& anOT) {}
void cAr2007::RawEndName(const cAuxAr2007& anOT) {}
bool cAr2007::Tagged() const {return mTagged;}
bool cAr2007::Input() const  {return mInput;}

cAr2007::cAr2007(bool Input,bool isTagged,bool isBinary) :
   mLevel   (0),
   mInput   (Input),
   mTagged  (isTagged),
   mBinary  (isBinary)
{
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

void AddData(const  cAuxAr2007 & anAux, size_t  &  aVal) {anAux.Ar().RawAddDataTerm(aVal); }
void AddData(const  cAuxAr2007 & anAux, int  &  aVal) {anAux.Ar().RawAddDataTerm(aVal); }
void AddData(const  cAuxAr2007 & anAux, double  &  aVal) {anAux.Ar().RawAddDataTerm(aVal); }
void AddData(const  cAuxAr2007 & anAux, std::string  &  aVal) {anAux.Ar().RawAddDataTerm(aVal); }
void AddData(const  cAuxAr2007 & anAux, cRawData4Serial  &  aVal) {anAux.Ar().RawAddDataTerm(aVal); }


void AddData(const  cAuxAr2007 & anAux, tREAL4  &  aVal) { anAux.Ar().TplAddDataTermByCast(anAux,aVal,(double*)nullptr); }

void AddData(const  cAuxAr2007 & anAux, tINT1  &  aVal) 
{ 
     anAux.Ar().TplAddDataTermByCast(anAux,aVal,(int*)nullptr); 
}
void AddData(const  cAuxAr2007 & anAux, tINT2  &  aVal) { anAux.Ar().TplAddDataTermByCast(anAux,aVal,(int*)nullptr); }
void AddData(const  cAuxAr2007 & anAux, tU_INT1  &  aVal) { anAux.Ar().TplAddDataTermByCast(anAux,aVal,(int*)nullptr); }
void AddData(const  cAuxAr2007 & anAux, tU_INT2  &  aVal) { anAux.Ar().TplAddDataTermByCast(anAux,aVal,(int*)nullptr); }
void AddData(const  cAuxAr2007 & anAux, bool     &  aVal) { anAux.Ar().TplAddDataTermByCast(anAux,aVal,(int*)nullptr); }




// void AddData(const  cAuxAr2007 & anAux, bool  &  aVal) {anAux.Ar().RawAddDataTerm(aVal); }

template <class Type> void AddTabData(const  cAuxAr2007 & anAux, Type *  aVD,int aNbVal)
{
    // A precaution, probably it work but need to test
    MMVII_INTERNAL_ASSERT_always(aNbVal,"Not Sur AddTabData work for NbVal=0, check....");
    anAux.Ar().OnBeginTab();
    
// StdOut() << "AddTabDataAddTabData\n";
    anAux.SetType(eTAAr::eFixTabNum);


    if (aNbVal)
       AddData(anAux,aVD[0]);
    for (int aK=1 ; aK<aNbVal ; aK++)
    {
        anAux.Ar().Separator();
        AddData(anAux,aVD[aK]);
    }
    anAux.Ar().OnEndTab();
}

template void AddTabData(const  cAuxAr2007 & anAux, size_t *  aVD,int aNbVal);
template void AddTabData(const  cAuxAr2007 & anAux, tREAL8 *  aVD,int aNbVal);
template void AddTabData(const  cAuxAr2007 & anAux, tREAL4 *  aVD,int aNbVal);

void cHomogCpleIm::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::AddData(anAux,mP1.x());
       anAux.Ar().Separator();
     MMVII::AddData(anAux,mP1.y());
       anAux.Ar().Separator();
     MMVII::AddData(anAux,mP2.x());
       anAux.Ar().Separator();
     MMVII::AddData(anAux,mP2.y());
}
void AddData(const  cAuxAr2007 & anAux,cHomogCpleIm & aCple)  {aCple.AddData(anAux);}



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


template  void AddData(const  cAuxAr2007 & anAux, cPtxd<tREAL8,4>  &  aVal) ;

#define MACRO_INSTANTIATE_AddDataPtxD(DIM)\
template  void AddData(const  cAuxAr2007 & anAux, cPtxd<tREAL4,DIM>  &  aVal) ;\
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

cAuxAr2007::cAuxAr2007 (const std::string & aName,cAr2007 & aTS2,eTAAr aType) :
    mName    (aName),
    mAr      (aTS2),
    mType    (aType)
{
    mAr.RawBeginName(*this);  // Indicate an opening tag
    mAr.mLevel++;             // Incremente the call level for indentatio,
}

cAuxAr2007::cAuxAr2007 (const std::string & aName, const cAuxAr2007 & anAux,eTAAr aType) :
    cAuxAr2007(aName,anAux.mAr,aType)
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

void cAuxAr2007::SetType(eTAAr aType) const
{
     (const_cast<cAuxAr2007*>(this))->mType = aType;
}

eTAAr cAuxAr2007::Type() const {return mType;}

/*============================================================*/
/*                                                            */
/*          cStreamIXml_Ar2007                                */
/*                                                            */
/*============================================================*/


/// Xml read archive
/**
    An archive for reading XML file saved by MMVII with cOXml_Ar2007
    Probably the more complicated class for cAr2007
*/

// class 

class cStreamIXml_Ar2007 : public cAr2007,
	                   public cXmlSerialTokenParser
{
     public :
          cStreamIXml_Ar2007(const std::string & aName,eTypeSerial aTypeS) : 
                cAr2007        (true,(aTypeS!=eTypeSerial::etxt),false), // Input, Tagged,Binary
                cXmlSerialTokenParser   (aName)
           {
           }

           bool IsFileOfFirstTag(bool Is2007,const std::string &);
     protected :

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
           /// Read next tag, if its what expected return 1, restore state of file
           int NbNextOptionnal(const std::string &) override;


           /// Get one tag
           bool GetTag(bool close,const std::string & aName);
	   
	  /// retunr string after skeep whit, comm .... accept "a b c" , 
          std::string  GetNextStdString();

        // Utilitaire de manipulation 


};

bool cStreamIXml_Ar2007::GetTag(bool aClose,const std::string & aName)
{
    SkeepWhite();
    std::string aTag = std::string(aClose ? "</" : "<") + aName + ">";
   
    return SkeepOneString(aTag.c_str());
}

std::string  cStreamIXml_Ar2007::GetNextStdString()
{
	return GetNextLex().mVal;
}



bool cStreamIXml_Ar2007::IsFileOfFirstTag(bool Is2007,const std::string  & aNameTag)
{

    bool aRes = false;
    try {
        aRes = ((!Is2007) || GetTag(false,TagMMVIISerial)) && GetTag(false,aNameTag);
    }
    catch (cEOF_Exception anE)
    {
        return false;
    }
    return aRes;
}

bool IsFileXmlOfGivenTag(bool Is2007,const std::string & aNameFile,const std::string & aNameTag)
{
  cSerialFileParser::TestFirstTag(aNameFile);

  if ((Postfix(aNameFile,'.',true) != "xml") || (! ExistFile(aNameFile)))
     return false;

  cStreamIXml_Ar2007 aFile (aNameFile,eTypeSerial::exml);
  return aFile.IsFileOfFirstTag(Is2007,aNameTag);
}


void cStreamIXml_Ar2007::RawAddDataTerm(size_t &    aSz) 
{
    FromS(GetNextStdString(),aSz);
}

void cStreamIXml_Ar2007::RawAddDataTerm(int &    anI) 
{
    FromS(GetNextStdString(),anI);
}
void cStreamIXml_Ar2007::RawAddDataTerm(double &    aD) 
{
    FromS(GetNextStdString(),aD);
}
void cStreamIXml_Ar2007::RawAddDataTerm(std::string &    aS) 
{
    aS =   GetNextStdString();
}

void cStreamIXml_Ar2007::RawAddDataTerm(cRawData4Serial  &    aRDS) 
{
   SkeepWhite();
   tU_INT1 * aPtr = static_cast<tU_INT1*>(aRDS.Adr());
   for (int aK=0 ; aK< aRDS.NbElem() ; aK++)
   {
       int aC1= FromHexaCode(GetNotEOF());
       int aC2= FromHexaCode(GetNotEOF());
       aPtr[aK] = aC1 * 16 + aC2;
   }
}


void  cStreamIXml_Ar2007::RawBeginName(const cAuxAr2007& anOT) 
{
     bool GotTag =  GetTag(false,anOT.Name());
     MMVII_INTERNAL_ASSERT_always(GotTag,"cStreamIXml_Ar2007 did not get entering tag=" +anOT.Name());
}


void  cStreamIXml_Ar2007::RawEndName(const cAuxAr2007& anOT) 
{
     bool GotTag =  GetTag(true,anOT.Name());
     MMVII_INTERNAL_ASSERT_always(GotTag,"cStreamIXml_Ar2007 did not get closing tag=" +anOT.Name());
}



int cStreamIXml_Ar2007::NbNextOptionnal(const std::string & aTag) 
{
    std::streampos  aPos = Ifs().tellg();
    bool GotTag = GetTag(false,aTag);
    Ifs().seekg(aPos);

    return GotTag ? 1 : 0;
}



/*============================================================*/
/*                                                            */
/*          cIBaseTxt_Ar2007                                  */
/*                                                            */
/*============================================================*/



class cIBaseTxt_Ar2007 : public cStreamIXml_Ar2007
{
     public :
        cIBaseTxt_Ar2007(std::string const  & aName) : 
		  cStreamIXml_Ar2007 (aName,eTypeSerial::etxt)
	{
	}
        void RawBeginName(const cAuxAr2007& anOT) override {}
        void RawEndName(const cAuxAr2007& anOT) override {}
        int NbNextOptionnal(const std::string &) override
	{
               return cStrIO<int>::FromStr(GetNextStdString());
	}
     private :
};




/*============================================================*/
/*                                                            */
/*          cOBaseTxt_Ar2007                                  */
/*                                                            */
/*============================================================*/

/**  Class for write/seriliaztion in text format, use to write
 * readable data with minimal syntax
 *
 *  Also used as base class to implement XML
 * */

class cOBaseTxt_Ar2007 : public cAr2007
{
     public :
	     cOBaseTxt_Ar2007(const std::string & aName,eTypeSerial aTypeS) ;
             inline std::ostream  & Ofs() {return mMMOs.Ofs();}
	     ~cOBaseTxt_Ar2007();
     protected :
        void DoIndent(); ///< add white correspond to xml level
        void Separator() override;  ///< put a ' ' between field of final non atomic type
				    //
         void RawBeginName(const cAuxAr2007& anOT)  override; ///< Put opening tag
         void RawEndName(const cAuxAr2007& anOT)  override;  ///< Put closing tag
							     //
	virtual std::string StrIndent() const;

	virtual void BDT() {} // Begin Data Term
        void RawAddDataTerm(int &    anI)  override;  ///< write int in text
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(double &    anI)  override;  ///< write double in text
        void RawAddDataTerm(std::string &    anI)  override; // write string
        void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
				    
        cMMVII_Ofs     mMMOs;  ///< secure oftsream to write values
        bool mXTerm;           ///< mXTerm is activated by RawAdds.. , it allow to put values on the same line
        bool mFirst;  ///< new line is done before <tag> or </tag>, mFirst is used to avoid at first one
		      
};

cOBaseTxt_Ar2007::~cOBaseTxt_Ar2007()
{
     // Ofs() << " ";
}

cOBaseTxt_Ar2007::cOBaseTxt_Ar2007(const std::string & aName,eTypeSerial aTypeS) : 
   cAr2007(false,(aTypeS!=eTypeSerial::etxt),false),  // Output, Tagged, Binary
   mMMOs(aName,false) ,
   mXTerm (false),
   mFirst(true) 
{
   mMMOs.Ofs().precision(15);
}

void cOBaseTxt_Ar2007::Separator() {Ofs() << ' ';}

void cOBaseTxt_Ar2007::RawAddDataTerm(size_t &    aSz) {BDT();Ofs() <<aSz; mXTerm=true;}
void cOBaseTxt_Ar2007::RawAddDataTerm(int &    anI) {BDT();Ofs() <<anI; mXTerm=true;}
void cOBaseTxt_Ar2007::RawAddDataTerm(double &  aD) {BDT();Ofs() <<aD; mXTerm=true;}
void cOBaseTxt_Ar2007::RawAddDataTerm(std::string &  anS) 
{  
	BDT();
    // To allow white in string, put it between ""
    Ofs() << '"' <<anS << '"'; mXTerm=true;
}

void cOBaseTxt_Ar2007::RawAddDataTerm(cRawData4Serial  &    aRDS) 
{
   BDT();
   tU_INT1 * aPtr = static_cast<tU_INT1*>(aRDS.Adr());
   for (int aK=0 ; aK< aRDS.NbElem() ; aK++)
   {
       int aICar = aPtr[aK];
       Ofs() << ToHexacode(aICar/16) << ToHexacode(aICar%16) ;
   }
   mXTerm=true;
}

std::string cOBaseTxt_Ar2007::StrIndent() const   { return " "; }

void cOBaseTxt_Ar2007::DoIndent()
{
     for (int aK=0 ; aK<mLevel ; aK++)
         Ofs()  << StrIndent();
}

void cOBaseTxt_Ar2007::RawBeginName(const cAuxAr2007& anOT)
{
    // if (mXTerm) Ofs()  << std::endl;
    // mFirst = false;
    // DoIndent();
}

void cOBaseTxt_Ar2007::RawEndName(const cAuxAr2007& anOT)
{
    if (mXTerm)  Ofs()  << std::endl; 
    mXTerm = false;
}

/*============================================================*/
/*                                                            */
/*          cOJSN_Ar2007                                      */
/*                                                            */
/*============================================================*/

class cOJSN_Ar2007 : public cOBaseTxt_Ar2007
{
     public :
        cOJSN_Ar2007(const std::string & aName) :
              cOBaseTxt_Ar2007 (aName,eTypeSerial::ejson),
	      mTabBegin (false)
	{
	}
         ~cOJSN_Ar2007()
	 {
             Ofs()  << std::endl;
	 }
         void Separator() override { Ofs()  << "";}
	 void OnBeginTab() override { mTabBegin= true;}
	 void OnEndTab() override { Ofs()  << "]";}

         void RawBeginName(const cAuxAr2007& anOT)  override; ///< Put opening tag
         void RawEndName(const cAuxAr2007& anOT)  override;  ///< Put closing tag
							     //
	 void BDT() override 
	 {
                Ofs()  << ",";
		if (mTabBegin) 
		{
			Ofs()  << "[";
			mTabBegin = false;
		}
	 }
	 std::string StrIndent() const override  {return "   ";}

	 bool mTabBegin;
};

void cOJSN_Ar2007::RawBeginName(const cAuxAr2007& anOT)
{
    if (!mFirst)
    {
	if (!mXTerm)
	{	
		Ofs() << ",";
	}
	Ofs()   << std::endl;
    }
    mFirst = false;
    DoIndent();
    Ofs()  << "[\"" << anOT.Name() << "\"";
}

void cOJSN_Ar2007::RawEndName(const cAuxAr2007& anOT)
{
    if (! mXTerm){  Ofs()  << std::endl; DoIndent(); }
    Ofs()  << "]";
    mXTerm = false;
}

/*============================================================*/
/*                                                            */
/*          cOXml_Ar2007                                      */
/*                                                            */
/*============================================================*/

/// Xml write archive
/**
    An archive for writing XML file 
    Much easier than reading ...
*/
class cOXml_Ar2007 : public cOBaseTxt_Ar2007
{
     public :
          cOXml_Ar2007(const std::string & aName) ;
          ~cOXml_Ar2007();

	 virtual void AddComment(const std::string &) override;
     protected :
	 std::string StrIndent() const override ;

         void RawBeginName(const cAuxAr2007& anOT)  override; ///< Put opening tag
         void RawEndName(const cAuxAr2007& anOT)  override;  ///< Put closing tag
};

void cOXml_Ar2007::AddComment(const std::string & aString) 
{
    mMMOs.Ofs() << "  " << TheXMLBeginCom  << aString << TheXMLEndCom;
}

cOXml_Ar2007::cOXml_Ar2007(const std::string & aName) : 
   cOBaseTxt_Ar2007 (aName,eTypeSerial::exml)
{
   // Not sure all that is usefull, however, untill now I skipp <? ?>
	/*
   mMMOs.Ofs() <<  "<?xml" 
               << " version=\"1.0\""
               << " encoding=\"ISO8859-1\"" 
               << " standalone=\"yes\"" 
               << " ?>" << std::endl;
	       */
   mMMOs.Ofs() <<  TheXMLHeader << std::endl;
}


cOXml_Ar2007::~cOXml_Ar2007()
{
   Ofs()  << std::endl;
}

std::string cOXml_Ar2007::StrIndent() const  { return "   "; }



void cOXml_Ar2007::RawBeginName(const cAuxAr2007& anOT)
{
    if (!mFirst) Ofs()  << std::endl;
    mFirst = false;
    DoIndent();
    Ofs()  << "<" << anOT.Name() << ">";
}

void cOXml_Ar2007::RawEndName(const cAuxAr2007& anOT)
{
    if (! mXTerm){  Ofs()  << std::endl; DoIndent(); }
    Ofs()  << "</" << anOT.Name() << ">";
    mXTerm = false;
}

/*============================================================*/
/*                                                            */
/*          cOXmlSpecif_Ar2007                                */
/*                                                            */
/*============================================================*/

class cOXmlSpecif_Ar2007 : public cOXml_Ar2007
{
     public :
          cOXmlSpecif_Ar2007(const std::string & aName) ;
          // inline std::ostream  & Ofs() {return mMMOs.Ofs();}
          ~cOXmlSpecif_Ar2007();
     private :
         void RawAddDataTerm(int &    anI)  override;  ///< write int in text
         void RawAddDataTerm(size_t &    anI) override;
         void RawAddDataTerm(double &    anI)  override;  ///< write double in text
         void RawAddDataTerm(std::string &    anI)  override; // write string
         void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
};

void cOXmlSpecif_Ar2007::RawAddDataTerm(int &    anI)          { Ofs() <<"int"   ; mXTerm=true;}
void cOXmlSpecif_Ar2007::RawAddDataTerm(size_t &    anI)       { Ofs() <<"size_t"; mXTerm=true;}
void cOXmlSpecif_Ar2007::RawAddDataTerm(double &    anI)       { Ofs() <<"real"  ; mXTerm=true;}
void cOXmlSpecif_Ar2007::RawAddDataTerm(std::string &    anI)  { Ofs() <<"string"; mXTerm=true;}
void cOXmlSpecif_Ar2007::RawAddDataTerm(cRawData4Serial & anI) { Ofs() <<"data"  ; mXTerm=true;}

cOXmlSpecif_Ar2007::~cOXmlSpecif_Ar2007(){}

cOXmlSpecif_Ar2007::cOXmlSpecif_Ar2007(const std::string & aName)  :
   cOXml_Ar2007(aName)
{
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
            cAr2007   (false,false,true),  // Is Not Input, Tagged, Binary
            mHashKey  (0),
            mOrdered  (Ordered)
        {
        }
        void RawAddDataTerm(int &    anI)  override;
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(double &    anI)  override;
        void RawAddDataTerm(std::string &    anI)  override;
        void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
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
            cAr2007(false,false,true),  // Is Not Input, Tagged,Binary
            mMMOs  (aName,false)
        {
        }
        // void RawAddDataTerm(tU_INT2 &    anI) override ; 
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(int &    anI)  override;
        void RawAddDataTerm(double &    anI)  override;
        void RawAddDataTerm(std::string &    anI)  override;
        void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
        
    private :
         cMMVII_Ofs     mMMOs;
};

void cOBin_Ar2007::RawAddDataTerm(size_t &    aSz) { mMMOs.Write(aSz); }
// void cOBin_Ar2007::RawAddDataTerm(tU_INT2 &    anI) { mMMOs.Write(anI); }
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
            cAr2007(true,false,true),  // Input, Tagged,Binary
            mMMIs  (aName)
        {
        }
        int NbNextOptionnal(const std::string &) override;
        void RawAddDataTerm(int &    anI)  override;
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(double &    anI)  override;
        void RawAddDataTerm(std::string &    anI)  override;
        void RawAddDataTerm(cRawData4Serial  &    anI) override;
        // void RawAddDataTerm(tU_INT2 &    anI) override ; 
        
    private :
         cMMVII_Ifs     mMMIs;
};

void cIBin_Ar2007::RawAddDataTerm(size_t &    aSz) { mMMIs.Read(aSz); }
// void cIBin_Ar2007::RawAddDataTerm(tU_INT2 &    anI) { mMMIs.Read(anI); }
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
       {
          // aRes =  new cStreamIXml_Ar2007(aName,eTypeSerial::exml);
          aRes =  Alloc_cIMakeTreeAr(aName,eTypeSerial::exml);
       }
       else
       {
          if (starts_with(FileOfPath(aName,false),PrefixSpecifXML))
             aRes =  new cOXmlSpecif_Ar2007(aName);
          else
             // aRes =  new cOXml_Ar2007(aName);
             aRes =  Alloc_cOMakeTreeAr(aName,eTypeSerial::exml);
       }
   }
   else if (UCaseEqual(aPost,PostF_DumpFiles) || UCaseEqual(aPost,"dat"))
   {
       if (Input)
          aRes =  new cIBin_Ar2007(aName);
       else
          aRes =  new cOBin_Ar2007(aName);
   }
   else if (UCaseEqual(aPost,"txt") )
   {
       if (Input)
       {
          aRes =  new cIBaseTxt_Ar2007(aName);
       }
       else
          aRes =  new cOBaseTxt_Ar2007(aName,eTypeSerial::etxt);
   }
   else if (UCaseEqual(aPost,"json") )
   {
       if (Input)
       {
          aRes =  Alloc_cIMakeTreeAr(aName,eTypeSerial::ejson);
       }
       else
          // aRes =  new cOJSN_Ar2007(aName);
          aRes =  Alloc_cOMakeTreeAr(aName,eTypeSerial::ejson);
   }
   else if (UCaseEqual(aPost,"tagt") )
   {
       if (Input)
       {
          // aRes =  new cIBaseTxt_Ar2007(aName);
       }
       else
          // aRes =  new cOJSN_Ar2007(aName);
          aRes =  Alloc_cOMakeTreeAr(aName,eTypeSerial::etagt);
   }
   else if (UCaseEqual(aPost,E2Str(eTypeSerial::exml2)))
   {
       if (Input)
       {
          //  aRes =  Alloc_cIMakeTreeAr(aName,eTypeSerial::exml2);
          aRes =  new cStreamIXml_Ar2007(aName,eTypeSerial::exml2);
       }
       else
       {
          // aRes =  Alloc_cOMakeTreeAr(aName,eTypeSerial::exml2);
          aRes =  new cOXml_Ar2007(aName);
       }
   }

   MMVII_INTERNAL_ASSERT_always(aRes!=0,"Do not handle postfix for "+ aName);
   return aRes;
}


/***********************************************************/


};

