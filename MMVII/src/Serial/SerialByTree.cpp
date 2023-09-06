#include "MMVII_Stringifier.h"
#include "Serial.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Class4Bench.h"



/** \file SerialByTree.cpp
    \brief Implementation of serialisation using a tree represention, instead
    of a streaming.

    Streaming if more efficient for big data, for example when used for exchanging data via binary file
    or unstructurd text file, but is more complicated 

    In SerialByTree we firt create a universall tree representation and then of the data and then
    transform

*/


namespace MMVII
{


const char * TheXMLBeginCom  = "<!--";
const char * TheXMLEndCom    = "-->";
const char * TheXMLBeginCom2 = "<?";
const char * TheXMLEndCom2   = "?>";
const char * TheXMLHeader = "<?xml version=\"1.0\" encoding=\"ISO8859-1\" standalone=\"yes\" ?>";

static const char *  FakeTopSerialTree = "FakeTopSerialTree";
// static const std::string  JSonComment =  "<!--comment-->" ;



int DecLevel(eLexP aLexP)
{
    switch (aLexP)
    {
       case eLexP::eUp    : return +1;
       case eLexP::eDown  : return -1;
       default :          return 0;
    }
}

/*============================================================*/
/*                                                            */
/*             cResLex                                        */
/*                                                            */
/*============================================================*/

cResLex::cResLex(std::string aVal,eLexP aLexP,eTAAr aTAAR) :
    mVal  (aVal),
    mLexP (aLexP),
    mTAAr (aTAAR)
{
}

/*============================================================*/
/*                                                            */
/*             cSerialGenerator                               */
/*                                                            */
/*============================================================*/

cResLex cSerialGenerator::GetNextLexSizeCont() 
{
   cResLex aRes = GetNextLex();
   MMVII_INTERNAL_ASSERT_tiny(aRes.mLexP == eLexP::eSizeCont,"GetNextLexSizeCont");

   return aRes;
}

cResLex cSerialGenerator::GetNextLexNotSizeCont() 
{
   cResLex aRes = GetNextLex();
   while (aRes.mLexP == eLexP::eSizeCont)
        aRes = GetNextLex();

   return aRes;
}

void cSerialGenerator::CheckOnClose(const cSerialTree &,const std::string &)  const
{
}



/*============================================================*/
/*                                                            */
/*             cSerialFileParser                             */
/*                                                            */
/*============================================================*/



cSerialFileParser::cSerialFileParser(const std::string & aName,eTypeSerial aTypeS) :
   mMMIs          (aName),
   mTypeS         (aTypeS)
{
}

cSerialFileParser::~cSerialFileParser()
{
}


cSerialFileParser *  cSerialFileParser::Alloc(const std::string & aName,eTypeSerial aTypeS)
{
    switch (aTypeS) 
    {
         case eTypeSerial::exml  : return new  cXmlSerialTokenParser(aName);
         case eTypeSerial::exml2 : return new  cXmlSerialTokenParser(aName);
         case eTypeSerial::ejson : return new  cJsonSerialTokenParser(aName);
	 default : {}
    }

    MMVII_UnclasseUsEr("Bad enum for cSerialFileParser::Alloc");
    return nullptr;
}

tTestFileSerial  cSerialFileParser::TestFirstTag(const std::string & aNameFile)
{
    if (!ExistFile(aNameFile))
       return tTestFileSerial(false,"");


    eTypeSerial aTypeS = Str2E<eTypeSerial>(LastPostfix(aNameFile),true);

    if ((aTypeS!=eTypeSerial::exml) && (aTypeS!=eTypeSerial::ejson))
       return tTestFileSerial(false,"");

    cSerialFileParser * aSFP = Alloc(aNameFile,aTypeS);
    cSerialTree  aTree(*aSFP);
    delete aSFP;


    if (aTypeS==eTypeSerial::exml)
    {
       return aTree.Xml_TestFirstTag();
    }

    if (aTypeS==eTypeSerial::ejson)
       return aTree.Json_TestFirstTag();

    return tTestFileSerial(false,"");
}

bool IsFileGivenTag(bool Is2007,const std::string & aNameFile,const std::string & aTag)
{
   if (! Is2007) return IsXmlV1FileGivenTag(aNameFile,aTag);

   tTestFileSerial  aTF =  cSerialFileParser::TestFirstTag(aNameFile);

   return aTF.first && (aTF.second == aTag);
}



void cSerialFileParser::Error(const std::string & aMesLoc)
{
    std::string aMesGlob =   aMesLoc + "\n"
                           + "while processing file=" +  mMMIs.Name() + " at char " + ToS(int(Ifs().tellg()));

    MMVII_UnclasseUsEr(aMesGlob);
}

int cSerialFileParser::GetNotEOF()
{
   int aC = Ifs().get();
   if (aC==EOF)
   {
       if (true)
          throw cEOF_Exception();
       else
       {
          Error("Unexpected EOF");
       }
   }
   return aC;
}



bool cSerialFileParser::SkeepOneString(const char * aString)
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


bool  cSerialFileParser::SkeepOneKindOfCom(const char * aBeg,const char * anEnd)
{
   if (! SkeepOneString(aBeg))
      return false;

   while (! SkeepOneString(anEnd))
   {
        GetNotEOF();
   }
   return true;
}

bool  cSerialFileParser::SkeepCom()
{
    return    SkeepOneKindOfCom(TheXMLBeginCom,TheXMLEndCom)
           || SkeepOneKindOfCom(TheXMLBeginCom2,TheXMLEndCom2);
}

int cSerialFileParser::SkeepWhite()
{
   int aC=' ';
   while (isspace(aC)|| (aC==0x0A)) // Apparement 0x0A est un retour chariot
   {
       while (SkeepCom());
       //aC = Ifs().get();
       aC = GetNotEOF();
   }
   Ifs().unget();
   return aC;

 }

std::string  cSerialFileParser::GetQuotedString()
{
   std::string aRes;
   for(;;)
   {
            int aC= GetNotEOF();
            if (aC=='"')  // End of "
            {
              return aRes;
            }
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

cResLex  cSerialFileParser::GetNextLex_NOEOF()
{

    SkeepWhite();
    std::string aRes;

    int aC =  GetNotEOF();

    if (BeginPonctuation(aC))
    {
            return AnalysePonctuation(aC);
    }
    if (aC=='"')
    {
          return cResLex(GetQuotedString(),eLexP::eStdToken_String,eTAAr::eUndef);
    }


    while ((!BeginPonctuation(aC)) && (!std::isspace(aC)))
    {
       aRes += aC;
       aC =  GetNotEOF();
    }
    Ifs().unget(); // put back < or ' '  etc ..

    //if (mTypeS!= eTypeSerial::etxt)  // else get EOF at end
    //   SkeepWhite();

    return cResLex(aRes,eLexP::eStdToken_UK,eTAAr::eUndef);
}

cResLex  cSerialFileParser::GetNextLex()
{
    try
    {
         return GetNextLex_NOEOF();
    }
    catch (cEOF_Exception anE)
    {
         return cResLex("",eLexP::eEnd,eTAAr::eUndef);
    }

}

/*============================================================*/
/*                                                            */
/*             cXmlSerialTokenParser                          */
/*                                                            */
/*============================================================*/

cXmlSerialTokenParser::cXmlSerialTokenParser(const std::string & aName) :
	cSerialFileParser(aName,eTypeSerial::exml)
{
}

bool  cXmlSerialTokenParser::BeginPonctuation(char aC) const { return aC=='<'; }

cResLex cXmlSerialTokenParser::AnalysePonctuation(char aC)
{
    aC =  GetNotEOF();
    eLexP aLex= eLexP::eDown;
    std::string aRes ;

    if (aC!='/')
    {
        aLex= eLexP::eUp;
        aRes += aC;
    }

    while (aC!='>')
    {
         aC =  GetNotEOF();
         if (aC!='>')
            aRes += aC;
    }

    return cResLex(aRes,aLex,eTAAr::eUndef);
}

void cXmlSerialTokenParser::CheckOnClose(const cSerialTree & aTree,const std::string & aStr)  const
{
     MMVII_INTERNAL_ASSERT_tiny(aTree.Value() == aStr,"Close tag unexpected");
}

/*============================================================*/
/*                                                            */
/*             cJsonSerialTokenParser                         */
/*                                                            */
/*============================================================*/

cJsonSerialTokenParser::cJsonSerialTokenParser(const std::string & aName) :
	cSerialFileParser(aName,eTypeSerial::ejson)
{
}

const  std::string  cJsonSerialTokenParser::OpenCars = "{[";
const  std::string  cJsonSerialTokenParser::CloseCars = "}]";
const  std::string  cJsonSerialTokenParser::SepCars = ",:";


bool cJsonSerialTokenParser::BeginPonctuation(char aC) const 
{
     static std::string JSonPonct = OpenCars+CloseCars+SepCars;
     return  JSonPonct.find(aC) !=  std::string::npos;
}

cResLex cJsonSerialTokenParser::AnalysePonctuation(char aC)
{
    std::string aStr(1,aC);

    if (OpenCars.find(aC) != std::string::npos)   return cResLex(aStr,eLexP::eUp,eTAAr::eUndef);

    if (CloseCars.find(aC) != std::string::npos)  return cResLex(aStr,eLexP::eDown,eTAAr::eUndef);

    return cResLex(aStr,eLexP::eSep,eTAAr::eUndef);
}

void cJsonSerialTokenParser::CheckOnClose(const cSerialTree & aTree,const std::string & aStr)  const
{
     char aCOpen = aTree.Value().at(0);
     char aCClose = aStr.at(0);

     size_t aIndOpen  = OpenCars.find( aCOpen);
     size_t aIndClose = CloseCars.find(aCClose);

     // is this case occurs this probably a MMVII-error
     MMVII_INTERNAL_ASSERT_tiny((aIndOpen!=std::string::npos) && (aIndClose!=std::string::npos),"Bad close in JSon");

     // other  case are user error

     //  case open dont match close
     MMVII_INTERNAL_ASSERT_User((aIndOpen==aIndClose),eTyUEr::eParseBadClose,"Bad open/close : \"" + aTree.Value() +"/" + aStr + "\" ");


     size_t aSz = aTree.Sons().size();
     if (aIndOpen==0) // we are in  a stuff like "{ "a":1 , ...}" 
     {
	     // test punctuations
         MMVII_INTERNAL_ASSERT_User((aSz==0) || (aSz%4==3),eTyUEr::eJSonBadPunct,"Bad size inside a json \"{...}\"");

	 for (size_t aKS=0 ; aKS<aSz ; aKS+=4)
	 {
	     std::string aS = aTree.Sons().at(aKS+1).Value() ;
             MMVII_INTERNAL_ASSERT_User(aS==":",eTyUEr::eJSonBadPunct,"Got \""+ aS + "\" instead of \":\"");
	     if (aKS+3<aSz)
	     {
	          aS = aTree.Sons().at(aKS+3).Value() ;
                  MMVII_INTERNAL_ASSERT_User(aS==",",eTyUEr::eJSonBadPunct,"Got \""+ aS + "\" instead of \",\"");
	     }
	 }
     }
     else if (aIndOpen==1)
     {
         MMVII_INTERNAL_ASSERT_User((aSz==0) || (aSz%2==1),eTyUEr::eJSonBadPunct,"Bad size inside a json \"{...}\"");
	 for (size_t aKS=1 ; aKS<aSz ; aKS+=2)
	 {
             std::string aS = aTree.Sons().at(aKS).Value() ;
             MMVII_INTERNAL_ASSERT_User(aS==",",eTyUEr::eJSonBadPunct,"Got \""+ aS + "\" instead of \",\"");
	 }
     }
}


/*============================================================*/
/*                                                            */
/*                       cSerialTree                          */
/*                                                            */
/*============================================================*/

static bool DEBUG=false;

cSerialTree::cSerialTree(const std::string & aValue,int aDepth,eLexP aLexP,eTAAr aTAAr) :
   mLexP      (aLexP),
   mTAAr      (aTAAr),
   mValue     (aValue),
   mDepth     (aDepth),
   mMaxDSon   (aDepth),
   mLength    (mValue.size())
{
}


bool cSerialTree::IsTerminalNode() const
{
        if (mLexP == eLexP::eUp)   return false;
	return  mDepth == mMaxDSon;
}

bool cSerialTree::IsTab() const
{
    if (mLexP != eLexP::eUp)  
        return false;

    for (const auto & aSon : mSons)
        if (! aSon.IsTerminalNode())
           return false;

    return true;
}

bool cSerialTree::IsSingleTaggedVal() const
{
	return IsTab() && (mSons.size()==1);
}


void cSerialTree::UpdateMaxDSon()
{
    UpdateMax(mMaxDSon,mSons.back().mMaxDSon);
    mLength += mSons.back().mLength + 1;
}


const std::string & cSerialTree::Value() const { return mValue; }
const std::vector<cSerialTree>&  cSerialTree::Sons() const {return mSons;}

cSerialTree:: cSerialTree(cSerialGenerator & aGenerator) :
   cSerialTree (aGenerator,FakeTopSerialTree,0,eLexP::eBegin,eTAAr::eUndef) 
{
}

cSerialTree::cSerialTree(cSerialGenerator & aGenerator,const std::string & aValue,int aDepth,eLexP aLexP,eTAAr aTAAr) :
   mLexP    (aLexP),
   mTAAr    (aTAAr),
   mValue   (aValue),
   mDepth   (aDepth),
   mMaxDSon (aDepth),
   mLength    (mValue.size())
{
    for(;;)
    {
// if (DEBUG) StdOut() << "cSerialTree::cSerialTree " << __LINE__ << "\n";
        cResLex aRL= aGenerator.GetNextLex();
	const std::string& aStr = aRL.mVal;
	eLexP aLex  = aRL.mLexP;

	if (aLex==eLexP::eEnd)
	{
// if (DEBUG) StdOut() << "cSerialTree::cSerialTree " << __LINE__ << "\n";
	    if (aDepth!=0)
	    {
               MMVII_UnclasseUsEr("cSerialTree unexpected EOF");
	    }
	    return;
	}
// if (DEBUG) StdOut() << "cSerialTree::cSerialTree " << __LINE__ << "\n";
        int aDec =  DecLevel(aLex);
	if (aDec>0)
	{
	    // StdOut() << "HHHHH: " <<  mSons.size() << " ::  " <<  aStr << "\n";
            mSons.push_back(cSerialTree(aGenerator,aStr,aDepth+1,aLex,aRL.mTAAr));
	    //mSons.back().mValue = aStr;
	    UpdateMaxDSon();
	}
	else if (aDec<0)
	{
	     mComment = aRL.mComment;
	     mTAAr = aRL.mTAAr;
	     aGenerator.CheckOnClose(*this,aStr);
             return;
	}
	else
	{
            mSons.push_back(cSerialTree(aStr,aDepth+1,aLex,aRL.mTAAr));
	    UpdateMaxDSon();
	}
    }
}

void  cSerialTree::Indent(cMMVII_Ofs & anOfs,int aDeltaInd) const
{
      for (int aK=0 ; aK<(mDepth-1+aDeltaInd) ; aK++)  
          anOfs.Ofs() << "   ";
}


     //=======================    XLM PRINTING ======================


tTestFileSerial  cSerialTree::Xml_TestFirstTag()
{
  if (mSons.size() != 1)
  {
     return tTestFileSerial(false,"");
  }

  const cSerialTree & aS0 = UniqueSon();

  // StdOut() << "Xml_TestFirstTag::: " << aS0.mValue << " " << aS0.mSons.size() << "\n";
  if (     (aS0.mValue!= TagMMVIIRoot ) 
        || (aS0.mSons.size() !=3) 
	|| (aS0.mSons.at(0).mValue != TagMMVIIType)
	|| (aS0.mSons.at(1).mValue != TagMMVIIVersion)
     )
     return tTestFileSerial(false,"");

  const cSerialTree & aS1 = aS0.mSons.at(2);
  if ((aS1.mValue!= TagMMVIIData) || ( aS1.mSons.size() !=1)) 
     return tTestFileSerial(false,"");

  const cSerialTree & aS2 = aS1.UniqueSon() ;


  return tTestFileSerial(true,aS2.mValue);
}


void  cSerialTree::Xml_PrettyPrint(cMMVII_Ofs & anOfs) const
{
     for (const auto & aSon : mSons)
         aSon.Rec_Xml_PrettyPrint(anOfs);
}

void  cSerialTree::Rec_Xml_PrettyPrint(cMMVII_Ofs & anOfs) const
{
     // bool IsTag = (mDepth!=0) && (!mSons.empty());
     bool IsTag = (mLexP == eLexP::eUp);
     bool OneLine = (mMaxDSon <= mDepth+1);


     if (mDepth!=0)
     {
	if (IsTag)
	{
	    Indent(anOfs,0);
            anOfs.Ofs()  << "<" << mValue << ">";
	}
	else 
	{
           if (!OneLine)
	       Indent(anOfs,0);
           anOfs.Ofs()<< mValue ;
	}

	if (!OneLine)
            anOfs.Ofs()  << "\n";
     }
     int aK=0;
     for (const auto & aSon : mSons)
     {
        if (OneLine && (aK!=0))
            anOfs.Ofs()  << " ";
		
        aSon.Rec_Xml_PrettyPrint(anOfs);
	aK++;
     }
     if (IsTag && (mDepth!=0) )  // && (!mSons.empty()))
     {
        if (!OneLine)
	    Indent(anOfs,0);
        anOfs.Ofs()<< "</" <<  mValue   << ">";
	if (mComment!="")
	{
            anOfs.Ofs()  << " " << TheXMLBeginCom << mComment << TheXMLEndCom;
	}
        anOfs.Ofs()<< "\n";
     }
}
     //=======================    TAGT PRINTING ======================

tTestFileSerial  cSerialTree::Json_TestFirstTag()
{
  if (mSons.size() != 1)
  {
     return tTestFileSerial(false,"");
  }
  const cSerialTree & aS0 = UniqueSon();

  // StdOut() << "Json_TestFirstTag::: " << aS0.mValue << " " << aS0.mSons.size() << "\n";
  // StdOut() << "Xml_TestFirstTag::: " << aS0.mValue << " " << aS0.mSons.size() << "\n";
  if ((aS0.mValue!= "{") || ( aS0.mSons.size() !=11)) 
     return tTestFileSerial(false,"");

   if (    (aS0.mSons[0].mValue!=TagMMVIIType)
        || (aS0.mSons[2].mValue!=TagMMVIISerial)
        || (aS0.mSons[4].mValue!=TagMMVIIVersion)
        || (aS0.mSons[8].mValue!=TagMMVIIData)
        || (aS0.mSons[10].mValue!="{")
      )
     return tTestFileSerial(false,"");

  const cSerialTree & aS1 = aS0.mSons[10];
  if ( aS1.mSons.size() !=3)
     return tTestFileSerial(false,"");

  return tTestFileSerial(true,aS1.mSons[0].mValue);
}


void  cSerialTree::Raw_PrettyPrint(cMMVII_Ofs & anOfs) const
{
     // bool OneLine = (mMaxDSon <= mDepth+1);
      Indent(anOfs,0);
      anOfs.Ofs() <<   mValue ;

      anOfs.Ofs() << " [" << E2Str(mTAAr) << "] : " << mLength ;
      /*
      if (IsTerminalNode())  anOfs.Ofs() << " (terminal)";
      else if (IsSingleTaggedVal())  anOfs.Ofs() << " (SingleTaggeVal)";
      else if (IsTab())  anOfs.Ofs() << " (tab)";
      */
      anOfs.Ofs() << "\n";
      for (const auto & aSon : mSons)
      {
           aSon.Raw_PrettyPrint(anOfs);
      }
}

     //=======================    JSON PRINTING ======================


static constexpr int MaxLength = 30;

void  cSerialTree::Json_Comment(cMMVII_Ofs & anOfs,bool Last,int & aCpt) const
{
    if ( (! Json_OmitKey()) && (mComment!="") )
    // if (  (mComment!="") )
    {
        Indent(anOfs,1);
        anOfs.Ofs() <<  "           " ;
	if (Last)  anOfs.Ofs() << ",";
        // anOfs.Ofs() <<  " "<<  Quote(JSonComment)  <<  ":" << Quote(mComment);
        anOfs.Ofs() <<  " "<<  Quote(TagJsonComment(aCpt))  <<  ":" << Quote(mComment);
	if (!Last)  anOfs.Ofs() << ",";
        anOfs.Ofs() <<  "\n";
    }
}

bool cSerialTree::Json_OmitKey() const
{
      return  (mValue==StrElCont) || (mValue==StrElMap) ;
}


void  cSerialTree::Json_PrintTerminalNode(cMMVII_Ofs & anOfs,bool Last,int & aCpt) const
{
      Indent(anOfs,1);
      if (!Json_OmitKey())
         anOfs.Ofs() <<   Quote(mValue) << " :";
      anOfs.Ofs() << UniqueSon().mValue;
      if (! Last) anOfs.Ofs() <<  " ,";
      anOfs.Ofs() <<  "\n";
      Json_Comment(anOfs,Last,aCpt);
}

void  cSerialTree::Json_PrintSingleTab(cMMVII_Ofs & anOfs,bool Last,int & aCpt) const
{
          Indent(anOfs,1);
          if (!Json_OmitKey())
             anOfs.Ofs() <<   Quote(mValue) << " :" ;
          anOfs.Ofs() <<   "[" ;
	  int aK=0;
	  for (const auto & aSon : mSons)
	  {
               if (aK!=0) anOfs.Ofs() <<  " , ";
               anOfs.Ofs() <<  aSon.mValue;
	       aK++;
	  }
          anOfs.Ofs() <<   "]" ;
          if (! Last) anOfs.Ofs() <<  " ,";
          anOfs.Ofs() <<   "\n" ;
          Json_Comment(anOfs,Last,aCpt);
}

void  cSerialTree::Rec_Json_PrettyPrint(cMMVII_Ofs & anOfs,bool IsLast,int & aCpt) const
{
      bool IsCont = ((mTAAr ==  eTAAr::eCont) ||  (mTAAr==eTAAr::eMap));
      Indent(anOfs,1);
      anOfs.Ofs() <<   (IsCont ? "[" : "{")  << "\n";
      int aK = mSons.size();
      for (const auto & aSon : mSons)
      {
          aK--;
          if (aSon.IsSingleTaggedVal())
          {
              aSon.Json_PrintTerminalNode(anOfs,aK==0,aCpt);
          }
          else  if (aSon.IsTab())
          {
              aSon.Json_PrintSingleTab(anOfs,aK==0,aCpt);
          }
          else
          {
               if (!aSon.Json_OmitKey())
	       {
                  aSon.Indent(anOfs,1);
                  anOfs.Ofs() <<   Quote(aSon.mValue) << " :" ;
                  anOfs.Ofs() <<   "\n" ;
	       }
               aSon.Rec_Json_PrettyPrint(anOfs,aK==0,aCpt);
               Json_Comment(anOfs,aK==0,aCpt);
          }
      }
      Indent(anOfs,1);
      anOfs.Ofs() <<   (IsCont ? "]" : "}") ;
      if (!IsLast) anOfs.Ofs() <<   " ," ;
      anOfs.Ofs() <<   "\n" ;
      Json_Comment(anOfs,IsLast,aCpt);
}


void  cSerialTree::Json_PrettyPrint(cMMVII_Ofs & anOfs) const
{
	int aCpt=0;
	Rec_Json_PrettyPrint(anOfs,true,aCpt);
}

std::string cSerialTree::TagJsonComment(int & aCpt)
{
    aCpt++;
    return "<!--comment"+ToStr(aCpt) +  "-->";
}

bool cSerialTree::IsJsonComment(const std::string& aName) 
{
    cMemManager::SetActiveMemoryCount(false);
    static  tNameSelector aPat = AllocRegex("<!--comment[0-9]+-->");
    cMemManager::SetActiveMemoryCount(true);

    return aPat.Match(aName);
}




const cSerialTree & cSerialTree::UniqueSon() const
{
    MMVII_INTERNAL_ASSERT_tiny(mSons.size()==1,"cSerialTree::UniqueSon");

    return *(mSons.begin());
}

void cSerialTree::Unfold(std::list<cResLex> & aRes,eTypeSerial aTypeS) const
{

    if (aTypeS==eTypeSerial::exml)
    {
        aRes.push_back(cResLex(mValue,mLexP,mTAAr));
        if (! IsTerminalNode())
           aRes.push_back(cResLex(ToStr(mSons.size()),eLexP::eSizeCont,eTAAr::eSzCont));

        // parse son for recursive call
        for (const auto & aSon : mSons)
            aSon.Unfold(aRes,aTypeS);

        // add potentiel closing tag
        if (mLexP==eLexP::eUp)
            aRes.push_back(cResLex(mValue,eLexP::eDown,mTAAr));
    }
    else if (aTypeS==eTypeSerial::ejson)
    {
         size_t aSz = mSons.size();
	 if (mValue == "{")
	 {
            int aNbCom = 0;
            for (size_t aK=0 ; aK<mSons.size() ; aK+=4)
	    {
                //if ( mSons.at(aK).mValue==JSonComment)
                if ( IsJsonComment(mSons.at(aK).mValue))
                   aNbCom ++;
	    }

            aRes.push_back(cResLex(ToStr((aSz+1)/4-aNbCom),eLexP::eSizeCont,eTAAr::eSzCont));
            for (size_t aK=0 ; aK<mSons.size() ; aK+=4)
            {
                 const cSerialTree  & aSonTag =  mSons.at(aK);
		 const std::string & aTag = aSonTag.mValue;
		 // if (aTag != JSonComment)
                 if ( !IsJsonComment(aTag))
		 {
                     aRes.push_back(cResLex(aTag,eLexP::eUp,eTAAr::eStd));
                     mSons.at(aK+2).Unfold(aRes,aTypeS);
                     aRes.push_back(cResLex(aTag,eLexP::eDown,eTAAr::eStd));
		 }
            }
	 }
	 else if (mValue == "[")
	 {
            aRes.push_back(cResLex(ToStr((aSz+1)/2),eLexP::eSizeCont,eTAAr::eSzCont));
            for (size_t aK=0 ; aK<mSons.size() ; aK+=2)
            {
                mSons.at(aK).Unfold(aRes,aTypeS);
	    }
	 }
	 else
	 {
            MMVII_INTERNAL_ASSERT_tiny(mSons.empty(),"Unfold : non empty sons");
            aRes.push_back(cResLex(mValue,eLexP::eStdToken_UK,eTAAr::eStd));
	 }
 
    }
}

/*============================================================*/
/*                                                            */
/*                cIMakeTreeAr                                */
/*                                                            */
/*============================================================*/

class cIMakeTreeAr : public cAr2007,
	             public cSerialGenerator
{
     public :
        cIMakeTreeAr(const std::string & aName,eTypeSerial aTypeS) ;
        //~cIMakeTreeAr();
     protected :
	typedef std::list<cResLex>   tContToken;
	typedef tContToken::iterator tIterCTk;

	cResLex GetNextLex() override;

        void RawBeginName(const cAuxAr2007& anOT)  override; ///< Put opening tag
        void RawEndName(const cAuxAr2007& anOT)  override;  ///< Put closing tag
							    //
        int NbNextOptionnal(const std::string & aTag) override;							    
	void AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux) override;

        void RawAddDataTerm(int &    anI)  override;  ///< write int in text
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(double &    anI)  override;  ///< write double in text
        void RawAddDataTerm(std::string &    anI)  override; // write string
        void RawAddDataTerm(cRawData4Serial  &    aRDS) override;


	void OnTag(const cAuxAr2007& anOT,bool IsUp);
	
	///  Put the size added to the list of token

	tContToken            mListRL;
	tIterCTk              mItLR;
        std::string           mNameFile;
	eTypeSerial           mTypeS;
};

cIMakeTreeAr::cIMakeTreeAr(const std::string & aName,eTypeSerial aTypeS)  :
    cAr2007   (true,true,false),
    mNameFile (aName),
    mTypeS    (aTypeS)
{
   DEBUG = true;

   cSerialFileParser *  aSTP = cSerialFileParser::Alloc(mNameFile,aTypeS);
   cSerialTree aTree(*aSTP);

   // StdOut() << "JJJJJUiOp " << mNameFile << "\n";
   aTree.UniqueSon().Unfold(mListRL,mTypeS);
   // Show(mListRL);
   /*
   if (mTypeS==eTypeSerial::ejson)
       mListRL.pop_front();
       */

   mItLR = mListRL.begin();

   if (0)
   {
       StdOut()  << "<<<<<<<<\n";
       for (auto & aL : mListRL)
       {
            StdOut() << aL.mVal << " " << (int) aL.mLexP << "\n";
       }
       StdOut()  << ">>>>>>>>\n";
   }

   delete aSTP;
}

cResLex cIMakeTreeAr::GetNextLex() 
{

   if (mItLR==mListRL.end())
   {
        MMVII_INTERNAL_ASSERT_tiny(false,"End of list in mListRL for :" + mNameFile);
   }
   
   return *(mItLR++);

}

void cIMakeTreeAr::OnTag(const cAuxAr2007& aTag,bool IsUp)
{
   if ((mTypeS==eTypeSerial::ejson)  && ( (aTag.Name() == StrElCont) || (aTag.Name()==StrElMap)))
       return;
   cResLex aRL = GetNextLexNotSizeCont();


   if (aRL.mLexP != (IsUp ? eLexP::eUp  : eLexP::eDown))
   {
        StdOut() <<  "LEX " << int(aRL.mLexP)  << "VALS ,got " << aRL.mVal  << " Exp=" <<  aTag.Name() << " F=" << mNameFile << "\n";
        MMVII_INTERNAL_ASSERT_tiny(false ,"Bad token cIMakeTreeAr::RawBegin-EndName");
   }
   if (aRL.mVal  != aTag.Name())
   {
      StdOut() <<  "LEX " << int(aRL.mLexP)  << "VALS ,got " << aRL.mVal  << " Exp=" <<  aTag.Name() << " F=" << mNameFile << "\n";
      MMVII_INTERNAL_ASSERT_tiny(false,"Bad tag cIMakeTreeAr::RawBegin-EndName");
   }
}


void cIMakeTreeAr::RawBeginName(const cAuxAr2007& anIT)
{
   OnTag(anIT,true);
	/*
   cResLex aRL = GetNextLexNotSizeCont();

   StdOut() <<  "LEX " << int(aRL.mLexP)  << "VALS ,got " << aRL.mVal  << " Exp=" <<  anIT.Name() << "\n";

   MMVII_INTERNAL_ASSERT_tiny(aRL.mLexP == eLexP::eUp  ,"Bad token cIMakeTreeAr::RawBeginName");
   MMVII_INTERNAL_ASSERT_tiny(aRL.mVal  == anIT.Name() ,"Bad tag cIMakeTreeAr::RawBeginName");
   */
}

void cIMakeTreeAr::RawEndName(const cAuxAr2007& anIT)
{
   OnTag(anIT,false);
   /*
   cResLex aRL = GetNextLexNotSizeCont();

   MMVII_INTERNAL_ASSERT_tiny(aRL.mLexP == eLexP::eDown  ,"Bad token cIMakeTreeAr::RawBeginName");
   MMVII_INTERNAL_ASSERT_tiny(aRL.mVal  == anIT.Name() ,"Bad tag cIMakeTreeAr::RawBeginName");
   */
}


void cIMakeTreeAr::AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux)
{
   cResLex aRL = GetNextLexSizeCont();

   aNb = cStrIO<int>::FromStr(aRL.mVal);
}

int cIMakeTreeAr::NbNextOptionnal(const std::string & aTag) 
{
    tIterCTk    aCurIt =  mItLR;
    int aResult = 0;


    cResLex aRL = GetNextLexNotSizeCont();
    if ((aRL.mVal==aTag) && (aRL.mLexP==eLexP::eUp))
        aResult = 1;

    mItLR =  aCurIt;

    return aResult;
}

void cIMakeTreeAr::RawAddDataTerm(int &    anI)  
{
   cResLex aRL = GetNextLexNotSizeCont();
   FromS(aRL.mVal,anI);
}

void cIMakeTreeAr::RawAddDataTerm(size_t &    aSz)  
{
   cResLex aRL = GetNextLexNotSizeCont();
   FromS(aRL.mVal,aSz);
}

void cIMakeTreeAr::RawAddDataTerm(double &    aD)  
{
   cResLex aRL = GetNextLexNotSizeCont();
   FromS(aRL.mVal,aD);
}

void cIMakeTreeAr::RawAddDataTerm(std::string &    aS)  
{
   cResLex aRL = GetNextLexNotSizeCont();
   aS = aRL.mVal;
}

void cIMakeTreeAr::RawAddDataTerm(cRawData4Serial &    aRDS)  
{
   cResLex aRL = GetNextLexNotSizeCont();
   const char * aCPtr = aRL.mVal.c_str();

   tU_INT1 * aPtr = static_cast<tU_INT1*>(aRDS.Adr());
   for (int aK=0 ; aK< aRDS.NbElem() ; aK++)
   {
       int aC1= FromHexaCode(*(aCPtr++));
       int aC2= FromHexaCode(*(aCPtr++));
       aPtr[aK] = aC1 * 16 + aC2;
   }
}

cAr2007 * Alloc_cIMakeTreeAr(const std::string & aName,eTypeSerial aTypeS) 
{
    return new cIMakeTreeAr(aName,aTypeS);
}

/*============================================================*/
/*                                                            */
/*                   cTokenGeneByList                         */
/*                                                            */
/*============================================================*/

/**   Class for 
 */

class cTokenGeneByList : public cSerialGenerator
{
      public :
	typedef std::list<cResLex>   tContToken;
	cResLex GetNextLex() override;
	cTokenGeneByList(tContToken &);

      private :
	tContToken *           mContToken;
	tContToken::iterator   mItToken;
};

cTokenGeneByList::cTokenGeneByList(tContToken & aCont) :
    mContToken  (& aCont),
    mItToken    (mContToken->begin())
{
}

cResLex cTokenGeneByList::GetNextLex() 
{
     return *(mItToken++);
}

							     
/*============================================================*/
/*                                                            */
/*                cOMakeTreeAr                                */
/*                                                            */
/*============================================================*/

/**  Class for creating  serializing an object by creating an explicit tree representation 
 *   (in a "cSerialTree"), instead of the more basic class using streaming.
 *
 *   Basically :
 *      - the method called in serialization "AddData.." and  "RawBeginName..." generate tokens
 *      that are memorized in a list (it contains the nature of the call and the string associated)
 *
 *      - at  destruction of object :
 *          *  an explicit cSerialTree is created from the object (which is a cSerialGenerator)
 *          * the this tree is exported in xml, json of whatever dialect which is implemented in  the "xxx_PrettyPrint" method
 *          of  "cSerialTree"
 *
 *    Crystal clear, isn't it ? ;-))
 */


class cOMakeTreeAr : public cAr2007
{
     public :
        cOMakeTreeAr(const std::string & aName,eTypeSerial aTypeS) ;
        ~cOMakeTreeAr();
     protected :
	typedef std::list<cResLex>   tContToken;

        void RawBeginName(const cAuxAr2007& anOT)  override; ///< Put opening tag
        void RawEndName(const cAuxAr2007& anOT)  override;  ///< Put closing tag

        void RawAddDataTerm(int &    anI)  override;  ///< write int in text
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(double &    anI)  override;  ///< write double in text
        void RawAddDataTerm(std::string &    anI)  override; // write string
        void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
	/// Do nothing because tree-struct contains the information for size
	void AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux) override;

	void AddComment(const std::string &) override;

	bool                  SkeepStrElCont(const cAuxAr2007& anOT) const;

	tContToken            mContToken;
	tContToken::iterator  mItToken;
        std::string           mNameFile;
	eTypeSerial           mTypeS;
	bool                  mSkeepStrElCont;
};


cOMakeTreeAr::cOMakeTreeAr(const std::string & aName,eTypeSerial aTypeS)  :
    cAr2007           (false,true,false),   // Input,  Tagged, Binary
    mNameFile         (aName),
    mTypeS            (aTypeS),
    mSkeepStrElCont   (false) // ((aTypeS == eTypeSerial::ejson) || (aTypeS == eTypeSerial::etagt))
{
}

bool  cOMakeTreeAr::SkeepStrElCont(const cAuxAr2007& anOT) const
{
	return mSkeepStrElCont && (anOT.Name()== StrElCont) ;
}

void cOMakeTreeAr::RawBeginName(const cAuxAr2007& anOT)  
{
   if ( !SkeepStrElCont(anOT))
       mContToken.push_back(cResLex(anOT.Name(),eLexP::eUp,anOT.Type()));
}

void cOMakeTreeAr::RawEndName(const cAuxAr2007& anOT)  
{
   // if (anOT.Name()!= StrElCont)
   if ( !SkeepStrElCont(anOT))
      mContToken.push_back(cResLex(anOT.Name(),eLexP::eDown,anOT.Type()));
}

void cOMakeTreeAr::RawAddDataTerm(int &    anI)           
{
    mContToken.push_back(cResLex(ToStr(anI),eLexP::eStdToken_Int,eTAAr::eStd)); 
}
void cOMakeTreeAr::RawAddDataTerm(size_t &    anS)        
{ 
     mContToken.push_back(cResLex(ToStr(anS),eLexP::eStdToken_Size_t,eTAAr::eStd)); 
}
void cOMakeTreeAr::RawAddDataTerm(double &    aD)         
{ 
     mContToken.push_back(cResLex(ToStr(aD),eLexP::eStdToken_Double,eTAAr::eStd)); 
}
void cOMakeTreeAr::RawAddDataTerm(std::string &    anS)   
{ 
    mContToken.push_back(cResLex(Quote(anS),eLexP::eStdToken_String,eTAAr::eStd)); 
}
void cOMakeTreeAr::RawAddDataTerm(cRawData4Serial & aRDS)   
{ 
   std::string aStr;
   tU_INT1 * aPtr = static_cast<tU_INT1*>(aRDS.Adr());
   for (int aK=0 ; aK< aRDS.NbElem() ; aK++)
   {
       int aICar = aPtr[aK];
       aStr +=  ToHexacode(aICar/16) ;
       aStr +=  ToHexacode(aICar%16) ;
   }
   mContToken.push_back(cResLex(aStr,eLexP::eStdToken_RD4S,eTAAr::eStd)); 

}

void cOMakeTreeAr::AddComment(const std::string & anS)
{
      // StdOut() <<  "CCCC " << anS << " " <<  mContToken.back().mVal << "\n";
       mContToken.back().mComment = anS; 
}


void cOMakeTreeAr::AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux)  {}

cOMakeTreeAr::~cOMakeTreeAr()
{
    mContToken.push_back(cResLex("",eLexP::eEnd,eTAAr::eUndef));
    mItToken = mContToken.begin();

    cTokenGeneByList aTGBL(mContToken);

    cSerialTree aTree(aTGBL);
    cMMVII_Ofs anOfs(mNameFile,false);

    if (mTypeS==eTypeSerial::exml)
    {
       anOfs.Ofs() <<  TheXMLHeader << std::endl;
       aTree.Xml_PrettyPrint(anOfs);
       // aTree.UniqueSon().Xml_PrettyPrint(anOfs);
    }
    else if (mTypeS==eTypeSerial::ejson)
    {
         aTree.Json_PrettyPrint(anOfs);
    }
    else if (mTypeS==eTypeSerial::etagt)
    {
         aTree.Raw_PrettyPrint(anOfs);
    }
}


cAr2007 * Alloc_cOMakeTreeAr(const std::string & aName,eTypeSerial aTypeS) 
{
    return new cOMakeTreeAr(aName,aTypeS);
}




};

