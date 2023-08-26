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


bool SkeepStrStrElCont = false;

const char * TheXMLBeginCom  = "<!--";
const char * TheXMLEndCom    = "-->";
const char * TheXMLBeginCom2 = "<?";
const char * TheXMLEndCom2   = "?>";
const char * TheXMLHeader = "<?xml version=\"1.0\" encoding=\"ISO8859-1\" standalone=\"yes\" ?>";

static const char *  FakeTopSerialTree = "FakeTopSerialTree";


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
/*             cSerialParser                          */
/*                                                            */
/*============================================================*/

cResLex cSerialParser::GetNextLexSizeCont() 
{
   cResLex aRes = GetNextLex();
   MMVII_INTERNAL_ASSERT_tiny(aRes.mLexP == eLexP::eSizeCont,"GetNextLexSizeCont");

   return aRes;
}

cResLex cSerialParser::GetNextLexNotSizeCont() 
{
   cResLex aRes = GetNextLex();
   while (aRes.mLexP == eLexP::eSizeCont)
        aRes = GetNextLex();

   return aRes;
}

void cSerialParser::OnClose(const cSerialTree &,const std::string &) const 
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
	 default : {}
    }

    MMVII_UnclasseUsEr("Bad enum for cSerialFileParser::Alloc");
    return nullptr;
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

void cXmlSerialTokenParser::OnClose(const cSerialTree & aTree,const std::string & aStr)  const
{
     MMVII_INTERNAL_ASSERT_tiny(aTree.Value() == aStr,"Close tag unexpected");
}

/*============================================================*/
/*                                                            */
/*             cJsonSerialTokenParser                         */
/*                                                            */
/*============================================================*/

/*
cJsonSerialTokenParser::cJsonSerialTokenParser(const std::string & aName) :
	cSerialFileParser(aName,eTypeSerial::ejson)
{
}

bool BeginPonctuation(char aC) const override
{
}
          cResLex AnalysePonctuation(char aC)  override;
          void OnClose(const cSerialTree &,const std::string &) const override;
	  */


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


cSerialTree::cSerialTree(cSerialParser & aGenerator,const std::string & aValue,int aDepth,eLexP aLexP,eTAAr aTAAr) :
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

	     // StdOut() << "GGgGG: " <<  E2Str(aRL.mTAAr) << "\n";
             // MMVII_INTERNAL_ASSERT_tiny(mValue == aStr,"Close tag unexpected");
	     mComment = aRL.mComment;
	     mTAAr = aRL.mTAAr;
	     aGenerator.OnClose(*this,aStr);
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

void  cSerialTree::Xml_PrettyPrint(cMMVII_Ofs & anOfs) const
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
		
        aSon.Xml_PrettyPrint(anOfs);
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

const cSerialTree * cSerialTree::RealSonOf(const cSerialTree * aSon) const
{
    if (mTAAr == eTAAr::eCont)
    {
	 StdOut() << "V=" << mValue << " L=" << mLength << "\n";
         MMVII_INTERNAL_ASSERT_tiny(aSon->mTAAr == eTAAr::eElemCont,"RealSonOf TAAr");
        // MMVII_INTERNAL_ASSERT_tiny(aSon->mSons.size()==1,"Size son");

        return &(aSon->mSons.at(0));
    }

    return aSon;
}


void  cSerialTree::Test_PrintTab(cMMVII_Ofs & anOfs,bool IsLast) const
{
      Indent(anOfs,1);
      anOfs.Ofs() << mValue <<   "[\n" ;

      int aK = mSons.size();
      for (const auto & aSon : mSons)
      {
           aK--;
	   RealSonOf(&aSon)->Test_Print(anOfs,(aK==0));
      }

      Indent(anOfs,1);
      anOfs.Ofs() <<   "]" ;
      if (!IsLast)  anOfs.Ofs() << ",";
      anOfs.Ofs() <<   "\n" ;
}

void  cSerialTree::Test_PrintAtomic(cMMVII_Ofs & anOfs,bool IsLast) const
{
      Indent(anOfs,1);
      anOfs.Ofs() <<   mValue << "\n";
}

void  cSerialTree::Test_Print(cMMVII_Ofs & anOfs,bool IsLast) const
{
      if (IsTerminalNode())
         Test_PrintAtomic(anOfs,IsLast);
      else 
         Test_PrintTab(anOfs,IsLast);
}



bool cSerialTree::Json_OmitKey() const
{
      return  (mValue==StrElCont) || (mValue==StrElMap) ;
}


void  cSerialTree::JSon_PrintTerminalNode(cMMVII_Ofs & anOfs,bool Last) const
{
      Indent(anOfs,1);
      if (!Json_OmitKey())
         anOfs.Ofs() <<   Quote(mValue) << " :";
      anOfs.Ofs() << UniqueSon().mValue;
      if (! Last) anOfs.Ofs() <<  " ,";
      anOfs.Ofs() <<  "\n";
}

void  cSerialTree::JSon_PrintSingleTaggedVal(cMMVII_Ofs & anOfs,bool Last) const
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
}

void  cSerialTree::Json_PrettyPrint(cMMVII_Ofs & anOfs,bool IsLast) const
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
              aSon.JSon_PrintTerminalNode(anOfs,aK==0);
          }
          else  if (aSon.IsTab())
          {
              aSon.JSon_PrintSingleTaggedVal(anOfs,aK==0);
          }
          else
          {
               if (!aSon.Json_OmitKey())
	       {
                  aSon.Indent(anOfs,1);
                  anOfs.Ofs() <<   Quote(aSon.mValue) << " :" ;
                  anOfs.Ofs() <<   "\n" ;
	       }
               aSon.Json_PrettyPrint(anOfs,aK==0);
          }
      }
      Indent(anOfs,1);
      anOfs.Ofs() <<   (IsCont ? "]" : "}") ;
      if (!IsLast) anOfs.Ofs() <<   " ," ;
      anOfs.Ofs() <<   "\n" ;
}





const cSerialTree & cSerialTree::UniqueSon() const
{
    MMVII_INTERNAL_ASSERT_tiny(mSons.size()==1,"cSerialTree::UniqueSon");

    return *(mSons.begin());
}

void cSerialTree::Unfold(std::list<cResLex> & aRes) const
{

    // add value
    aRes.push_back(cResLex(mValue,mLexP,mTAAr));
    if (! IsTerminalNode())
       aRes.push_back(cResLex(ToStr(mSons.size()),eLexP::eSizeCont,eTAAr::eSzCont));

    // parse son for recursive call
    for (const auto & aSon : mSons)
        aSon.Unfold(aRes);

    // add potentiel closing tag
    if (mLexP==eLexP::eUp)
        aRes.push_back(cResLex(mValue,eLexP::eDown,mTAAr));
}

/*============================================================*/
/*                                                            */
/*                cIMakeTreeAr                                */
/*                                                            */
/*============================================================*/

// cResLex cSerialParser::GetNextLexSizeCont() 
// cResLex cSerialParser::GetNextLexNotSizeCont() 
//
class cIMakeTreeAr : public cAr2007,
	             public cSerialParser
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
   cSerialTree aTree(*aSTP,FakeTopSerialTree,0,eLexP::eBegin,eTAAr::eUndef);

   // StdOut() << "JJJJJUiOp " << mNameFile << "\n";
   aTree.UniqueSon().Unfold(mListRL);

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
   if (SkeepStrStrElCont && (aTag.Name() == StrElCont))
       return;
   cResLex aRL = GetNextLexNotSizeCont();

   // StdOut() <<  "LEX " << int(aRL.mLexP)  << "VALS ,got " << aRL.mVal  << " Exp=" <<  aTag.Name() << " F=" << mNameFile << "\n";

   if (aRL.mLexP != (IsUp ? eLexP::eUp  : eLexP::eDown))
   {
        StdOut() <<  "LEX " << int(aRL.mLexP)  << "VALS ,got " << aRL.mVal  << " Exp=" <<  aTag.Name() << " F=" << mNameFile << "\n";
        MMVII_INTERNAL_ASSERT_tiny(false ,"Bad token cIMakeTreeAr::RawBegin-EndName");
   }
   MMVII_INTERNAL_ASSERT_tiny(aRL.mVal  == aTag.Name() ,"Bad tag cIMakeTreeAr::RawBegin-EndName");
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

class cTokenGeneByList : public cSerialParser
{
      public :
	typedef std::list<cResLex>   tContToken;
	cResLex GetNextLex() override;
	cTokenGeneByList(tContToken &);
	// cListTokenGenerator(

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
 *          *  an explicit cSerialTree is created from the object (which is a cSerialParser)
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

    cSerialTree aTree(aTGBL,FakeTopSerialTree,0,eLexP::eBegin,eTAAr::eUndef);
    cMMVII_Ofs anOfs(mNameFile,false);

    if (mTypeS==eTypeSerial::exml)
    {
       anOfs.Ofs() <<  TheXMLHeader << std::endl;
       aTree.UniqueSon().Xml_PrettyPrint(anOfs);
    }
    else if (mTypeS==eTypeSerial::ejson)
    {
         aTree.Json_PrettyPrint(anOfs,true);


	 if (0)
         {
             cMMVII_Ofs anOfs2(LastPrefix(mNameFile)+".json2",false);
             aTree.Test_Print(anOfs2,false);
	 }
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


void BenchSerialJson()
{
	/*
         SkeepStrStrElCont = false;

         cTestSerial0 aS0;
         SaveInFile(aS0,"toto_S0_.tagt");

         SaveInFile(GentTestMasSerial(),"toto_Map.tagt");

	 std::vector< std::vector<int> > aVI {{1},{},{1,2}};
         SaveInFile(aVI,"toto_VI.tagt");

	 exit(EXIT_SUCCESS);
	 */

    for  (bool Skeep : {false})
    {
         // SkeepStrStrElCont = Skeep;
         cTestSerial0 aS0;

	 std::string aPost = Skeep ? "Skeep" : "NoSk";


         SaveInFile(aS0,"toto_S0_" + aPost + ".xml");
         SaveInFile(aS0,"toto_S0_" + aPost + ".json");
         SaveInFile(aS0,"toto_S0_" + aPost + ".tagt");
// StdOut() << "SkeepSkeepSkeep\n"; getchar();

	 std::vector< std::vector<int> > aVI {{1},{},{1,2}};
         SaveInFile(aVI,"toto_VI_" + aPost + ".xml");
         SaveInFile(aVI,"toto_VI_" + aPost + ".tagt");
         SaveInFile(aVI,"toto_VI_" + aPost + ".json");

         SaveInFile(GentTestMasSerial(),"toto_Map_" + aPost + ".xml");
         SaveInFile(GentTestMasSerial(),"toto_Map_" + aPost + ".tagt");
         SaveInFile(GentTestMasSerial(),"toto_Map_" + aPost + ".json");
    }
	/*
    cTestSerial0 aS0;
    // SaveInFile(aS0,"toto.json");
    SaveInFile(aS0,"toto.xml");
    StdOut() << "SAVE--- BenchSerialJson \n"; 
    ReadFromFile(aS0,"toto.xml");
    StdOut() << "BenchSerialJson \n"; getchar();

     SkeepStrStrElCont = false;
    */
}



};

