#include "MMVII_Stringifier.h"
#include "Serial.h"


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

cResLex::cResLex(std::string aVal,eLexP aLexP) :
    mVal  (aVal),
    mLexP (aLexP)
{
}

/*============================================================*/
/*                                                            */
/*             cSerialTokenGenerator                          */
/*                                                            */
/*============================================================*/

cResLex cSerialTokenGenerator::GetNextLexSizeCont() 
{
   cResLex aRes = GetNextLex();
   MMVII_INTERNAL_ASSERT_tiny(aRes.mLexP == eLexP::eSizeCont,"cSerialTree::UniqueSon");

   return aRes;
}

cResLex cSerialTokenGenerator::GetNextLexNotSizeCont() 
{
   cResLex aRes = GetNextLex();
   while (aRes.mLexP == eLexP::eSizeCont)
        aRes = GetNextLex();

   return aRes;
}


/*============================================================*/
/*                                                            */
/*             cSerialTokenParser                             */
/*                                                            */
/*============================================================*/



cSerialTokenParser::cSerialTokenParser(const std::string & aName,eTypeSerial aTypeS) :
   mMMIs          (aName),
   mTypeS         (aTypeS)
{
}

cSerialTokenParser::~cSerialTokenParser()
{
}


cSerialTokenParser *  cSerialTokenParser::Alloc(const std::string & aName,eTypeSerial aTypeS)
{
    switch (aTypeS) 
    {
         case eTypeSerial::exml : return new  cXmlSerialTokenParser(aName);
	 default : {}
    }

    MMVII_UnclasseUsEr("Bad enum for cSerialTokenParser::Alloc");
    return nullptr;
}


void cSerialTokenParser::Error(const std::string & aMesLoc)
{
    std::string aMesGlob =   aMesLoc + "\n"
                           + "while processing file=" +  mMMIs.Name() + " at char " + ToS(int(Ifs().tellg()));

    MMVII_UnclasseUsEr(aMesGlob);
}

int cSerialTokenParser::GetNotEOF()
{
   int aC = Ifs().get();
   if (aC==EOF)
   {
       if (true)
          throw cEOF_Exception();
       else
       {
               StdOut() << "jjjjjjjjjjjjjjj " << __LINE__ << "\n";
          Error("Unexpected EOF");
               StdOut() << "jjjjjjjjjjjjjjj " << __LINE__ << "\n";
       }
   }
   return aC;
}



bool cSerialTokenParser::SkeepOneString(const char * aString)
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


bool  cSerialTokenParser::SkeepOneKindOfCom(const char * aBeg,const char * anEnd)
{
   if (! SkeepOneString(aBeg))
      return false;

   while (! SkeepOneString(anEnd))
   {
        GetNotEOF();
   }
   return true;
}

bool  cSerialTokenParser::SkeepCom()
{
    return    SkeepOneKindOfCom(TheXMLBeginCom,TheXMLEndCom)
           || SkeepOneKindOfCom(TheXMLBeginCom2,TheXMLEndCom2);
}

int cSerialTokenParser::SkeepWhite()
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

std::string  cSerialTokenParser::GetQuotedString()
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

cResLex  cSerialTokenParser::GetNextLex_NOEOF()
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
          return cResLex(GetQuotedString(),eLexP::eStdToken_String);
    }


    while ((!BeginPonctuation(aC)) && (!std::isspace(aC)))
    {
       aRes += aC;
       aC =  GetNotEOF();
    }
    Ifs().unget(); // put back < or ' '  etc ..

    //if (mTypeS!= eTypeSerial::etxt)  // else get EOF at end
    //   SkeepWhite();

    return cResLex(aRes,eLexP::eStdToken_UK);
}

cResLex  cSerialTokenParser::GetNextLex()
{
    try
    {
         return GetNextLex_NOEOF();
    }
    catch (cEOF_Exception anE)
    {
         return cResLex("",eLexP::eEnd);
    }

}

/*============================================================*/
/*                                                            */
/*             cXmlSerialTokenParser                          */
/*                                                            */
/*============================================================*/

cXmlSerialTokenParser::cXmlSerialTokenParser(const std::string & aName) :
	cSerialTokenParser(aName,eTypeSerial::exml)
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

    return cResLex(aRes,aLex);
}

/*============================================================*/
/*                                                            */
/*                       cSerialTree                          */
/*                                                            */
/*============================================================*/

static bool DEBUG=false;

cSerialTree::cSerialTree(const std::string & aValue,int aDepth,eLexP aLexP) :
   mLexP    (aLexP),
   mValue   (aValue),
   mDepth   (aDepth),
   mMaxDSon (aDepth)
{
}

bool cSerialTree::IsTerminalNode() const
{
	return mDepth == mMaxDSon;
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
}

cSerialTree::cSerialTree(cSerialTokenGenerator & aGenerator,int aDepth,eLexP aLexP) :
   mLexP    (aLexP),
   mDepth   (aDepth),
   mMaxDSon (aDepth)
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
            mSons.push_back(cSerialTree(aGenerator,aDepth+1,aLex));
	    mSons.back().mValue = aStr;
	    UpdateMaxDSon();
	}
	else if (aDec<0)
	{
	     mComment = aRL.mComment;
             return;
	}
	else
	{
            mSons.push_back(cSerialTree(aStr,aDepth+1,aLex));
	    UpdateMaxDSon();
	}
    }
}

void  cSerialTree::Indent(cMMVII_Ofs & anOfs,int aDeltaInd) const
{
      for (int aK=0 ; aK<(mDepth-1+aDeltaInd) ; aK++)  
          anOfs.Ofs() << "   ";
}

void  cSerialTree::Xml_PrettyPrint(cMMVII_Ofs & anOfs) const
{
     bool IsTag = (mDepth!=0) && (!mSons.empty());
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

void  cSerialTree::Raw_PrettyPrint(cMMVII_Ofs & anOfs) const
{
     // bool OneLine = (mMaxDSon <= mDepth+1);
      Indent(anOfs,0);
      anOfs.Ofs() <<   mValue ;
      if (IsTerminalNode())  anOfs.Ofs() << " *";
      else if (IsSingleTaggedVal())  anOfs.Ofs() << " @";
      else if (IsTab())  anOfs.Ofs() << " #";
      anOfs.Ofs() << "\n";
      for (const auto & aSon : mSons)
      {
           aSon.Raw_PrettyPrint(anOfs);
      }
}

void  cSerialTree::PrintTerminalNode(cMMVII_Ofs & anOfs,bool Last) const
{
      Indent(anOfs,1);
      anOfs.Ofs() <<   Quote(mValue) << " :" << UniqueSon().mValue;
      if (! Last) anOfs.Ofs() <<  " ,";
      anOfs.Ofs() <<  "\n";
}

void  cSerialTree::PrintSingleTaggedVal(cMMVII_Ofs & anOfs,bool Last) const
{
          Indent(anOfs,1);
          anOfs.Ofs() <<   Quote(mValue) << " :[" ;
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
      Indent(anOfs,1);
      anOfs.Ofs() <<   "{\n" ;
      int aK = mSons.size();
      for (const auto & aSon : mSons)
      {
          aK--;
          if (aSon.IsSingleTaggedVal())
          {
              aSon.PrintTerminalNode(anOfs,aK==0);
          }
          else  if (aSon.IsTab())
          {
              aSon.PrintSingleTaggedVal(anOfs,aK==0);
          }
          else
          {
               aSon.Indent(anOfs,1);
               anOfs.Ofs() <<   Quote(aSon.mValue) << " :\n" ;
               aSon.Json_PrettyPrint(anOfs,aK==0);
          }
      }
      Indent(anOfs,1);
      anOfs.Ofs() <<   "}" ;
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
    aRes.push_back(cResLex(mValue,mLexP));

    // parse son for recursive call
    for (const auto & aSon : mSons)
        aSon.Unfold(aRes);

    // add potentiel closing tag
    if (mLexP==eLexP::eUp)
        aRes.push_back(cResLex(mValue,eLexP::eDown));
}

/*============================================================*/
/*                                                            */
/*                cIMakeTreeAr                                */
/*                                                            */
/*============================================================*/

// cResLex cSerialTokenGenerator::GetNextLexSizeCont() 
// cResLex cSerialTokenGenerator::GetNextLexNotSizeCont() 
//
class cIMakeTreeAr : public cAr2007,
	             public cSerialTokenGenerator
{
     public :
        cIMakeTreeAr(const std::string & aName,eTypeSerial aTypeS) ;
        //~cIMakeTreeAr();
     protected :
	typedef std::list<cResLex>   tContToken;

	cResLex GetNextLex() override;

        void RawBeginName(const cAuxAr2007& anOT)  override; ///< Put opening tag
	/*
        void RawEndName(const cAuxAr2007& anOT)  override;  ///< Put closing tag

        void RawAddDataTerm(int &    anI)  override;  ///< write int in text
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(double &    anI)  override;  ///< write double in text
        void RawAddDataTerm(std::string &    anI)  override; // write string
        void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
	/// Do nothing because tree-struct contains the information for size
	void AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux) override;

	void AddComment(const std::string &) override;



	tContToken            mContToken;
	tContToken::iterator  mItToken;
	*/

	tContToken            mListRL;
	tContToken::iterator  mItLR;
        std::string           mNameFile;
	eTypeSerial           mTypeS;
};

cIMakeTreeAr::cIMakeTreeAr(const std::string & aName,eTypeSerial aTypeS)  :
    cAr2007   (true,true,false),
    mNameFile (aName),
    mTypeS    (aTypeS)
{
   DEBUG = true;
   StdOut() << "cIMakeTreeAr " << mNameFile << "\n";

   cSerialTokenParser *  aSTP = cSerialTokenParser::Alloc(mNameFile,aTypeS);
   cSerialTree aTree(*aSTP,0,eLexP::eBegin);

   aTree.UniqueSon().Unfold(mListRL);

   mItLR = mListRL.begin();

   delete aSTP;
}

cResLex cIMakeTreeAr::GetNextLex() 
{
   MMVII_INTERNAL_ASSERT_tiny(mItLR!=mListRL.end(),"End of list in mListRL");
   
   return *(mItLR++);

}

void cIMakeTreeAr::RawBeginName(const cAuxAr2007& anIT)
{
   cResLex aRL = GetNextLexNotSizeCont();

   MMVII_INTERNAL_ASSERT_tiny(aRL.mLexP == eLexP::eUp  ,"Bad token cIMakeTreeAr::RawBeginName");
   MMVII_INTERNAL_ASSERT_tiny(aRL.mVal  == anIT.Name() ,"Bad tag cIMakeTreeAr::RawBeginName");
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
 *          *  an explicit cSerialTree is created from the object (which is a cSerialTokenGenerator)
 *          * the this tree is exported in xml, json of whatever dialect which is implemented in  the "xxx_PrettyPrint" method
 *          of  "cSerialTree"
 *
 *    Crystal clear, isn't it ? ;-))
 */

class cOMakeTreeAr : public cAr2007,
	             public cSerialTokenGenerator
{
     public :
        cOMakeTreeAr(const std::string & aName,eTypeSerial aTypeS) ;
        ~cOMakeTreeAr();
     protected :
	typedef std::list<cResLex>   tContToken;

	cResLex GetNextLex() override;
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



	tContToken            mContToken;
	tContToken::iterator  mItToken;
        std::string           mNameFile;
	eTypeSerial           mTypeS;
};


cOMakeTreeAr::cOMakeTreeAr(const std::string & aName,eTypeSerial aTypeS)  :
    cAr2007     (false,true,false),   // Input,  Tagged, Binary
    mNameFile   (aName),
    mTypeS      (aTypeS)
{
}

void cOMakeTreeAr::RawBeginName(const cAuxAr2007& anOT)  
{
   if (anOT.Name()!= "el")
      mContToken.push_back(cResLex(anOT.Name(),eLexP::eUp));
}

void cOMakeTreeAr::RawEndName(const cAuxAr2007& anOT)  
{
   if (anOT.Name()!= "el")
      mContToken.push_back(cResLex(anOT.Name(),eLexP::eDown));
}

void cOMakeTreeAr::RawAddDataTerm(int &    anI)           { mContToken.push_back(cResLex(ToStr(anI),eLexP::eStdToken_Int)); }
void cOMakeTreeAr::RawAddDataTerm(size_t &    anS)        { mContToken.push_back(cResLex(ToStr(anS),eLexP::eStdToken_Size_t)); }
void cOMakeTreeAr::RawAddDataTerm(double &    aD)         { mContToken.push_back(cResLex(ToStr(aD),eLexP::eStdToken_Double)); }
void cOMakeTreeAr::RawAddDataTerm(std::string &    anS)   
{ 
    mContToken.push_back(cResLex(Quote(anS),eLexP::eStdToken_String)); 
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
   mContToken.push_back(cResLex(aStr,eLexP::eStdToken_RD4S)); 

}

void cOMakeTreeAr::AddComment(const std::string & anS)
{
      // StdOut() <<  "CCCC " << anS << " " <<  mContToken.back().mVal << "\n";
       mContToken.back().mComment = anS; 
}


void cOMakeTreeAr::AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux)  {}

cResLex cOMakeTreeAr::GetNextLex() 
{
     return *(mItToken++);
}

cOMakeTreeAr::~cOMakeTreeAr()
{
    mContToken.push_back(cResLex("",eLexP::eEnd));
    mItToken = mContToken.begin();

    cSerialTree aTree(*this,0,eLexP::eBegin);
 
    {
        std::string aNewName = Prefix(mNameFile)+"_2.xml";
        {
            cMMVII_Ofs anOfs(aNewName,false);
            aTree.UniqueSon().Xml_PrettyPrint(anOfs);
        }

        // cIMakeTreeAr aIAMR(aNewName,eTypeSerial::exml);
    }
    {
        cMMVII_Ofs anOfs(Prefix(mNameFile)+"_raw.txt",false);
        aTree.UniqueSon().Raw_PrettyPrint(anOfs);
    }
    {
        cMMVII_Ofs anOfs(Prefix(mNameFile)+"_2.json",false);
        aTree.Json_PrettyPrint(anOfs,true);
    }
}


cAr2007 * Alloc_cOMakeTreeAr(const std::string & aName,eTypeSerial aTypeS) 
{
    return new cOMakeTreeAr(aName,aTypeS);
}




};

