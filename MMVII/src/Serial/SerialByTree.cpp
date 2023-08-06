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
/*             cSerialTokenParser                             */
/*                                                            */
/*============================================================*/



cSerialTokenParser::cSerialTokenParser(const std::string & aName,eTypeSerial aTypeS) :
   mMMIs          (aName),
   mTypeS         (aTypeS)
{
}

void cSerialTokenParser::Error(const std::string & aMesLoc)
{
    std::string aMesGlob =   aMesLoc + "\n"
                           + "while processing file=" +  mMMIs.Name() + " at char " + ToS(int(Ifs().tellg()));

    MMVII_INTERNAL_ASSERT_bench(false,aMesGlob);
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

tResLex  cSerialTokenParser::GetNextLex_NOEOF()
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
          return tResLex(GetQuotedString(),eLexP::eStdToken_String);
    }


    while ((!BeginPonctuation(aC)) && (!std::isspace(aC)))
    {
       aRes += aC;
       aC =  GetNotEOF();
    }
    Ifs().unget(); // put back < or ' '  etc ..

    //if (mTypeS!= eTypeSerial::etxt)  // else get EOF at end
    //   SkeepWhite();

    return tResLex(aRes,eLexP::eStdToken_UK);
}

tResLex  cSerialTokenParser::GetNextLex()
{
    try
    {
         return GetNextLex_NOEOF();
    }
    catch (cEOF_Exception anE)
    {
         return tResLex("",eLexP::eEnd);
    }

}

/*============================================================*/
/*                                                            */
/*             cXmlSerialTokenParser                          */
/*                                                            */
/*============================================================*/

cXmlSerialTokenParser::cXmlSerialTokenParser(const std::string & aName,eTypeSerial aTypeS) :
	cSerialTokenParser(aName,aTypeS)
{
}


bool  cXmlSerialTokenParser::BeginPonctuation(char aC) const { return aC=='<'; }

tResLex cXmlSerialTokenParser::AnalysePonctuation(char aC)
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

    return tResLex(aRes,aLex);
}

/*============================================================*/
/*                                                            */
/*                       cSerialTree                          */
/*                                                            */
/*============================================================*/

cSerialTree::cSerialTree(const std::string & aValue,int aDepth) :
   mValue (aValue),
   mDepth (aDepth)
{
}

cSerialTree::cSerialTree(cSerialTokenGenerator & aGenerator,int aDepth) :
   mDepth (aDepth)
{
    for(;;)
    {
        auto [aStr,aLex] = aGenerator.GetNextLex();

	if (aLex==eLexP::eEnd)
	{
	    if (aDepth!=0)
	    {
               MMVII_UnclasseUsEr("cSerialTree unexpected EOF");
	    }
	    return;
	}
        int aDec =  DecLevel(aLex);
	if (aDec>0)
	{
            mSons.push_back(cSerialTree(aGenerator,aDepth+1));
	    mSons.back().mValue = aStr;
	}
	else if (aDec<0)
	{
             return;
	}
	else
	{
            mSons.push_back(cSerialTree(aStr,aDepth+1));
	}
    }
}

void  cSerialTree::Indent(cMMVII_Ofs & anOfs) const
{
      for (int aK=0 ; aK<(mDepth-1) ; aK++)  
          anOfs.Ofs() << "   ";
}

void  cSerialTree::Xml_PrettyPrint(cMMVII_Ofs & anOfs) const
{
     bool IsTag = (mDepth!=0) && (!mSons.empty());
     if (mDepth!=0)
     {
	Indent(anOfs);
	if (IsTag)
            anOfs.Ofs()  << "<" << mValue << ">\n";
	else 
           anOfs.Ofs()<< mValue << "\n";
     }
     for (const auto & aSon : mSons)
     {
        aSon.Xml_PrettyPrint(anOfs);
     }
     if ((mDepth!=0) && (!mSons.empty()))
     {
	Indent(anOfs);
	if (IsTag) 
           anOfs.Ofs()<< "</" <<  mValue   << ">\n";
     }
}

/*============================================================*/
/*                                                            */
/*                cOMakeTreeAr                                */
/*                                                            */
/*============================================================*/

class cOMakeTreeAr : public cAr2007,
	             public cSerialTokenGenerator
{
     public :
        cOMakeTreeAr(const std::string & aName,eTypeSerial aTypeS) ;
        ~cOMakeTreeAr();
     protected :
	typedef std::list<tResLex>   tContToken;

	tResLex GetNextLex() override;
        void RawBeginName(const cAuxAr2007& anOT)  override; ///< Put opening tag
        void RawEndName(const cAuxAr2007& anOT)  override;  ///< Put closing tag

        void RawAddDataTerm(int &    anI)  override;  ///< write int in text
        void RawAddDataTerm(size_t &    anI) override;
        void RawAddDataTerm(double &    anI)  override;  ///< write double in text
        void RawAddDataTerm(std::string &    anI)  override; // write string
        void RawAddDataTerm(cRawData4Serial  &    aRDS) override;
	/// Do nothing because tree-struct contains the information for size
	void AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux) override;


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
    mContToken.push_back(tResLex(anOT.Name(),eLexP::eUp));
}

void cOMakeTreeAr::RawEndName(const cAuxAr2007& anOT)  
{
    mContToken.push_back(tResLex(anOT.Name(),eLexP::eDown));
}

void cOMakeTreeAr::RawAddDataTerm(int &    anI)           { mContToken.push_back(tResLex(ToStr(anI),eLexP::eStdToken_Int)); }
void cOMakeTreeAr::RawAddDataTerm(size_t &    anS)        { mContToken.push_back(tResLex(ToStr(anS),eLexP::eStdToken_Size_t)); }
void cOMakeTreeAr::RawAddDataTerm(double &    aD)         { mContToken.push_back(tResLex(ToStr(aD),eLexP::eStdToken_Double)); }
void cOMakeTreeAr::RawAddDataTerm(std::string &    anS)   { mContToken.push_back(tResLex(anS,eLexP::eStdToken_String)); }
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
   mContToken.push_back(tResLex(aStr,eLexP::eStdToken_RD4S)); 

}

void cOMakeTreeAr::AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux)  {}

tResLex cOMakeTreeAr::GetNextLex() 
{
     return *(mItToken++);
}

cOMakeTreeAr::~cOMakeTreeAr()
{
    mContToken.push_back(tResLex("",eLexP::eEnd));
    mItToken = mContToken.begin();

    cSerialTree aTree(*this,0);

    cMMVII_Ofs anOfs(mNameFile,false);
    aTree.Xml_PrettyPrint(anOfs);
}


cAr2007 * Alloc_cOMakeTreeAr(const std::string & aName,eTypeSerial aTypeS) 
{
    return new cOMakeTreeAr(aName,aTypeS);
}

/*============================================================*/
/*                                                            */
/*                           ::                               */
/*                                                            */
/*============================================================*/

/*
void TestcNodeSerial()
{
	cSerialTree aN("",0);
	aN.mSons.push_back(aN);
}

void TestGenerikPARSE(const std::string& aName)
{
    cXmlSerialTokenParser aXmlParse(aName,eTypeSerial::exml);

    StdOut() << "\n\n";
    cSerialTree aTree(aXmlParse,0);
    // aTree.Xml_PrettyPrint();
    getchar();
}
*/


};

