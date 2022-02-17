/*Header-MicMac-eLiSe-25/06/2007

MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
eLiSe  : ELements of an Image Software Environnement

www.micmac.ign.fr


Copyright : Institut Geographique National
Author : Marc Pierrot Deseilligny
Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
"A multiresolution and optimization-based image matching approach:
An application to surface reconstruction from SPOT5-HRS stereo imagery."
In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
(With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
d'images, adapte au contexte geograhique" to appears in 
Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

MicMac est un logiciel de mise en correspondance d'image adapte 
au contexte de recherche en information geographique. Il s'appuie sur
la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

MicMac is an open source software specialized in image matching
for research in geographic information. MicMac is built on the
eLiSe image library. MicMac is governed by the  "Cecill-B licence".
See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/



#include "StdAfx.h"



#include <map>
#include <locale>

std::string TheEliseDirXmlSpec=string("include")+ELISE_CAR_DIR+"XML_GEN"+ELISE_CAR_DIR;
bool ValInitNameDecl = false;

std::vector<std::string> VCurXmlFile;


/***********************************************************/
/*                                                         */
/*                    POLONAISE INVERSE                    */
/*                                                         */
/***********************************************************/

typedef const char * tCCP;

double PolonaiseInverse(tCCP & aC,bool & OK);
std::vector<double> GetNPolI(tCCP & aC,bool & OK,int aNB);

std::vector<double> GetNPolI(tCCP & aC,bool & OK,int aNB)
{
   std::vector<double> aRes;
   for (int aK=0 ; OK && (aK<aNB); aK++)
   {
       aRes.push_back(PolonaiseInverse(aC,OK));
   }
   return aRes;
}

typedef double (*tFoncPoli)(const std::vector<double> &);

class cOpPolI
{
    public :
       cOpPolI (int aNb,const std::string & aName,tFoncPoli aF) :
           mNb (aNb),
           mName (aName),
           mF (aF)
       {
       }
       int          mNb;
       std::string  mName;
       tFoncPoli    mF;

};

static double FLog2(const std::vector<double> & aV) { return log2(aV[0]);}
static double FRoundNI(const std::vector<double> & aV) { return round_ni(aV[0]);}
static double FTrue(const std::vector<double> & aV) { return 1; }
static double FFalse(const std::vector<double> & aV) { return 0; }
static double FNot(const std::vector<double> & aV) { return aV[0]==0; }

static double FEq(const std::vector<double> & aV) { return aV[0]==aV[1]; }
static double FNotEq(const std::vector<double> & aV) { return aV[0]!=aV[1]; }
    
static double FSom(const std::vector<double> & aV) { return aV[0]+aV[1]; }
static double FPow(const std::vector<double> & aV) { return pow(aV[0],aV[1]); }
static double FBarPow(const std::vector<double> & aV) { return  pow(aV[0],aV[2]) * pow(aV[1],1-aV[2]);}


static double FOr(const std::vector<double> & aV) { return (aV[0]!=0)||(aV[1]!=0); }
static double FAnd(const std::vector<double> & aV) { return (aV[0]!=0)&&(aV[1]!=0); }
static double FMul(const std::vector<double> & aV) { return aV[0]*aV[1]; }
static double FInfEq(const std::vector<double> & aV) { return aV[0]<=aV[1]; }
static double FSupEq(const std::vector<double> & aV) { return aV[0]>=aV[1]; }
static double FInfStrict(const std::vector<double> & aV) { return aV[0]<aV[1]; }
static double FSupStrict(const std::vector<double> & aV) { return aV[0]>aV[1]; }
static double FIf(const std::vector<double> & aV) { return aV[0]?aV[1] : aV[2]; }
static double FDiv(const std::vector<double> & aV) 
{
   ELISE_ASSERT(aV[1]!=0,"Null divisor in / (Polonaise invert)");
   return aV[0]/aV[1];
}
static double FMoins(const std::vector<double> & aV) { return aV[0]-aV[1]; }

const std::vector<cOpPolI> & OpPolI()
{
   static  std::vector<cOpPolI>  aRes;
   if (aRes.empty())
   {
       aRes.push_back(cOpPolI(3,"BarPow",FBarPow));
       aRes.push_back(cOpPolI(2,"Pow",FPow));
       aRes.push_back(cOpPolI(2,"+",FSom));
       aRes.push_back(cOpPolI(2,"Or",FOr));
       aRes.push_back(cOpPolI(2,"And",FAnd));
       aRes.push_back(cOpPolI(2,"*",FMul));
       aRes.push_back(cOpPolI(2,"InfEq",FInfEq));
       aRes.push_back(cOpPolI(2,"SupEq",FSupEq));
       aRes.push_back(cOpPolI(2,"Inf",FInfStrict));
       aRes.push_back(cOpPolI(2,"Sup",FSupStrict));
       aRes.push_back(cOpPolI(3,"?",FIf));
       aRes.push_back(cOpPolI(2,"/",FDiv));
       aRes.push_back(cOpPolI(0,"true",FTrue));
       aRes.push_back(cOpPolI(0,"false",FFalse));
       aRes.push_back(cOpPolI(1,"!",FNot));
       aRes.push_back(cOpPolI(2,"-",FMoins));
       aRes.push_back(cOpPolI(2,"Eq",FEq));
       aRes.push_back(cOpPolI(2,"Neq",FNotEq));
       aRes.push_back(cOpPolI(1,"Log2",FLog2));
       aRes.push_back(cOpPolI(1,"Int",FRoundNI));
   }
   return aRes;
}


// ATTENTION isblank non defini !!!  == >>  EN FAIT PAS VRAIMENT  :
//  sous linux  :    isblank defini ,  ISBLANK non defini
//
 bool PolIBlank(int aC) {return (ElIsBlank(aC)) || (aC=='"');}
//bool PolIBlank(int aC) {return ((aC) == ' ' || (aC) == '\t') || (aC=='"');}
void PasserPolIBlank(tCCP & aC)
{
   while (PolIBlank(*aC)) aC++;
}

double PolonaiseInverse(tCCP & aC,bool & OK)
{
   // while (isblank(*aC)) aC++;
   PasserPolIBlank(aC);

   if (isdigit(*aC))
   {
       char * aNew;
       double aRes = strtod(aC,&aNew);
       if (aNew==aC)
       {
          OK = false;
          return 0;
       }
       aC = aNew;
       return aRes;
   }


   const std::vector<cOpPolI> & aVOp = OpPolI();
   for (int aKop=0 ; aKop<int(aVOp.size()) ; aKop++)
   {
       const cOpPolI & anOp = aVOp[aKop];
       const char * anOpName = anOp.mName.c_str();
       if (IsPrefix(anOpName,aC))
       {
          aC += strlen(anOpName);
          
          std::vector<double> aV;
          for (int aK=0 ;  (aK<anOp.mNb); aK++)
          {
              aV.push_back(PolonaiseInverse(aC,OK));
              if (! OK)
                 return 0;
          }
          return anOp.mF(aV);
       }
   }

   OK = false;
   return 0;
}

double PolonaiseInverse(const std::string & aStr)
{
    const char * aC = aStr.c_str();
    bool OK=true;
    double aRes = PolonaiseInverse(aC,OK);
    if (!OK)
    {
       std::cout << "for " << aStr << "\n";
       ELISE_ASSERT(false,"Syntax error in PolonaiseInverse");
    }
    return aRes;
}


std::string PolISubst(const std::string & aStr)
{
   double aD = PolonaiseInverse(aStr);
   int anI = int(aD);
   return  (anI==aD) ? ToString(anI) : ToString(aD);
}





/***********************************************************/
/*                                                         */
/*                    eElXMLKindTree                       */
/*                                                         */
/***********************************************************/

eElXMLKindTree MergeForComp(eElXMLKindTree aKind)
{
	if (aKind == eXMLClone) return eXMLBranche;
	return aKind;
}

/***********************************************************/
/*                                                         */
/*                    cElXMLToken                          */
/*                                                         */
/***********************************************************/
char fgetcNN(cVirtStream * aFp)
{
	int aC = aFp->my_getc();

	if (aC==aFp->my_eof())
	{
		std::cout << "NAME=" << aFp->Name() << "\n";
		ELISE_ASSERT(aC!=aFp->my_eof(),"Unexpected EOF in cElXMLToken");
	}
/*
if (!isascii(aC))
{
   std::cout << "NON ASCII " << int(aC) << "\n";
}
*/
	return aC;
}



bool PasserString(cVirtStream * aFp,const char * aString)
{
	std::vector<int> aVPileCarLus;
	while (*aString)
	{
		int aC = aFp->my_getc();
		aVPileCarLus.push_back(aC);
		if (aC != *aString)
		{
			while (! aVPileCarLus.empty())
			{
				aFp->my_ungetc(aVPileCarLus.back());
				aVPileCarLus.pop_back();
			}
			return false;
		}
		++aString;
	}
	return true;
}



bool XML_PasserCommentaire(cVirtStream * aFp,const char * aBeg,const char * anEnd)
{
	if (! PasserString(aFp,aBeg))
		return false;


	while (! PasserString(aFp,anEnd))
	{
		fgetcNN(aFp);
	}
	return true;
}

static const char * aXMLBeginCom = "<!--";
static const char * aXMLEndCom = "-->";
static const char * aXMLBeginCom2 = "<?";
static const char * aXMLEndCom2 = "?>";

bool XML_PasserCommentaire(cVirtStream * aFp)
{
	return    XML_PasserCommentaire(aFp,aXMLBeginCom,aXMLEndCom)
		|| XML_PasserCommentaire(aFp,aXMLBeginCom2,aXMLEndCom2);
}



char XML_fgetcNN(cVirtStream * aFp)
{
	while (XML_PasserCommentaire(aFp));
	int aC = fgetcNN(aFp);

	// 
	if (aC==aFp->my_eof())
	{
		std::cout << "For name " << aFp->Name() << "\n";
		ELISE_ASSERT(false,"Unexpected EOF in cElXMLToken");
	}
	return aC;
}

int XML_passer_blanc(cVirtStream * aFp)
{
	int aC=' ';
	while (isspace(aC)|| (aC==0x0A))// retour arriere MPD : ca plante sous Linux ?? 
		// while (std::isspace(aC, std::locale("C")))/* || (aC==0x0A)*/ //GILLES:en fait 0A correspond au retour chariot; il faut au moins modifier cela en 0x0A
	{
		while (XML_PasserCommentaire(aFp));
		aC = aFp->my_getc();
	}
	aFp->my_ungetc(aC);
	return aC;
}

bool BoolVAl(const std::string& aStr)
{
	if ((aStr=="1") || (aStr=="true"))
		return  true;
	if ((aStr=="0") || (aStr=="false"))
		return  false;

	ELISE_ASSERT(false,"Bad val for Subst-Attr");
	return  false;
}


void XML_Attributs_speciaux
	(
	bool              & aNameDecl,
	cArgCreatXLMTree  & anArg,
	const std::string & aNameAttr,
	const std::string & aValAttrSpec,
	bool &  aUseSubst
	)
{

	if ( aNameDecl)
	{
		std::string aValAttr = aValAttrSpec;
		if (aUseSubst)
			anArg.DoSubst(aValAttr);
		anArg.SetDico(aNameAttr,aValAttr,false);
		//std::cout << "SYMB " << aNameAttr << "<=>" << aValAttr << "\n";
		// getchar();
		return;
	}


	if (
		(aValAttrSpec[0] != '@') 
		|| (aValAttrSpec[1] != '$') 
		|| (aValAttrSpec[2] != '#') 
		)
		return;



	std::string aValAttr(aValAttrSpec.c_str()+3);


	if (aNameAttr=="NameDecl")
	{
		aNameDecl=BoolVAl(aValAttr); 
	}
	else if (aNameAttr=="ExitOnBrkp")
	{
		TheExitOnBrkp = BoolVAl(aValAttr);
	}
	else if (aNameAttr=="DirXmlSpec")
	{
		// Beware : TheEliseDirXmlSpec is considered relative to MMDir()
		TheEliseDirXmlSpec = aValAttr;
	}
	else if (aNameAttr=="Subst")
	{
		aUseSubst=BoolVAl(aValAttr);
	}

}

void XML_GetAttr 
	(
	bool &aNameDecl,
	cArgCreatXLMTree  & anArg,
	cVirtStream *aFp,
	std::string & aRes1, 
	std::string & aRes2,
	bool & aUseSubst
	)
{
	bool GotEq  = false;
	bool GotQuote  = false;

	while(1)
	{
		int aC= XML_fgetcNN(aFp);
		bool withBckSlh = (aC=='\\');
		if (withBckSlh)
		{
			aC= XML_fgetcNN(aFp);
		}
		if ((aC== '"') && (! withBckSlh))
		{
			if (! GotEq)
			{
				std::cout << "For file = " <<  aFp->Name() << "\n";
				ELISE_ASSERT(GotEq,"Error in XML_GetAttr");
			}
			if (GotQuote) 
			{
				XML_Attributs_speciaux(aNameDecl,anArg,aRes1,aRes2,aUseSubst);
				return;
			}
			GotQuote = true;
		}
		else if (GotQuote)
		{
			aRes2 += aC;
		}
		else if (isspace(aC) || (aC=='>'))
		{
			ELISE_ASSERT(aRes1!="","Internal Error in XML_GetAttr");
			aFp->my_ungetc(aC);
			XML_Attributs_speciaux(aNameDecl,anArg,aRes1,aRes2,aUseSubst);
			return;
		}
		else if (aC=='=')
		{
			GotEq = true;
		}
		else
		{
			if (GotEq)
				aRes2 += aC;
			else
				aRes1 += aC;
		}
	}
}

void VirerBlancFinal(std::string & aStr)
{
	std::string::iterator anIt = aStr.end();
	while (1)
	{
		if(aStr.empty()) return;
		anIt--;
		if (!isspace (*anIt)) return;
		anIt = aStr. erase(anIt);
	}
}


/*
   Une sequence binaire est faite de TheNbDieseBinSeq '#',
   puis 4 caractere qui seront utilise peut etre plus tard ,
   pour l'instant '0000' . Elle contient des #, donc ne peut jamais
   apparaitres dans les sequences binaire 
*/
                                //0123456789012345678901234567890123
static const char *  TheBinSeq = "#0ArTuQdI(RvO6z{Z[a924]Uh}mBgD)k37";
static std::vector<const void *>  TheVecDataXMLBin;

const void * GetXMLDataBin(int aK)
{
   ELISE_ASSERT((aK>=0)&&(aK<int(TheVecDataXMLBin.size())),"GetXMLDataBin");
   return TheVecDataXMLBin[aK];
}

inline int LengthBinSeq()
{
    static int aRes = (int)strlen(TheBinSeq);
    return aRes;
}


typedef const char * tConstCharPtr;
inline bool AnalyseBinSeq(tConstCharPtr & aSeq,const int &   aCar)
{
   if (*aSeq!=aCar) 
   {
      aSeq = TheBinSeq;
      return false;
   }

   aSeq++;

   if (*aSeq==0) 
   {
      aSeq = TheBinSeq;
      return true;
   }
  
   return false;
}

/*
   Dans le fichier 
      TheBinSeq0000[%NbOct]lkjslkfbeiuriouerio....
   En sortie
      TheBinSeq0000[%NbOct][%Num]

   Avec 
      TheVecDataXMLBin[%Num]=lkjslkfbeiuriouerio.....
*/

int GetNumEntreCrochet(cVirtStream * aFp,std::string * aPush)
{
   int aC= XML_fgetcNN(aFp);
   if (aPush)
    *aPush += char(aC);
   ELISE_ASSERT(aC=='[',"Expected [ in binary sequence count");
   int aNb = 0;
   bool Cont = true;
   while (Cont)
   {
       aC = XML_fgetcNN(aFp);
       if (aPush)
         *aPush += char(aC);
       if (isdigit(aC))
       {
          aNb  = aNb *10 + (aC-'0') ;
       }
       else if (aC==']')
       {
          Cont = false;
       }
       else
       {
             ELISE_ASSERT(false,"Expected ] in binary sequence count");
       } 
   }
   return aNb;
}

void cElXMLToken::GetSequenceBinaire(cVirtStream * aFp)
{
   for (int aK=0 ; aK<4 ; aK++)
   {
        int aC= XML_fgetcNN(aFp);
        mVal += char(aC);
        ELISE_ASSERT(aC=='0',"cElXMLToken::GetSequenceBinaire expect 0000");
   }
   int aNb = GetNumEntreCrochet(aFp,&mVal);
   void * aPtr = malloc(aNb);
   TheVecDataXMLBin.push_back(aPtr);
   aFp->fread(aPtr,aNb);
   mVal = mVal + "["+ToString(int(TheVecDataXMLBin.size()-1)) + "]";
   
}


void PutCharPtrWithTraitSeqBinaire(FILE * aFp,const  char * aVal)
{
    while (*aVal)
    {
         if ((*aVal=='#') && (!strncmp(aVal,TheBinSeq,LengthBinSeq())))
         {
              fprintf(aFp,"%s",TheBinSeq);
              aVal += LengthBinSeq();
              for (int aK=0 ; aK<4 ; aK++) 
              {
                  fputc(*aVal,aFp);
                  aVal++;
              }
              cVirtStream*  aVStr = cVirtStream::VStreamFromCharPtr(aVal);
              int aNbOct = GetNumEntreCrochet(aVStr,0);
              int anAdr = GetNumEntreCrochet(aVStr,0);
              fprintf(aFp,"[%d]",aNbOct);
              fwrite(GetXMLDataBin(anAdr),1,aNbOct,aFp);
              aVal = aVStr->Ending();
              delete aVStr;
         }
         else
         {
            fputc(*aVal,aFp);
            aVal++;
         }
    }
    // fprintf(aFp,"%s",aVal);
}

void PutDataInXMLString(std::ostringstream & anOs,const void * aData,int aNbOct)
{
    anOs << TheBinSeq  << "0000" << "["<<aNbOct<<"][" <<TheVecDataXMLBin.size() << "]";
    TheVecDataXMLBin.push_back(aData);
}

const void * GetDataInXMLString(std::istringstream & aStream,int aNbExpOct)
{
  char aC = ' ';
  while (isspace(aC))
  {
      aStream >> aC;
  }
  if (aC=='#')
  {
     const char * aStr = TheBinSeq+1;
     while (*aStr)
     {
         aStream >> aC;
         ELISE_ASSERT(*aStr==aC,"Bad binary sequence");
         aStr++;
     }
     for (int aK=0 ; aK<4 ; aK++)
     {
         aStream >> aC;
         ELISE_ASSERT(aC=='0',"GetDataInXMLString, expect 0000");
     }
     cVirtStream*  aVStr = cVirtStream::VStreamFromIsStream(aStream);
     int aNbOct = GetNumEntreCrochet(aVStr,0);
     ELISE_ASSERT(aNbOct==aNbExpOct,"Incoheren byte count in GetDataInXMLString");
     int anAdr = GetNumEntreCrochet(aVStr,0);

     delete aVStr;
     return GetXMLDataBin(anAdr);
  }

  aStream.putback(aC);
  return 0;
}


cElXMLToken::cElXMLToken
	(
	cArgCreatXLMTree & anArg,
	cVirtStream * aFp,
	bool & aUseSubst
	)
{
	XML_passer_blanc(aFp);
	int aC = aFp->my_getc();
        const char * CurBinSeq = TheBinSeq;

        // AnalyseBinSeq(CurBinSeq,aC);

	if (aC==aFp->my_eof())
	{
		mKind = eXMLEOF;
		return;
	}

	mKind = eXMLStd;
	if (aC == '<')
	{
		XML_passer_blanc(aFp);
		aC = XML_fgetcNN(aFp);
		mKind = eXMLOpen;
		if (aC ==  '/')
		{
			mKind = eXMLClose;
			aC = XML_fgetcNN(aFp);
		}
	}


	bool aNameDecl=ValInitNameDecl;
	int aLastC = -1;
	while (1)
	{
		std::string anAttrStr;
		if (mKind==eXMLStd)
		{
			//if (isspace(aC)) return;
			if (aC=='<')
			{
				aFp->my_ungetc(aC);
				VirerBlancFinal(mVal);
				return;
			}
			mVal += char(aC);
                       if (AnalyseBinSeq(CurBinSeq,aC))
                       {
                            GetSequenceBinaire(aFp);
                       }
		}
		else
		{
			if (aC=='>')
			{
				// std::cout << "C=> " << aLastC << " " << int('/') << "\n";
				const char * aCV = mVal.c_str();
				char aCL = aCV[mVal.size()-1];
				if (aCL=='/')
				{
					mKind = eXMLOpenClose;
					mVal = mVal.substr(0,mVal.size()-1);
				}
				if (aLastC=='/')
				{
					mKind = eXMLOpenClose;
				}
				return;
			}
			if (isspace(aC)) 
			{
				int aC2 = XML_passer_blanc(aFp);
				if (aC2=='/')
				{
					aC2 = XML_fgetcNN(aFp);
					aC2 = XML_fgetcNN(aFp);
					ELISE_ASSERT(aC2=='>',"No > after / ");
					mKind = eXMLOpenClose;
					return;
				}
				if (aC2 != '>')
				{
					cElXMLAttr anAttr;
					XML_GetAttr(aNameDecl,anArg,aFp,anAttr.mSymb,anAttr.mVal,aUseSubst);

// std::cout << "DDDDD " << aUseSubst << " " << anAttr.mSymb << " " << anAttr.mVal << "\n";
		                        if (aUseSubst)
                                        {
			                        anArg.DoSubst(anAttr.mSymb,true);
			                        anArg.DoSubst(anAttr.mVal,true);
                                        }
					mAttrs.push_back(anAttr);
					// std::cout << "Atttrrrrrrr" << anAttr.mSymb << "::" << anAttr.mVal <<"\n";
				}
			}
			else
			{
				mVal += char(aC);
			}
		}
		aLastC = aC;
		aC = XML_fgetcNN(aFp);
	}
}

const std::string &  cElXMLToken::Val() const {return mVal;}
eElXMLKindToken  cElXMLToken::Kind() const    {return mKind;}

const std::list<cElXMLAttr> & cElXMLToken::Attrs() const
{
	return mAttrs;
}


/***********************************************************/
/*                                                         */
/*                    cElXMLTreeFilter                     */
/*                                                         */
/***********************************************************/

bool cElXMLTreeFilter::Select(const cElXMLTree &) const
{
	return true;
}

/***********************************************************/
/*                                                         */
/*                    cArgCreatXLMTree                     */
/*                                                         */
/***********************************************************/


cArgCreatXLMTree::~cArgCreatXLMTree()
{
	// NE PEUT PAS LE DETRUIRE CAR UTILISE ENSUITE PAR GENERATEUR DE CODE !!
	// DeleteAndClear(mAddedTree);
}

// convert upper-case caracter into lower-case
void tolower( std::string &io_str )
{
	std::string::iterator it = io_str.begin();
	while ( it!=io_str.end() )
	{
		*it = tolower( *it );
		it++;
	}
}

std::string tolower( const std::string &i_str )
{
	std::string str = i_str;
	tolower( str );
	return str;
}

// convert a filename into a unique representation
// (don't do anything unless under windows because unix's filenames are already unique)
void filename_normalize( std::string &io_filename )
{
#if (ELISE_windows)
	std::string::iterator it = io_filename.begin();
	while ( it!=io_filename.end() )
	{
		if ( (*it)=='\\' )
			*it = '/';
		else
			*it = tolower( *it );
		it++;
	}
#endif
}

std::string filename_normalize( const std::string &i_filename)
{
	std::string str = i_filename;
	filename_normalize( str );
	return str;
}

// return true if i_str starts with i_start (case sensitive)
bool startWith( const std::string &i_str, const std::string &i_start )
{
	if ( i_str.length()<i_start.length() ) return false;
	string strStart = i_str.substr( 0, i_start.length() );
	return ( strStart.compare( i_start )==0 );
}

std::string StdGetFileXMLSpec(const std::string & aName)
{
	return MMDir() + TheEliseDirXmlSpec+ aName;
}

void cArgCreatXLMTree::AddRefs(const std::string & aTag,const std::string & aFileSeul)
{
	std::string aFile = StdGetFileXMLSpec(aFileSeul);
	if (BoolFind(mAddedFiles,aFile))
		return;

	mAddedFiles.push_back(aFile);


	cElXMLTree * aTree = new cElXMLTree(aFile,this);
	mAddedTree.push_back(aTree);
}

cArgCreatXLMTree::cArgCreatXLMTree
	(
	const std::string & aNF,
	bool aModifTree,
	bool aModifDico
	) :
mNF           (aNF),
	mModifTree    (aModifTree),
	mModifDico    (aModifDico)
{
	if (mModifDico)
	{

               std::string aNFW = aNF;
#if (ELISE_windows)
               replace( aNFW.begin(), aNFW.end(),  '/','\\' );
#endif
		SetDico("ThisDir",aNFW,false);

		// SetDico("ThisFile",aNFile);
	}
}

bool  cArgCreatXLMTree::ModifTree() const {return mModifTree;}
bool  cArgCreatXLMTree::ModifDico() const {return mModifDico;}

void cArgCreatXLMTree::SetDico(const std::string & aKey,std::string  aVal,bool IsMMCALL)
{
	if (aVal.length()!=0 && aVal[0] =='@')
	{
		aVal = aVal.substr(1,std::string::npos);
	}

	if (aVal.length()!=0 && aVal[0]=='"')
	{
		const char * aVc = aVal.c_str();
		int aL = (int)strlen(aVc);
		if (aL>1 && aVc[aL-1]=='"')
		{
			aVal = aVal.substr(1,aL-2);
		}
	}

	// Pour l'instant, on ne modifie pas  les symbole micmac
	if (mSymbMMCall.find(aKey)!=mSymbMMCall.end())
	{
		return;
	}

	mDicSubst[aKey] = aVal;
	if (IsMMCALL)
	{
		mSymbMMCall.insert(aKey);
	}

	/*
	// std::cout << "DICO ; " << aKey << " => " << aVal << "\n";
	// Si la vale commence par @, cela signifie ne modifier que si nouveau
	if (aVal[0]=='@')
	{
	if (! DicBoolFind(mDicSubst,aKey))
	{
	mDicSubst[aKey] =aVal.substr(1,std::string::npos);
	}
	}
	else
	{
	mDicSubst[aKey] = aVal;
	}

	if (IsInvar)
	*/
}

void cArgCreatXLMTree::Add2EntryDic
	(
	cElXMLTree * aTree,
	const std::string & aName
	)
{
	ELISE_ASSERT(mDico[aName]==0,"multiple Add2EntryDic");
	mDico[aName] = aTree;
}

cElXMLTree* cArgCreatXLMTree::ReferencedVal(const std::string & aName)
{
	return mDico[aName];
}

void cArgCreatXLMTree::DoSubst(std::string & aStr)
{
	DoSubst(aStr,false);
}

void cArgCreatXLMTree::DoSubst(std::string & aStr,bool ForceSubst)
{
	if ((! mModifTree) && (!ForceSubst))
		return;
	char aJoker = '$';
	char aOpen = '{';
	char aClose = '}';

	if (find(aStr.begin(),aStr.end(),aJoker)==aStr.end())
		return;
	if (find(aStr.begin(),aStr.end(),aOpen)==aStr.end())
		return;
	//std::cout << "33333\n";

	std::string aRes;
	for (const char * aInp = aStr.c_str() ; *aInp ; aInp++)
	{
		if ((*aInp==aJoker) && (aInp[1]==aOpen))
		{
			aInp+=2;
			std::string aEntry;
			while (*aInp && *aInp!=aClose)
			{
				aEntry.push_back(*aInp);
				aInp++;
			}
			ELISE_ASSERT(*aInp,"Unclosed substitution");
			if (! DicBoolFind(mDicSubst,aEntry))
			{
				std::cout << "For " << aEntry << "\n";
				ELISE_ASSERT(false,"Cannot handle substitution");
			}
			aRes = aRes + mDicSubst[aEntry];
		}
		else
		{
			aRes.push_back(*aInp);
		}
	}
	// std::cout << aStr << " ==> " << aRes << "\n"; getchar();
	aStr = aRes;
}

/***********************************************************/
/*                                                         */
/*                    cElXMLTree                           */
/*                                                         */
/***********************************************************/
cElXMLTree::~cElXMLTree()
{
	if (mKind != eXMLClone)
		DeleteAndClear(mFils);
}

bool cElXMLTree:: IsFeuille() const
{
	return    (mKind==eXMLFeuille);
}

cElXMLTree * cElXMLTree::ReTagThis(const std::string & aNameTag)
{
	mValTag = aNameTag;
	return this;
}

bool cElXMLTree::IsBranche() const
{
	return    (mKind==eXMLBranche)
		|| (mKind==eXMLClone);
}

bool ValInitUseSubst =true;



cElXMLTree::cElXMLTree(const std::string & aName,cArgCreatXLMTree * anArgEx,bool DoFileInclu) 
{
        VCurXmlFile.push_back(aName);

	cArgCreatXLMTree anArg00(aName,false,false);
	if (anArgEx== 0)
		anArgEx = & anArg00;
	anArgEx->mNF = aName;

	mKind = eXMLTop;
	mValTag = aName;
	mPere = 0;

	cVirtStream * aFp  = cVirtStream::StdOpen(aName);

	while(1)
	{
		bool        aUseSubst = ValInitUseSubst;
		cElXMLToken aToken(*anArgEx,aFp,aUseSubst);
		if (aToken.Kind() == eXMLEOF)
		{
			VerifCreation();
			delete aFp;
                        VCurXmlFile.pop_back();
			return;
		}
		//cArgCreatXLMTree anArg(aName);

		mFils.push_back(new cElXMLTree(DoFileInclu,aUseSubst,aFp,aToken,this,*anArgEx ));
	}
        VCurXmlFile.pop_back();
}

void cElXMLTree::ExpendRef
	(
	cArgCreatXLMTree &anArg,
	const std::string &  aNameAttr,
	const std::string &  aTagExpendFile,
	bool aMustBeEmpty
	)
{
	static std::string aStrEmpty="";
	const std::string & aValAttr = ValAttr(aNameAttr,aStrEmpty);
	if (aValAttr=="") return;


	const std::string & aRefFile = ValAttr(aTagExpendFile,aStrEmpty);
	if (aRefFile!=aStrEmpty)
	{
		anArg.AddRefs(aNameAttr,aRefFile);
	}

	if (aMustBeEmpty)
	{
		ELISE_ASSERT
			(
			mFils.empty(),
			"Try to ref-expend non empty fils"
			);
	}
	cElXMLTree * aRefVal = anArg.ReferencedVal(aValAttr);
	if (aRefVal==0)
	{
		std::cout << "-----FOR TAG = " << mValTag<< "\n";
		ELISE_ASSERT(aRefVal!=0,"Cannot find Referenced Val");
	}

	mKind = eXMLClone;
	mFils = aRefVal->mFils;
}







void TestSpecialTags(const std::string & aMes,cElXMLTree * aTree,cVirtStream * aFP,cArgCreatXLMTree &anArg,bool UsePolI)
{
	if (aFP->IsFileSpec()) 
	{
		return;
	}


	if (aTree->ValTag()=="ExitOnBrkp")
	{
		TheExitOnBrkp =true;  // A priori si on l'a modifie c'est pour le changer
		TheExitOnBrkp = BoolVAl(aTree->GetUniqueVal());
	}

        bool IsSymb = (aTree->ValTag()=="Symb");
        bool IsEvSymb = (aTree->ValTag()=="eSymb");
	if (IsSymb || IsEvSymb)
	{
		std::string aSymb,aVal;

		SplitIn2ArroundEq(aTree->GetUniqueVal(),aSymb,aVal);

		// std::cout << aMes  << aVal << " => " << aSymb << "\n"; getchar();

		//GERALD ATTENTION PLANTE SI PAS VALEUR PAR DEFAUT DANS Apero-Glob.xml!!!!
		if(aVal.size()!=0)
                {
                  if (IsEvSymb&&anArg.ModifTree())
                     aVal = PolISubst(aVal);
	  	  anArg.SetDico(aSymb,aVal,false);
                }
		else
		{
			std::cout << "Probleme avec le parametre " << aSymb << ", dans le fichier " << anArg.mNF << "\n";
			ELISE_ASSERT(false,"Valeur par defaut non definie");
		}

	}

}

static const std::string StrIF =     "#IF";
static const std::string StrWHEN =   "#WHEN";
static const std::string StrSWITCH = "#SWITCH";

std::list<cElXMLTree *>  cElXMLTree::Interprete()
{
   std::list<cElXMLTree *> aRes;
   if (
              (mValTag==StrIF)
           || (mValTag==StrWHEN)
           || (mValTag==StrSWITCH)
      )
   {

        const std::string & aVTest = ValAttr("VTEST");
       if (
              (mValTag==StrIF)
           || (mValTag==StrWHEN)
         )
         {
              bool aVal = Str2BoolForce(aVTest); 
              // ELISE_ASSERT(mFils.size()==2,"Bd size in #IF-tag");
              int aCpt = 0;

              int aLim = 1000000000; // Cas WHEN
              if (mValTag==StrIF)
              {
                 std::string aStrLim   = ValAttr("GOTO","1"); 
                 FromString(aLim,aStrLim);
              }
              for 
              (
                        std::list<cElXMLTree *>::iterator anIt= mFils.begin();
                        anIt!= mFils.end();
                        anIt++
              )
              {
                 if (aVal ^ (aCpt >= aLim))
                    aRes.push_back(*anIt);
                 aCpt++;
              }
  
              return aRes;
         }
         if (mValTag==StrSWITCH)
         {
              std::list<cElXMLTree *> aRDef;
              for 
              (
                        std::list<cElXMLTree *>::iterator anIt= mFils.begin();
                        anIt!= mFils.end();
                        anIt++
              )
              {
                 std::string aStrCase   = (*anIt)->ValAttr("CASE"); 
                 if (aStrCase==aVTest)
                    aRes.push_back(*anIt);
                 else if (aStrCase=="DEFAULT")
                    aRDef.push_back(*anIt);
              }
              return aRes.empty() ? aRDef : aRes;
         }     
   }

   aRes.push_back(this);
   return aRes;
}

cElXMLTree::cElXMLTree
	(
	bool   DoFileCinclu,
	bool   aUseSubstTree,
	cVirtStream * aFp,
	const cElXMLToken &  aTok,
	cElXMLTree * aPere,
	cArgCreatXLMTree &anArg
	) 
{
	mValTag = aTok.Val();
	mPere = aPere;
	if (aTok.Kind() == eXMLStd)
	{
// std::cout << "UST " << aUseSubstTree << " " << mValTag << "\n";
		if (aUseSubstTree)
			anArg.DoSubst(mValTag);
		mKind = eXMLFeuille;
		VerifCreation();
		return;
	}
	mKind = eXMLBranche;

	if (aTok.Kind() == eXMLOpen)
	{
		mAttrs = aTok.Attrs();
	}

	if (aTok.Kind() ==eXMLOpenClose)
	{
		for (std::list<cElXMLAttr>::const_iterator itA = aTok.Attrs().begin() ; itA!= aTok.Attrs().end() ; itA++)
		{
			cElXMLTree* aFils = ValueNode(itA->mSymb,itA->mVal);
			TestSpecialTags("xx",aFils,aFp,anArg,aUseSubstTree);
			AddFils(aFils);
		}
		TestSpecialTags("AA",this,aFp,anArg,aUseSubstTree);
		return;
	}

	if (aTok.Kind() != eXMLOpen)
	{
		std::cout << "PERE =" << aPere->mValTag << "\n";
		std::cout << "FILE " << aFp->Name() << "\n";
		ELISE_ASSERT(false,"Not Open in cElXMLTree::cElXMLTree");
	}
	while (1)
	{	   
		bool  aUseSubst=aUseSubstTree;
		cElXMLToken aNewTok(anArg,aFp,aUseSubst);

		if (aNewTok.Kind() == eXMLClose)
		{
			if (aNewTok.Val() != aTok.Val())
			{
				cout << "<" <<  aTok.Val() <<">#</"<< aNewTok.Val() <<"> dans "<<aFp->Name()<<"\n";
				  ELISE_ASSERT( false, "bad Close Tag in cElXMLTree::cElXMLTree" );
			}
			VerifCreation();
			{
				ExpendRef(anArg,"RefType","RefFile",true);
				static std::string aStrFalse="false";
				if (ValAttr("ToReference",aStrFalse)=="true")
				{
					anArg.Add2EntryDic(this,mValTag);
				}
			}
			TestSpecialTags("BB",this,aFp,anArg,aUseSubstTree);
			return;
		}
		if (aNewTok.Kind()==eXMLEOF)
		{
			std::cout << "FILE =" << anArg.mNF 
				<< " ; TAG-PERE =" << aPere->mValTag << "\n";
			ELISE_ASSERT
				(
				false,
				"Unexptecd EOF in cElXMLTree::cElXMLTree"
				);
		}

		cElXMLTree * aFils = new cElXMLTree(DoFileCinclu,aUseSubst,aFp,aNewTok,this,anArg);

		//aFils->IsBranche();

		if (	   aFils
			&& aFils->IsBranche()
			&&  DoFileCinclu
			&& ( (aFils->mValTag=="IncludeFile")
			|| (aFils->mValTag=="IncludeFileLoc")
			)
			)
		{
			std::list<cElXMLTree *>::iterator it,it2;
			for ( it=aFils->mFils.begin(); it!=aFils->mFils.end() ; it++ )
			{
				ELISE_ASSERT((*it)->IsFeuille(),"Error in Include File");
				std::string aNF = (*it)->mValTag;
				if (aFils->mValTag=="IncludeFileLoc")
					aNF  = DirOfFile(aFp->Name()) + aNF;

				if (aUseSubstTree)
					anArg.DoSubst(aNF,true);
                
				cElXMLTree aPetitFils(aNF,&anArg);
                                for ( it2=aPetitFils.mFils.begin(); it2!=aPetitFils.mFils.end(); it2++ )
				{
					mFils.push_back(*it2);
					(*it2)->mPere = this;
				}
				aPetitFils.mFils.clear(); // sinon il vont etre detruits
			}  
			delete aFils;
		}
		else
                {
                   if (anArg.ModifTree())
                   {
                       std::list<cElXMLTree *> aNewFils = aFils->Interprete();
                       for 
                       (
                            std::list<cElXMLTree *>::const_iterator anIt=aNewFils.begin();
                            anIt!=aNewFils.end();
                            anIt++
                       )
	                   mFils.push_back(*anIt);
                    }
                    else
                    {
                       mFils.push_back(aFils);
                    }
                }
	}


	VerifCreation();
}

cElXMLTree::cElXMLTree
	(
	cElXMLTree * aPere,
	const std::string & aVal,
	eElXMLKindTree  aKind
	)  :
mValTag  (aVal),
	mPere    (aPere),
	mKind    (aKind)
{
}


void cElXMLTree::VerifCreation()
{
	if (!mFils.empty())
	{
		eElXMLKindTree aKind = (*mFils.begin())->mKind;
		for 
			(
			std::list<cElXMLTree *>::iterator it=mFils.begin(); 
		it!=mFils.end() ; 
		it++
			)
		{
			if (MergeForComp(aKind)!=MergeForComp((*it)->mKind))
			{
				if (aKind!=(*it)->mKind)
				{
					std::cout << "Tag=<" << mValTag << ">\n";
					ELISE_ASSERT
						(
						false,
						"Pas de support pour les noeud mixtes"
						);
				}
			}
		}
	}
}


// extern bool BUGGET;

cElXMLTree * cElXMLTree::Get(const std::string & aName,int aDepthMax)
{
// if (BUGGET) std::cout <<  "cElXMLTree::Get " << mValTag  << " " << aName << "\n";
	if (aName==mValTag)
		return this;

	if (aDepthMax==0) 
		return 0;

	for (std::list<cElXMLTree *>::iterator it=mFils.begin(); it!=mFils.end() ; it++)
	{
		cElXMLTree * aRes = (*it)->Get(aName,aDepthMax-1);
		if (aRes)
			return aRes;
	}
	return 0;
}

void cElXMLTree::GetAll(const std::string & aName,std::list<cElXMLTree *> & aRes,bool byAttr,int aDepthMax)
{
	if (IsBranche())
	{
		if (byAttr)
		{
			bool Got;
			const std::string & aVId = StdValAttr("Id",Got);
			if (Got && (aVId== aName))
				aRes.push_back(this);
		}
		else
		{
			if (aName==mValTag)
				aRes.push_back(this);
		}
	}

	if (aDepthMax==0) 
		return;


	for (std::list<cElXMLTree *>::iterator it=mFils.begin(); it!=mFils.end() ; it++)
	{
		(*it)->GetAll(aName,aRes,byAttr,aDepthMax-1);
	}
}

std::list<cElXMLTree *>  cElXMLTree::GetAll(const std::string & aName,bool byAttr,int aDepthMax)
{
	std::list<cElXMLTree *> aRes;
	GetAll(aName,aRes,byAttr,aDepthMax);
	return aRes;
}


const  std::list<cElXMLTree *>   &  cElXMLTree::Fils() const {return mFils;}
std::list<cElXMLTree *>   &  cElXMLTree::Fils() {return mFils;}


cElXMLTree * cElXMLTree::GetUnique(const std::string & aName,bool ByAttr)
{
	std::list<cElXMLTree *> aRes = GetAll(aName,ByAttr);
	if (aRes.size()!=1)
	{
		std::cout << " #####  Name-TAG = " << aName << " Nb= " << aRes.size() << "\n";
		ELISE_ASSERT(false,"cElXMLTree::GetUnique");
	}
	return *(aRes.begin());
}

cElXMLTree * cElXMLTree::GetUniqueFils()
{
	ELISE_ASSERT(mFils.size()==1,"cElXMLTree::GetUniqueFils");
	return *(mFils.begin());
}


cElXMLTree * cElXMLTree::GetOneOrZero(const std::string & aName)
{
	std::list<cElXMLTree *> aRes = GetAll(aName);
	if (aRes.empty())
		return 0;
	else if (aRes.size() == 1)
		return *(aRes.begin());
	ELISE_ASSERT(false,"cElXMLTree::GetOneOrZero");
	return 0;
}



const std::string & cElXMLTree::ValTag() const
{
	return mValTag;
}


bool  cElXMLTree::IsVide() const
{
	return  (mFils.size() == 0) ;
}
const std::string & cElXMLTree::Contenu() const
{
	bool OK = (mFils.size() == 1) && ((*(mFils.begin()))->IsFeuille());

	if (!OK)
	{
		cout << "--------FOR TAG = " << mValTag << "\n";
		ELISE_ASSERT ( OK, "Error in cElXMLTree::KemeContenu");
	}
	return (*(mFils.begin()))->mValTag;
}

std::string & cElXMLTree::NCContenu() 
{
	return const_cast<std::string &>(Contenu());
}

void cElXMLTree::AddFils(cElXMLTree * aFils)
{
	aFils->mPere = this;
	mFils.push_back(aFils);
}

cElXMLTree * cElXMLTree::ValueNode
	(
	const std::string & aNameTag,
	const std::string & aVal
	)
{
	cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)NULL,aNameTag,eXMLBranche);
	aRes->AddFils(new cElXMLTree(aRes,aVal,eXMLFeuille));

	return aRes;
}

cElXMLTree * cElXMLTree::MakePereOf(const std::string & aNameTag,cElXMLTree * aFils)
{
	cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)NULL,aNameTag,eXMLBranche);
	aRes->AddFils(aFils);

	return aRes;
}



std::string & cElXMLTree::GetUniqueVal() 
{

	if (mFils.size() != 1)
	{
		std::cout << "----------- [TAG=" << mValTag << "]\n";
		ELISE_ASSERT(mFils.size() == 1,"cElXMLTree::GetUniqueVal");
	}
	cElXMLTree * theFils =mFils.front();
	ELISE_ASSERT(theFils->mFils.size() == 0,"cElXMLTree::GetUniqueVal");

	return theFils->mValTag;

}


INT cElXMLTree::GetUniqueValInt()
{
	const std::string & aName = GetUniqueVal();
	return atoi(aName.c_str());
}
INT cElXMLTree::GetUniqueValInt(const std::string & aName)
{
	return Get(aName)->GetUniqueValInt();
}

double cElXMLTree::GetUniqueValDouble()
{
	const std::string & aName = GetUniqueVal();
	return atof(aName.c_str());
}
double cElXMLTree::GetUniqueValDouble(const std::string & aName)
{
	return Get(aName)->GetUniqueValDouble();
}

Pt2di cElXMLTree::GetPt2di()
{
	return Pt2di(GetUniqueValInt("x"),GetUniqueValInt("y"));
}


Box2di  cElXMLTree::GetDicaRect()
{
	int anX = GetUniqueValInt("x");
	int anY = GetUniqueValInt("y");
	int aW  = GetUniqueValInt("w");
	int aH  = GetUniqueValInt("h");

	return Box2di(Pt2di(anX,anY),Pt2di(anX+aW,anY+aH));
}

Pt2dr cElXMLTree::GetPt2dr()
{
	return Pt2dr(GetUniqueValDouble("x"),GetUniqueValDouble("y"));
}


ElCplePtsHomologues  cElXMLTree::GetCpleHomologues()
{
	return ElCplePtsHomologues
		(
		Pt2dr(GetUniqueValDouble("x1"),GetUniqueValDouble("y1")),
		Pt2dr(GetUniqueValDouble("x2"),GetUniqueValDouble("y2")),
		GetUniqueValDouble("pds")
		);
}

ElPackHomologue  cElXMLTree::GetPackHomologues(const std::string & aName)
{
	ElPackHomologue aRes;
	cElXMLTree * aTr = Get(aName);
	for 
		(
		std::list<cElXMLTree *>::const_iterator itTree = aTr->mFils.begin();
	itTree != aTr->mFils.end();
	itTree++ 
		)
	{
		// std::cout << "TAG = " << (*itTree)->mValTag << "\n";
		if ((*itTree)->mValTag =="CpleHom")
			aRes.Cple_Add((*itTree)->GetCpleHomologues());
	}
	int aSwap=0;
	cElXMLTree * aTrSwap=GetOneOrZero("SwapP1P2");
	if (aTrSwap)
	{
		aSwap = GetUniqueValInt("SwapP1P2");
	}
	if (aSwap)
		aRes.SelfSwap();
	return aRes;
}

cElComposHomographie cElXMLTree::GetElComposHomographie()
{
	return cElComposHomographie
		(
		GetUniqueValDouble("cX"),
		GetUniqueValDouble("cY"),
		GetUniqueValDouble("c1")
		);
}

cElHomographie cElXMLTree::GetElHomographie()
{
	return cElHomographie
		(
		GetUnique("XFonc")->GetElComposHomographie(),
		GetUnique("YFonc")->GetElComposHomographie(),
		GetUnique("ZFonc")->GetElComposHomographie()
		);
}


Monome2dReal cElXMLTree::GetElMonome2D(double & aCoeff,double anAmpl)
{
	aCoeff = GetUniqueValDouble("Coeff");
	return Monome2dReal
		(
		GetUniqueValInt("dX"),
		GetUniqueValInt("dY"),
		anAmpl
		);
}

Polynome2dReal cElXMLTree::GetPolynome2D()
{
	int aDegre = GetUniqueValInt("DegreTotal");
	double Ampl = GetUniqueValDouble("Amplitude");

	Polynome2dReal aPol(aDegre,Ampl);
	int aKP=0;
	for 
		(
		std::list<cElXMLTree *>::const_iterator itTree = mFils.begin();
	itTree != mFils.end();
	itTree++ 
		)
	{
		cElXMLTree & aTr = **itTree;
		if (aTr.ValTag()==  "Monome")
		{
			double aC;
			Monome2dReal aMon = aTr.GetElMonome2D(aC,Ampl);
			ELISE_ASSERT
				(
				(aMon.DegreX() == aPol.DegreX(aKP))
				&& (aMon.DegreY() == aPol.DegreY(aKP)),
				"Incoherence in XMP Pol2d"
				);
			aPol.SetCoeff(aKP,aC);

			aKP++;
		}
	}
	return aPol;
}



RImGrid* cElXMLTree::GetRImGrid(const std::string & aTagNameFile,const std::string& aDir)
{
	// cElXMLTree * aT = GetUnique("size"); aT->GetPt2di();
	// Pt2di aS2 = GetUnique("size")->GetPt2di();

	Pt2di aSz = GetUnique("size")->GetPt2di();
	Pt2dr anOri = GetUnique("origine")->GetPt2dr();
	Pt2dr  aStep = GetUnique("step")->GetPt2dr();
	std::string aName = aDir + GetUnique("filename")->Get(aTagNameFile)->GetUniqueVal();

	int Adapt =GetUniqueValInt("StepIsAdapted");


	RImGrid  * aRes  = new RImGrid
		(
		bool(Adapt != 0),
		anOri,
		anOri+ Pt2dr(aSz).mcbyc(aStep),
		aStep,
		aTagNameFile,
		aSz
		);


	ELISE_fp aFile(aName.c_str(),ELISE_fp::READ);
	aFile.read(aRes->DataGrid().data_lin(),sizeof(REAL8),aSz.x*aSz.y);
	aFile.close();
	return aRes;
}

PtImGrid cElXMLTree::GetPtImGrid(const std::string& aDir)
{
	return PtImGrid
		(
		GetRImGrid("x",aDir),
		GetRImGrid("y",aDir),
		"Toto"
		);
}

// INT  cElXMLTree::GetInt(const std::string & aName) { }




INT cElXMLTree::Profondeur() const
{
	INT aRes = 0;
	if (IsBranche()) aRes=1;
	for 
		(
		std::list<cElXMLTree *>::const_iterator itF=mFils.begin();
	itF != mFils.end();
	itF++
		)
		aRes = ElMax(aRes,1+(*itF)->Profondeur());

	return aRes;
}

void cElXMLTree::ShowOpeningTag(FILE * aFile)
{
	fprintf(aFile,"<%s",mValTag.c_str());
	for 
		(
		std::list<cElXMLAttr>::const_iterator itA=mAttrs.begin();
	itA!=mAttrs.end();
	itA++
		) 
	{
		fprintf(aFile," %s=\"%s\"",itA->mSymb.c_str(),itA->mVal.c_str());
	}
	fprintf(aFile,">");
}

void cElXMLTree::Show
	(
	const std::string & mIncr,
	FILE * aFile, 
	INT aCpt,
	INT LevMin,
	bool  isTermOnLine,
	const cElXMLTreeFilter & aSelector
	)
{
	if (! aSelector.Select(*this))
		return;

	if (LevMin<=0)
	{
		for (INT aK=0 ; aK<aCpt ; aK++)
			fprintf(aFile,"%s",mIncr.c_str());
	}
	if (isTermOnLine && (Profondeur() <= 1) && (! IsFeuille() ))
	{
		ShowOpeningTag(aFile);
		for 
			(
			std::list<cElXMLTree *>::iterator itF=mFils.begin();
		itF != mFils.end();
		itF++
			)
                {
                      PutCharPtrWithTraitSeqBinaire(aFile,(*itF)->mValTag.c_str());
/*
void PutCharWithSeqBinaire(FILE * aFp,const std::string & aVal)
			fprintf(aFile,"%s",(*itF)->mValTag.c_str());
*/
                }
		fprintf(aFile,"</%s>\n",mValTag.c_str());
		return;
	}
	// if (mFils.size()==0)
	if (IsFeuille())
	{
		if (LevMin<=0)
			fprintf(aFile,"%s\n",mValTag.c_str());
		return;
	}
	if (LevMin<=0)
	{
		ShowOpeningTag(aFile);
		fprintf(aFile,"\n");
	}

	for 
		(
		std::list<cElXMLTree *>::iterator itF=mFils.begin();
	itF != mFils.end();
	itF++
		)
	{
		(*itF)->Show(mIncr,aFile,aCpt+1,LevMin-1,isTermOnLine,aSelector);
	}

	if (LevMin<=0)
	{
		for (INT aK=0 ; aK<aCpt ; aK++)
			fprintf(aFile,"%s",mIncr.c_str());
		fprintf(aFile,"</%s>\n",mValTag.c_str());
	}
}

void cElXMLTree::Show
	(
	const std::string & mIncr,
	FILE * aFile, 
	INT aCpt,
	INT LevMin,
	bool  isTermOnLine
	)
{
	cElXMLTreeFilter aSelector;
	Show(mIncr,aFile,aCpt,LevMin,isTermOnLine,aSelector);
}

extern bool AutorizeTagSpecial(const std::string &);

void  cElXMLTree::StdShow(const std::string & aNameFile)
{
	FILE * aFP = ElFopen(aNameFile.c_str(),"wb");
	if (aFP==0)
	{
		std::cout << "FILE=[" << aNameFile << "]\n";
		ELISE_ASSERT(false,"Cannot Open File in cElXMLTree::StdShow");
	}
	fprintf(aFP,"<?xml version=\"1.0\" ?>\n");
	Show("     ",aFP,0,0,true);
	ElFclose(aFP);
}

cElXMLTree *  cElXMLTree::Missmatch
	(
	cElXMLTree *   aT2,
	bool           isSpecOn2,
	std::string &  aMes
	)
{
// std::cout << "VerifM " <<  ValTag() << " " << aTSpecif->ValTag() << "\n";
        if (
                 (aT2->ValAttr("Type","")=="XmlXml")
              || (ValAttr("Type","")=="XmlXml")
           )
          return 0;

	std::string aStrFalse = "false";
	bool  aUnionType =     (!isSpecOn2)
		&& (ValAttr("UnionType",aStrFalse)=="true") ;
	if (aUnionType && (aT2->mFils.size() != 1))
	{
		aMes = "UnionType must have exactly on descendant, Tag : " + aT2->ValTag();
		return this;
	}
	for 
		(
		std::list<cElXMLTree *>::iterator itF1=mFils.begin();
	itF1 != mFils.end();
	itF1++
		)
	{
		if (! (*itF1)->IsFeuille())
		{
			int aNbMatch =0;
			for 
				(
				std::list<cElXMLTree *>::iterator itF2=aT2->mFils.begin();
			itF2 != aT2->mFils.end();
			itF2++
				)
			{
				if ((*itF1)->mValTag == (*itF2)->mValTag)
				{
					aNbMatch++;
					cElXMLTree * aRes = (*itF1)->Missmatch(*itF2,isSpecOn2,aMes);
					if (aRes)
					{
						return aRes;
					}
				}
			}
			if (isSpecOn2)
			{
				if (aNbMatch!=1)
				{
					if (AutorizeTagSpecial((*itF1)->mValTag))
						return 0;
					//  if ((*itF1)->mValTag=="AutorizeSplitRec") return 0;
					// std::cout << "TAG = " << (*itF1)->mValTag << "\n";
					aMes = (aNbMatch==0) ? 
						"Value has no match in specifs" : 
					"Value has multiple match in specifs" ; 
					return *itF1;
				}
			}
			else
			{
				const std::string & aPat = (*itF1)->ValAttr("Nb");
				if (! ValidNumberOfPattern(aPat,aNbMatch))
				{
					aMes =    "Number " + ToString(aNbMatch) 
						+ " does not match pattern [" + aPat +"]"; 
					return *itF1;
				}
				if (aUnionType && (aPat != "?"))
				{
					aMes = "Number spec must be [?] with union Type";
					return *itF1;
				}
			}
		}
		else
		{
		}
	}
	return 0;
}

bool  cElXMLTree::VerifMatch(cElXMLTree* aTSpecif,bool SVP)
{

	std::string aMes;

	cElXMLTree * aMM = Missmatch(aTSpecif,true,aMes);

	if (! aMM) 
	{
		aMM = aTSpecif->Missmatch(this,false,aMes);
	}

	if (! aMM) return true;
        if (SVP) return false;

	cout << "*********************************************************\n";
	cout << "\n";
	cout << "   Erreur dans le matching entre XML et XML-Specif\n";
	cout << "   Error = [" << aMes << "]\n";
	aMM->ShowAscendance(stdout);
	cout << "*********************************************************\n";
	std::cout << "SPECIF:" ; aTSpecif->ShowAscendance(stdout);
	ELISE_ASSERT(false,"Exit XML Matching Specif Error");

        return false;
}

bool  cElXMLTree::TopVerifMatch
	(
	   const std::string & aNameObj,
	   cElXMLTree* aTSpecif,
	   const std::string & aNameType,
	   bool  ByAttr,
           bool SVP
	)
{
	std::list<cElXMLTree *> aL1 = GetAll(aNameObj,ByAttr);
	std::list<cElXMLTree *> aL2 = aTSpecif->GetAll(aNameType);

	if ((aL1.size() !=1) || (aL2.size() !=1))
	{
            if (SVP)
            {
               return false;
            }
            else
            {
		ShowAscendance(stdout);
		cout << "ERROR at top level in TopVerifMatch for " 
			<< aNameObj << "-" << aNameType << "\n";
		cout << " Found " << (unsigned int) aL1.size() << " in instance \n";
		cout << " Found " << (unsigned int) aL2.size() << " in specif \n";
		ELISE_ASSERT(false,"ERROR at top level in TopVerifMatch");
            }
	}
	return (*(aL1.begin()))->VerifMatch(*(aL2.begin()),SVP);
}

bool  cElXMLTree::TopVerifMatch(cElXMLTree* aTSpecif,const std::string & aName,bool SVP)
{
	return TopVerifMatch(aName,aTSpecif,aName,false,SVP);
}


void cElXMLTree::ShowAscendance(FILE * aFp)
{
	INT aK=0;
	for(cElXMLTree * aTr = this;aTr; aTr=aTr->mPere)
	{
		if (aK)
			fprintf(aFp," -> ");
		fprintf(aFp,"%s",aTr->mValTag.c_str());
		aK++;
	}
	fprintf(aFp,"\n");
        if (VCurXmlFile.size())
           fprintf(aFp,"in file : %s\n",VCurXmlFile.back().c_str());
}

const std::string & cElXMLTree::ValAttr
	(
	const std::string & anAttr,
	const std::string * aDef
	) const
{
	for 
		(
		std::list<cElXMLAttr>::const_iterator itA = mAttrs.begin();
	itA != mAttrs.end();
	itA++
		)
	{
		if (itA->mSymb == anAttr)
			return itA->mVal;
	}

	if (aDef==0)
	{
		std::cout << "Tag=" << mValTag << " Attibut manquant=" << anAttr << "\n";
		ELISE_ASSERT(false,"No Match in cElXMLTree::ValAttr");
	}
	return *aDef;
}

bool cElXMLTree::SetAttr
	(
	const std::string & aSymb,
	const std::string & aVal
	)
{
	for 
		(
		std::list<cElXMLAttr>::iterator itA = mAttrs.begin();
	itA != mAttrs.end();
	itA++
		)
	{
		if (itA->mSymb == aSymb)
		{
			itA->mVal = aVal;
			return false;
		}
	}

	cElXMLAttr anAttr;
	anAttr.mSymb = aSymb;
	anAttr.mVal = aVal;
	mAttrs.push_back(anAttr);

	return true;
}



const std::string & cElXMLTree::ValAttr
	(
	const std::string & anAttr,
	const std::string & aDef
	) const
{
	return ValAttr(anAttr,&aDef);
}

const std::string & cElXMLTree::StdValAttr
	(
	const std::string & aName,
	bool & aGot
	) const
{
	static std::string aNoDef = "{}MP$$@#//[]ubamshaioxPKh167c_";
	const std::string & aRes = ValAttr(aName,aNoDef);
	aGot = (aRes != aNoDef);
	return aRes;
}
bool cElXMLTree:: HasAttr(const std::string & aName) const
{
	bool aGot;
	StdValAttr(aName,aGot);
	return aGot;
}

const std::string & cElXMLTree::ValAttr
	(
	const std::string & anAttr
	) const
{
	return ValAttr(anAttr,0);
}

bool cElXMLTree::ValidNumberOfPattern(const std::string & aPat,int aN)
{
	if (aPat=="1") return aN == 1;
	if (aPat=="?") return aN <= 1;
	if (aPat=="+") return aN >= 1;
	if (aPat=="*") return aN >= 0;

	ELISE_ASSERT(false,"Bad pattern");

	return false;
}


bool SplitIn2ArroundEqSvp
	(
	const std::string  &  a2Stplit,
	char            aCar,
	std::string  &  aBefore,
	std::string  &  aAfter
	)
{
	aBefore="";
	aAfter="";
	bool GotEq = false;
	const char * anArg = a2Stplit.c_str();
	while (*anArg)
	{
		if (*anArg==aCar) 
		{
			if (GotEq)
				aAfter  += *anArg;
			GotEq = true;
		}
		else
		{
			if (GotEq)
				aAfter  += *anArg;
			else
				aBefore += *anArg;
		}
		anArg++;
	}
	return GotEq;
}




bool SplitIn2ArroundCar
	(
	const std::string  &  a2Stplit,
	char                  aSpliCar,
	std::string  &  aBefore,
	std::string  &  aAfter,
	bool            AcceptNoCar  // Est on OK pour ne pas trouver aSpliCar
	// dans ce cas  aAfter est vide
	)
{
	bool aGot = SplitIn2ArroundEqSvp(a2Stplit,aSpliCar,aBefore,aAfter);
	if ((!aGot) && (!AcceptNoCar))
	{
		std::cout << "STRING=[" << a2Stplit << "] CAR=" << aSpliCar << "\n";
if (MPD_MM())
{
    std::cout << "MPD_MMMPD_MMMPD_MM\n";
    getchar();
}
		ELISE_ASSERT(false,"Cannot split");
	}
	return aGot;
}

void  SplitInNArroundCar
	(
	const std::string  &  a2Stplit,
	char                  aSpliCar,
	std::string   &             aR0,
	std::vector<std::string>  &  aRAux
	)
{
	aRAux.clear();
	std::string aRest;
	bool aGotSplit = SplitIn2ArroundCar(a2Stplit,aSpliCar,aR0,aRest,true);
	while (aGotSplit)
	{
		std::string aBefore,aAfter;
		aGotSplit = SplitIn2ArroundCar(aRest,aSpliCar,aBefore,aAfter,true);
		aRAux.push_back(aBefore);
		aRest = aAfter;
	}
}


void NewSplit( const std::string  &  a2Stplit,std::string & aK0,std::vector<std::string>  & aSup)
{
	int aNbPar = 0;
	aK0 = "";
	aSup.clear();

	bool GotEq = false;
	const char * anArg = a2Stplit.c_str();
	while (*anArg)
	{
		if ((anArg[0]=='[') && (anArg[1]=='['))
		{
			anArg++;
			aNbPar++;
		}
		else if ((anArg[0]==']') && (anArg[1]==']'))
		{
			anArg++;
			aNbPar--;
			ELISE_ASSERT(aNbPar>=0,"Badly parenthesis expr in NewSplit")
		}
		else if ((anArg[0]=='@') && (aNbPar==0))
		{
			aSup.push_back("");
			GotEq = true;
		}
		else
		{
			if (GotEq)
				aSup.back()  += *anArg;
			else
				aK0 += *anArg;
		}
		anArg++;
	}
	ELISE_ASSERT(aNbPar==0,"Badly parenthesis expr in NewSplit")
}



void SplitIn2ArroundEq
	(
	const std::string  &  a2Stplit,
	std::string  &  aBefore,
	std::string  &  aAfter
	)
{
	SplitIn2ArroundCar(a2Stplit,'=',aBefore,aAfter,false);

	/*
	bool aGot=SplitIn2ArroundEqSvp(a2Stplit,'=',aBefore,aAfter);
	ELISE_ASSERT(aGot,"Pas trouve = dans SplitIn2ArroundEq");
	*/
}

std::string  GetValLC
	(
	int argc,
	char **argv,
	const std::string & aKey,
	const std::string & aDef
	)
{
	for (int aK=0 ; aK<argc; aK++)
	{
            if (argv[aK][0] != cInterfChantierNameManipulateur::theCharSymbOptGlob)
            {
		std::string aSymb;
		std::string aVal;
		SplitIn2ArroundEq(std::string(argv[aK]),aSymb,aVal);
		if (aSymb==aKey)
			return aVal;
            }
	}
	return aDef;
}


// Modification par ligne de commande

void cElXMLTree::ModifLC(char * anArg,cElXMLTree * aSpecif)
{
        if (anArg[0] == cInterfChantierNameManipulateur::theCharSymbOptGlob ) return;
// std::cout << "cElXMLTree::ModifLC " << anArg << "\n";

	if (anArg[0] == cInterfChantierNameManipulateur::theCharModifDico)
		return;
	// std::cout << "ARG="<<  anArg<< "\n";
	char * anArgInit = anArg;
	bool aUseSpecif = true;
	bool aUseAttr   = false;

	if ((anArg[0] == '@') || (anArg[0] == '%'))
	{
		aUseSpecif = false;
		if (anArg[0] == '%')
			aUseAttr  = true;
		anArg++;
	}


	std::string aSymb;
	std::string aVal;
	SplitIn2ArroundEq(std::string(anArg),aSymb,aVal);


	cElXMLTree * aSymSp=0;

	if (aUseSpecif)
	{
		std::list<cElXMLTree *>  aLSymSp = aSpecif->GetAll(aSymb);

		if (aLSymSp.size() != 1)
		{
                        if (aSpecif) aSpecif->StdShow("ShowSpec.xml");
			cout << "ModifLC :: MATCH ERROR FOR SYMB [" << aSymb << "] : ";
			if (aLSymSp.empty())
				cout << "NO MATCH\n";
			else
                        {
				cout << "MULTIPLE MATCH " << aLSymSp.size() << "\n";
                        }
			ELISE_ASSERT(false,"MATCH ERROR dans la ligne de commande");
		}
		aSymSp = *(aLSymSp.begin());

		{
			const std::string & aPat = aSymSp->ValAttr("Nb");

			ELISE_ASSERT 
				(
				(aPat=="1") || (aPat=="?"),
				"XML-MATCH ERROR dans la ligne de commande (Specif)\n"
				"Le symbole a modifier autorise la cardinalite multiple"
				);
		}

		if (aSymSp->Profondeur()!=1)
		{
			cout << "ModifLC :: MATCH ERROR FOR SYMB [" << aSymb << "] (Specif)\n ";
			cout << aSymSp->Profondeur() << "\n";
			cout << "Le symbole n'est pas terminal\n";
			ELISE_ASSERT(false,"NON TERMINAL SYMBOL");
		}

		for 
			(
			cElXMLTree * aS = aSymSp->mPere ; 
		aS && (aS->IsBranche()) ; 
		aS= aS->mPere
			)
		{
			std::string aS1("1");
			const std::string & aPat = aS->ValAttr("Nb",aS1);
			ELISE_ASSERT 
				(
				(aPat=="1"),
				"XML-MATCH ERROR dans la ligne de commande (Specif)\n"
				"Un ascendant du symbole a modifier autorise les cardinalites differente de 1"
				);
		}
	}

	std::list<cElXMLTree *>  aLSymVal = GetAll(aSymb,aUseAttr);

	ELISE_ASSERT
		(
		aLSymVal.size() <=1,
		"XML-MODIF LC MATCH ERROR, Multiple found (Value)"
		);

	if (aLSymVal.empty())
	{
		if (aSymSp==0)
		{
			std::cout << anArgInit << "\n";
			ELISE_ASSERT
				(
				aSymSp,
				"Pas de gestion implicite en mode par valeur"
				);
		}
		cElXMLTree * aPereSymSp = aSymSp->mPere;
		ELISE_ASSERT(aPereSymSp,"XML-MODIF LC MATCH ERROR, STRANGE ERROR !!");
		aLSymVal = GetAll(aPereSymSp->mValTag);

		if (aLSymVal.size() > 1)
		{
			cout << "ModifLC :: MATCH ERROR FOR SYMB [" << aSymb << "] : ";
			if (aLSymVal.empty())
				cout << "NO MATCH\n";
			else
				cout << "MULTIPLE MATCH\n";
			ELISE_ASSERT(false,"MATCH ERROR dans la ligne de commande");
		}

		cElXMLTree * aPereSVal = *aLSymVal.begin();
		aPereSVal->mFils.push_back
			(
			new cElXMLTree(aPereSVal,aSymb,eXMLBranche)
			);

		aLSymVal = GetAll(aSymb);
	}
	cElXMLTree * aSymVal = *(aLSymVal.begin());

	while (! aSymVal->mFils.empty())
	{
		delete aSymVal->mFils.back();
		aSymVal->mFils.pop_back();
	}
	aSymVal->mFils.push_back
		(
		new cElXMLTree(aSymVal,aVal,eXMLFeuille)
		);
}

void  cElXMLTree::Debug(const std::string & aMes)
{
	for 
		(
		std::list<cElXMLTree *>::iterator itF1=mFils.begin();
	itF1 != mFils.end();
	itF1++
		)
		cout << "Fils " <<  aMes << " " << (*itF1)->mValTag 
		<< " " << (void *) (*itF1)
		<< "\n";
	cout << "This = [" << mValTag << "] " << (void *) this << "\n";
}

void cElXMLTree::ModifLC(int argc,char ** argv,cElXMLTree * aSpecif)
{
	for (int aK=0 ; aK<argc ; aK++)
		ModifLC(argv[aK],aSpecif);
}


bool GetOneModifLC
	(
	int argc,
	char ** argv,
	const std::string & aNameSymb,
	std::string &       aVal
	)
{
	std::string aBefore;
	std::string aAfter;
	for (int aK=0 ; aK<argc ; aK++)
	{
            if (argv[aK][0]!= cInterfChantierNameManipulateur::theCharSymbOptGlob)
            {
		SplitIn2ArroundEq(argv[aK],aBefore,aAfter);
		if (aBefore==aNameSymb)
		{
			aVal=aAfter;
			return true;
		}
            }
	}
	return false;
}

void cElXMLTree::breadthFirstFunction( Functor &i_functor )
{
	list<cElXMLTree *> nodeList;
	cElXMLTree *node;

	// first node is the current node
	nodeList.push_back(this);
	unsigned int nbNodes = 1;

	while ( nbNodes!=0 ){
		// get first node from the list and remove it
		node = nodeList.front();
		nodeList.pop_front();
		nbNodes--;

		// add node's children to the list
		list<cElXMLTree *>::iterator itChild = node->mFils.begin();
		while ( itChild!=node->mFils.end() ){
			nodeList.push_back( *itChild++ );
			nbNodes++;
		}

		// process node
		i_functor( *node );
	}
}


/***********************************************************/
/*                                                         */
/*                    cElXMLFileIn                         */
/*                                                         */
/***********************************************************/

cElXMLFileIn::cElXMLFileIn(const std::string & aName) :
mFp (ElFopen(aName.c_str(),"wb")),
	mCurIncr   (0),
	mStrIncr   ("   ")
{
	if (mFp==0)
	{
		cout << "Name [" << aName <<  "]\n";
		ELISE_ASSERT(mFp!=0,"Cannot Open cElXMLFileIn");
	}
}

cElXMLFileIn::~cElXMLFileIn()
{
	ElFclose(mFp);
}

//   Grids ...

void cElXMLFileIn::PutGrid(const PtImGrid & aGr,const std::string & aName)
{
	cTag aTag(*this,aName);aTag.NoOp();
	{
		cTag aTagGr(*this,"grid");aTagGr.NoOp();
		{
			cTag aTagFName(*this,"filename");aTagFName.NoOp();
			PutString(aGr.NameX()+".dat","x");
			PutString(aGr.NameY()+".dat","y");
		}
		PutPt2di(aGr.SzGrid(),"size");
		PutPt2dr(aGr.Origine(),"origine");
		PutPt2dr(aGr.Step(),"step");
		PutInt(aGr.StepAdapted(),"StepIsAdapted");
	}

}
void cElXMLFileIn::PutDbleGrid
	(
	bool XMLAutonome,
	const cDbleGrid & aGR2,
	const std::string & aName
	)
{
	cTag aTag(*this,aName);aTag.NoOp();
	if (XMLAutonome)
	{
		cGridDirecteEtInverse aXMLGr = ToXMLExp(aGR2);
		cElXMLTree * aTree = ToXMLTree(aXMLGr);
		PutTree(aTree);
		delete aTree;
	}
	else
	{
		PutGrid(aGR2.GrDir(),"grid_directe");
		PutGrid(aGR2.GrInv(),"grid_inverse");
	}
}

void cElXMLFileIn::SensorPutDbleGrid
	(
	Pt2di aSzIm,
	bool XMLAutonome,
	cDbleGrid & aGR2,
	const char * aNameTF,
	const char * aNameXMLCapteur,
	ElDistRadiale_PolynImpair * aDistRad,
	Pt2dr * aPPExt,
	double  * aFocExt
	)
{
	cTag aTag(*this,"sensor");aTag.NoOp();
	if (aNameTF)
	{
		ThomParam aTP(aNameTF);
		if (! aNameXMLCapteur)
		{
			PutString(std::string("camera ")+ToString(aTP.mCAMERA),"name");
			PutString(aTP.ORIGINE,"origine");
			PutString(aTP.DATE,"calibration-date");
			PutString("LaTeteAToto","serial-number");
		}
		PutString("0","argentique");
		PutString(aTP.OBJECTIF,"objectif");
	}

	if (aNameXMLCapteur)
	{
		cElXMLTree aTree(aNameXMLCapteur);
		aTree.Show(mStrIncr,mFp,-1,2,true);
	}

	REAL aFocale = aFocExt ? (*aFocExt) : aGR2.Focale();
	Pt2dr  aPP   = aPPExt  ? (*aPPExt)  : aGR2.PP();
	{
		cTag aTag(*this,"focal");aTag.NoOp();
		{
			cTag aTag(*this,"pt3d");aTag.NoOp();
			PutDouble(aPP.x,"x");
			PutDouble(aPP.y,"y");
			PutDouble(aFocale,"z");
		}
	}
	if (aDistRad)
	{
		PutString("Radiale","TypeDistortion");
		PutDist(*aDistRad);
	}
	else
	{
		PutString("Non Modelisee","TypeDistortion");
	}
	PutDbleGrid(XMLAutonome,aGR2);

	if (XMLAutonome && aDistRad)
	{
		cCalibrationInternConique aCIC;
		cCalibrationInterneRadiale aCIR;
		aCIC.PP() = aPP;
		aCIC.F()  = aFocale;
		aCIC.SzIm() = aSzIm;
		aCIR.CDist() = aDistRad->Centre();
		for (INT aK=0 ; aK<aDistRad->NbCoeff() ; aK++)
		{
			double aCK = aDistRad->Coeff(aK);
			// std::cout << aCK << "\n"; getchar();
			aCIR.CoeffDist().push_back(aCK);
		}
		cCalibDistortion aCD;
		aCD.ModRad().SetVal(aCIR);
		aCD.ModPhgrStd().SetNoInit();
		aCD.ModUnif().SetNoInit();

		aCIC.CalibDistortion().push_back(aCD);

		cElXMLTree * aTree = ToXMLTree(aCIC);
		PutTree(aTree);
		delete aTree;
	}
}
//   Cam  etc ...

void cElXMLFileIn::PutDist(const ElDistRadiale_PolynImpair & aDist)
{
	cTag aTag(*this,"distortion");aTag.NoOp();
	for (INT aK=0 ; aK<aDist.NbCoeff() ; aK++)
		PutDouble(aDist.Coeff(aK),std::string("r")+ToString(3+2*aK));
	PutPt2dr(aDist.Centre());
}

void cElXMLFileIn::PutCamGen(const CamStenope  & aCam)
{
	Pt2dr aPP = aCam.PP();

	cTag aTag(*this,"focal"); aTag.NoOp();
	PutPt3dr(Pt3dr(aPP.x,aPP.y,aCam.Focale()));
}

void cElXMLFileIn::PutCam(const cCamStenopeDistRadPol  & aCam)
{
	cTag aTag(*this,"sensor");aTag.NoOp();
	PutCamGen(aCam);
	PutDist(aCam.DRad());
}


// -------------------------------
//   PutPt2dr PutPt3dr
// -------------------------------

void cElXMLFileIn::PutPt2dr(const Pt2dr & aP,const std::string & aNameTag)
{
	cTag aTag(*this,aNameTag);aTag.NoOp();
	PutDouble(aP.x,"x");
	PutDouble(aP.y,"y");
}

void cElXMLFileIn::PutPt2di(const Pt2di & aP,const std::string & aNameTag)
{
	cTag aTag(*this,aNameTag);aTag.NoOp();
	PutInt(aP.x,"x");
	PutInt(aP.y,"y");
}

void cElXMLFileIn::PutPt3dr(const Pt3dr & aP,const std::string & aNameTag)
{
	cTag aTag(*this,aNameTag);aTag.NoOp();
	PutDouble(aP.x,"x");
	PutDouble(aP.y,"y");
	PutDouble(aP.z,"z");
}

// PutString  PutInt PutDouble

void cElXMLFileIn::PutString(const std::string & aVal,const std::string & aNameTag)
{
	cTag aTag(*this,aNameTag,true);aTag.NoOp();
	fprintf(mFp,"%s",aVal.c_str());
}
void cElXMLFileIn::PutInt(const INT & aVal,const std::string & aNameTag)
{
	cTag aTag(*this,aNameTag,true);aTag.NoOp();
	fprintf(mFp,"%d",aVal);
}

void cElXMLFileIn::PutTabInt(const std::vector<INT> & aTab,const std::string & aNameTag)
{
	cTag aTag(*this,aNameTag,true);aTag.NoOp();
	for (INT aK=0 ; aK<INT(aTab.size()) ;aK++)
	{
		if (aK>=1)
			fprintf(mFp," ");
		fprintf(mFp,"%d",aTab[aK]);
	}

}

void cElXMLFileIn::PutCpleHom
	(
	const ElCplePtsHomologues & aCple,
	const std::string & aNameTag
	)
{
	cTag aTag(*this,aNameTag,false);aTag.NoOp();
	PutDouble(aCple.P1().x,"x1");
	PutDouble(aCple.P1().y,"y1");
	PutDouble(aCple.P2().x,"x2");
	PutDouble(aCple.P2().y,"y2");
	PutDouble(aCple.Pds(),"pds");
}

void cElXMLFileIn::PutPackHom
	(
	const ElPackHomologue & aPack,
	const std::string & aNameTag
	)
{
	cTag aTag(*this,aNameTag,false);aTag.NoOp();
	for 
		(
		ElPackHomologue::const_iterator anIt = aPack.begin();
	anIt!= aPack.end();
	anIt++
		)
		PutCpleHom(anIt->ToCple());
}



void cElXMLFileIn::PutDouble(const double & aVal,const std::string & aNameTag)
{
	cTag aTag(*this,aNameTag,true);aTag.NoOp();
	fprintf(mFp,"%.9e",aVal);
}

//  ##    Show(const std::string & mIncr,FILE *,INT aCpt,INT aLevelMin,bool isTermOnLine)
//  StdShow
void cElXMLFileIn::PutTree (cElXMLTree * aTree)
{
	aTree->Show(mStrIncr,mFp,mCurIncr,0,true);
}




void cElXMLFileIn::PutMonome
	(
	const Monome2dReal & aMonome,
	const double & aCoeff
	)
{
	cTag aTag(*this,"Monome",false);aTag.NoOp();
	PutInt(aMonome.DegreX(),"dX");
	PutInt(aMonome.DegreY(),"dY");
	PutDouble(aCoeff,"Coeff");
}


void cElXMLFileIn::PutPoly
	(
	const Polynome2dReal & aPoly,
	const std::string &  aName
	)
{
	cTag aTag(*this,aName,false);aTag.NoOp();
	PutDouble(aPoly.Ampl(),"Amplitude");
	PutInt(aPoly.DMax(),"DegreTotal");
	for (int aK=0 ; aK<aPoly.NbMonome() ; aK++)
		PutMonome(aPoly.KthMonome(aK),aPoly.Coeff(aK));
}


void cElXMLFileIn::PutElComposHomographie
	(
	const cElComposHomographie & aCmp,
	const std::string & aNameTag
	)
{
	cTag aTag(*this,aNameTag);aTag.NoOp();
	{
		PutDouble(aCmp.CoeffX(),"cX");
		PutDouble(aCmp.CoeffY(),"cY");
		PutDouble(aCmp.Coeff1(),"c1");
	}
}

void cElXMLFileIn::PutElHomographie
	(
	const cElHomographie & aHom,
	const std::string &    aNameTag
	)
{
	cTag aTag(*this,aNameTag);aTag.NoOp();
	{
		PutElComposHomographie(aHom.HX(),"XFonc");
		PutElComposHomographie(aHom.HY(),"YFonc");
		PutElComposHomographie(aHom.HZ(),"ZFonc");
	}
}





// Basic : PutIncr  PutTagBegin  PutTagEnd

void cElXMLFileIn::PutIncr()
{
	for (INT kI =0 ; kI<mCurIncr ; kI++)
		fprintf(mFp,"%s",mStrIncr.c_str());
}

void cElXMLFileIn::PutTagBegin(const std::string & aTag,bool SimpleTag)
{
	PutIncr();
	fprintf(mFp,"<%s>",aTag.c_str());
	if (! SimpleTag)
		fprintf(mFp,"\n");
	mCurIncr++;
}

void cElXMLFileIn::PutTagEnd(const std::string & aTag,bool SimpleTag)
{
	mCurIncr--;
	if (! SimpleTag)
		PutIncr();
	fprintf(mFp,"</%s>",aTag.c_str());
	fprintf(mFp,"\n");
}


//-------------------------------------
//            cTag
//-------------------------------------

cElXMLFileIn::cTag::cTag(cElXMLFileIn & aFile,const std::string & aName,bool aSimpleTag) :
mFile      (aFile),
	mName      (aName),
	mSimpleTag (aSimpleTag)
{
	mFile.PutTagBegin(mName,mSimpleTag);
}

cElXMLFileIn::cTag::~cTag()
{
	mFile.PutTagEnd(mName,mSimpleTag);
}
void cElXMLFileIn::cTag::NoOp(){}


void PreloadXML
	(
	const std::string & aDir,
	const std::string & aNameFile,
	const std::string & aNameTagFileSup,
	const std::string & // aNameTagFileSup
	)
{
}









/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
