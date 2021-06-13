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




std::string RegExAdapt(const std::string & aStrInit,bool CaseSens)
{
    std::string aNameExprIn = CaseSens ? aStrInit : tolower(aStrInit);
    const char * aC = aNameExprIn.c_str();
    int aNb = (int)strlen(aC);

    return
                 ( ((aNb==0) || (aC[0]!='^'))  ? "^" : "")
             +   aNameExprIn
             +   ( ((aNb==0) || (aC[aNb-1]!='$'))  ? "$" : "") ;
}

string regcompErrorToString(int aRegcompResult)
{
	switch (aRegcompResult)
	{
	case REG_BADPAT: return "BADPAT";
	case REG_ECOLLATE: return "ECOLLATE";;
	case REG_ECTYPE: return "ECTYPE";
	case REG_EESCAPE: return "EESCAPE";
	case REG_ESUBREG: return "ESUBREG";
	case REG_EBRACK: return "EBRACK";
	#ifdef REG_ENOSYS
		case REG_ENOSYS: return "ENOSYS";
	#endif
	case REG_EPAREN: return "EPAREN";
	case REG_EBRACE: return "EBRACE";
	case REG_BADBR: return "BADBR";
	case REG_ERANGE: return "ERANGE";
	case REG_ESPACE: return "ESPACE";
	case REG_BADRPT: return "BADRPT";
	#ifdef REG_EMPTY
		case REG_EMPTY: return "REG_EMPTY";
	#endif
	#ifdef REG_ASSERT
		case REG_ASSERT: return "REG_ASSERT";
	#endif
	#ifdef REG_INVARG
		case REG_INVARG: return "REG_INVARG";
	#endif
	default: return string("UNKNOWN_ERROR(") + ToString(aRegcompResult) + ")";
	}
}

cElRegex::cElRegex(const std::string & aNameExprIn,int aNbMatchMax,int aCFlag,bool CaseSens) :
     mNameExpr      (RegExAdapt(aNameExprIn,CaseSens)),
     mCaseSensitive (CaseSens)
{

   mResMatch = -1;
   mOkReplace = false;

   mResCreate = regcomp(&mAutom,mNameExpr.c_str(),aCFlag);
   
	#ifdef __DEBUG
		if (mResCreate != 0)
		{
			stringstream ss;
			ss << "regcomp(" << mNameExpr << ") = ";
			ELISE_DEBUG_WARNING(true, "cElRegex::cElRegex", string("regcomp(") + mNameExpr + ") = " << regcompErrorToString(mResCreate));
		}
	#endif

   if (! IsOk())
      return;
   regmatch_t aMatch;
   mVMatch.reserve(aNbMatchMax);
   for (int aK=0; aK<aNbMatchMax ; aK++)
       mVMatch.push_back(aMatch);
}

cElRegex::~cElRegex()
{
   regfree(&mAutom);
}

bool cElRegex::IsOk() const
{
   return mResCreate==0;
}

void cElRegex::AssertOk() const
{
   if (!IsOk())
   {
      std::cout << "EXPR=" << mNameExpr << "\n";
      ELISE_ASSERT(false,"Expression is not valide");
   }
   // assert(IsOk());
}

bool cElRegex::IsMatched() const
{
   return mResMatch==0;
}

void cElRegex::AssertIsMatched() const
{
   ELISE_ASSERT(IsMatched(),"cElRegex::AssertIsMatched");
   // assert(IsMatched());
}

bool cElRegex::IsReplaced() const
{
    return mOkReplace;
}
void cElRegex::AssertIsReplaced() const
{
   ELISE_ASSERT(IsReplaced(),"cElRegex::AssertIsReplaced");
   // assert(IsReplaced());
}

bool cElRegex::Match(const std::string & aNameInit,int aCFlag)  
{
   std::string aName = mCaseSensitive ? aNameInit : tolower(aNameInit);
   AssertOk();
   mResMatch = regexec(&mAutom,aName.c_str(),mVMatch.size(),&mVMatch[0],aCFlag);
   if (IsMatched())
      mLastNameMatched = aName;
   return IsMatched();
}

const std::vector<regmatch_t>  & cElRegex::VMatch() const
{
   AssertIsMatched();
   return mVMatch;
}

//int Str2I(const std::string & aStr)
//{
//}

template <class Type> bool FromString(Type& x,const std::string & s);
template <class Type> std::string ToString(const Type &);


std::string cElRegex::KIemeExprPar(int aNum)
{
    if (! ((aNum >=0) && (aNum<int(mVMatch.size()))))
    {
       std::cout << "NUM="<< aNum << " NB=" << mVMatch.size() << "\n";
       ELISE_ASSERT
       (
          (aNum >=0) && (aNum<int(mVMatch.size())),
          "Bad Num in cElRegex::KIemeExprPar"
       );
    }
    int aI0 = mVMatch[aNum].rm_so;
    int aI1 = mVMatch[aNum].rm_eo;

              
    ELISE_ASSERT
    (
        (aI0>=0)&&(aI1>=0),
	"Internal error KIemeExprPar"
    );
    return   mLastNameMatched.substr(aI0,aI1-aI0);
}

static const char * Month[12] = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};

double  cElRegex::VNumKIemeExprPar(int aNum,bool aVerif,double * aDefError)
{
  
   double aRes;

   bool Inv= false;
   std::string aStr = KIemeExprPar(aNum);


   if ((aStr.size() > 2) && ((aStr[0]=='1')||(aStr[0]=='f')) && (aStr[1]=='/') )
   {
       aStr = aStr.substr(2,aStr.size());
       Inv=true;
   }


   bool aOK=  FromString(aRes,aStr);
   if ((! aOK) && (aVerif))
   {
      for (int aK=0 ; aK<12; aK++)
          if (aStr==Month[aK])
             return aK+1;

       if (aStr=="inf")
       {
          if (Inv) return 0;
          return InfRegex;
       }
      {
         if (aDefError) 
            return *aDefError;
         ELISE_ASSERT
         (
             false,
             "Not Num in cElRegex::VNumKIemeExprPar"
         );
      }
   }
   return  Inv ? (1/aRes)  :aRes;
}

std::string  ToStringNBD(int aNb,int aNbDig)
{
   std::string aRes = ToString(aNb);
   int aSz = (int)aRes.size();
   ELISE_ASSERT(aSz<=aNbDig,"Pas assez de digit dans ToStringNBD");
   for (;aSz<aNbDig ; aSz++)
   {
       aRes = "0" + aRes;
   }
   return aRes;
}

const std::string & cElRegex::Error() const
{
   return mError;
}

bool  cElRegex::Replace(const std::string & aMotifInit)  
{
   std::string aMotif = mCaseSensitive ? aMotifInit : tolower(aMotifInit);
   mError = "";
   AssertIsMatched();
   mOkReplace = false;
   mLastReplace = "";
   for (const char * aC=aMotif.c_str(); *aC ; aC++)
   {
       if (*aC=='$')
       {
          aC++;
          if (*aC=='$')
          {
             mLastReplace += *aC;
          }
          else
          {
             int aNum=-1;

             int aOffset=0;
	     bool isArithm = false;
	     int  aNbDigit = -1;

             char aCAritm = '0';

	     //  $+234+1 -> ajoute 234 a $1
	     //  $-456-A -> enleve 456 a $A
	     //  $%5+234+6 -> ajoute 234 a $6 et ecrit le resultat sur 5 chiffres
	     if ((*aC=='+') || (*aC=='-') || (*aC=='/') || (*aC=='%'))
	     {
                  aCAritm = *aC;
	          isArithm = true;
		  if (*aC=='%')
		  {
		      aC++;
		      if (isdigit(*aC))
                      {
		         aNbDigit = *aC-'0';
                         aC++;
			 if ((*aC!='+') && (*aC!='-') && (*aC!='/'))
			 {
			      mError = "No + or - after %x";
                             return false;
                         }
                      }
		      else
                      {
                         mError = "No Digit after %";
                         return false;
                      }
		  }

                  const char *  aC0  = aC;
		  aC++;
		  std::string aS;
		  while (*aC != *aC0)
		  {
		      if (isdigit(*aC))
		          aS = aS + *aC;
		      else
                      {
                         mError = "No Digit before closing +or-";
                         return false;
                      }
                      aC++;
		  }
		  aC++;
		  if (! FromString(aOffset,aS))
		  {
                         mError = "Internal error, should be a digit";
                         return false;
		  }
	     }


             if (isdigit(*aC))
             {
                aNum = *aC-'0';
             }
             else if (isalpha(*aC))
             {
                aNum = 10 +toupper(*aC)-'A';
             }
             if ((aNum<0) || (aNum>=int(mVMatch.size())))
             {
	         mError = "Nor digit nor alpha after $";
                 return false;
             }
             int aI0 = mVMatch[aNum].rm_so;
             int aI1 = mVMatch[aNum].rm_eo;

              
             if ((aI0<0) || (aI1<0))
                 return false;
             std::string aToInsert =  mLastNameMatched.substr(aI0,aI1-aI0);
	     if (isArithm)
	     {
	         int aVal;
		 FromString(aVal,aToInsert);
                 if (aCAritm=='+')
		    aVal += aOffset;
                 if (aCAritm=='-')
		    aVal -= aOffset;
                 if (aCAritm=='/')
		    aVal /= aOffset;
		 if (aNbDigit==-1)
		 {
		     aNbDigit = (int)aToInsert.size();
		 }
		 aToInsert = ToString(aVal);
		 if (int(aToInsert.size()) >aNbDigit)
		 {
		    mError ="No enough digit for computed number";
		    return false;
                 }
		 while (int(aToInsert.size()) <aNbDigit)
		     aToInsert = '0' + aToInsert;
	     }
             mLastReplace += aToInsert;
          }
       }
       else
       {
          mLastReplace += *aC;
       }
   }
   mOkReplace = true;
   return true;
}

const std::string & cElRegex::LastReplaced() const
{
   AssertIsReplaced();
   return mLastReplace;
}

const std::string & cElRegex::NameExpr() const
{
   return mNameExpr;
}


std::list<cElRegex *> CompilePats(const  std::list<std::string> & aLS)
{
    std::list<cElRegex *> aRes;

    for
    (
        std::list<std::string>::const_iterator itS=aLS.begin();
        itS!=aLS.end();
        itS++
    )
    {
           aRes.push_back(new cElRegex(*itS,10));
    }
    return aRes;
}

bool AuMoinsUnMatch(const std::list<cElRegex *> & aLR,const  std::string & aName)
{
   for
   (
           std::list<cElRegex *>::const_iterator itR=aLR.begin();
           (itR!=aLR.end()) ;
           itR++
   )
   {
         if ((*itR)->Match(aName))
	    return true;
   }
   return false;
}


std::string  GetStringFromLineExprReg
             (
                  cElRegex_Ptr & aReg,
                  const std::string & aNameFile,
                  const std::string & aNameExpr,
                  const std::string & aMotif,
                  int * aPtrNbMatch
             )
{
  std::string aRes;
  string aBuf; //char aBuf[1000]; TEST_OVERFLOW
  if (aReg == 0)
     aReg = new cElRegex(aNameExpr,15);


  bool endof=false;
  ELISE_fp aFile(aNameFile.c_str(),ELISE_fp::READ);
  int aNbMatch = 0;

  while (!endof)
  {
      if (aFile.fgets(aBuf,endof)) //if (aFile.fgets(aBuf,1000,endof)) TEST_OVERFLOW
      {
          // printf("UgRl[%s,%s]\n",aBuf,aNameExpr.c_str());
          if (aReg->Match(aBuf))
	  {
               aNbMatch++;
               aRes = MatchAndReplace(*aReg,aBuf,aMotif);
	  }
      }
  }

  aFile.close();

  if (aPtrNbMatch)
  {
     *aPtrNbMatch= aNbMatch;
  }
  else
  {
     if (aNbMatch!=1)
     {
         std::cout << "NB MATCH = " << aNbMatch << "\n";
         std::cout << "  EXPR  = " << aNameExpr  << "\n";
         ELISE_ASSERT(aNbMatch==1,"NbMatch != 1 in GetStringFromLineExprReg");
     }
  }

  return aRes;
}



std::vector<double> GetValsNumFromLineExprReg
                    (
                       cElRegex_Ptr & aReg,
                       const std::string & aNameFile,
                       const std::string & aNameExpr,
                       const std::string & aVInd,
                       int * aPtrNbMatch
                    )
{
  std::vector<double> aRes;
  string aBuf; //char aBuf[1000];
  if (aReg == 0)
     aReg = new cElRegex(aNameExpr,15);


  bool endof=false;
  ELISE_fp aFile(aNameFile.c_str(),ELISE_fp::READ);
  int aNbMatch = 0;

  while (!endof)
  {
      if (aFile.fgets(aBuf,endof)) //if (aFile.fgets(aBuf,1000,endof))
      {
          if (aReg->Match(aBuf))
	  {
             aNbMatch++;
	     for (const char * aC=aVInd.c_str(); *aC ; aC++)
	     {
                if (aPtrNbMatch)
                {
                    double aDef = -81234.89e15;
                    double OneRes = aReg->VNumKIemeExprPar(*aC-'0',true,&aDef);

                    if (OneRes==aDef)
                    {
                         *aPtrNbMatch = 0;
                          aRes.clear();
                          aFile.close();
                          return aRes;
                    }
                    else
	                aRes.push_back(OneRes);
                }
                else
                {
	            aRes.push_back(aReg->VNumKIemeExprPar(*aC-'0'));
                }
             }
	  }
      }
  }

  aFile.close();

  if (aPtrNbMatch)
  {
     *aPtrNbMatch = aNbMatch;
  }
  else
  {
      if (aNbMatch!=1)
      {
          std::cout << "NB MATCH = " << aNbMatch << "\n";
          std::cout << "  EXPR  = " << aNameExpr  << "\n";
          ELISE_ASSERT(aNbMatch==1,"NbMatch != 1 in GetValsNumFromLineExprReg");
      }
  }

  return aRes;
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
