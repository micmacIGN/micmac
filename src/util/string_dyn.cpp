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

Tprov_char * dup_name_std(const char * name)
{
   Tprov_char * res;

   res = AllocTprov<char>::alloc_big_tprov(1+(int) strlen(name));
   strcpy(res->coord(),name);
   return res;
}

void * std_dup(void * out,const void * in,INT sz_nb)
{
    memcpy(out,in,sz_nb);
    return out;
}
char * std_ch_dup(const char * ch)
{
     INT l = 1 + (int) strlen(ch);
     return STD_NEW_TAB_DUP(ch,char,l);
}
/************************************************************/
/*                                                          */
/*                complement a std::string                  */
/*                                                          */
/************************************************************/

INT   IndPostfixed  (const ElSTDNS string & str,char  c)
{
    size_t ind = str.rfind(c);
    if ( ind ==  string::npos)
        Tjs_El_User.ElAssert
        (
            false,
            EEM0 << "Unpostfixed string " << str.c_str()
        );
    return (int) ind;
}

bool IsPostfixed(const ElSTDNS string & str,char  c)
{
     return   str.rfind(c) !=  ElSTDNS string::npos;
}

bool    IsPostfixedBy  (const ElSTDNS string & aName,const std::string & aPost)
{
    return IsPostfixed(aName) && (StdPostfix(aName)==aPost);
}

bool IsFileDmp(const std::string & aName)
{
    return IsPostfixedBy(aName,"dmp");
}


ElSTDNS string StdPrefix(const ElSTDNS string & str,char c)
{
    return str.substr(0,IndPostfixed(str,c));
}

ElSTDNS string StdPostfix(const ElSTDNS string & str,char c)
{
    return str.substr(IndPostfixed(str,c)+1,str.length());
}

//remove extension
ElSTDNS string StdPrefixGen(const ElSTDNS string & aStr,char c)
{
        if (IsPostfixed(aStr,c))
        return aStr.substr(0,IndPostfixed(aStr,c));
        return aStr;
}

bool IsPrefix(const char * aPref,const char *aStr)
{
   while (*aPref && *aStr && (*aPref==*aStr))
   {
       aPref++;
       aStr++;
   }
   return (*aPref==0);
}

template <>  std::string ToString(const bool & aBool)
{
   return  aBool ? "1" : "0";
}
template <>  std::string ToString(const int & anI)
{
    static char aBuf[50];
    sprintf(aBuf,"%d",anI);
    return aBuf;
}

template <>  std::string ToString(const double & aD)
{
    static char aBuf[50];
    sprintf(aBuf,"%f",aD);
    return aBuf;
}

template <> std::string ToString(const Pt2di & aP)
{
   return "[" + ToString<int>(aP.x) + "," + ToString<int>(aP.y) + "]";
}

template <> std::string ToString(const Pt3di & aP)
{
   return "[" + ToString<int>(aP.x) + "," + ToString<int>(aP.y) + "," + ToString<int>(aP.z) + "]";
}

template <> std::string ToString(const Pt2dr & aP)
{
   return "[" + ToString<double>(aP.x) + "," + ToString<double>(aP.y) + "]";
}

template <> std::string ToString(const Pt3dr & aP)
{
   return "[" + ToString<double>(aP.x) + "," + ToString<double>(aP.y) + "," + ToString<double>(aP.z) + "]";
}

template <class Type> std::string  Vec2String(const std::vector<Type> & aV)
{
   std::string aRes= "[";
   for (int aK=0 ; aK<int(aV.size()) ; aK++)
   {
      if (aK!=0) aRes = aRes + ",";
      aRes = aRes + ToString(aV[aK]);
   }
   aRes =  aRes + "]";
   return aRes;
}
template <> std::string ToString(const std::vector<double> & aV)
{
    return Vec2String(aV);
}



void CmpByEnd(const char * aName1,const char * aName2,INT & aK1,INT &aK2)
{
    aK1=(int) strlen( aName1);
    aK2=(int) strlen(aName2 );
    while
    (
        (aK1>=0)&&( aK2>=0) && (aName1[aK1]==aName2[aK2])
    )
    {
       --aK1;
       aK2--;
    }
}
bool N2IsEndN1(const char * aName1,const char * aName2)
{
     INT  aK1,aK2;
     CmpByEnd(aName1,aName2,aK1,aK2);
     return aK2==-1;
}

// Evolution de NameWithoutDir, pour etre compatible W-Linux, on
// extrait une chaine sans / ni \ , en esperant qu'il n'y a pas
// d'effet collateral. A priori c'est bon car :
//
//   - peu utilisee ; uniquement sur de la generation de cpp
//    (specif Linux) et des fichiers genere par MicMac (on a
//    utilise en dur le /, d'ou le bug en recherchant ELISE_CAR_DIR)
//
//   - il faut chercher les ennuis pour mettre des \ dans les noms
//   linus ou / dans les noms W
//

std::string PostFixWithout(const std::string & aStr,char aSep)
{
 std::string::size_type aPos = aStr.rfind(aSep);
 if (aPos == std::string::npos)
    return aStr;
 return aStr.substr(aPos+1,std::string::npos);
}


std::string StrToLower(const std::string & aStr)
{
   std::string aRes;
   for (const char * aC=aStr.c_str(); *aC; aC++)
   {
      aRes += (isupper(*aC) ?  tolower(*aC) : *aC);
   }
   return aRes;
}

std::string NameWithoutDir(const std::string & aStr)
{
return PostFixWithout(PostFixWithout(aStr,'/'),'\\');
}

/*
void MakeNameDir(std::string & aStr)
{
    if ((aStr.size()==0) ||  (aStr[aStr.size()-1] != ELISE_CAR_DIR))
      aStr +=  ELISE_CAR_DIR;
}
*/

std::string AddDirIfRequired(const std::string & aDir,const std::string & aFile)
{
    if (ELISE_fp::exist_file(aDir+ aFile)) return aDir + aFile;
    return aFile;
}

std::string StdWorkdDir(const std::string & aValWD,const std::string & aNameFile)
{
  // std::cout << "XXXXXXXXXX\n";

   if (aValWD!="ThisDir")
      return aValWD;

/*
   std::string aDir,aFile;
   SplitDirAndFile(aDir,aFile,aNameFile);
   return aDir;
*/
   return DirOfFile(aNameFile);

}
void MakeFileDirCompl(std::string & aD)
{
   const char * aC = aD.c_str();
   int aL = (int)strlen(aC);
   if ( (aL==0) || ( (aC[aL-1]!='/') && (aC[aL-1]!='\\') ) )
   {
          aD = aD + '/';
   }
}

std::string DirOfFile(const std::string & aNameFile)
{
   std::string aDir,aFile;
   SplitDirAndFile(aDir,aFile,aNameFile);
   return aDir;
}


std::string ToCommande(int argc,char ** argv)
{
   std::string aRes;
   for (int aK=0 ; aK<argc ; aK++)
   {
     if (aK!=0) aRes = aRes + " ";
     aRes = aRes + argv[aK];
   }
   return aRes;
}



void SplitDirAndFile
     (
           std::string & aNameDir,
           std::string & aNameFile,
           const std::string & aStr
     )
{


    // ======
    string strCpy = aStr;

    #if (ELISE_windows)
        replace( strCpy.begin(), strCpy.end(), '\\', '/' );
    #endif



    // ====== Regle speciale, si on veut passer des argument de Type  NKS-Set-OfFile, on doit
    // pouvoir specifier un split special, on le fait avec un %
    {
         std::string::size_type aPos  = strCpy.rfind('%');
         if (aPos != std::string::npos)
         {
             aNameDir = strCpy.substr(0,aPos);
             aNameFile = strCpy.substr(aPos+1,std::string::npos);
             return;
         }
    }


    std::string::size_type aPos  = strCpy.rfind('/');

    if (aPos == std::string::npos)
    {
        aNameDir = ELISE_Current_DIR ;
        aNameFile = strCpy;
    }
    else
    {
       aNameDir = strCpy.substr(0,aPos+1);
       aNameFile = strCpy.substr(aPos+1,std::string::npos);
    }
}


std::string AddPrePost(const std::string & aFull,const std::string & aPref,const std::string & aPost)
{
    std::string aDir,aName;
    SplitDirAndFile(aDir,aName,aFull);

   return aDir + aPref +StdPrefix(aName) + aPost + "." +StdPostfix(aName);

}

std::string MatchAndReplace
            (
                  cElRegex & anAutom,
                  const std::string & aName,
                  const std::string & aMotif
            )
{
    bool Ok = anAutom.Match(aName);
    if (!Ok)
    {
       std::cout << "EXPR=[" << anAutom.NameExpr() << "]\n";
       std::cout << "To Match=[" << aName << "]\n";

       ELISE_ASSERT(false,"No Match in expr");
    }
    Ok = anAutom.Replace(aMotif);
    if (!Ok)
    {
       std::cout << "EXPR=[" << anAutom.NameExpr()  << "]\n";
       std::cout << "To Match=[" << aName << "]\n";
       std::cout << "MOtif=[" << aMotif << "]\n";
       std::cout << "Error=[" << anAutom.Error() << "]\n";

       ELISE_ASSERT(false,"No Match in motif");
    }
    return anAutom.LastReplaced();
}


/*******************************************************/
/*                                                     */
/*                 cCpleString                         */
/*                                                     */
/*******************************************************/

cCpleString::cCpleString(const std::string& aN1,const std::string& aN2) :
   mN1 (aN1),
   mN2 (aN2)
{
}



cCpleString::cCpleString() :
   mN1 (),
   mN2 ()
{
}

cCpleString cCpleString::AddPrePost
            (
                const std::string& aPre,
                const std::string& aPost
            )  const
{
   return cCpleString(aPre+mN1+aPost,aPre+mN2+aPost);
}


const std::string &  cCpleString::N1() const
{
  return mN1;
}

const std::string &  cCpleString::N2() const
{
  return mN2;
}

bool cCpleString::operator < (const cCpleString & aCpl2) const
{
   if (mN1 <aCpl2.mN1)
      return true;

   if (mN1 >aCpl2.mN1)
      return false;

   return mN2<aCpl2.mN2;
}

bool cCpleString::operator == (const cCpleString & aCpl2) const
{
    return (mN1 == aCpl2.mN1) && (mN2==aCpl2.mN2);
}



bool IsSep(char aC)
{
   return (aC==' ') || (aC=='\\') || (aC=='\t');
}

std::vector<char *> ToArgMain(const std::string & aStr)
{
    // std::cout <<"SZ " << aStr.size() << "\n";
    char * aDup = new char [aStr.size()+1] ;
    memcpy(aDup,aStr.c_str(),aStr.size()+1);
    std::vector<char *> aRes;
    aRes.push_back(aDup);

    while (IsSep(*aDup)) aDup++;
    if (*aDup==0)
    {
        return aRes;
    }

    bool Cont = true;
    while (Cont)
    {
        aRes.push_back(aDup);
        while (!IsSep(*aDup) && *aDup) aDup++;
        char * aL = aDup;
        while (IsSep(*aDup)) aDup++;
        Cont = (*aDup!=0);
        *aL = 0;
    }

    return aRes;
}


int  StringDcraw(const std::string & aStr)
{
    std::vector<char *> anAM= ToArgMain(aStr);

    int aRes = el_dcraw_main((int)(anAM.size() - 1),const_cast<const char **>(&anAM[1]));

    delete [] anAM[0];
    return aRes;
}

void GlobStdAdapt2Crochet(std::string & aStr)
{
    if (aStr=="") return;

    if (
           (aStr.find('@')!=(std::string::npos))
         && (aStr[0]!='[')
       )
    {
         aStr = "[["+aStr+"]]";
    }
}




std::string QUOTE(const std::string & aStr)
{
   const char * aC = aStr.c_str();
   int aL = (int)strlen(aC);
   if (aL==0) return aStr;


   return    ((aC[0]=='"'   ) ? "" : "\"" )
           + aStr
           + ((aC[aL-1]=='"') ? "" : "\"" );
}

bool needCommandlineProtection( const string &aStr )
{
	const char *it = aStr.c_str();
	size_t i = aStr.length();
        if (i && (*it=='"')) 
           return false;
	while (i--)
	{
		const char &c = *it++;
		if (c == '[' || c == '(' || c == ')' || c == ']') return true;
	}
	return false;
}

string PATTERN_QUOTE( const string &aStr )
{
	return (needCommandlineProtection(aStr) ? QUOTE(aStr): aStr);
}

std::vector<std::string> VecStrFromFile(const std::string & aFilePtsIn)
{
    std::vector<std::string>  aRes;
    ELISE_fp aFIn(aFilePtsIn.c_str(),ELISE_fp::READ);
    char * aLine;

    while ((aLine = aFIn.std_fgets()))
    {
        aRes.push_back(aLine);
    }

    return aRes;
}

std::string getBanniereMM3D()
{
    std::string banniere = "\n";
    banniere += "************************************************************************ \n";
    banniere += "**                                                                    ** \n";
    banniere += "**    MicMac: a  free open source project  for photogrammetry         ** \n";
    banniere += "**     hosted at Ecole Nationale des Sciences Geographiques           ** \n";
    banniere += "**               in Marne-la-Vallee, for IGN-France                   ** \n";
    banniere += "**                                                                    ** \n";
    banniere += "**                                                                    ** \n";
    banniere += "**  The project is funded by :                                        ** \n";
    banniere += "**                                                                    ** \n";
    banniere += "**  - Institut National de l'Information Geographique et Forestiere   ** \n";
    banniere += "**    (IGN main funder since 2003)                                    ** \n";
    banniere += "**  - the french FUI Project \"Culture 3D Cloud\" (and MAP-CNRS)        ** \n";
    banniere += "**  - the french ANR Project \"MONUMENTUM\"(collaborating with MAP-CNRS)** \n";
    banniere += "**                                                                    ** \n";
    banniere += "**  Research also currently supported by :                            ** \n";
    banniere += "**  - CNES (French Space Agency) - via TOSCA Committee (and IPGP)     ** \n";
    banniere += "**  - Compagnie Nationale du Rhone                                    ** \n";
    banniere += "**  - Vinci-Construction-Terrassement                                 ** \n";
    banniere += "**  - ERC Advanced Grant A.Kaeaeb \"ICEMASS\" (University of Oslo)      ** \n";
    banniere += "**                                                                    ** \n";
    banniere += "**                                                                    ** \n";
    banniere += "**  Current Team: MP Deseilligny, D Jouin, J Belvaux, G Maillet,      ** \n";
    banniere += "**    L Girod, E Rupnik, JM Muller, M Daakir, TG Nguyen               ** \n";
    banniere += "**                                                                    ** \n";
    banniere += "**    Contact for participating : Marc.Pierrot-Deseilligny@ensg.eu    ** \n";
    banniere += "**                                                                    ** \n";
    banniere += "**    Hope you enjoy, todo list in case of any problem using MicMac : ** \n";
    banniere += "**      (0) Take a Pastis                                             ** \n";
    banniere += "**      (1) Switch your computer off and on again                     ** \n";
    banniere += "**      (2) Install it on Gnu-Linux (work better on)                  ** \n";
    banniere += "**      (3) See the wiki at http://micmac.ensg.eu/                    ** \n";
    banniere += "**      (4) Contact the forum http://forum-micmac.forumprod.com/      ** \n";
    banniere += "**                                                                    ** \n";
    banniere += "************************************************************************ \n";

    return banniere;
}

void BanniereMM3D()
{
    std::cout << getBanniereMM3D();
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
