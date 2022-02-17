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



int wildmat(const char *s, const char *p) ;


static int Star(const char *s, const char *p)
{
    while (wildmat(s, p) == false)
		if (*++s == '\0')
	    	return(false);
    return(true);
}


int wildmat(const char *s, const char *p) 
{
    int  last;
    int  matched;
    int  reverse;

    for ( ; *p; s++, p++)
	switch (*p) {
	    case '\\':
		/* Literal match with following character; fall through. */
		p++;
	    default:
		if (*s != *p)
		    return(false);
		continue;
	    case '?':
		/* Match anything. */
		if (*s == '\0')
		    return(false);
		continue;
	    case '*':
		/* Trailing star matches everything. */
		return(*++p ? Star(s, p) : true);
	    case '[':
		/* [^....] means inverse character class. */
		if ((reverse = (p[1] == '^')))
		    p++;
		for (last = 0400, matched = false; *++p && *p != ']'; last = *p)
		    /* This next line requires a good C compiler. */
		    if (*p == '-' ? *s <= *++p && *s >= last : *s == *p)
				matched = true;
		if (matched == reverse)
		    return(false);
		continue;
	}

    /* For "tar" use, matches that end at a slash also work. --hoptoad!gnu */
    return(*s == '\0' || *s == '/');
}

bool MatchPattern
     (
         const std::string & aName,
         const std::string & aPattern
     )
{
   return (wildmat(aName.c_str(),aPattern.c_str()) != 0);
}


/*****************************************************/
/*                                                   */
/*            cListDirAndMatch                       */
/*                                                   */
/*****************************************************/


class cListDirAndMatch : public ElActionParseDir
{
     public :
         cListDirAndMatch
         (
                const std::string  & aPattern,
                const std::string  & aDir, 
                bool  NameCompl,
                bool  isRegexMode
          ) :
             mPattern  (aPattern),
             mDir      (aDir),
             mNameCompl (NameCompl),
             mRegex     (
                           isRegexMode ?
                           new cElRegex(aPattern,2,REG_EXTENDED) :
                           0
                        )
         {
         }
         const std::list<std::string> & Res() const {return mRes;}
         virtual ~cListDirAndMatch() 
         {
                delete mRegex;
         }

     private :
         bool Match(const char * aName)
         {
              return mRegex ?
                     mRegex->Match(aName):
                     (wildmat(aName,mPattern.c_str()) != 0);
         }
         
         void act(const ElResParseDir &);
         const std::string mPattern;
         const std::string mDir;
         bool  mNameCompl;
         cElRegex * mRegex;
         std::list<std::string>  mRes;
  
};


void cListDirAndMatch::act(const ElResParseDir & aRes)
{
   if ( aRes.is_dir() )
      return;


  if (mNameCompl)
  {
      // if (! wildmat(aRes.name(),mPattern.c_str()))
      if (! Match(aRes.name()))
        return;

      mRes.push_back(aRes.name());
  }
  else
  {
       std::string aNR (aRes.name());
#if (ELISE_windows)
	   replace( aNR.begin(), aNR.end(), '\\', '/' );
#endif
       aNR = aNR.substr(mDir.size(),aNR.size());

      // if (! wildmat(aNR.c_str(),mPattern.c_str()))
      if (! Match(aNR.c_str()))
        return;
      mRes.push_back(aNR);
  }
}



std::list<std::string>  ListFileMatchGen
                        (
                               const std::string & aDir,
                               const std::string & aPattern,
                               INT NivMax,
                               bool NameCompl,
                               bool isModeRegex
                        ) 
{ 
   
    cListDirAndMatch aLDAM(aPattern,aDir,NameCompl,isModeRegex);
    char BufDir[1000];
    sprintf(BufDir,"%s",aDir.c_str());

    ElParseDir(BufDir,aLDAM,NivMax);

    std::vector<std::string> aV(aLDAM.Res().begin(),aLDAM.Res().end());

    for (int aKS=0 ; aKS<int(aV.size()) ; aKS++)
    {
        std::replace
        (
              aV[aKS].begin(),
              aV[aKS].end(),
              '\\',
              '/'
        );
    }

    std::sort(aV.begin(),aV.end());
    std::list<std::string> aL(aV.begin(),aV.end());

    return aL;
    // return aLDAM.Res();
}

std::list<std::string>  ListFileMatch
                        (
                               const std::string & aDir,
                               const std::string & aPattern,
                               INT NivMax,
                               bool NameCompl
                        ) 
{ 
   return ListFileMatchGen(aDir,aPattern,NivMax,NameCompl,false);
}


std::list<std::string>  RegexListFileMatch
                        (
                               const std::string & aDir,
                               const std::string & aPattern,
                               INT NivMax,
                               bool NameCompl
                        ) 
{ 
   return ListFileMatchGen(aDir,aPattern,NivMax,NameCompl,true);
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
