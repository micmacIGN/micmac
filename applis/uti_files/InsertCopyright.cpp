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
#include "general/all.h"
#include "private/all.h"



class cCpleString
{
    public :
      cCpleString
      (
          const std::string & anOuv,
          const std::string & aFerm
      ) :
          mOuv (anOuv),
	  mFerm (aFerm)
      {
      }
      std::string mOuv;
      std::string mFerm;
};

class cAppliInsertCopyright
{
    public :
        cAppliInsertCopyright
	(
	    int argc,
	    int argv
	) :
	  mCurFile (0)
	{
	}

        void ReadParam();
	void DoOneDir(const std::string&);
	void DoAll();
	void DoOneFile(const std::string&);

	const std::string * LL_GetCurLine();
	const std::string * GetCurLine();
    private :
        char mBuf[100000];
	std::string aResBuf;

	std::string mFileLicence;
	std::string mFilePostLic;
        std::list<cCpleString> mLCple;

        ELISE_fp * mCurFile;
	std::string mDir;
	std::string mPat;
	std::list<std::string> mFiles;
};

const std::string * cAppliInsertCopyright::LL_GetCurLine()
{
   bool endof;
   if (mCurFile->fgets(mBuf,100000,endof))
   {
       if (endof) 
          return 0;
       aResBuf = mBuf;
       return &aResBuf;
   }
   ELISE_ASSERT(false,"cAppliInsertCopyright::LL_GetCurLine");
   return 0;
}

const std::string * cAppliInsertCopyright::GetCurLine()
{
   const std::string * aRes = 0;
   while (aRes == 0)
   {
      aRes = LL_GetCurLine();
      if  (aRes)
      {
         for 
         (
             std::list<cCpleString>::iterator itC= mLCple.begin();
	     (itC != mLCple.end()) && (aRes!=0);
	     itC++
         )
         {
             if (*aRes==itC->mOuv)
	     {
	        aRes = 0;
		bool aCont = true;
	        while (aCont)
		{
		   const std::string * aL = LL_GetCurLine();
		   if (aL==0)
		   {
		      ELISE_ASSERT(false,"Get Current Line");
		   }
                   aCont = (*aL != itC->mFerm);
		}
	     }
         }
      }
      else
          return 0;
   }
   return aRes;
}

void cAppliInsertCopyright::DoOneFile(const std::string& aName)
{
    FILE * aFp = ElFopen("Tmp.txt","w");
    const std::string * aLine;

    mCurFile = new ELISE_fp(mFileLicence.c_str(),ELISE_fp::READ);
    while ((aLine=LL_GetCurLine()) !=0)
    {
        fprintf(aFp,"%s\n",aLine->c_str());
    }
    mCurFile->close();
    delete mCurFile;


    mCurFile = new ELISE_fp(aName.c_str(),ELISE_fp::READ);
    while ((aLine=GetCurLine()) !=0)
    {
        fprintf(aFp,"%s\n",aLine->c_str());
    }
    mCurFile->close();
    delete mCurFile;


    mCurFile = new ELISE_fp(mFilePostLic.c_str(),ELISE_fp::READ);
    while ((aLine=LL_GetCurLine()) !=0)
    {
        fprintf(aFp,"%s\n",aLine->c_str());
    }
    mCurFile->close();
    delete mCurFile;




    ElFclose(aFp);

    std::string aCom = std::string("mv Tmp.txt ") + aName;
    system(aCom.c_str());
}

void cAppliInsertCopyright::DoOneDir(const std::string & aDir)
{
   std::list<std::string>  aList= RegexListFileMatch
                                  (
				      aDir,
				      mPat,
				      10
				  );

   for 
   (
       std::list<std::string>::iterator itS=aList.begin();
       itS!=aList.end();
       itS++
   )
   {
      std::cout << *itS<< "\n";
      DoOneFile(*itS);
   }
}

void cAppliInsertCopyright::DoAll()
{
    DoOneDir(mDir);
}

void cAppliInsertCopyright::ReadParam()
{
    mPat = "(.*\\.cpp)|(.*\\.h)";
    mDir = "../TmpMicMac";
    mFileLicence ="Documentation/Header-Licence-MicMac.txt";
    mFilePostLic ="Documentation/Footer-Licence-MicMac.txt";
    mLCple.push_back(cCpleString("/*eLiSe06/05/99","eLiSe06/05/99*/"));
    mLCple.push_back(cCpleString("/*Header-MicMac-eLiSe-25/06/2007","Header-MicMac-eLiSe-25/06/2007*/"));
    mLCple.push_back(cCpleString("/*Footer-MicMac-eLiSe-25/06/2007","Footer-MicMac-eLiSe-25/06/2007*/"));
}

int main(int argc,int argv)
{
   cAppliInsertCopyright anAppli(argc,argv);
   anAppli.ReadParam();
   //anAppli.DoOneFile("../TMP/toto.cpp");
   anAppli.DoAll();
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
