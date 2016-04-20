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


//  "BoldoIsation" du code 
//
//        - supprimer 
#include "general/all.h"
#include "private/all.h"


class cOneOperSupress
{
      public : 
          cOneOperSupress(const std::string & AutomNameToApply);
          void AddAutom(int aCarMajic,const std::string & aName2Supress);

          void DoOneDir(const std::string & aFile);
      private : 

          const std::string * GetCurLine();
          bool  KAcceptLine(const std::string &  aLine,int aK) const;
          bool AcceptLine(const std::string &  aLine);


          void DoOneFile(const std::string & aFile);
          char mBuf[100000];
	  std::string aResBuf;

          std::vector<int>         mCarMajic;
          std::vector<cElRegex *>  mExprLine2Supr;
          std::string              mExprFile;
          ELISE_fp *               mCurFile;
};

void cOneOperSupress::AddAutom(int aCarMajic,const std::string & aName2Supress)
{
   mCarMajic.push_back(aCarMajic);
   mExprLine2Supr.push_back(new cElRegex(aName2Supress,10));
}

const std::string * cOneOperSupress::GetCurLine()
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

bool  cOneOperSupress::KAcceptLine(const std::string &  aLine,int aK) const
{
   if ( (mCarMajic[aK] != -1) && (aLine.c_str()[0] != mCarMajic[aK] ))
     return  true;
  return ! mExprLine2Supr[aK]->Match(aLine) ;
}

bool cOneOperSupress::AcceptLine(const std::string &  aLine)
{
   for (int aK=0 ; aK<int(mExprLine2Supr.size()) ; aK++)
      if (!KAcceptLine(aLine,aK))
         return  false;

   return  true;
}



void cOneOperSupress::DoOneFile(const std::string & aFile)
{

   FILE * aFp = ElFopen("Tmp.txt","w");
   mCurFile = new ELISE_fp(aFile.c_str(),ELISE_fp::READ);

   const std::string * aL;
   int aNbSup =0;
   while ((aL=GetCurLine()))
   {
       if (AcceptLine(*aL))
          fprintf(aFp,"%s\n",aL->c_str());
       else 
          aNbSup++;
   }
   ElFclose(aFp);
   mCurFile->close();
   delete mCurFile;

   std::string aCom = "cp Tmp.txt " + aFile;
   System(aCom);
  std::cout << "DONE FILE " << aFile << " " << aNbSup<< "\n";
}

cOneOperSupress::cOneOperSupress(const std::string & AutomNameToApply) :
   mExprFile(AutomNameToApply)
{
}

void cOneOperSupress::DoOneDir(const std::string & aDir)
{
    std::list<std::string>  aLN= RegexListFileMatch(aDir,mExprFile,1,true);

    for 
    (
        std::list<std::string>::const_iterator itN=aLN.begin();
        itN!=aLN.end();
        itN++
    )
    {
      DoOneFile(*itN);
    }
}



int main(int argc,char ** argv)
{
   {
      cOneOperSupress aOS1(".*\\.(cpp|h)");

       aOS1.AddAutom('#',"#ifndef.*");
       aOS1.AddAutom('#',"#define.*");
       aOS1.AddAutom('#',"#endif.*");

       aOS1.DoOneDir("/home/pierrot/TMP/CG/");
   }
   //anAppli.DoOneFile("../TMP/toto.cpp");
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
