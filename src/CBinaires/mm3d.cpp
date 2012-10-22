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

    MicMa cis an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"

#define DEF_OFSET -12349876

typedef int (*tCommande)  (int,char**);

std::string StrToLower(const std::string & aStr)
{
   std::string aRes;
   for (const char * aC=aStr.c_str(); *aC; aC++)
   {
      aRes += (isupper(*aC) ?  tolower(*aC) : *aC);
   }
   return aRes;
}

// CMMCom is a descriptor of a MicMac Command
class cMMCom
{
   public :
      cMMCom (const std::string & aName,tCommande  aCommand,const std::string & aComment) :
          mName     (aName),
          mLowName  (StrToLower(aName)),
          mCommand  (aCommand),
          mComment  (aComment)
      {
      }

      std::string  mName;
      std::string  mLowName;
      tCommande    mCommand;
      std::string  mComment;
};





const std::vector<cMMCom> & getAvailableCommands()
{
   static std::vector<cMMCom> aRes;
   if (aRes.empty())
   {
       aRes.push_back(cMMCom("AperiCloud",AperiCloud_main," Visualisation of camerai in ply file"));
       aRes.push_back(cMMCom("Apero",Apero_main," Do some stuff"));
       aRes.push_back(cMMCom("Bascule",Bascule_main," Do some stuff"));
       aRes.push_back(cMMCom("BatchFDC",BatchFDC_main," Do some stuff"));
       aRes.push_back(cMMCom("CmpCalib",CmpCalib_main," Do some stuff"));
       aRes.push_back(cMMCom("Dequant",Dequant_main," Do some stuff"));
       aRes.push_back(cMMCom("Devlop",Devlop_main," Do some stuff"));
       aRes.push_back(cMMCom("ElDcraw",ElDcraw_main," Do some stuff"));
       aRes.push_back(cMMCom("GCPBascule",GCPBascule_main," Do some stuff"));
       aRes.push_back(cMMCom("GenXML2Cpp",GenXML2Cpp_main," Do some stuff"));
       aRes.push_back(cMMCom("Gri2Bin",Gri2Bin_main," Do some stuff"));
       aRes.push_back(cMMCom("GrShade",GrShade_main," Do some stuff"));
       aRes.push_back(cMMCom("MakeGrid",MakeGrid_main," Do some stuff"));
       aRes.push_back(cMMCom("Malt",Malt_main," Simplified matching (inteface to MicMac)"));
       aRes.push_back(cMMCom("MapCmd",MapCmd_main," Do some stuff"));
       aRes.push_back(cMMCom("MICMAC",MICMAC_main," Do some stuff"));
       aRes.push_back(cMMCom("MpDcraw",MpDcraw_main," Do some stuff"));
       aRes.push_back(cMMCom("Nuage2Ply",Nuage2Ply_main," Do some stuff"));
       aRes.push_back(cMMCom("Pasta",Pasta_main," Do some stuff"));
       aRes.push_back(cMMCom("PastDevlop",PastDevlop_main," Do some stuff"));
       aRes.push_back(cMMCom("Pastis",Pastis_main," Do some stuff"));
       aRes.push_back(cMMCom("Porto",Porto_main," Do some stuff"));
       aRes.push_back(cMMCom("Reduc2MM",Reduc2MM_main," Do some stuff"));
       aRes.push_back(cMMCom("ReducHom",ReducHom_main," Do some stuff"));
       aRes.push_back(cMMCom("RepLocBascule",RepLocBascule_main," Do some stuff"));
       aRes.push_back(cMMCom("SBGlobBascule",SBGlobBascule_main," Do some stuff"));
       aRes.push_back(cMMCom("ScaleIm",ScaleIm_main," Do some stuff"));
       aRes.push_back(cMMCom("ScaleNuage",ScaleNuage_main," Do some stuff"));
       aRes.push_back(cMMCom("Tapas",Tapas_main," Do some stuff"));
       aRes.push_back(cMMCom("Tapioca",Tapioca_main," Do some stuff"));
       aRes.push_back(cMMCom("Tarama",Tarama_main," Do some stuff"));
       aRes.push_back(cMMCom("Tawny",Tawny_main," Do some stuff"));
       aRes.push_back(cMMCom("TestCam",TestCam_main," Do some stuff"));
       aRes.push_back(cMMCom("tiff_info",tiff_info_main," Do some stuff"));
       aRes.push_back(cMMCom("to8Bits",to8Bits_main," Do some stuff"));
   }
   return aRes;
}

class cSuggest
{
     public :
        cSuggest(const std::string & aName,const std::string & aPat) :
             mName (aName),
             mPat  (aPat),
             mAutom (mPat,10)
        {
        }
        void Test(const cMMCom & aCom)
        {
            if (mAutom.Match(aCom.mLowName))
               mRes.push_back(aCom);
        }

        std::string          mName;
        std::string          mPat;
        cElRegex             mAutom;
        std::vector<cMMCom>  mRes;
};


int main(int argc,char ** argv)
{

   const std::vector<cMMCom> & aVComs = getAvailableCommands();
   if ((argc==1) || ((argc==2) && (std::string(argv[1])=="-help")))
   {
       std::cout << "mm3d : Allowed commands \n";
       for (unsigned int aKC=0 ; aKC<aVComs.size() ; aKC++)
       {
            std::cout  << " " << aVComs[aKC].mName << "\t" << aVComs[aKC].mComment << "\n";
       }
       return 0;
   }

   std::string aCom = argv[1];
   std::string aLowCom = StrToLower(aCom);

   std::vector<cSuggest *> mSugg;
   mSugg.push_back(new cSuggest("Pattern Match",aLowCom));
   mSugg.push_back(new cSuggest("Prefix Match",aLowCom+".*"));
   mSugg.push_back(new cSuggest("Subex Match",".*"+aLowCom+".*"));

   for (unsigned int aKC=0 ; aKC<aVComs.size() ; aKC++)
   {
       if (StrToLower(aVComs[aKC].mName)==StrToLower(aCom))
       {
          return (aVComs[aKC].mCommand(argc-1,argv+1));
       }
       for (int aKS=0 ; aKS<int(mSugg.size()) ; aKS++)
       {
            mSugg[aKS]->Test(aVComs[aKC]);
       }
   }


   for (unsigned int aKS=0 ; aKS<int(mSugg.size()) ; aKS++)
   {
      if (! mSugg[aKS]->mRes.empty())
      {
           std::cout << "Suggest by " << mSugg[aKS]->mName << "\n";
           for (int aKC=0 ; aKC<mSugg[aKS]->mRes.size() ; aKC++)
           {
                std::cout << "    " << mSugg[aKS]->mRes[aKC].mName << "\n";
           }
           return -1;
      }
   }



   std::cout << "For command = " << argv[1] << "\n";
   ELISE_ASSERT(false,"Unkown command in mm3d");
   return -1;
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
