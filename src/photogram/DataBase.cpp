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







static const int  TheCurNumVersion= 0;


std::string cInterfChantierNameManipulateur::NameDataBase(const std::string & anExt)
{
   return mDir+mMkDB->NameFile().Val()+anExt;
}



void  cInterfChantierNameManipulateur::CreateDataBase(const std::string & aNameDB)
{
   ELISE_fp aFile(aNameDB.c_str(),ELISE_fp::WRITE,ELISE_fp::eBinTjs);

   // Numero de version
   aFile.write(TheCurNumVersion);

   // 
   const std::vector<std::string> * aVN = Get(mMkDB->KeySetCollectXif());

   aFile.write(int(aVN->size()));
   aFile.write(int(mMkDB->KeyAssocNameSup().size()));

   for (int aKN=0 ; aKN<int(aVN->size()) ; aKN++)
   {
        const std::string & aNF = (*aVN)[aKN];
        cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(mDir+aNF);
        aFile.write(aMDP.FocMm());
        aFile.write(aNF);
        for 
        (
            std::list<std::string>::const_iterator iTS=mMkDB->KeyAssocNameSup().begin();
            iTS!=mMkDB->KeyAssocNameSup().end();
            iTS++
        )
        {
            std::string aNA =  Assoc1To1(*iTS,aNF,true);
            aFile.write(aNA);
        }
   }

   aFile.close();
}

void cInterfChantierNameManipulateur::MkDataBase()
{
   if (! mMkDB) 
      return;

   std::string aNameDB = NameDataBase(".dat");

   if (! ELISE_fp::exist_file(aNameDB))
   {
       CreateDataBase(aNameDB);
   }


   mDB = new cMMDataBase;

   ELISE_fp aFile(aNameDB.c_str(),ELISE_fp::READ,ELISE_fp::eBinTjs);

   int NV = aFile.read_INT4();
   if (NV != TheCurNumVersion)
   {
      aFile.close();

      CreateDataBase(aNameDB);

      aFile.ropen(aNameDB.c_str());
      NV = aFile.read_INT4();
      ELISE_ASSERT
      (
           NV==TheCurNumVersion,
           "incoh in num version after re-creation"
      );
   }
   
   int aNbStr = aFile.read_INT4();
   int aNbAux = aFile.read_INT4();


   for (int aKS=0 ; aKS <aNbStr ; aKS++)
   {
       cXmlExivEntry * aXEE = new cXmlExivEntry;
       aXEE->Focale() = aFile.read_REAL8();


       std::string aNF = aFile.read((std::string *)0);

       mDB->mExivs[aNF] = aXEE;
       for (int aKA=0 ; aKA<aNbAux ; aKA++)
       {
            aNF = aFile.read((std::string *)0);
            mDB->mExivs[aNF] = aXEE;
       }
   }

}

cXmlExivEntry * cInterfChantierNameManipulateur::GetXivEntry(const std::string & aName)
{
   ELISE_ASSERT
   (
      mDB,
      "Data Base required but do not exist"
   );

   
   cXmlExivEntry * aXEE = mDB->mExivs[aName];
   if (! aXEE)
   {
       std::cout << "For Name=" << aName << "\n";
       ELISE_ASSERT
       (
          false,
          "No Entry found in data base"
       );
   }

   return aXEE;
}


/*
void  cInterfChantierNameManipulateur::AddMTD2Name
      (
           std::string & aName,
           const std::string & aSep,
           double aMul
      )
{
   // Gestion d'Id simple
   {
      std::string anId =  Assoc1To1("NKS-Assoc-StdIdCam",aName,true);
      if (anId!="")
      {
          aName = aName + aSep + anId;
          return;
      }
   }


   const cMetaDataPhoto &  aMDP = cMetaDataPhoto::CreateExiv2(mDir+aName);

   bool NewMode = (MMUserEnv().VersionNameCam().Val()>=1) ;
   std::string aSepName = "";
   if (NewMode)
   {
         aMul = 1000;
         aSepName="_";
   }

   if (aMul>0)
   {
       double aFoc = aMDP.FocMm();
       if (aFoc<=0)
       {
           std::cout << "For name " << aName << "\n";
           ELISE_ASSERT(aFoc>0,"No Xif Focale found in NameTransfo");
       }
       aName = aName + aSep + aSepName + ToString(round_ni(aMul*aFoc));

       if (NewMode)
       {
           const std::string &  aNameCamInit = aMDP.Cam(true);
           std::string aNameCamUsed;
           for (const char * aC = aNameCamInit.c_str() ; *aC; aC++)
           {
               if (isalnum(*aC)) aNameCamUsed += *aC;
               else if (isblank(*aC)) aNameCamUsed += "_";
           }
           aName = aName +"_" + aNameCamUsed;
       }
       {
           std::string anId =  Assoc1To1("NKS-Assoc-StdIdAdditionnelCam",aName,true);
           if (anId!="")
             aName = aName + "_" + anId;
       }
   }
}
*/

std::string  cInterfChantierNameManipulateur::DBNameTransfo
     (
           const std::string & aNameInit,
           const cTplValGesInit<cDataBaseNameTransfo> &  aTplDBNT
     )
{
   if (! aTplDBNT.IsInit() ) return aNameInit;
   const cDataBaseNameTransfo & aDBNT = aTplDBNT.Val();

   std::string aName = aNameInit;
   std::string aSep = aDBNT.Separateur().Val();
   int aMode = MMUserEnv().VersionNameCam().ValWithDef(1);
   const cMetaDataPhoto &  aMDP = cMetaDataPhoto::CreateExiv2(mDir+aName);
   double aFoc = aMDP.FocMm(true);

   std::string aCompl = "";

   // A L'ancienne 
   if (aMode==0)
   {
        if (aFoc>0)
        {
             double aMul  = aDBNT.AddFocMul().ValWithDef(-1);
             if (aMul>0)
             {
                  aCompl += ToString(round_ni(aMul*aFoc));
             }
        }
   }
   else
   {
      std::string aUserId="";
      if (aDBNT.NewKeyId().IsInit())
      {
          aUserId=  Assoc1To1(aDBNT.NewKeyId().Val(),aName,true);
      }

      if (aUserId!="")
      {
         aCompl = aUserId;
      }
      else
      {
         if (aFoc>0)
         {
             double aMul  = aDBNT.NewFocMul().ValWithDef(-1);
             if (aMul>0)
             {
                  aCompl += "_Foc-"+ ToString(round_ni(aMul*aFoc));
             }
         }

         if (aDBNT.NewAddNameCam().ValWithDef(false))
         {
              const std::string &  aNameCamInit = aMDP.Cam(true);
              if (aNameCamInit!="")
              {
                 std::string aNameCamUsed ="_Cam-";
                 bool IsLastBlk= false;
                 for (const char * aC = aNameCamInit.c_str() ; *aC; aC++)
                 {
                      if (isalnum(*aC)) 
                      {
                          aNameCamUsed += *aC;
                          IsLastBlk = false;
                      }
                      else if (ElIsBlank(*aC))		
                      { 
                          if (!IsLastBlk)
                          {
                             aNameCamUsed += "_";
                          }
                          IsLastBlk = true;
                      }
                 }
                 aCompl +=  aNameCamUsed;
             }
         }
         
         if (aDBNT.NewKeyIdAdd().IsInit())
         {
             aCompl +=  Assoc1To1(aDBNT.NewKeyIdAdd().Val(),aName,true);
         }
      }
   }


   aName = aName + aSep + aCompl;


   return aName ;
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
