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


class cAppli_TNR_Main  : public  cElErrorHandlor
{
    public :
        cAppli_TNR_Main(int argc,char ** argv);
    private :

        void OnError();

        

        void DoOneGlobTNR(const std::string &);
        void TestOneCom(const cXmlTNR_OneTest & aOneT);
        cElemAppliSetFile mEASF;
        std::string mPattern;
        std::string mCurDirRef;
        std::string mCurDirTNR;

        cXmlTNR_GlobTest  mXML_CurGT;

        void  Error(const std::string &);
        std::string mCurCom;
};

void cAppli_TNR_Main::OnError()
{
    std::cout << "ERROR WHEN EXE " << mCurCom << "\n";
    exit(EXIT_SUCCESS);
}
  

void cAppli_TNR_Main::Error(const std::string & aMes)
{
    std::cout << aMes << "\n";
}


void cAppli_TNR_Main::TestOneCom(const cXmlTNR_OneTest & aOneT)
{
      mCurCom = aOneT.Cmd();
      int aVal = System(mCurCom,true);

std::cout << "VVVVVVVVVvvvvvvv " << aVal << "\n";
      if ((aVal != EXIT_SUCCESS)  && (aOneT.TestReturnValue().Val()))
      {
             Error("Cannot exe "+ mCurCom);
      }
}

void cAppli_TNR_Main::DoOneGlobTNR(const std::string & aNameFile)
{
    
      mXML_CurGT = StdGetFromSI(mEASF.mDir+aNameFile,XmlTNR_GlobTest);

      mCurDirRef = mEASF.mDir+  "TNR-Ref-" + mXML_CurGT.Name() + "/";
      mCurDirTNR = mEASF.mDir+  "TNR-Exe-" + mXML_CurGT.Name() + "/";

      // On fait une directory TNT vide
      if (mXML_CurGT.PurgeExe().Val())
      {
          ELISE_fp::PurgeDir(mCurDirTNR);
          ELISE_fp::MkDirSvp(mCurDirTNR);

/*
const std::list<std::string> & aLN = mXML_CurGT.PatFileInit();
*/
          for (std::list<std::string>::const_iterator itN=mXML_CurGT.PatFileInit().begin() ; itN!=mXML_CurGT.PatFileInit().end() ; itN++)
          {
              std::string aComCp = "cp " + (mCurDirRef + *itN) + " " + mCurDirTNR;
              std::cout << aComCp << "\n";
              System(aComCp);
          }
      }


      for (std::list<cXmlTNR_OneTest>::const_iterator it1T= mXML_CurGT.Tests().begin() ; it1T!= mXML_CurGT.Tests().end() ;  it1T++)
      {
           TestOneCom(*it1T);
      }
}




cAppli_TNR_Main::cAppli_TNR_Main(int argc,char ** argv)  
{
    TheCurElErrorHandlor = this;
    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mPattern, "Pattern of Xml test specification",  eSAM_IsPatFile),
         LArgMain()  
    );

    mEASF.Init(mPattern);
    const cInterfChantierNameManipulateur::tSet * aSetIm = mEASF.SetIm();


    for (int aK=0 ; aK<int(aSetIm->size()) ; aK++)
    {
         DoOneGlobTNR((*aSetIm)[aK]);
    }
}


//----------------------------------------------------------------------------

int TNR_main(int argc,char ** argv)
{
  cAppli_TNR_Main anAppli(argc,argv);
  return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

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
   Footer-MicMac-eLiSe-25/06/2007/*/
