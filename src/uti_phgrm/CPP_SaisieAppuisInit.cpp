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

#if (ELISE_X11 || ELISE_QT)
void SaisieAppuisInit(int argc, char ** argv,
                      Pt2di &aSzW,
                      Pt2di &aNbFen,
                      std::string &aFullName,
                      std::string &aDir,
                      std::string &aName,
                      std::string &aNamePt,
                      std::string &anOri,
                      std::string &aModeOri,
                      std::string &anOut,
                      std::string &aNameAuto,
                      std::string &aPrefix2Add,
                      bool &aForceGray,
                      double &aZMoy,
                      double &aZInc,
                      std::string & aInputSec,
                      bool          & WithMaxMin,
                      double &aGama
                      )
{
    MMD_InitArgcArgv(argc,argv);

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aFullName,"Full name (Dir+Pattern)", eSAM_IsPatFile)
                      << EAMC(anOri,"Orientation ; NONE if not used",eSAM_IsExistDirOri)
                      << EAMC(aNamePt,"Point name, or point file name", eSAM_None)
                      << EAMC(anOut,"Output",eSAM_IsOutputFile),
          LArgMain()  << EAM(aSzW,"SzW",true,"Sz of window")
                      << EAM(aNbFen,"NbF",true,"Nb of sub window (Def depends of number of images with max of 2x2)")
                      << EAM(aNameAuto,"NameAuto",true," Prefix for automatic point creation")
                      << EAM(aPrefix2Add,"Pref2Add",true," Prefix to add during import (for bug correction ?)")
                      << EAM(aForceGray,"ForceGray",true," Force gray image, def =true")
                      << EAM(aGama,"Gama",true," Gama adjustment (def=1.0)")
                      << EAM(aModeOri,"OriMode", true, "Orientation type (GRID) (Def=Std)")
                      << EAM(aZMoy,"ZMoy",true,"Average Z, Mandatory in PB", eSAM_NoInit)
                      << EAM(aZInc,"ZInc",true,"Incertitude on Z, Mandatory in PB", eSAM_NoInit)
                      << EAM(aInputSec,"InputSec",true,"PIMs filter used for visibility", eSAM_NoInit)
                      << EAM(WithMaxMin,"WMM",true,"With max-min option for point seizing", eSAM_NoInit)
    );

    if (!MMVisualMode)
    {
        SplitDirAndFile(aDir,aName,aFullName);

        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
        if ((anOri!="NONE") && (aModeOri != "GRID"))
           aICNM->CorrecNameOrient(anOri);
        const cInterfChantierNameManipulateur::tSet  *  aSet = aICNM->Get(aName);

        //std::cout << "Nb Image =" << aSet->size() << "\n";
        ELISE_ASSERT(aSet->size()!=0,"No image found");

        if (aNbFen.x<0)
        {
            if (aSet->size() == 1)
            {
                aNbFen = Pt2di(1,2);
            }
            else if (aSet->size() == 2)
            {
                Tiff_Im aTF = Tiff_Im::StdConvGen(aDir+(*aSet)[0],1,false,true);
                Pt2di aSzIm = aTF.sz();
                aNbFen = (aSzIm.x>aSzIm.y) ? Pt2di(1,2) : Pt2di(2,1);
            }
            else
            {
                aNbFen = Pt2di(2,2);
            }
        }

        cResulMSO aRMSO = aICNM->MakeStdOrient(anOri,true);

        if (0)
        {
           std::cout  << "RMSO; Cam "  << aRMSO.Cam()
                      << " Nuage " <<  aRMSO.Nuage()
                      << " Ori " <<  aRMSO.IsKeyOri()
                      << "\n";
           getchar();
        }
    }
}
#endif // (ELISE_X11 || ELISE_QT)

#if ELISE_X11
int SaisieAppuisInit_main(int argc,char ** argv)
{
  Pt2di aSzW(800,800);
  Pt2di aNbFen(-1,-1);
  std::string aFullName,aNamePt,anOri,anOut, aNameAuto, aPrefix2Add, aDir, aName, aModeOri;
  aNameAuto = "NONE";
  aPrefix2Add = "";
  bool aForceGray = true;
  double aZMoy,aZInc;
  std::string aInputSec;
  bool  WithMaxMin=false;
  double aGama = 1.0;


  SaisieAppuisInit(argc, argv, aSzW, aNbFen, aFullName, aDir, aName, aNamePt, anOri, aModeOri, anOut, aNameAuto, aPrefix2Add, aForceGray, aZMoy, aZInc,aInputSec,WithMaxMin,aGama);

  if (!MMVisualMode)
  {
      std::string aCom =     MMDir() +"bin/SaisiePts "
                          +  MMDir() +"include/XML_MicMac/SaisieInitiale.xml "
                          +  std::string(" DirectoryChantier=") + aDir
                          +  std::string(" +Image=") + QUOTE(aName)
                          +  std::string(" +Ori=") + anOri
                          +  std::string(" +NamePt=") + aNamePt
                          +  std::string(" +NameAuto=") + aNameAuto
                          +  std::string(" +Sauv=") + anOut
                          +  std::string(" +SzWx=") + ToString(aSzW.x)
                          +  std::string(" +SzWy=") + ToString(aSzW.y)
                          +  std::string(" +UseMinMaxPt=") + ToString(WithMaxMin)
                          +  std::string(" +NbFx=") + ToString(aNbFen.x)
                          +  std::string(" +NbFy=") + ToString(aNbFen.y) ;

      if (aModeOri == "GRID")
      {
          aCom += " +ModeOriIm=eGeomImageGrille"
                  + std::string(" +Conik=false")
                  +  std::string(" +ZIncIsProp=false")
                  //+ " +PostFixOri=GRIBin"
                  + " +Px1Inc="+ ToString(aZInc) + std::string(" ")
                  + " +Px1Moy="+ ToString(aZMoy) + std::string(" ");

          //aCom += std::string(" +Geom=eGeomMNTFaisceauIm1ZTerrain_Px1D");
      }

      if (EAMIsInit(&aForceGray))
         aCom = aCom + " +ForceGray=" + ToString(aForceGray);

      if (EAMIsInit(&aPrefix2Add))
         aCom = aCom + " +Pref2Add=" + aPrefix2Add;


       if (EAMIsInit(&aInputSec))
       {
           aCom = aCom + " +WithInputSec=true  +InputSec=" + aInputSec + " ";
       }

       if (EAMIsInit(&aGama))
       {
           aCom = aCom + " +Gama=" + ToString(aGama) + " ";
       }


      std::cout << aCom << "\n";

      int aRes = system(aCom.c_str());

      return aRes;
  }
  else
      return EXIT_SUCCESS;
}
#endif
/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
