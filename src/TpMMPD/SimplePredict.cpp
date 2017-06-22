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
#include "SimplePredict.h"

cOrientedImage::cOrientedImage
  (   std::string aOriIn,
   std::string aName,
   cInterfChantierNameManipulateur * aICNM):
    mName(aName),mOriFileName(aOriIn+"Orientation-"+mName+".xml")
{
   mCam=CamOrientGenFromFile(mOriFileName,aICNM);
}
/**
 * SimplePredict: project ground points on oriented cameras
 *
 * Inputs:
 *  - pattern of images
 *  - Ori
 *  - ground points xml file
 *
 * Output:
 *  - 2d predicted points on all images
 *
 * Call example:
 *   mm3d SimplePredict ".*.tif" Ori-Basc/ Polygone_GCP.xml
 *
 *   Full commands for IGN polygone calibration:
 *   create Polygone_GCP.xml with: ./polygone2GCP.py IGN-2009.xml
 *   exiftool *.tif -'FocalLength'=60
 *   rm *_original
 *   needs MicMac-LocalChantierDescripteur.xml to describe CameraEntry
 *   Tapioca All ".*.tif" 2000
 *   Tapas RadialStd ".*.tif"
 *   AperiCloud ".*tif" RadialStd
 *   mm3d SaisieAppuisInitQT "GS1_15_1_60_rrx00.1.*.tif" Ori-RadialStd/ Dico-Appuis_Polyg2009.xml saisieInit.xml
 *   GCPBascule ".*.tif" Ori-RadialStd/ Ori-Basc/ Polygone_GCP.xml saisieInit.xml
 *   mm3d SimplePredict ".*.tif" Ori-Ori-Basc/ Polygone_GCP.xml  ExportPolyIGN=1 PrefixeNomImageSize=18
 *
 * */

/**
 To create a GCP xml file from IGN-2009.xml file, use this python sript:

from lxml import etree
import sys
in_tree = etree.parse("IGN-2009.xml")
GCP = etree.Element("Global")
GCP_dico = etree.SubElement(GCP, "DicoAppuisFlottant")
for cible in in_tree.xpath("/PolygoneCalib/Cibles"):
    GCP_OneAppuisDAF = etree.SubElement(GCP_dico, "OneAppuisDAF")
    GCP_OneAppuisDAF_Id = etree.SubElement(GCP_OneAppuisDAF, "NamePt")
    GCP_OneAppuisDAF_Id.text=cible.xpath("Id")[0].text
    GCP_OneAppuisDAF_Pos = etree.SubElement(GCP_OneAppuisDAF, "Pt")
    GCP_OneAppuisDAF_Pos.text=cible.xpath("Position")[0].text
    GCP_OneAppuisDAF_Inc = etree.SubElement(GCP_OneAppuisDAF, "Incertitude")
    GCP_OneAppuisDAF_Inc.text="0.001 0.001 0.001"

f = open("Polygone_GCP.xml", 'w')
f.write( etree.tostring(GCP, pretty_print=True) )
f.close()

**/

int SimplePredict_main(int argc,char ** argv)
{
  std::string aFullPattern;//pattern of all scanned images
  std::string aGCPFileName;//3d points
  std::string aOriIn;//Orientation containing all images and calibrations
  bool aExportPolyIGN=false;
  int aPrefixeNomImageSize=18;

  std::cout<<"SimplePredict: project ground points on oriented cameras"<<std::endl;

  ElInitArgMain
    (
     argc,argv,
     //mandatory arguments
     LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                 << EAMC(aOriIn, "Directory orientation",  eSAM_IsExistDirOri)
                 << EAMC(aGCPFileName, "Ground points file", eSAM_IsExistFile),
     //optional arguments
     LArgMain()  /*<< EAM(aTargetHalfSzPx,"TargetHalfSize",true,"Target half size in pixels (Def=64)")
                 << EAM(aSearchIncertitudePx,"SearchIncertitude",true,"Search incertitude in pixels (Def=5)")
                 << EAM(aSearchStepPx,"SearchStep",true,"Search step in pixels (Def=0.5)")*/
                 << EAM(aExportPolyIGN,"ExportPolyIGN",true, "Export PointeInitIm files for IGN Polygon calibration method (Def=false)", eSAM_IsBool)
                 << EAM(aPrefixeNomImageSize,"PrefixeNomImageSize",true,"Size of PrefixeNomImage in param.txt")
    );

  if (MMVisualMode) return EXIT_SUCCESS;

  MakeFileDirCompl(aOriIn);
  std::cout<<"OrinIn dir: "<<aOriIn<<std::endl;


  // Initialize name manipulator & files
  std::string aDirXML,aDirImages,aPatIm;
  //std::string aGCPFileTmpName;
  //SplitDirAndFile(aDirXML,aGCPFileTmpName,aGCPFileName);
  SplitDirAndFile(aDirImages,aPatIm,aFullPattern);
  std::cout<<"Working dir: "<<aDirImages<<std::endl;
  std::cout<<"Images pattern: "<<aPatIm<<std::endl;


  cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
  const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));


  //read xml file
  //see cDicoAppuisFlottant definition in include/XML_GEN/ParamChantierPhotogram.xml
  cDicoAppuisFlottant aDico = StdGetFromPCP(aDirImages + aGCPFileName,DicoAppuisFlottant);
  std::list< cOneAppuisDAF > aOneAppuisDAFList= aDico.OneAppuisDAF();
  std::cout<<"On "<<aGCPFileName<<", found 3d points:\n";

  for (std::list< cOneAppuisDAF >::iterator itP=aOneAppuisDAFList.begin(); itP != aOneAppuisDAFList.end(); itP ++)
  {
    std::cout<<" - "<<itP->NamePt()<<" "<<itP->Pt()<<"\n";
  }


  std::cout<<"Found pictures:\n";
  for (unsigned int i=0;i<aSetIm.size();i++)
  {
    std::cout<<" - "<<aSetIm[i]<<"\n";
  }

  //create structure for 2d points:
  cSetOfMesureAppuisFlottants aMes2dList;

  std::ofstream outFile;//for ExportPolyIGN

  std::cout<<"point Name Ground coords => L3: 3D coords camera frame =>  C2: image frame (no distorsion) =>  F2: image frame with distorsion\n";
  for (unsigned int i=0;i<aSetIm.size();i++)
  {
    std::cout<<"For image "<<aSetIm[i]<<std::endl;
    cOrientedImage aIm(aOriIn,aSetIm[i],aICNM);
    cMesureAppuiFlottant1Im aListMes1Im;
    aListMes1Im.NameIm()=aSetIm[i];

    if (aExportPolyIGN)
    {
      std::string aPointeInitFileName=std::string("PointeInitIm.")+aSetIm[i].substr(aPrefixeNomImageSize,9);
      std::cout<<"  => "<<aPointeInitFileName<<std::endl;
      outFile.open( aPointeInitFileName.c_str() );
      ELISE_ASSERT(outFile.is_open(), ("Impossible to create \""+aPointeInitFileName+"\" file!").c_str() );
    }

    for (std::list< cOneAppuisDAF >::iterator itP=aOneAppuisDAFList.begin(); itP != aOneAppuisDAFList.end(); itP ++)
    {
      std::cout<<"  point "<<itP->NamePt()<<" "<<itP->Pt()<<" => ";
      std::cout<<" L3: "<<aIm.getCam()->R3toL3(itP->Pt())<<"  => ";
      std::cout<<"  C2: "<<aIm.getCam()->R3toC2(itP->Pt())<<" => ";

      Pt2dr aPtProj = aIm.getCam()->R3toF2(itP->Pt());
      std::cout<<"  F2: "<<aPtProj<<"\n";
      if (!  aIm.getCam()->Devant(itP->Pt()) )
      {
         //std::cout<<"       On the back\n";
         continue;
      }
      if (! aIm.getCam()->PIsVisibleInImage(itP->Pt()) )
        continue;

      //save it only inside picture
      if ((aPtProj.x>=0) && (aPtProj.y>=0) && (aPtProj.x<aIm.getCam()->Sz().x) && (aPtProj.y<aIm.getCam()->Sz().y) )
      {
        cOneMesureAF1I aOneMesIm;
        aOneMesIm.NamePt()=itP->NamePt();
        aOneMesIm.PtIm()=aPtProj;
        aListMes1Im.OneMesureAF1I().push_back( aOneMesIm );
        if (aExportPolyIGN)
          outFile<<itP->NamePt()<<" "<<aPtProj.x<<" "<<aPtProj.y<<"\n";
      }
    }
    aMes2dList.MesureAppuiFlottant1Im().push_back( aListMes1Im );

    outFile.close();
  }

  MakeFileXML(aMes2dList, aDirImages + "SimplePredict.xml");

  //export for calib poly:
  //  GS1_15_1_60_rrx00001_00001.tif => PointeInitIm.001_00001


  std::cout<<"Quit"<<std::endl;

  return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilite au code source et des droits de copie,
   de modification et de redistribution accordes par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitee.  Pour les mêmes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement,  a  l'utilisation,  a  la modification et/ou au
   developpement et a  la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a  des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a  charger  et  tester  l'adequation  du
   logiciel a  leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
