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

/**********************************************************/
/*	   Author: Luca Penasa, after Girod's Apero2PMVS 	  */
/**********************************************************/

#include "StdAfx.h"
#include <algorithm>


void Apero2Meshlab_Banniere()
{
  std::cout << "\n  *************************************************\n";
  std::cout << "  **                                             **\n";
  std::cout << "  **                   Apero                     **\n";
  std::cout << "  **                     2                       **\n";
  std::cout << "  **                  Meshlab                    **\n";
  std::cout << "  **                                             **\n";
  std::cout << "  **              Contribution by:               **\n";
  std::cout << "  **    Luca Penasa, after Girod's Apero2PMVS    **\n";
  std::cout << "  **                                             **\n";
  std::cout << "  *************************************************\n";
}

///
/// \brief The MeshlabRasterProjectWriter class provides a simple way to write
/// a valid mlp file for meshlab containing raster informations.
/// The best solution would be to write it using the xml support within micmac
/// but it seems to be pretty complicated and an hard-coded solution have been used!
/// For using it remember to call :
/// MeshlabRasterProjectWriter writer;
/// writer.setOutputFileName(meshlabProjectFileName);
/// writer.openFile();
/// writer.writeHeaderStuff();
///
///then for each raster:
/// writer.writeMLRasterStart(aRawName + ".jpg");
/// writer.writeVCGCamera(Trans,  NRot, distortions, viewport, pixelsize, principal_point, focal);
/// writer.writePlaneSemantic(aRawName + ".jpg");
/// writer.writeMLRasterEnd();
///
/// and at the end:
/// writer.writeFooterStuff();
/// writer.writeAllToFile();
///
///
class MeshlabRasterProjectWriter
{
public:

  //some useful types
  typedef Pt3d<double> centerT;
  typedef Pt2d<double> vector2d;
  typedef Pt2d<int> vector2dInt;

  typedef ElMatrix<double> rotationT;

  ///
  /// \brief MeshlabProjectWriter def constructor
  ///
  MeshlabRasterProjectWriter(){}

  void openFile()
  {
    file_.open(filename_.c_str());
  }

  void setOutputFileName(std::string filename)
  {
    filename_ = filename;
  }

  void writeHeaderStuff()
  {
    writeMeshlabDocumentHeader();
    writerMeshlabProjectStart();
    writeRasterGroupStart();
  }

  void writeMLRasterStart(const std::string filename_and_path)
  {
    std::string text = std::string("<MLRaster label=\"" ) + filename_and_path + std::string("\">");
    writeString(text.c_str());
  }

  void writeVCGCamera(centerT center, rotationT rotation, vector2d distortions, vector2dInt viewportPX, vector2d pixelsizeMM, vector2d principal_pointPX, double focalMM)
  {
    stringstream s;

    s << "<VCGCamera TranslationVector=\"" << center.x << " " << center.y << " " << center.z << " 1\"" ;
    s << " LensDistortion=\"" << distortions.x << " " << distortions.y << "\"";
    s << " ViewportPx=\"" << viewportPX.x << " " << viewportPX.y << "\"";
    s << " PixelSizeMm=\"" << pixelsizeMM.x << " " << pixelsizeMM.y << "\"" ;
    s << " CenterPx=\"" << principal_pointPX.x << " " << principal_pointPX.y << "\"" ;
    s << " FocalMm=\"" << focalMM << "\"" ;
    s << " RotationMatrix=\"";
    for (int i = 0; i < rotation.tx(); ++i)
      {
        for (int j=0; j < rotation.ty(); ++j)
          {
            s << rotation(i, j) << " ";
          }
        s << 0.0 << " ";
      }

    s << 0.0 << " " <<   0.0 << " " << 0.0 << " " << 1.0;

    s << "\"/>";

    writeString(s.str().c_str());
  }

  void writePlaneSemantic(std::string filename, std::string semantic= "rgb")
  {
    std::string s = std::string("<Plane semantic=\"") + semantic + "\" " + "fileName=\"" + filename + "\"/>";

    writeString(s.c_str());
  }

  void writeMLRasterEnd()
  {
    writeString("</MLRaster>");
  }

  void writeFooterStuff()
  {
    writeRasterGroupEnd();
    writeMeshlabProjectEnd();
  }

  void writeAllToFile()
  {
    file_ << stream_.str().c_str();
    file_.close();
  }


private:

  void writeMeshlabDocumentHeader()
  {
    writeString("<!DOCTYPE MeshLabDocument>") ;
  }

  void writerMeshlabProjectStart()
  {
    writeString("<MeshLabProject>") ;
  }

  void writeRasterGroupStart()
  {
    writeString("<RasterGroup>");
  }



  void writeRasterGroupEnd()
  {
    writeString("</RasterGroup>");
  }

  void writeMeshlabProjectEnd()
  {
    writeString("</MeshLabProject>");
  }



  void writeString(const char * str)
  {
    stream_ << str << "\n";
  }



private:

  ofstream file_;
  std::string filename_;

  stringstream stream_;

};

void Apero2Meshlab(string aFullPattern, string aOri, int aUnDist)
{
  string aPattern,aNameDir;
  SplitDirAndFile(aNameDir,aPattern,aFullPattern);


  //Bulding the output file system
  ELISE_fp::MkDirRec(aNameDir + "meshlabRast-"+ aOri + "/");

  //Reading the list of input files
  list<string> ListIm=RegexListFileMatch(aNameDir,aPattern,1,false);
  int nbIm = (int)ListIm.size();
  cout<<"Images to process: "<<nbIm<<endl;

  string cmdDRUNK, cmdConv, cmdRM, cmdCopyMeta;
  list<string> ListDrunk, ListConvert, ListRM, ListCopyMeta;

  string meshlabProjectFileName=aNameDir + "/meshlabRast-"+ aOri + "/meshlabProj.mlp";


  MeshlabRasterProjectWriter writer;
  writer.setOutputFileName(meshlabProjectFileName);
  writer.openFile();
  writer.writeHeaderStuff();

  //prepare the list of command if undistortion is required. If not just write out the right things in the mlp file
  for(int i=0;i<nbIm;++i)
    {
      //Reading the images list
      string aFullName=ListIm.front();

      int lastindex = (int)aFullName.find_last_of(".");
      string aRawName = aFullName.substr(0, lastindex);

      cout<<aFullName<<" ("<<i+1<<" of "<<nbIm<<")"<<endl;
      ListIm.pop_front();
      string aMeshLabImageFullName = aNameDir + "meshlabRast-" + aOri + ELISE_STR_DIR + aFullName;
      string aMeshLabImageRawName = aNameDir + "meshlabRast-" + aOri + ELISE_STR_DIR + aRawName;
      if (aUnDist) //!!!!JM
        {
          //Creating the numerical format for the output files names
          char nb[12];
          sprintf(nb, "%08d", i);

          //Creating the lists of DRUNK and Convert commands
      cmdDRUNK = MM3dBinFile("Drunk") + " " + aNameDir + aFullName + " " + aOri + " Out=" + "meshlabRast-" + aOri + ELISE_STR_DIR + " Talk=0";
          ListDrunk.push_back(cmdDRUNK);

      cmdConv = g_externalToolHandler.get("convert").callName() + " " + aMeshLabImageFullName + ".tif " + aMeshLabImageRawName  + ".jpg";
          ListConvert.push_back(cmdConv);

      //Remove temporary tiffs created by Drunk
      cmdRM = string(SYS_RM)+ " " + aMeshLabImageFullName + ".tif";
      ListRM.push_back(cmdRM);

          //also copy the metadata from the original
      cmdCopyMeta = g_externalToolHandler.get("exiftool").callName() + " -overwrite_original -TagsFromFile " + aNameDir
          + aFullName + " " + aMeshLabImageRawName + ".jpg";
          ListCopyMeta.push_back(cmdCopyMeta);
        }


      //Formating the camera name
      string aNameCam="Ori-" + aOri + ELISE_STR_DIR + "Orientation-" + aFullName + ".xml";

      //Loading the camera
      cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
      CamStenope * aCS = CamOrientGenFromFile(aNameCam, anICNM);

      ElCamera * aCam = Cam_Gen_From_File(aNameDir + aNameCam, "OrientationConique" , anICNM);


      ElMatrix<double> Rot(3,3,0.0);
      ElMatrix<double> RotX(3,3,0.0);

      Rot=aCS->Orient().Mat();
      //rotate the above matrix with this one:
      RotX(0,0) = 1;
      RotX(1,1) = -1;
      RotX(2,2) = -1;
      ElMatrix<double> NRot = RotX * Rot;
      NRot = NRot.transpose();

      //get the translation - this is ok as it is!  just get the negatives
      Pt3d<double> Trans;
      Trans= -aCS->OrigineProf();

      //Loading EXIF data, to get FocMm and Calculate PixelSizeMm
      cMetaDataPhoto aMetaData = cMetaDataPhoto::CreateExiv2(aNameDir + aFullName);
      double rap35MmThisFrame = aMetaData.Foc35()/aMetaData.FocMm();
      double PxSize35Mm = euclid(Pt2di(24,36))/euclid(aMetaData.TifSzIm());
      double PxSizeMm = PxSize35Mm / rap35MmThisFrame;


      //the focal
      double focal = aCS->Focale()*PxSizeMm;

      //and the other internal parameters
      Pt2d<double> distortions, pixelsize, principal_point;
      Pt2di viewport;
      viewport = aCS->Sz();

      pixelsize.x = PxSizeMm; pixelsize.y = PxSizeMm;

      if (aUnDist)
        {
          distortions.x = 0.0 ; distortions.y = 0.0; //images have been undistorted!
          principal_point = viewport / 2.0;
        }
      else
        {
          cmdConv = g_externalToolHandler.get("convert").callName() + " " + aNameDir + aFullName  + " " + aMeshLabImageRawName + ".jpg";
          ListConvert.push_back(cmdConv);

          NS_ParamChantierPhotogram::cCalibrationInternConique cal_xml = aCam->ExportCalibInterne2XmlStruct(aCam->Sz());

          vector<cCalibDistortion> dists = cal_xml.CalibDistortion();

          cCalibDistortion dist = dists.at(0); //get the first one
          cTplValGesInit<cCalibrationInterneRadiale> qua = dist.ModRad();

          cCalibrationInterneRadiale rad = qua.Val();
          vector<double> coeffs = rad.CoeffDist();

          distortions.x = coeffs.at(0);
          distortions.y = coeffs.at(1);

          //in principle this will be the principal point of the camera
          principal_point = cal_xml.PP();

          //BUT it seems that is causing problems when importing in meshlab.

          principal_point = viewport / 2.0;
        }

      writer.writeMLRasterStart(aRawName + ".jpg");
      writer.writeVCGCamera(Trans,  NRot, distortions, viewport, pixelsize, principal_point, focal);
      if (aUnDist)
        writer.writePlaneSemantic(aRawName + ".jpg");
      else
        writer.writePlaneSemantic(aRawName + ".jpg");

      writer.writeMLRasterEnd();
    }//end of "for each image"

  writer.writeFooterStuff();

  writer.writeAllToFile();

  if (aUnDist)
    {
      //Undistorting the images with Drunk
      cout<<"Undistorting the images with Drunk"<<endl;
      cEl_GPAO::DoComInParal(ListDrunk,aNameDir + "MkDrunk");

      //Converting into .jpg
      cout<<"Converting into .jpg"<<endl;
      cEl_GPAO::DoComInParal(ListConvert,aNameDir + "MkConvert");

    //Remove tiffs from drunk
    cEl_GPAO::DoComInSerie(ListRM);

      //Copy the metadata - it is better to keep metadatas sync
      cout<<"Copying metadata"<<endl;
      cEl_GPAO::DoComInParal(ListCopyMeta,aNameDir + "MkCopyMetadata");
    }
  else
    {
      cout<<"Converting into .jpg"<<endl;
      cEl_GPAO::DoComInParal(ListConvert,aNameDir + "MkConvert");
    }


  Apero2Meshlab_Banniere();
}


int Apero2Meshlab_main(int argc,char ** argv)
{
  MMD_InitArgcArgv(argc,argv);

  //Reading the arguments
  string aFullPattern, aOri;
  bool aUnDist = false;

  ElInitArgMain
      (
        argc,argv,
        LArgMain()  << EAMC(aFullPattern,"Images' name pattern", eSAM_IsPatFile)
                    << EAMC(aOri,"Orientation name", eSAM_IsExistDirOri),
        LArgMain()  << EAM(aUnDist,"UnDist", true, "Undistort images using Drunk", eSAM_IsBool)
        );

  if (!MMVisualMode)
  {
      string aDir, aPat;
      SplitDirAndFile(aDir, aPat, aFullPattern);
      StdCorrecNameOrient(aOri, aDir);

    //if aUnDist == true, we need to run convert and exiftool
    if(aUnDist)
    {
      if(!g_externalToolHandler.get("convert").isCallable())
      {
        ELISE_ERROR_RETURN("convert command unavailable")
      }
      if(!g_externalToolHandler.get("exiftool").isCallable())
      {
        ELISE_ERROR_RETURN("exiftool command unavailable")
      }
    }
      Apero2Meshlab(aFullPattern, aOri, aUnDist);

      return EXIT_SUCCESS;
  }
  else
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
   offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement, a l'utilisation, a la modification et/ou au
   developpement et a la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a charger  et  tester  l'adequation  du
   logiciel a leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
