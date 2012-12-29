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

class cArgLockCom
{
    public :

        cArgLockCom(int aNumArg) :
            mNumArgDir ( aNumArg)
        {
        }

        int mNumArgDir ;

        static const cArgLockCom NoLock;
};

const cArgLockCom  cArgLockCom::NoLock(-1);



FILE * FileLockMM3d(const std::string & aDir)
{
    return  FopenNN(aDir+"mm3d-LockFile.txt","a+","Lock File");
}

#include <ctime>

void LockTime(FILE * aFp,const std::string & aMes)
{

  time_t rawtime;
  struct tm * timeinfo;

  time ( &rawtime );
  timeinfo = localtime ( &rawtime );

  fprintf(aFp,"   %s %s",aMes.c_str(),asctime (timeinfo));
}

void LockIn(int  argc,char **  argv,const std::string & aDir)
{
   FILE * aFp = FileLockMM3d(aDir);

   fprintf(aFp,"=================================================================\n");
   for (int aK=0 ; aK< argc ; aK++)
       fprintf(aFp,"%s ",argv[aK]);
   fprintf(aFp,"\n");
   LockTime(aFp,"[Beginning at ]");

   fclose(aFp);
}

void LockOut(int aRes,const std::string & aDir)
{
   FILE * aFp = FileLockMM3d(aDir);
   std::string aMes;
   if (aRes==0)
      aMes = "[Ending correctly at]";
   else 
      aMes =  std::string("[Failing with code ") + ToString(aRes) +   " at ]" ;
   LockTime(aFp,aMes);
   fclose(aFp);
}




// CMMCom is a descriptor of a MicMac Command
class cMMCom
{
   public :
      cMMCom 
      (
             const std::string & aName,
             tCommande  aCommand,
             const std::string & aComment,
             const cArgLockCom& aLock=cArgLockCom::NoLock
      ) :
          mName     (aName),
          mLowName  (StrToLower(aName)),
          mCommand  (aCommand),
          mComment  (aComment),
          mLock     (aLock)
      {
      }


    
      std::string  mName;
      std::string  mLowName;
      tCommande    mCommand;
      std::string  mComment;
      cArgLockCom  mLock;
};





const std::vector<cMMCom> & getAvailableCommands()
{
   static std::vector<cMMCom> aRes;
   if (aRes.empty())
   {
       aRes.push_back(cMMCom("AperiCloud",AperiCloud_main," Visualisation of camera in ply file"));
       aRes.push_back(cMMCom("Apero",Apero_main," Compute external and internal orientations"));
       aRes.push_back(cMMCom("AperoChImSecMM",AperoChImMM_main,"Select secondary images for MicMac "));
       aRes.push_back(cMMCom("Bascule",Bascule_main," Generate orientations coherent with some physical information on the scene"));
       aRes.push_back(cMMCom("BatchFDC",BatchFDC_main," Tool for batching a set of commands"));
       aRes.push_back(cMMCom("Campari",Campari_main," Interface to Apero , for compensation of heterogenous measures",cArgLockCom(3)));
       aRes.push_back(cMMCom("CmpCalib",CmpCalib_main," Do some stuff"));
       aRes.push_back(cMMCom("cod",cod_main," Do some stuff"));
       aRes.push_back(cMMCom("CreateEpip",CreateEpip_main," Tool create epipolar images"));
       aRes.push_back(cMMCom("Dequant",Dequant_main," Tool for dequantifying an image"));
       aRes.push_back(cMMCom("Devlop",Devlop_main," Do some stuff"));
       aRes.push_back(cMMCom("ElDcraw",ElDcraw_main," Do some stuff"));
       aRes.push_back(cMMCom("GCPBascule",GCPBascule_main," Use ground control points (GCP) to make a global transformation from a general orientation to an orientation in the system of the GCP"));


       aRes.push_back(cMMCom("GCPConvert",GCP_Txt2Xml_main," Use ground control points (GCP) to make a global transformation from a general orientation to an orientation in the system of the GCP"));

       aRes.push_back(cMMCom("GenXML2Cpp",GenXML2Cpp_main," Do some stuff"));
       aRes.push_back(cMMCom("GrShade",GrShade_main," Compute shading from depth image"));
       aRes.push_back(cMMCom("Gri2Bin",Gri2Bin_main," Do some stuff"));
       aRes.push_back(cMMCom("MakeGrid",MakeGrid_main," Generate orientations in a grid format"));
       aRes.push_back(cMMCom("Malt",Malt_main," Simplified matching (interface to MicMac)",cArgLockCom(2)));
       aRes.push_back(cMMCom("MapCmd",MapCmd_main," Transforms a command working on a single file in a command working on a set of files"));
	   aRes.push_back(cMMCom("Mascarpone",Mascarpone_main," Automatic mask tests"));
	   aRes.push_back(cMMCom("MergePly",MergePly_main," Merge ply files"));
       aRes.push_back(cMMCom("MICMAC",MICMAC_main," Computes image matching from oriented images"));
       aRes.push_back(cMMCom("MMPyram",MMPyram_main," Computes pyram for micmac (internal use)"));
       aRes.push_back(cMMCom("MpDcraw",MpDcraw_main," Interface to dcraw"));
       aRes.push_back(cMMCom("MMInitialModel",MMInitialModel_main,"Initial Model for MicMac "));

       aRes.push_back(cMMCom("MyRename",MyRename_main,"File renaming using posix regular expression "));


       aRes.push_back(cMMCom("Nuage2Ply",Nuage2Ply_main," Convert depth map into point cloud"));
       aRes.push_back(cMMCom("NuageBascule",NuageBascule_main,"To Channge geometry of depth map "));



       aRes.push_back(cMMCom("Pasta",Pasta_main," Do some stuff"));
       aRes.push_back(cMMCom("PastDevlop",PastDevlop_main," Do some stuff"));
       aRes.push_back(cMMCom("Pastis",Pastis_main," Tie points detection"));
       aRes.push_back(cMMCom("Porto",Porto_main," Generates a global ortho-photo"));
       aRes.push_back(cMMCom("Reduc2MM",Reduc2MM_main," Do some stuff"));
       aRes.push_back(cMMCom("ReducHom",ReducHom_main," Do some stuff"));
       aRes.push_back(cMMCom("RepLocBascule",RepLocBascule_main," Tool to define a local repair without changing the orientation"));
       aRes.push_back(cMMCom("SBGlobBascule",SBGlobBascule_main," Tool for 'scene based global' bascule"));
       aRes.push_back(cMMCom("ScaleIm",ScaleIm_main," Tool for scaling image"));
       aRes.push_back(cMMCom("ScaleNuage",ScaleNuage_main," Tool for scaling internal representation of point cloud"));
       aRes.push_back(cMMCom("Tapas",Tapas_main," Interface to Apero to compute external and internal orientations",cArgLockCom(3)));
       aRes.push_back(cMMCom("Tapioca",Tapioca_main," Interface to Pastis for tie point detection and matching",cArgLockCom(3)));
       aRes.push_back(cMMCom("Tarama",Tarama_main," Do some stuff"));
       aRes.push_back(cMMCom("Tawny",Tawny_main," Interface to Porto to generate ortho-image"));
       aRes.push_back(cMMCom("TestCam",TestCam_main," Test camera orientation convention"));
       aRes.push_back(cMMCom("tiff_info",tiff_info_main," Tool for giving information about a tiff file"));
       aRes.push_back(cMMCom("to8Bits",to8Bits_main," Tool for converting 16 or 32 bit image in a 8 bit image."));
       aRes.push_back(cMMCom("Undist",Undist_main," Tool make undistorted images"));

       aRes.push_back(cMMCom("CheckDependencies",CheckDependencies_main," check dependencies to third-party tools"));

#if (ELISE_X11)
       aRes.push_back(cMMCom("MPDtest",MPDtest_main," My own test"));
       aRes.push_back(cMMCom("SaisieAppuisInit",SaisieAppuisInit_main,"Interactive tool for initial capture of GCP"));
       aRes.push_back(cMMCom("SaisieAppuisPredic",SaisieAppuisPredic_main,"Interactive tool for assisted capture of GCP "));
       aRes.push_back(cMMCom("SaisieBasc",SaisieBasc_main,"Interactive tool to cature information on the scene"));
       aRes.push_back(cMMCom("SaisieMasq",SaisieMasq_main,"Interactive tool to capture  masq"));
       aRes.push_back(cMMCom("SaisiePts",SaisiePts_main,"Tool to capture GCP (low level, not recommanded)"));
       aRes.push_back(cMMCom("SEL",SEL_main,"Tool to visualise tie points"));
       aRes.push_back(cMMCom("MICMACSaisieLiaisons",MICMACSaisieLiaisons_main,"low level version of SEL, not recommanded"));
	   
#ifdef ETA_POLYGON

	   //Etalonnage polygone
	   aRes.push_back(cMMCom("Compens",Compens_main," Do some stuff"));
	   aRes.push_back(cMMCom("CatImSaisie",CatImSaisie_main," Do some stuff"));
	   aRes.push_back(cMMCom("CalibFinale",CalibFinale_main," Do some stuff"));
	   aRes.push_back(cMMCom("CalibInit",CalibInit_main," Do some stuff"));
	   aRes.push_back(cMMCom("ConvertPolygone",ConvertPolygone_main," Do some stuff"));
	   aRes.push_back(cMMCom("PointeInitPolyg",PointeInitPolyg_main," Do some stuff"));
	   aRes.push_back(cMMCom("RechCibleDRad",RechCibleDRad_main," Do some stuff"));
	   aRes.push_back(cMMCom("RechCibleInit",RechCibleInit_main," Do some stuff"));
	   aRes.push_back(cMMCom("ScriptCalib",ScriptCalib_main," Do some stuff"));
#endif

#endif
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

   // MPD : deplace sinon core dump qd argc==1
   MMD_InitArgcArgv( argc, argv );

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
          cArgLockCom aLock = aVComs[aKC].mLock;
          bool DoLock = (aLock.mNumArgDir >0) && (aLock.mNumArgDir<argc);
          if (DoLock)
          {
               LockIn(argc,argv,DirOfFile(argv[aLock.mNumArgDir]));
          }
          int aRes =  (aVComs[aKC].mCommand(argc-1,argv+1));

          if (DoLock)
          {
               LockOut(aRes,DirOfFile(argv[aLock.mNumArgDir]));
          }
          return aRes;
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
           for (unsigned int aKC=0 ; aKC<mSugg[aKS]->mRes.size() ; aKC++)
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
