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


#include "TiepTri.h"


cParamAppliTieTri::cParamAppliTieTri():
   mDistFiltr         (TT_DefSeuilDensiteResul),
   mNumInterpolDense  (-1),
   mDoRaffImInit      (false),
   mNbByPix           (1),
   mSzWEnd            (6),
   mNivLSQM           (-1),
   mRandomize         (0.0),
   mNoTif             (false),
   mFilSpatial        (true),
   mFilAC             (true),
   mFilFAST           (true),
   mTT_SEUIL_SURF_TRI (TT_SEUIL_SURF_TRI_PIXEL),
   mTT_SEUIL_CORREL_1PIXSUR2 (TT_SEUIL_CORREL_1PIXSUR2),
   mTT_SEUIl_DIST_Extrema_Entier (TT_SEUIl_DIST_Extrema_Entier),
   mEtapeInteract     (-1),
   mLastEtape         (2),
   mFlagFS            ((1<<16) - 1),
   mHomolOut          ("_TiepTri"),
   mSurfDiffAffHomo   (Pt2dr(mTT_SEUIL_SURF_TRI, DBL_MAX)),
   mUseHomo           (false),
   mMaxErr            (1.0)
{
}


int TiepTri_Main(int argc,char ** argv)
{
   std::string aFullNameXML,anOri;
   Pt3di       aSzW;
   bool        aDebug=false;
   Pt2dr       aPtsSel;
   std::vector<int> aNumSel;
   std::string      aKeyMasqIm;
   cParamAppliTieTri aParam;
   bool              UseABCorrel = false;
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aFullNameXML, "Name XML for Triangu",  eSAM_IsPatFile)
                     << EAMC(anOri,        "Orientation dir"),
         LArgMain()   
                      << EAM(aParam.mSzWEnd,  "SzWEnd",true,"SzW Final")
                      << EAM(aParam.mDistFiltr,"DistF",true,"Average distance between tie points")
                      << EAM(aParam.mNumInterpolDense,"IntDM",true," Interpol for Dense Match, -1=NONE, 0=BiL, 1=BiC, 2=SinC")
                      << EAM(aParam.mDoRaffImInit,"DRInit",true," Do refinement on initial images, instead of resampled")
                      << EAM(aParam.mNivLSQM,"LSQC",true,"Test LSQ,-1 None (Def), Flag 1=>Affine Geom, Flag 2=>Affin Radiom")
                      << EAM(aParam.mNbByPix,"NbByPix",true," Number of point inside one pixel")
                      << EAM(aParam.mRandomize,  "Randomize",true,"Level of random perturbation (LSQ Match), def=1.0 in interactive, else 0.0  ")               
                      << EAM(aKeyMasqIm,"KeyMasqIm",true,"Key for masq, Def=NKS-Assoc-STD-Masq, set NONE or key with NONE result")

                      << EAM(aSzW,         "SzW",true,"if visu [x,y,Zoom]")
                      << EAM(aDebug,       "Debug",true,"If true do debuggibg")
                      << EAM(aPtsSel,  "PSelectT",true,"for selecting triangle")
                      << EAM(aNumSel,  "NumSelIm",true,"for selecting imade")
                      << EAM(UseABCorrel,  "UseABCorrel",true,"Tuning use correl in mode A*v1+B=v2 ")

                      << EAM(aParam.mEtapeInteract,"StepInt",true,"Step of Intection (apply when SzW is init) see FFS")
                      << EAM(aParam.mLastEtape,  "LastStep",true,"Last step (tuning essentially)")
                      << EAM(aParam.mNoTif,  "NoTif",true,"Not an image TIF - read img in Tmp-MM-Dir")
                      << EAM(aParam.mFilSpatial,  "FilSpatial",true,"Use filter spatial ? (def = true)")
                      << EAM(aParam.mFilFAST,  "FilFAST",true,"Use FAST condition ? (def = true)")
                      << EAM(aParam.mFilAC,  "FilAC",true,"Use Autocorrelation condition ? (def = true)")
                      << EAM(aParam.mTT_SEUIL_SURF_TRI,  "surfTri",true,"Surface min to eliminate too small triangle (def = 100 unit)")
                      << EAM(aParam.mTT_SEUIL_CORREL_1PIXSUR2,  "correlBrut",true,"Threshold of correlation score 1pxl/2 (def = 0.7)")
                      << EAM(aParam.mTT_SEUIl_DIST_Extrema_Entier,  "distWithExtr",true,"Distant limit b/w max correl point & extrema (def=1.5pxl)")
                      << EAM(aParam.mFlagFS,  "FFS",true,"Flag spatial  [1= after int,2= after bilin ,4=after dense, 8= at end]")

                      << EAM(aParam.mHomolOut,  "Out",true,"Suffix for Homol Out Folder (def=_TiepTri)")
                      << EAM(aParam.mUseHomo,  "useHomo",true,"Use homography for triangle")
                      << EAM(aParam.mSurfDiffAffHomo,  "surfDiff",true,"range of triangle surface for evaluation diff Aff-Homogr")
                      << EAM(aParam.mMaxErr,  "MaxErr",true,"Max Error Aff/Homogr (to do statistic)")
               );

   bool WithInteract = EAMIsInit(& aSzW);

   if (WithInteract  && (!EAMIsInit(&aParam.mEtapeInteract)))
      aParam.mEtapeInteract = 1;

   if (EAMIsInit(&aParam.mSurfDiffAffHomo))
   {
       aParam.mUseHomo = true;
   }
   if (! EAMIsInit(&aParam.mDoRaffImInit))
   {
       aParam.mDoRaffImInit =  (aParam.mNivLSQM >= 0);
   }
   if (! EAMIsInit(&aParam.mRandomize))
   {
       aParam.mRandomize =  (WithInteract) ? 1.0 : 0.0;
   }

   if ((aParam.mNivLSQM>=0) || UseABCorrel)
   {
      USE_SCOR_CORREL = false;
   }

   std::string aDir,aNameXML;

   SplitDirAndFile(aDir,aNameXML,aFullNameXML);


   if (!  StdCorrecNameOrient(anOri,aDir,true))
   {
      StdCorrecNameOrient(anOri,"./");
      aDir = "./";
   }


   cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);


       cXml_TriAngulationImMaster aTriang =   StdGetFromSI(aFullNameXML,Xml_TriAngulationImMaster);

       cAppliTieTri  anAppli(aParam,anICNM,aDir,anOri,aTriang);
       anAppli.Debug() = aDebug;


       if (EAMIsInit(&aPtsSel))
           anAppli.SetPtsSelect(aPtsSel);
       if (EAMIsInit(&aNumSel))
           anAppli.SetNumSelectImage(aNumSel);

       if (EAMIsInit(&aSzW))
       {
           anAppli.SetSzW(Pt2di(aSzW.x,aSzW.y),aSzW.z);
       }

       if (EAMIsInit(&aKeyMasqIm))
          anAppli.SetMasqIm(aKeyMasqIm);


       anAppli.DoAllTri(aTriang);

   return EXIT_SUCCESS;

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
aooter-MicMac-eLiSe-25/06/2007*/
