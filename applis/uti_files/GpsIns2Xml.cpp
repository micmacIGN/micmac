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
#include "XML_GEN/all.h"


/*
bin/GpsIns2Xml /home/mpierrot/Data/Trieve/gps_imageselection.txt UDS

*/


using namespace NS_ParamChantierPhotogram;


void T(const std::string & aStr)
{
   double x;
   std::istringstream i(aStr);
   i >> x;

   std::cout << "X=" << x << " S=" << aStr << " BAD =" << i.fail() << "\n";

}

int main(int argc,char ** argv)
{
/*
   T("u3.14");
   T("3.14");
   T("2,71");
exit(1);
*/

   const double ZDEF=-1e60;

   std::string aNameIn,aType,aFileCalib;
   double aZ = ZDEF;
   double aP = ZDEF;

   eConventionsOrientation aCOR = eConvInconnue;
   eConventionsOrientation aCOR_Def = eConvInconnue;


   std::string aSubDir = "Orient/";
   std::string aPrefix = "GpsIns-";
   std::string aPostFix = ".xml";

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAM(aNameIn) 
                    << EAM(aFileCalib) 
                    << EAM(aType) ,
        LArgMain()  << EAM(aZ,"Alti",true)
                    << EAM(aP,"Prof",true)
                    << EAM(aSubDir,"SubDir",true)
                    << EAM(aPrefix,"Prefix",true)
                    << EAM(aPostFix,"PostFix",true)
   );

   cElRegex * anAutom=0;
   int * aNum=0;
   int aNumUDS[7] = {1,2,4,6,8,10,12};

   std::string aDir = DirOfFile(aNameIn);

   aFileCalib = aDir + aFileCalib;
   if (aSubDir!="")
      ELISE_fp::MkDirSvp(aDir+aSubDir);

   if (aType== "UDS")
   {
        anAutom = cElRegex::AutomUDS();
        aNum = aNumUDS;
        aCOR_Def = eConvAngErdas_Grade;
   }
   else
   {
      ELISE_ASSERT(false,"Unknown type");
   }


   if (aCOR==eConvInconnue)
   {
       ELISE_ASSERT(aCOR_Def!=eConvInconnue,"Impossible determiner convention orientation");
       aCOR=aCOR_Def;
   }



   std::vector<cLine_N_XYZ_WPK> aVOr = cLine_N_XYZ_WPK::FromFile(anAutom,aNum,aNameIn,true);
   cCalibrationInternConique  aCIC = StdGetObjFromFile<cCalibrationInternConique>
                                     (
                                           aFileCalib,
                                           StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                           "CalibrationInternConique",
                                           "CalibrationInternConique"
                                     );

   for (int aK=0 ; aK<int(aVOr.size()) ; aK++)
   {
        
        cOrientationConique  anOr;


        anOr.TypeProj().SetVal(eProjStenope);
        anOr.Interne() = aCIC;


        {
           cOrientationExterneRigide aCER;
           aCER.Centre() = aVOr[aK].mXYZ;
           aCER.ParamRotation().CodageAngulaire().SetVal(aVOr[aK].mWPK);
           ElRotation3D aRot = Std_RAff_C2M(aCER,aCOR);
           aCER = From_Std_RAff_C2M(aRot,true);

           if (aZ!=ZDEF)
              aCER.AltiSol().SetVal(aZ);
           if (aP!=ZDEF)
              aCER.Profondeur().SetVal(aP);

           anOr.Externe() = aCER;
         }

        anOr.ConvOri().KnownConv().SetVal(eConvApero_DistM2C);

        std::string aNameFileExp = aDir+aSubDir+aPrefix+aVOr[aK].mName +aPostFix;
        MakeFileXML(anOr,aNameFileExp,"Export-GpsIns2Xml");

         // std::cout  << "NAME=" << aVOr[aK].mName  << "\n";

   }

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
