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


/*********************************************/
/*                                           */
/*                ::                         */
/*                                           */
/*********************************************/


int Ori2XML_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string aFullOri,anOut;
    std::string toto;


    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aFullOri,"Full pattern", eSAM_IsPatFile)
                      << EAMC(anOut,"Dir for result"),
           LArgMain() << EAM(toto,"toto",true, "todo", eSAM_InternalUse)
    );

    std::string aDir,aFileOriIn;
    SplitDirAndFile(aDir,aFileOriIn,aFullOri);

    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

   std::string aKeyOut = "NKS-Assoc-Im2Orient@-" + anOut;


    const std::vector<std::string> * aVName = anICNM->Get(aFileOriIn);

    for (int aK=0 ; aK<int(aVName->size()) ; aK++)
    {
        std::string aNameIn = (*aVName)[aK];
        std::string aNameFile = StdPrefix(aNameIn) + ".tif";

        CamStenope * aCam = CamStenope::StdCamFromFile(true,aNameIn,anICNM);
        // CamStenope * aCam = CamOrientGenFromFile(aNameIn,anICNM);


        std::string aNameOut = anICNM->Assoc1To1(aKeyOut,aNameFile,true);
        std::cout << "FFF= " << aCam->Focale() << " " << aNameOut  << "\n";
        //cCalibrationInternConique  aCIO = aCam->ExportCalibInterne2XmlStruct(aCam->Sz());
        cOrientationConique anOC = aCam->StdExportCalibGlob();

        MakeFileXML(anOC,aDir+aNameOut);
    }
    return 0;
}

/************************************************************/
/*                                                          */
/*     CONVERSION  Matis => MICMAC                          */
/*                                                          */
/************************************************************/

   // Conversions elementaires

Pt2di Mat2MM(const cXmlMatis_image_size & aSz)
{
   return Pt2di(aSz.width(),aSz.height());
}
Pt2dr Mat2MM(const cXmlMatis_P2d_cl & aSz)
{
   return Pt2dr(aSz.c(),aSz.l());
}
Pt3dr Mat2MM(const cXmlMatis_ppa & aPPa)
{
   return Pt3dr(aPPa.c(),aPPa.l(),aPPa.focale());
}

Pt3dr Mat2MM(const cXmlMatis_sommet & aSom)
{
   return Pt3dr(aSom.easting(),aSom.northing(),aSom.altitude());
}

Pt3dr Mat2MM(const cXmlMatis_pt3d & aP)
{
    return Pt3dr(aP.x(),aP.y(),aP.z());
}
ElMatrix<double> Mat2MM(const  cXmlMatis_mat3d & aMat)
{
    ElMatrix<double> aRes(3,3);
    SetLig(aRes,0,Mat2MM(aMat.l1().pt3d()));
    SetLig(aRes,1,Mat2MM(aMat.l2().pt3d()));
    SetLig(aRes,2,Mat2MM(aMat.l3().pt3d()));
    return aRes;
}

cXmlDate  Mat2MM(const cXmlMatis_image_date & aMatis)
{
   cXmlDate aMM;

   aMM.Y() =  aMatis.year();
   aMM.M() = aMatis.month();
   aMM.D() = aMatis.day();
   aMM.Hour().H() = aMatis.hour();
   aMM.Hour().M() = aMatis.minute();
   aMM.Hour().S() = aMatis.second();

   return aMM;
}

cCamStenopeDistRadPol * Mat2MMConik(const corientation &  aMatis)
{
    const cXmlMatis_geometry& aGeom = aMatis.geometry();
    const cXmlMatis_intrinseque& aIntr = aGeom.intrinseque();
    ELISE_ASSERT(aIntr.sensor().IsInit(),"No sensor found");
    const cXmlMatis_sensor& aSens = aIntr.sensor().Val();
    const cXmlMatis_distortion aDist = aSens.distortion();

    Pt2di aSz = Mat2MM(aSens.image_size());
    double aRay = (euclid(Pt2dr(aSz)) /2.0) *1.1;


    ElDistRadiale_Pol357 aDistMM(
                            aRay,
                            Mat2MM(aDist.pps()),
                            aDist.r3(),
                            aDist.r5(),
                            aDist.r7()
                         );


     Pt3dr aPPa = Mat2MM(aSens.ppa());
     std::vector<double> aParamAF;

     cCamStenopeDistRadPol* aCamMM = new cCamStenopeDistRadPol
                            (
                                false,
                                aPPa.z,
                                Pt2dr(aPPa.x,aPPa.y),
                                aDistMM,
                                aParamAF,
                                0,
                                aSz
                            );
     const cXmlMatis_extrinseque & anExtr = aGeom.extrinseque();
     Pt3dr aSom = Mat2MM(anExtr.sommet());
     ElMatrix<double> aMat = Mat2MM(anExtr.rotation().mat3d());
     ElRotation3D aR(aSom,aMat,true);
     aCamMM->SetOrientation(aR.inv());

     return aCamMM;
}



int MatisOri2MM_main(int argc,char ** argv)
{
    static cElHour TheH0(0,0,0);
    static cElDate TheT0(1,1,2000,TheH0);

    std::string aFullNameIn,aNameOut;
    double      aRounding=0 ;
    Pt3dr Offset(0,0,0);
    // bool SepCalib = true;
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aFullNameIn,"Full pattern", eSAM_IsPatFile)
                      << EAMC(aNameOut,"Orientation destination", eSAM_IsExistDirOri),
           LArgMain() << EAM(aRounding,"RoundingOffset",true, "Rounding factor, def none")
                      << EAM(Offset,"Offset", true, "Value of offset")
    );

    if (MMVisualMode) return EXIT_SUCCESS;

    std::string aKeyMM = "NKS-Assoc-Im2Orient@-" + aNameOut;
    std::string aKeyMATIS = "NKS-MATIS-Assoc-Im2Orient";
    cElemAppliSetFile  anEASF(aFullNameIn);

    std::string aKeyCalib = "NKS-Assoc-FromFocMm@Ori-"+ aNameOut + "/AutoCal@.xml";

    const cInterfChantierNameManipulateur::tSet * aVSet = anEASF.SetIm();
    for (int aK=0 ; aK<int(aVSet->size()) ; aK++)
    {
        std::string aNameIm = (*aVSet)[aK];
        std::string aNameMatCam = anEASF.mICNM->Assoc1To1(aKeyMATIS,aNameIm,true);
        corientation aXMLMatis = StdGetFromPCP(anEASF.mDir+aNameMatCam,orientation);
        std::string aNameMMCam = anEASF.mICNM->Assoc1To1(aKeyMM,aNameIm,true);
        cCamStenopeDistRadPol * aMMCam =  Mat2MMConik(aXMLMatis);
        cOrientationConique  aXMLMM = aMMCam->StdExportCalibGlob() ;

        if (EAMIsInit(&aRounding) || EAMIsInit(&Offset))
        {
            Pt3dr & aCentre = aXMLMM.Externe().Centre();
            if ((aK==0) && (! EAMIsInit(&Offset)))
            {
                Offset.x = arrondi_ni(aCentre.x,aRounding);
                Offset.y = arrondi_ni(aCentre.y,aRounding);
                Offset.z = arrondi_ni(aCentre.z,aRounding);
            }
            aCentre  = aCentre - Offset;
            aXMLMM.Externe().OffsetCentre().SetVal(Offset);
        }


         // Creer une structur qui contient le xif ou equivalent
        cMetaDataPhoto aMTD = cMetaDataPhoto::CreateExiv2(anEASF.mDir+aNameIm);
        std::string aMMNameIntr = "Ori-" + aNameOut + "/AutoCal"
                                 + ToString(round_ni(aMTD.FocMm() *10)) + ".xml";

         aMMNameIntr =  anEASF.mICNM->Assoc1To1(aKeyCalib,aNameIm,true);

        //
        aXMLMM.Interne().SetNoInit();
        cXmlDate aMMDate = Mat2MM(aXMLMatis.auxiliarydata().image_date());
        aXMLMM.Externe().Date().SetVal(aMMDate);
        aXMLMM.Externe().Time().SetVal(cElDate::FromXml(aMMDate).DifInSec(TheT0));
        aXMLMM.FileInterne().SetVal(aMMNameIntr);
        aXMLMM.RelativeNameFI().SetVal(true);

        MakeFileXML(aXMLMM,anEASF.mDir+aNameMMCam);

        cCalibrationInternConique aXMLi = aMMCam->ExportCalibInterne2XmlStruct(aMMCam->Sz());

        MakeFileXML(aXMLi,anEASF.mDir+aMMNameIntr);
    }
    std::cout << " Offset " << Offset << "\n";

    return EXIT_SUCCESS;
}





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
