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

/*
void TestOneCorner(ElCamera * aCam,const Pt2dr&  aP, const Pt2dr&  aG)
{
     Pt2dr aQ0 = aCam->DistDirecte(aP);
     Pt2dr aQ1 = aCam->DistDirecte(aP+aG);

     std::cout <<  " Grad " << (aQ1-aQ0 -aG) / euclid(aG) << " For " << aP << "\n";
}


void TestOneCorner(ElCamera * aCam,const Pt2dr&  aP)
{
    TestOneCorner(aCam,aP,Pt2dr(1,0));
    TestOneCorner(aCam,aP,Pt2dr(0,1));
    std::cout << "=======================================\n";
}

void CamInv_TestOneCorner(ElCamera * aCam)
{
    Pt2dr aSz = Pt2dr(aCam->Sz());

    TestOneCorner(aCam,Pt2dr(0,0));
    TestOneCorner(aCam,Pt2dr(aSz.x,0));
    TestOneCorner(aCam,Pt2dr(0,aSz.y));
    TestOneCorner(aCam,Pt2dr(aSz.x,aSz.y));
    TestOneCorner(aCam,Pt2dr(aSz.x/2.0,aSz.y/2.0));

    TestOneCorner(aCam,Pt2dr(1072,712));
}


void TestDistInv(ElCamera * aCam,const Pt2dr & aP)
{
    std::cout << "Test Dis Inv , aP " << aP << "\n";
    std::cout << "Res =  " << aCam->DistInverse(aP) << "\n";

}

void TestDirect(ElCamera * aCam,Pt3dr aPG)
{
    {
         std::cout.precision(10);

         std::cout << " ---PGround  = " << aPG << "\n";
         Pt3dr aPC = aCam->R3toL3(aPG);
         std::cout << " -0-CamCoord = " << aPC << "\n";
         Pt2dr aIm1 = aCam->R3toC2(aPG);

         std::cout << " -1-ImSsDist = " << aIm1 << "\n";
         Pt2dr aIm2 = aCam->DComplC2M(aCam->R3toF2(aPG));

         std::cout << " -2-ImDist 1 = " << aIm2 << "\n";

         Pt2dr aIm3 = aCam->OrGlbImaC2M(aCam->R3toF2(aPG));

         std::cout << " -3-ImDist N = " << aIm3 << "\n";

         Pt2dr aIm4 = aCam->R3toF2(aPG);
         std::cout << " -4-ImFinale = " << aIm4 << "\n";
    }
}

extern void TestCamCHC(ElCamera & aCam);
*/


     // ========== FOV ===============

double CamDemiFOV(CamStenope * aCam,const Pt2dr aP0)
{
   Pt3dr aDir0 = vunit(aCam->F2toDirRayonR3(Pt2dr(aCam->Sz())/2.0));
   Pt3dr aDir1 = vunit(aCam->F2toDirRayonR3(aP0));

   double  aScal = scal(aDir0,aDir1);

   return acos(aScal);
}

double CamFOV(CamStenope * aCam,const Pt2dr aP0)
{

   return CamDemiFOV(aCam,aP0) +  CamDemiFOV(aCam,Pt2dr(aCam->Sz())-aP0);
}

double CamFOV(CamStenope * aCam)
{
    Pt2dr aSz = Pt2dr(aCam->Sz());
    return ElMax(CamFOV(aCam,Pt2dr(0,0)),CamFOV(aCam,Pt2dr(aSz.x,0)));
}



     // ========== FOV ===============

double CamOuvVert(CamStenope * aCam,const Pt2dr aP0)
{
   Pt3dr aDir0 = vunit(aCam->F2toDirRayonR3(aP0));
   return aDir0.z;
}

double CamOuvVert(CamStenope * aCam)
{
    Box2dr aBox(Pt2dr(0,0),Pt2dr(aCam->Sz()));
    Pt2dr aC[4];
    aBox.Corners(aC);

    double aMinZ =  -1;
    for (int aK=0 ; aK<4 ; aK++)
       ElSetMax(aMinZ,CamOuvVert(aCam,aC[aK]));



   return acos(-aMinZ);

}



   //   =================================


class cAppli_TestChantier : cAppliWithSetImage
{
    public :
        cAppli_TestChantier (int argc,char ** argv);
    private :
};


void Chantier_TestOneCam(const std::string & aName, CamStenope * aCam,cAppli_TestChantier *)
{
   double aFOV = CamFOV(aCam);
   double aOuvV = CamOuvVert(aCam);


   std::cout << " Cam=" << aName << " FOV=" << aFOV << " OuvV " << aOuvV << "\n";
   // Pt2dr aP0 = 
}



cAppli_TestChantier::cAppli_TestChantier(int argc,char ** argv) :
    cAppliWithSetImage(argc,argv,0)
{
    for (int aKS=0 ; aKS<int(mVSoms.size()) ; aKS++)
    {
       cImaMM & anI = *(mVSoms[aKS]->attr().mIma);
       Chantier_TestOneCam(anI.mNameIm,anI.CamSNN(),this);
    }
}


int TestChantier_main(int argc,char ** argv)
{
    cAppli_TestChantier anAppli(argc-1,argv+1);
    return EXIT_SUCCESS;
}

//---------------------------------------------------------------------//


void AlphaGet27_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " ****************************************\n";
    std::cout <<  " *     G-éolocalisation                 *\n";
    std::cout <<  " *     E-n l'air                        *\n";
    std::cout <<  " *     T-étradimensionnelle             *\n";
    std::cout <<  " *     2-01                             *\n";
    std::cout <<  " *     7                                *\n";
    std::cout <<  " ****************************************\n\n";
}

int AlphaGet27_main(int argc,char ** argv)
{
    std::cout << "\n";

//    MMD_InitArgcArgv(argc,argv);

    std::string anOriFolder, aGCPfile, aSaisiefile, aFullDir;
    std::string aPathOut = "./AlphaGet27.xml";
    std::string aNamePt = "";

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir, "Full Directory (Dir+Pattern)", eSAM_IsPatFile)
                    << EAMC(anOriFolder, "Orientation folder (in projected coordinates)", eSAM_IsExistDirOri)
                    << EAMC(aGCPfile, "Ground Control Points file (XML)", eSAM_IsExistFileRP)
                    << EAMC(aSaisiefile, "Saisie Appuis file (XML)", eSAM_IsExistFileRP),
        LArgMain()  << EAM(aPathOut, "Out", true, "Output path (default = './AlphaGet27.xml')", eSAM_IsOutputFile)
                    << EAM(aNamePt, "GCPid", true, "GCP name for computation of coordinates (default : first in GCP file)")
    );

    std::string aDir, aPat;
    SplitDirAndFile(aDir,aPat,aFullDir);
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aSetIm = *(anICNM->Get(aPat));

    // Aspro treatments
    std::string aComAspro = MM3dBinFile_quotes("Aspro")
                            + " " + aFullDir
                            + " " + anOriFolder
                            + " " + aGCPfile
                            + " " + aSaisiefile;
//                            + " && "
//                            + MM3dBinFile_quotes("OriExport")
//                            + " " + aFullDir
//                            + " Ori-Aspro/Orientation.*xml"
//                            ;

//    std::cout << aComAspro << "\n";
    System(aComAspro);

    // TestDistortion treatments
    cDicoAppuisFlottant aDAF = StdGetFromPCP(aGCPfile, DicoAppuisFlottant);
    cSetOfMesureAppuisFlottants aSOMAF = StdGetFromPCP(aSaisiefile, SetOfMesureAppuisFlottants);

//    std::cout << "\n GCP id not OK !";

    if(aNamePt == "")
    {
        aNamePt = aSOMAF.MesureAppuiFlottant1Im().front().OneMesureAF1I().front().NamePt();
    }
    else
    {
        bool GCPnameOK = false;
        for(std::list<cOneAppuisDAF>::const_iterator itA = aDAF.OneAppuisDAF().begin() ; itA != aDAF.OneAppuisDAF().end() ; itA++)
        {
            cOneAppuisDAF anOA = *itA;
            if(aNamePt == anOA.NamePt()) GCPnameOK = true;
        }
        ELISE_ASSERT(GCPnameOK,"GCP name not in GCP input file");
    }


//    std::string aNameCalib = anOriFolder + "AutoCal@" + ".xml";
//    CamStenope * aCalib =  CamOrientGenFromFile(aNameCalib, anICNM);
    cDicoAppuisFlottant aDAFout;
    Pt3dr aFaisc = Pt3dr(0,0,0);

    for(unsigned aC=0; aC<aSetIm.size(); aC++)
    {
        CamStenope * aCam = CamOrientGenFromFile(anOriFolder + "/Orientation-" + aSetIm[aC] + ".xml", anICNM);
//        std::cout << "XML = " << anOriFolder + "/Orientation-" + aSetIm[aC] + ".xml" << "\n";
        ElMatrix<double> aMatRot = aCam->Orient().Mat();
        std::cout << "Image = " << aSetIm[aC] << "\n";

        for(std::list<cMesureAppuiFlottant1Im>::const_iterator itF = aSOMAF.MesureAppuiFlottant1Im().begin() ; itF != aSOMAF.MesureAppuiFlottant1Im().end() ; itF++)
        {
            cMesureAppuiFlottant1Im aMAF = *itF;
            if(aMAF.NameIm() == aSetIm[aC])
            {
                for(std::list<cOneMesureAF1I>::const_iterator itM = aMAF.OneMesureAF1I().begin() ; itM != aMAF.OneMesureAF1I().end() ; itM++)
                {
                    cOneMesureAF1I anOM = *itM;
                    if(anOM.NamePt() == aNamePt)
                    {
                        Pt2dr aPtIm = anOM.PtIm();
//                        aCalib->C2toDirRayonL3()
                        aFaisc = aCam->ImEtProf2Terrain(aPtIm, aCam->Focale());
                        double aFNorm = sqrt( pow(aFaisc.x, 2) + pow(aFaisc.y, 2) + pow(aFaisc.z, 2) );
                        aFaisc = aFaisc / aFNorm;
                    }
                }
            }
        }

        cOrientationConique anOCAspro = StdGetFromPCP("Ori-Aspro/Orientation-" + aSetIm[aC] + ".xml", OrientationConique);
        Pt3dr aPosIm = anOCAspro.Externe().Centre();
        double aDist = sqrt( pow(aPosIm.x, 2) + pow(aPosIm.y, 2) + pow(aPosIm.z, 2) );

        std::cout << "Distance camera-objet = " << aDist << " metres\n";

        Pt3dr aVecDir = aFaisc * aDist;

        std::cout << "Vecteur directeur camera-objet = {" << aVecDir.x << " ; " << aVecDir.y << " ; " << aVecDir.z << "}\n";

//        std::cout << aMatRot(0,0) << " ; " << aMatRot(0,1) << " ; " << aMatRot(0,2) << "\n";
//        std::cout << aMatRot(1,0) << " ; " << aMatRot(1,1) << " ; " << aMatRot(1,2) << "\n";
//        std::cout << aMatRot(2,0) << " ; " << aMatRot(2,1) << " ; " << aMatRot(2,2) << "\n";

        double aB2Ltab[3] = {aMatRot(0,0)*aVecDir.x + aMatRot(0,1)*aVecDir.y + aMatRot(0,1)*aVecDir.z,
                             aMatRot(1,0)*aVecDir.x + aMatRot(1,1)*aVecDir.y + aMatRot(1,2)*aVecDir.z,
                             aMatRot(2,0)*aVecDir.x + aMatRot(2,1)*aVecDir.y + aMatRot(2,2)*aVecDir.z};

//        std::cout << aB2Ltab[0] << " ; " << aB2Ltab[1] << " ; " << aB2Ltab[2] << std::endl;
        Pt3dr aLevier = Pt3dr::FromTab(aB2Ltab);

        std::cout << "Bras de levier terrain camera-objet = {" << aLevier.x << " ; " << aLevier.y << " ; " << aLevier.z << "}\n";

        cOneAppuisDAF anAp;
        anAp.Pt() = aCam->VraiOpticalCenter() + aLevier;
        anAp.NamePt() = aSetIm[aC];
        anAp.Incertitude() = Pt3dr(1,1,1);
        aDAFout.OneAppuisDAF().push_back(anAp);
    }

    MakeFileXML(aDAFout, aPathOut);

    AlphaGet27_Banniere();

    return EXIT_SUCCESS;
}


//---------------------------------------------------------------------//


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
