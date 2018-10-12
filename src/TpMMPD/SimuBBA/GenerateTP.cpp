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
#include "string.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"
#include "../schnaps.h"
#include <random>
#include <ctime>
#include <algorithm>

struct StructHomol {
    int IdIm;
    CamStenope * Cam;
    vector<int> VIdIm2;
    vector<ElPackHomologue> VElPackHomol;
};

struct TPConfig
{
    uint NbIm;
    std::vector<string> VNameIm;
    std::vector<Pt3dr> VPt3D;
    std::vector<std::vector<Pt2dr>> VVPt2dr;
};

// Function for comparing two students according
// to given rules
bool compareTwoConfig(TPConfig a, TPConfig b)
{
    if (a.NbIm != b.NbIm )
        return a.NbIm > b.NbIm;
    return a.VPt3D.at(0).x > b.VPt3D.at(0).x;
}
/*
// Fills total marks and ranks of all Students
void computeRanks(Student a[], int n)
{
    // To calculate total marks for all Students
    for (int i=0; i<n; i++)
        a[i].total = a[i].math + a[i].phy + a[i].che;

    // Sort structure array using user defined
    // function compareTwoStudents()
    sort(a, a+5, compareTwoStudents);

    // Assigning ranks after sorting
    for (int i=0; i<n; i++)
        a[i].rank = i+1;
}
*/

// check if one image is in the image list
bool IsInList(const std::vector<std::string> aVImgs, std::string aNameIm)
{
    int aFind = 0;
    for (uint i=0; i<aVImgs.size();i++)
    {
        if (aVImgs.at(i).compare(aNameIm)!=0) continue;
        else aFind=1;
    }
    if (aFind==0) return false;
    else return true;
}

bool IsInImage(Pt2dr aPt, Pt2dr aImgSz)
{
    if ((aPt.x >=0) && (aPt.x < aImgSz.x) && (aPt.y >=0) && (aPt.y < aImgSz.y))
        return true;
    else
        return false;
}

int GenerateTP_main(int argc,char ** argv)
{
    string aPatImgs,aDir,aImgs,aSH,aOri,aSHOut="Gen";
    vector<double> aNoiseGaussian(4,0.0);
    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aPatImgs,"Image Pattern",eSAM_IsExistFile)
                << EAMC(aSH, "PMul File",  eSAM_IsExistFile)
                << EAMC(aOri, "Ori",  eSAM_IsExistDirOri),
                LArgMain() << EAM(aSHOut,"Out",false,"Output name of generated tie points, Def=Gen")
                << EAM(aNoiseGaussian,"NoiseGaussian",false,"[meanX,stdX,meanY,stdY]")
                );

    SplitDirAndFile(aDir,aImgs,aPatImgs);

    // get directory
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    const std::vector<std::string> aVImgs = *(aICNM->Get(aImgs));

    //generation of noise on X & Y
    std::default_random_engine generator(time(0)); //seed
    std::normal_distribution<double> distributionX(aNoiseGaussian[0],aNoiseGaussian[1]);
    std::normal_distribution<double> distributionY(aNoiseGaussian[2],aNoiseGaussian[3]);


    //1. lecture of tie points and orientation
    StdCorrecNameOrient(aOri, aDir);
    const std::string  aSHInStr = aSH;
    std::cout << aSH << endl;
    cSetTiePMul * aSHIn = new cSetTiePMul(0);
    aSHIn->AddFile(aSHInStr);

    std::cout<<"Total : "<<aSHIn->DicoIm().mName2Im.size()<<" imgs"<<endl;
    std::map<std::string,cCelImTPM *> VName2Im = aSHIn->DicoIm().mName2Im;

    // load cam for all Img
    // Iterate through all elements in std::map
    std::map<std::string,cCelImTPM *>::iterator it = VName2Im.begin();
    vector<CamStenope*> aVCam (VName2Im.size());

    std::cout << "Loading tie points... \n";
    while(it != VName2Im.end())
    {
        //std::cout<<it->first<<" :: "<<it->second->Id()<<std::endl;
        string aNameIm = it->first;
        int aIdIm = it->second->Id();
        CamStenope * aCam = aICNM->StdCamStenOfNames(aNameIm, aOri);
        aCam->SetNameIm(aNameIm);
        aVCam[aIdIm] = aCam;
        it++;
    }

    std::cout << "Finished loading tie points! \n";
    std::cout<<"VPMul - Nb Config: "<<aSHIn->VPMul().size()<<endl;


    // declare aVCH to stock generated tie points
    vector<ElPackHomologue> aVPack (VName2Im.size());
    vector<int> aVIdIm2 (VName2Im.size(),-1);
    StructHomol aStructH;
    aStructH.VElPackHomol = aVPack;
    aStructH.VIdIm2 = aVIdIm2;
    vector<StructHomol> aVStructH (VName2Im.size(),aStructH);

    //2. get 3D position of tie points
    std::cout << "Filling ElPackHomologue... !\n";

    // parse Configs aVCnf
    std::cout << "Gaussian Noise: " << aNoiseGaussian << endl;
    std::vector<cSetPMul1ConfigTPM *> aVCnf = aSHIn->VPMul();
    for (uint aKCnf=1; aKCnf<aVCnf.size(); aKCnf++)
    {
        cSetPMul1ConfigTPM * aCnf = aVCnf[aKCnf];
        std::vector<int> aVIdIm =  aCnf->VIdIm();

        // Parse all images in one Config
        for (uint aKPtCnf=0; aKPtCnf<uint(aCnf->NbPts()); aKPtCnf++)
        {
            vector<Pt2dr> aVPtInter;
            vector<CamStenope*> aVCamInter;
            vector<int> aVIdImInter;

            for (uint aKImCnf=0; aKImCnf<aVIdIm.size(); aKImCnf++)
            {

                aVPtInter.push_back(aCnf->Pt(aKPtCnf, aKImCnf));
                aVCamInter.push_back(aVCam[aVIdIm[aKImCnf]]);
                aVIdImInter.push_back(aVIdIm[aKImCnf]);
            }

            //Intersect aVPtInter:

            ELISE_ASSERT(aVPtInter.size() == aVCamInter.size(), "Size not coherent");
            ELISE_ASSERT(aVPtInter.size() > 1 && aVCamInter.size() > 1, "Nb faiseaux < 2");
            Pt3dr aPInter3D = Intersect_Simple(aVCamInter , aVPtInter);

            // reproject aPInter3D sur tout les images dans aVCamInter
            vector<Pt2dr> aVP2d;
            vector<CamStenope *> aVCamInterVu;
            vector<int> aVIdImInterVu;
            for (uint itVCI=0; itVCI < aVCamInter.size(); itVCI++)
            {
                CamStenope * aCam = aVCamInter[itVCI];
                Pt2dr aPt2d = aCam->R3toF2(aPInter3D);
                //std::cout << aPt2d << "----------------";

                // add noise
                if (EAMIsInit(&aNoiseGaussian))
                {
                    aPt2d.x += distributionX(generator);
                    aPt2d.y += distributionY(generator);
                }
                //std::cout << aPt2d << endl;

                //check if the point is in the camera view
                if (aCam->PIsVisibleInImage(aPInter3D))
                {
                    aVP2d.push_back(aPt2d);
                    aVCamInterVu.push_back(aCam);
                    aVIdImInterVu.push_back(aVIdImInter[itVCI]);
                    cout.precision(17);
                    std::cout << aPInter3D << endl;
                }


            }
            // parse images to fill ElPackHomologue


            for (uint it1=0; it1 < aVCamInterVu.size(); it1++)
            {
                int aIdIm1=aVIdImInterVu.at(it1);
                aVStructH.at(aIdIm1).IdIm=aIdIm1;

                for (uint it2=0; it2 < aVCamInterVu.size(); it2++)
                {
                    if (it1==it2) continue;

                    int aIdIm2=aVIdImInterVu.at(it2);

                    ElCplePtsHomologues aCPH (aVP2d[it1],aVP2d[it2]);
                    aVStructH.at(aIdIm1).VElPackHomol.at(aIdIm2).Cple_Add(aCPH);
                    aVStructH.at(aIdIm1).VIdIm2.at(aIdIm2)=aIdIm2;
                }
            }


        }
    }


    std::cout << "ElPackHomologue filled !\n";

    //writing of new tie points
    std::cout << "Writing Homol files... \n";
    //key for tie points
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
            + "_"
            +  std::string(aSHOut)
            +  std::string("@")
            +  std::string("dat");


    for (uint itVSH=0; itVSH < aVStructH.size(); itVSH++)
    {
        int aIdIm1 = aVStructH.at(itVSH).IdIm;
        CamStenope * aCam1 = aVCam.at(aIdIm1);
        std::string aNameIm1 = aCam1->NameIm();
        if (IsInList(aVImgs,aNameIm1))
        {
            //std::cout << "Master Im: " << aNameIm1 << "    IdIm1 : " << aIdIm1 << endl;
            for (uint itVElPH=0; itVElPH < aVStructH.at(itVSH).VElPackHomol.size(); itVElPH++)
            {
                int aIdIm2 = aVStructH.at(itVSH).VIdIm2.at(itVElPH);
                if (aIdIm2 == -1) continue;
                CamStenope * aCam2 = aVCam.at(aIdIm2);
                std::string aNameIm2 = aCam2->NameIm();
                if (IsInList(aVImgs,aNameIm1))
                {
                    std::string aHmOut= aICNM->Assoc1To2(aKHOut, aNameIm1, aNameIm2, true);
                    ElPackHomologue aPck = aVStructH.at(aIdIm1).VElPackHomol.at(aIdIm2);
                    aPck.StdPutInFile(aHmOut);
                }
            }
        }

    }
    std::cout << "Finished writing Homol files ! \n";

    return EXIT_SUCCESS;
}

int GenerateMAF_main(int argc,char ** argv)
{
    string aPatImgs,aDir,aOri,aImgs,aGCPFile,aMAFOut;

    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aPatImgs,"Image pattern", eSAM_IsExistFile)
                << EAMC(aOri, "Ori",  eSAM_IsExistDirOri)
                << EAMC(aGCPFile, "File containning GCP coordinates",eSAM_IsExistFile),
                LArgMain() << EAM(aMAFOut,"Out",false,"Output name of the generated MAF file, Def=Gen_MAF_Ori.xml")
                );

    SplitDirAndFile(aDir,aImgs,aPatImgs);

    // get directory
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    const vector<string> aVImg = *(aICNM->Get(aImgs));

    StdCorrecNameOrient(aOri, aDir);
    std::cout << "Ori: Ori-" << aOri << "/" << endl;

    std::cout << "GCP file : " << aGCPFile << endl;

    if (!EAMIsInit(&aMAFOut))
        aMAFOut =  "Gen_MAF_" + aOri + ".xml";

    std::cout << "Output File : " << aMAFOut << endl;


    //read GCP coordinates
    cDicoAppuisFlottant aDicoAF = StdGetObjFromFile<cDicoAppuisFlottant>
            (
                aGCPFile,
                StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                "DicoAppuisFlottant",
                "DicoAppuisFlottant"
                );

    //output MAF file
    cSetOfMesureAppuisFlottants aDicoOut;
    std::list<cMesureAppuiFlottant1Im> aLMAFOut;
    std::list<cOneMesureAF1I> aMesOut;

    for (uint itVImg=0; itVImg < aVImg.size(); itVImg++)
    {
        std::string aOriName = "Ori-"+aOri+"/Orientation-"+aVImg.at(itVImg)+".xml";
        std::cout << aOriName << endl;
        if (!ELISE_fp::exist_file(aOriName)) continue;
        CamStenope * aCam = CamOrientGenFromFile(aOriName,aICNM);

        for
                (
                 std::list<cOneAppuisDAF>::iterator itDAF=aDicoAF.OneAppuisDAF().begin();
                 itDAF!=aDicoAF.OneAppuisDAF().end();
                 itDAF++
                 )
        {
            if (aCam->PIsVisibleInImage(itDAF->Pt()))
            {
                Pt2dr aPt2d = aCam->R3toF2(itDAF->Pt());
                cOneMesureAF1I aOMAF1I;
                aOMAF1I.NamePt() = itDAF->NamePt();
                aOMAF1I.PtIm() = aPt2d;
                aMesOut.push_back(aOMAF1I);
            }
        }
        cMesureAppuiFlottant1Im   aMAF1Im;
        aMAF1Im.NameIm() = aVImg.at(itVImg);
        aMAF1Im.OneMesureAF1I() = aMesOut;
        aLMAFOut.push_back(aMAF1Im);
        aMesOut.clear();

    }

    aDicoOut.MesureAppuiFlottant1Im() = aLMAFOut;
    MakeFileXML(aDicoOut,aMAFOut);
    return EXIT_SUCCESS;

}

int CompMAF_main(int argc,char ** argv)
{
    string aPatImgs,aDir,aImgs,aMAF1,aMAF2;
    string aOut="CmpMAF.xml";

    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aPatImgs,"Image pattern", eSAM_IsExistFile)
                << EAMC(aMAF1, "MAF File 1",  eSAM_IsExistFile)
                << EAMC(aMAF2, "MAF File 2",eSAM_IsExistFile),
                LArgMain() << EAM(aOut,"Out",false,"Output name of the generated CmpMAF file, Def=CmpMAF.xml")
                );
    SplitDirAndFile(aDir,aImgs,aPatImgs);

    cSetOfMesureAppuisFlottants aDico1 = StdGetFromPCP(aMAF1,SetOfMesureAppuisFlottants);
    cSetOfMesureAppuisFlottants aDico2 = StdGetFromPCP(aMAF2,SetOfMesureAppuisFlottants);


    std::list<cMesureAppuiFlottant1Im> & aLMAF1 = aDico1.MesureAppuiFlottant1Im();
    std::list<cMesureAppuiFlottant1Im> & aLMAF2 = aDico2.MesureAppuiFlottant1Im();


    for (std::list<cMesureAppuiFlottant1Im>::iterator iT1=aLMAF1.begin();iT1 != aLMAF1.end(); iT1++)
    {
        if (iT1->OneMesureAF1I().size()==0)continue;
        string aNameIm1=iT1->NameIm();
        std::list<cOneMesureAF1I> & aMes1 = iT1->OneMesureAF1I();
        //find iT2
        for(auto iT2=aLMAF2.begin();iT2!=aLMAF2.end();iT2++)
        {
            string aNameIm2=iT2->NameIm();
            if (aNameIm1.compare(aNameIm2)) continue;
            std::list<cOneMesureAF1I> & aMes2 = iT2->OneMesureAF1I();
            for (std::list<cOneMesureAF1I>::iterator itMes1 = aMes1.begin() ; itMes1 != aMes1.end() ; itMes1 ++)
            {
                std::string aNamePt1 = itMes1->NamePt();
                for(auto itMes2=aMes2.begin();itMes2!=aMes2.end();itMes2++)
                {
                    string aNamePt2 = itMes2->NamePt();
                    if (aNamePt1.compare(aNamePt2)!=0) continue;
                    itMes1->NamePt()+="*";
                    itMes1->PtIm().x-= itMes2->PtIm().x;
                    itMes1->PtIm().y-= itMes2->PtIm().y;
                }
            }
        }
    }
    MakeFileXML(aDico1,aOut);

    return EXIT_SUCCESS;
}


int GenerateOriGPS_main(int argc,char ** argv)
{
    string aDir,aImg,aOri,aOut="GPS_Gen";
    std::vector<std::string> aVImgPattern;
    std::string aFileGpsLa;

    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aVImgPattern,"Image pattern, grouped by lever-arm", eSAM_IsExistFile)
                << EAMC(aOri,"Orientation file",eSAM_IsExistDirOri)
                << EAMC(aFileGpsLa,"CSV file containing GPS Lever-arms for every image pattern",eSAM_IsExistFile),
                LArgMain() << EAM(aOut,"Out",false,"Output name of the generated Ori file, Def=Ori-GPS_Gen/")
                );

    SplitDirAndFile(aDir,aImg,aVImgPattern[0]);

    // get directory
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    // read image patterns
    std::vector<std::vector<std::string>> aVVImg;

    for (uint i=0; i<aVImgPattern.size(); i++)
    {
        vector<string> aVImg = *(aICNM->Get(aVImgPattern[i]));
        aVVImg.push_back(aVImg);
    }

    std::cout << "Get " << aVVImg.size() << " Image groupes!\n" ;


    // read CSV file containing GPS lever-arms
    std::ifstream aFile((aDir+aFileGpsLa).c_str());

    std::vector<Pt3dr> aVGpsLa;

    if(aFile)
    {
        double aX,aY,aZ;

        while(aFile >> aX >> aY >> aZ)
        {
            aVGpsLa.push_back(Pt3dr(aX,aY,aZ));
        }
        aFile.close();
    }
    std::cout << "Read " << aVGpsLa.size() << " GPS lever-arms!" << endl;

    ELISE_ASSERT(aVImgPattern.size()==aVGpsLa.size(),"Number of image patterns and Lever-arms not matched!");


    //for each image, read orientation file and create simulated orientation file
    StdCorrecNameOrient(aOri, aDir);
    std::cout << "Ori: Ori-" << aOri << "/" << endl;
    std::string aKey = "NKS-Assoc-Im2Orient@-" + aOut;

    for(uint iVV=0; iVV < aVVImg.size(); iVV++)
    {
        Pt3dr aGpsLa = aVGpsLa[iVV];
        std::vector<std::string> aVImg = aVVImg[iVV];
        for (uint iV=0; iV < aVImg.size(); iV++)
        {
            CamStenope * aCam = aICNM->StdCamStenOfNames(aVImg[iV], aOri);
            Pt3dr aGpsLaR3 = aCam->L3toR3(aGpsLa);
            ElRotation3D anOriGpsLa(aGpsLaR3,aCam->Orient().Mat().transpose(),false);
            aCam->SetOrientation(anOriGpsLa.inv());
            cOrientationConique  anOC = aCam->StdExportCalibGlob();
            std::string aOriOut = aICNM->Assoc1To1(aKey,aVImg[iV],true);
            MakeFileXML(anOC,aOriOut);

        }
    }


    return EXIT_SUCCESS;

}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
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
associés au chargement,  a  l'utilisation,  a  la modification et/ou au
développement et a  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe a
manipuler et qui le réserve donc a  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités a  charger  et  tester  l'adéquation  du
logiciel a  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
a  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder a cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
