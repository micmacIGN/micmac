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

#include <random>
#include <ctime>
#include<iostream>
#include<fstream>
#include <algorithm>

#include "StdAfx.h"
#include "string.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"
#include "../schnaps.h"
#include "../TpPPMD.h"
#include "SimuBBA.h"


int GenerateTP_main(int argc,char ** argv)
{
    string aPatImgs,aDir,aImgs,aSH,aOri,aSHOut="simulated",aNameImNX,aNameImNY,aExportP3D;
    vector<double> aNoiseGaussian(4,0.0);
    int aSeed;
    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aPatImgs,"Image Pattern",eSAM_IsExistFile)
                           << EAMC(aSH, "PMul File",  eSAM_IsExistFile)
                           << EAMC(aOri, "Ori",  eSAM_IsExistDirOri),
                LArgMain() << EAM(aSHOut,"Out",false,"Output name of generated tie points, Def=simulated")
                           << EAM(aSeed,"Seed",false,"Seed for generating random noise")
                           << EAM(aNoiseGaussian,"NoiseGaussian",false,"[meanX,stdX,meanY,stdY]")
                           << EAM(aNameImNX,"ImNX",false,"image containing noise on X-axis")
                           << EAM(aNameImNY,"ImNY",false,"image containing noise on Y-axis")
                           << EAM(aExportP3D,"TP3D",false,"Output 3D positions of tie points without distortion.")
                );


    // get directory
    SplitDirAndFile(aDir,aImgs,aPatImgs);
    StdCorrecNameOrient(aOri, aDir);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aVImgs = *(aICNM->Get(aImgs));

    // read images containing noise
    Im2D<double,double>  aImNX, aImNY;

    if(EAMIsInit(&aNameImNX))
    {
        //read ImNX

        aImNX = Im2D<double,double>::FromFileStd(aDir + aNameImNX);
        Pt2di aSzImNX = aImNX.sz();
        std::cout << "NX" << aNameImNX << " : " << aSzImNX << endl;

    }

    if(EAMIsInit(&aNameImNY))
    {
        //read ImNY

        aImNY = Im2D<double,double>::FromFileStd(aDir + aNameImNY);
        Pt2di aSzImNY = aImNY.sz();
        std::cout << "NY" << aNameImNY << " : " << aSzImNY << endl;

    }


    //generation of noise on X & Y
    if(!EAMIsInit(&aSeed))  aSeed=time(0);

    std::default_random_engine generator(aSeed);

    std::normal_distribution<double> distributionX(aNoiseGaussian[0],aNoiseGaussian[1]);
    std::normal_distribution<double> distributionY(aNoiseGaussian[2],aNoiseGaussian[3]);

    // output 3D positions of tie points
    ofstream aTP3Dfile;
    if(EAMIsInit(&aExportP3D))
    {
        std::cout << "Output 3D positions of tie points! \n";
        aTP3Dfile.open (aExportP3D);
    }


    //1. lecture of tie points and orientation
    std::cout << "Loading tie points + orientation...   ";
    cSetTiePMul * pSH = new cSetTiePMul(0);
    pSH->AddFile(aSH);
    std::map<std::string,cCelImTPM *> aVName2Im = pSH->DicoIm().mName2Im;

    // load cam for all Img
    // Iterate through all elements in std::map
    vector<CamStenope*> aVCam (aVName2Im.size());
    for(auto &aName2Im:aVName2Im)
    {
        CamStenope * aCam = aICNM->StdCamStenOfNames(aName2Im.first,aOri);
        aCam->SetNameIm(aName2Im.first);
        aVCam[aName2Im.second->Id()] = aCam;
    }

    std::cout << "Finish loading " << pSH->VPMul().size() << " CONFIG\n";

    // declare aVStructH to stock generated tie points
    vector<ElPackHomologue> aVPack (aVName2Im.size());
    vector<int> aVIdImS (aVName2Im.size(),-1);
    StructHomol aStructH;
    aStructH.VElPackHomol = aVPack;
    aStructH.VIdImSecond = aVIdImS;
    vector<StructHomol> aVStructH (aVName2Im.size(),aStructH);

    //2. get 2D/3D position of tie points
    std::cout << "Filling ElPackHomologue...   ";


    if (EAMIsInit(&aNoiseGaussian))
    {
        std::cout << "Gaussian Noise: " << aNoiseGaussian << endl;
    }    


    // parse Configs aVCnf
    std::vector<cSetPMul1ConfigTPM *> aVCnf = pSH->VPMul();
    for (auto &aCnf:aVCnf)
    {
        std::vector<int> aVIdIm =  aCnf->VIdIm();

        // Parse all pts in one Config
        for (uint aKPtCnf=0; aKPtCnf<uint(aCnf->NbPts()); aKPtCnf++)
        {
            vector<Pt2dr> aVPtInter;
            vector<CamStenope*> aVCamInter;
            vector<int> aVIdImInter;

            // Parse all imgs for one pts
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


            if(EAMIsInit(&aExportP3D))
            {
                aTP3Dfile << setprecision(17) << aPInter3D.x << " " << aPInter3D.y << " " << aPInter3D.z << endl;
            }


            // reproject aPInter3D sur tout les images dans aVCamInter
            std::vector<Pt2dr> aVP2d;
            std::vector<CamStenope *> aVCamInterVu;
            std::vector<int> aVIdImInterVu;

            for (uint itVCI=0; itVCI < aVCamInter.size(); itVCI++)
            {
                CamStenope * aCam = aVCamInter[itVCI];
                Pt2dr aPt2d = aCam->R3toF2(aPInter3D);
                //std::cout << aPt2d << "----------------";

                // add noise
                if (EAMIsInit(&aNameImNX))
                {
                    aPt2d.x += aImNX.data()[int(round(aPt2d.y))][int(round(aPt2d.x))];
                }
                if (EAMIsInit(&aNameImNY))
                {
                    aPt2d.y += aImNY.data()[int(round(aPt2d.y))][int(round(aPt2d.x))];
                }
                if (EAMIsInit(&aNoiseGaussian))
                {
                    aPt2d.x += distributionX(generator);
                    aPt2d.y += distributionY(generator);
                }
                //std::cout << aPt2d << endl;

                //check if the point is in the camera view
                if (aCam->PIsVisibleInImage(aPInter3D) && IsInImage(aCam->Sz(),aPt2d))
                {
                    aVP2d.push_back(aPt2d);
                    //std::cout << aPt2d << endl;
                    aVCamInterVu.push_back(aCam);
                    aVIdImInterVu.push_back(aVIdImInter[itVCI]);
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
                    aVStructH.at(aIdIm1).VIdImSecond.at(aIdIm2)=aIdIm2;
                }
            }


        }
    }

    std::cout << "ElPackHomologue filled !\n";

    if(EAMIsInit(&aExportP3D))
    {
        aTP3Dfile.close();
        std::cout << "Finish outputing 3D positions of tie points ! \n";
    }


    //writing of new tie points
    std::cout << "Writing Homol files...   ";
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
            for (uint itVElPH=0; itVElPH < aVStructH.at(itVSH).VElPackHomol.size(); itVElPH++)
            {
                int aIdIm2 = aVStructH.at(itVSH).VIdImSecond.at(itVElPH);
                if (aIdIm2 == -1) continue;
                CamStenope * aCam2 = aVCam.at(aIdIm2);
                std::string aNameIm2 = aCam2->NameIm();
                if (IsInList(aVImgs,aNameIm2))
                {
                    std::string aHmOut= aICNM->Assoc1To2(aKHOut, aNameIm1, aNameIm2, true);
                    ElPackHomologue aPck = aVStructH.at(aIdIm1).VElPackHomol.at(aIdIm2);
                    aPck.StdPutInFile(aHmOut);
                }
            }
        }

    }

    // write seed file
    ofstream aSeedfile;
    aSeedfile.open ("Homol_"+aSHOut+"/Seed.txt");
    std::cout << "Homol_"+aSHOut+"/Seed.txt" << endl;
    aSeedfile << aSeed << endl;
    aSeedfile.close();
    std::cout << "Finished writing Homol files ! \n";

    // convert Homol folder into new format
    std::string aComConvFH = MM3dBinFile("TestLib ConvNewFH")
                           + aImgs
                           + " All SH=_"
                           + aSHOut
                           + " ExportBoth=1";
    system_call(aComConvFH.c_str());
    std::cout << aComConvFH << endl;

    return EXIT_SUCCESS;
}

int GenerateMAF_main(int argc,char ** argv)
{
    string aPatImgs,aDir,aOri,aImgs,aGCPFile,aMAFOut,aNameImNX,aNameImNY,aOriRS;
    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aPatImgs,"Image pattern", eSAM_IsExistFile)
                           << EAMC(aOri, "Ori",  eSAM_IsExistDirOri)
                           << EAMC(aGCPFile, "File containning GCP coordinates",eSAM_IsExistFile),
                LArgMain() << EAM(aMAFOut,"Out",false,"Output name of the generated MAF file, Def=Gen_MAF_Ori.xml")
                           << EAM(aNameImNX,"ImNX",false,"image containing noise on X-axis")
                           << EAM(aNameImNY,"ImNY",false,"image containing noise on Y-axis")
                           << EAM(aOriRS,"OriRS",false,"If generate image measurement file for rolling shutter, give generated Ori name")
                );
    // get directory
    SplitDirAndFile(aDir,aImgs,aPatImgs);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    // get image pattern
    const vector<string> aVImg = *(aICNM->Get(aImgs));

    StdCorrecNameOrient(aOri, aDir);
    std::cout << "Ori: Ori-" << aOri << "/" << endl;

    if(EAMIsInit(&aOriRS))
    {
        StdCorrecNameOrient(aOriRS, aDir);
        std::cout << "Ori: Ori-" << aOriRS << "/" << endl;
    }


    std::cout << "GCP file : " << aGCPFile << endl;

    if (!EAMIsInit(&aMAFOut))
        aMAFOut =  "Gen_MAF_" + aOri + ".xml";

    std::cout << "Output File : " << aMAFOut << endl;

    // read images containing noise
    Im2D<double,double>  aImNX, aImNY;

    if(EAMIsInit(&aNameImNX))
    {
        //read ImNX

        aImNX = Im2D<double,double>::FromFileStd(aDir + aNameImNX);
        Pt2di aSzImNX = aImNX.sz();
        std::cout << "NX" << aNameImNX << " : " << aSzImNX << endl;

    }

    if(EAMIsInit(&aNameImNY))
    {
        //read ImNY

        aImNY = Im2D<double,double>::FromFileStd(aDir + aNameImNY);
        Pt2di aSzImNY = aImNY.sz();
        std::cout << "NY" << aNameImNY << " : " << aSzImNY << endl;

    }


    //read GCP coordinates
    cDicoAppuisFlottant aDicoAF = StdGetObjFromFile<cDicoAppuisFlottant>
            (
                aGCPFile,
                StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                "DicoAppuisFlottant",
                "DicoAppuisFlottant"
                );

    //output MAF file
    cSetOfMesureAppuisFlottants aDicoOut; // image measurement file
    std::list<cMesureAppuiFlottant1Im> aLMAFOut; // list of (NameIm + aLMesOut)
    std::list<cOneMesureAF1I> aLMesOut; //list of (NPt + Pt2D) for 1 image


    // Parse images
    for (uint itVImg=0; itVImg < aVImg.size(); itVImg++)
    {
        CamStenope * aCam = aICNM->StdCamStenOfNames(aVImg[itVImg],aOri);
        CamStenope * aCambis = aICNM->StdCamStenOfNames(aVImg[itVImg],aOri); //P0

        if(EAMIsInit(&aOriRS))
        {
            aCambis = aICNM->StdCamStenOfNames(aVImg[itVImg],aOriRS);
        }


        // Parse GCP list
        for
                (
                 std::list<cOneAppuisDAF>::iterator itDAF=aDicoAF.OneAppuisDAF().begin();
                 itDAF!=aDicoAF.OneAppuisDAF().end();
                 itDAF++
                 )
        {
            if (aCam->PIsVisibleInImage(itDAF->Pt()))
            {
                Pt2dr aPt2d = aCam->R3toF2(itDAF->Pt());//P0
                if(EAMIsInit(&aOriRS))
                {
                    Pt2dr aPt2d1 = aCambis->R3toF2(itDAF->Pt());//P1

                    // Pl = l*P1 + (1-l)P0
                    double aY = aCam->Sz().y * aPt2d.y /( aCam->Sz().y - aPt2d1.y + aPt2d.y );
                    double aRatio = aY / aCam->Sz().y;
                    double aX = aRatio * (aPt2d1.x-aPt2d.x) + aPt2d.x;
                    aPt2d = Pt2dr(aX,aY);
                }
                if (IsInImage(aCam->Sz(),aPt2d))
                {
                    // add noise
                    if (EAMIsInit(&aNameImNX))
                    {
                        aPt2d.x += aImNX.data()[int(round(aPt2d.y))][int(round(aPt2d.x))];
                    }
                    if (EAMIsInit(&aNameImNY))
                    {
                        aPt2d.y += aImNY.data()[int(round(aPt2d.y))][int(round(aPt2d.x))];
                    }

                    if (IsInImage(aCam->Sz(),aPt2d))
                    {
                        cOneMesureAF1I aOMAF1I;
                        aOMAF1I.NamePt() = itDAF->NamePt();
                        aOMAF1I.PtIm() = aPt2d;
                        aLMesOut.push_back(aOMAF1I);
                    }
                }
            }
        }
        cMesureAppuiFlottant1Im   aMAF1Im;
        aMAF1Im.NameIm() = aVImg.at(itVImg);
        aMAF1Im.OneMesureAF1I() = aLMesOut;
        aLMAFOut.push_back(aMAF1Im);
        aLMesOut.clear();

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
