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
#include <iostream>
#include <string>
#include "../uti_phgrm/TiepTri/TiepTri.h"
#include "../uti_phgrm/TiepTri/MultTieP.h"

struct ImgT{
    std::string ImgName;
    double ImgTemp;
};

struct CamCoord {
    CamStenope * Cam;
    Pt2dr coord2d;
};

struct AllPts {
    int NumPt;
    std::vector<CamCoord> CAC;
};

std::vector<ImgT> ReadImgTFile(string & aDir, string aTempFile, std::string aExt)
{
    std::vector<ImgT> aVSIT;
    ifstream aFichier((aDir + aTempFile).c_str());
    if(aFichier)
    {
        std::string aLine;
        
        while(!aFichier.eof())
        {
            getline(aFichier,aLine,'\n');
            if(aLine.size() != 0)
            {
                char *aBuffer = strdup((char*)aLine.c_str());
                std::string aImage = strtok(aBuffer,"	");
                std::string aTemp = strtok(NULL, "	");

                ImgT aImgT;
                if(aExt != "")
                    aImgT.ImgName = aImage + aExt;
                else
                    aImgT.ImgName = aImage;

                aImgT.ImgTemp = atof(aTemp.c_str());

                aVSIT.push_back(aImgT);
            }
        }
        aFichier.close();
    }
    else
    {
        std::cout<< "Error While opening file" << '\n';
    }
    return aVSIT;
}

int ReechHomol_main(int argc, char **argv)
{
    std::string aDir, aSHIn = "", aTempFile, aExt= ".thm.tif", aPrefix="Reech_";

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aDir, "Directory", eSAM_IsDir)
                            << EAMC(aSHIn, "Input Homol folder, "" means Homol", eSAM_IsExistFile)
                            << EAMC(aTempFile, "file containing image name & corresponding temperature", eSAM_IsExistFile),
                LArgMain()  << EAM(aExt,"Ext",true,"Extension of Imgs, Def = .thm.tif")
                            << EAM(aPrefix,"Prefix",true,"Prefix for resampled Imgs, Def = Reech_")
            );

    if ((aSHIn.substr(aSHIn.size()-1,1)).compare("/"))
        aSHIn = aSHIn.substr(0,aSHIn.size());

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    // read temperature file
    std::vector<ImgT> aVSIT = ReadImgTFile(aDir, aTempFile, aExt);
    cout << "Temperature file size : " << aVSIT.size() << endl;

    // get all Patis folders
    std::string aDirHomolIn = aDir + aSHIn;
    std::list<cElFilename> aLPastis;
    ctPath * aPathHomol = new ctPath(aDirHomolIn);
    aPathHomol->getContent(aLPastis);

    cout << "Pastis File size : " << aLPastis.size() << endl;

    // for one Patis folder
    for (std::list<cElFilename>::iterator iTLP = aLPastis.begin() ; iTLP != aLPastis.end() ; iTLP++)
    {
        // master image
        string aIm1 = iTLP->m_basename.substr (6,iTLP->m_basename.length()-6);
        cout << "Image 1 : " << aIm1 << endl;
        std::string aNameMap1 = "PolOfTXY-" + ToString(aVSIT.at(0).ImgTemp) + ".xml";
        cElMap2D * aMapIm1 = cElMap2D::FromFile(aNameMap1);

        // match the temperature for master image
        for (uint aV=0; aV < aVSIT.size(); aV++)
        {
            if (aVSIT.at(aV).ImgName.compare(aIm1) == 0)
            {
                aNameMap1 = "PolOfTXY-" + ToString(aVSIT.at(aV).ImgTemp) + ".xml";
                aMapIm1->FromFile(aNameMap1);
                // * aMapIm1->FromFile(aNameMap1);  Warn unused
                //cout << "Im : " << aVSIT.at(aV).ImgName << " Temp : " << aVSIT.at(aV).ImgTemp << endl;
            }
        }
        // get all .dat files of the master image
        std::string aDirPastis = aDirHomolIn + "/" + iTLP->m_basename;
        cout << "Dir-Pastis" << aDirPastis << endl;
        cInterfChantierNameManipulateur * aICNMP = cInterfChantierNameManipulateur::BasicAlloc(aDirPastis);
        vector<string> aLFileP = *(aICNMP->Get(".*"));
        cout << ".dat File size : " << aLFileP.size() << endl;

        if (aLFileP.size()!=0)
        {
            // matche the temperature for secondary image and correct with maps
            for (uint aL=0; aL < aLFileP.size(); aL++)
            {
                // secondary image
                string aIm2 = aLFileP.at(aL).substr(1,aLFileP.at(aL).length()-5);
                cout << "Image 2 : " << aIm2 << endl;
                std::string aNameMap2 = "PolOfTXY-" + ToString(aVSIT.at(0).ImgTemp) + ".xml";
                cElMap2D * aMapIm2 = cElMap2D::FromFile(aNameMap2);

                for (uint aT=0; aT < aVSIT.size(); aT++)
                {
                    // match the temperature for secondary image
                    if (aVSIT.at(aT).ImgName.compare(aIm2) == 0)
                    {
                        aNameMap2 = "PolOfTXY-" + ToString(aVSIT.at(aT).ImgTemp) + ".xml";
                        aMapIm2->FromFile(aNameMap2);  
                        // * aMapIm2->FromFile(aNameMap2);  Warn unused
                        //cout << "Im : " << aVSIT.at(aT).ImgName << " Temp : " << aVSIT.at(aT).ImgTemp << endl;
                    }
                }

                // read Pts Homologues data
                ElPackHomologue aPckIn, aPckOut;
                std::string aPostfix = aSHIn.substr(5,aSHIn.size()-5);
                std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(aPostfix)
                        +  std::string("@")
                        +  std::string("dat");
                std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string("_Reech")
                        +  std::string("@")
                        +  std::string("dat");
                std::string aHmIn= aICNM->Assoc1To2(aKHIn, aIm1, aIm2, true);

                aPckIn =  ElPackHomologue::FromFile(aHmIn);
                cout << "Pts Homologues size : " << aPckIn.size() << endl;

                // apply maps to Pts Homologues data
                for(ElPackHomologue::iterator iTH = aPckIn.begin(); iTH != aPckIn.end(); iTH++)
                {
                    Pt2dr aP1 = (iTH->P1())*2-(*aMapIm1)(iTH->P1());
                    Pt2dr aP2 = (iTH->P2())*2-(*aMapIm2)(iTH->P2());
                    ElCplePtsHomologues aPH (aP1,aP2);
                    aPckOut.Cple_Add(aPH);
                }

                std::string aIm1Out = aPrefix + aIm1;
                std::string aIm2Out = aPrefix + aIm2;
                std::string aHmOut= aICNM->Assoc1To2(aKHOut, aIm1Out, aIm2Out, true);
                aPckOut.StdPutInFile(aHmOut);
            }
        }

    }

    return EXIT_SUCCESS;
}

int ExtraitHomol_main(int argc, char ** argv)
{
    std::string aDir, aPatImgs, aFullPat, aSHIn="Homol", aSHOut="Extrait";
    std::vector<int> aVImgNum;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullPat, "Full Imgs Pattern", eSAM_IsExistFile)
                    << EAMC(aVImgNum, "Num of Img, e.g., [0,1,9,10]"),
        LArgMain()  << EAM(aSHIn,"SHIn",true,"Input Homol folder, Def = Homol")
                    << EAM(aSHOut,"SHOut",true,"Output Homol folder, Def = Homol_Extrait")
    );

    SplitDirAndFile(aDir,aPatImgs,aFullPat);

    // read corresponding imgs files
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aVImg = *(aICNM->Get(aPatImgs));

    std::string aVSImg = "\"";
    for (uint iVS = 0; iVS < aVImgNum.size(); iVS++)
    {
        int aNum = aVImgNum.at(iVS);
        if (iVS == 0)
            aVSImg += aVImg.at(aNum);
        else
        {
            aVSImg += "|";
            aVSImg += aVImg.at(aNum);
        }
    }

    aVSImg += "\"";
    cout << aVSImg << endl;

    // convert Homol folder into new format
    std::string aComConvFH = MM3dBinFile("TestLib ConvNewFH ")
                           + aVSImg
                           + " "
                           + aSHOut
                           + " Bin=0";

    cout << aComConvFH << endl;
    system_call(aComConvFH.c_str());

    // read PMul.txt and output new Homol files
    const std::string  aSHExtrStr = aSHIn +"/PMul" + aSHOut + ".txt";
    cSetTiePMul * aSHExtr = new cSetTiePMul(0);
    aSHExtr->AddFile(aSHExtrStr);
    std::vector<cSetPMul1ConfigTPM *> aVCnf = aSHExtr->VPMul();

    std::string aDirHomolExtr = "_" + aSHOut;
    cout << "aDirhomolExtr" << aDirHomolExtr << endl;
    std::string aKHExtr =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(aDirHomolExtr)
            +  std::string("@")
            +  std::string("dat");
    cout << "aKHExtr" << aKHExtr << endl;

    for (uint aKCnf=1; aKCnf < aVCnf.size(); aKCnf++)
    {
        cSetPMul1ConfigTPM * aCnf = aVCnf[aKCnf];
        if (uint (aCnf->NbIm()) == aVImgNum.size())
        {
            cout << "Cnf : " << aKCnf << " - Nb Imgs : " << aCnf->NbIm() << " - Nb Pts : " << aCnf->NbPts() << endl;
            std::vector<int> aVIdIm =  aCnf->VIdIm();

            for (uint i1=0; i1 < aVIdIm.size(); i1++)
            {
                for (uint i2=0; i2 < aVIdIm.size(); i2++)
                {
                    if (i1 != i2)
                    {
                        std::string aIm1 = aVImg.at(aVImgNum.at(i1));
                        std::string aIm2 = aVImg.at(aVImgNum.at(i2));
                        ElPackHomologue * aPckExtr =new ElPackHomologue();

                        for (uint aKPtCnf=0; aKPtCnf < uint(aCnf->NbPts()); aKPtCnf++)
                        {
                            Pt2dr aPt1 = aCnf->Pt(aKPtCnf,i1);
                            Pt2dr aPt2 = aCnf->Pt(aKPtCnf,i2);
                            ElCplePtsHomologues aPtFH (aPt1,aPt2);
                            aPckExtr->Cple_Add(aPtFH);
                        }

                        std::string aHmExtr= aICNM->Assoc1To2(aKHExtr, aIm1, aIm2, true);
                        aPckExtr->StdPutInFile(aHmExtr);
                    }
                }
            }
        }
    }
    return EXIT_SUCCESS;
}

Pt3dr IntersectionFaisceaux
       (
            const std::vector<CamStenope *> & aVCS,
            const std::vector<Pt2dr> & aNPts2D
        )
{
    //vecteur d'éléments segments 3d
    std::vector<ElSeg3D> aVSeg;

    for (int aKR=0 ; aKR < int(aVCS.size()) ; aKR++)
    {
        ElSeg3D aSeg = aVCS.at(aKR)->F2toRayonR3(aNPts2D.at(aKR));
        aVSeg.push_back(aSeg);
    }
    //std::cout<<"Intersect "<<aVSeg.size()<<" bundles...\n";
    Pt3dr aRes =  ElSeg3D::L2InterFaisceaux(0,aVSeg,0);
    return aRes;
}

int IntersectHomol_main(int argc, char ** argv)
{
    std::string aDir, aPatImgs, aFullPat, aOriIn, aSHIn="Homol", aOut="Extrait";
    std::vector<int> aDebut, aFin;
    bool aXmlExport=true;
    Pt3dr aInc(1,1,1);

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullPat, "Full Imgs Pattern", eSAM_IsExistFile)
                    << EAMC(aDebut, "Num of Img for group 1, e.g., [0,1]")
                    << EAMC(aFin, "Num of Img for group 2, e.g., [26,27]")
                    << EAMC(aOriIn, "Directory of input orientation",  eSAM_IsExistDirOri),
        LArgMain()  << EAM(aSHIn,"SHIn",true,"Input Homol folder, Def = Homol")
                    << EAM(aOut,"SHOut",true,"Output result, Def = Extrait")
                    << EAM(aXmlExport,"XmlOut",true,"Export in .xml format to use as GCP file (Def=true)")
    );

    SplitDirAndFile(aDir,aPatImgs,aFullPat);

    // read corresponding imgs files
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aVImg = *(aICNM->Get(aPatImgs));

    MakeFileDirCompl(aOriIn);
    ELISE_ASSERT(ELISE_fp::IsDirectory(aOriIn),"ERROR: Input orientation not found!");

    std::vector<int> aVImgNum = aDebut;
    aVImgNum.insert( aVImgNum.end(), aFin.begin(), aFin.end() );

    std::string aVSImg = "\"";
    for (uint iVS = 0; iVS < aVImgNum.size(); iVS++)
    {
        int aNum = aVImgNum.at(iVS);
        if (iVS == 0)
            aVSImg += aVImg.at(aNum);
        else
        {
            aVSImg += "|";
            aVSImg += aVImg.at(aNum);
        }
    }

    aVSImg += "\"";
    cout << aVSImg << endl;

    // convert Homol folder into new format
    std::string aComConvFH = MM3dBinFile("TestLib ConvNewFH ")
                           + aVSImg
                           + " "
                           + aOut
                           + " Bin=0";

    cout << aComConvFH << endl;
    system_call(aComConvFH.c_str());

    // read PMul.txt and output new Homol files
    const std::string  aSHExtrStr = aSHIn +"/PMul" + aOut + ".txt";
    cSetTiePMul * aSHExtr = new cSetTiePMul(0);
    aSHExtr->AddFile(aSHExtrStr);
    std::vector<cSetPMul1ConfigTPM *> aVCnf = aSHExtr->VPMul();

    // generate list of CamCoord
    std::vector<CamCoord> v1CameraEtCoord;
    std::vector<AllPts> aVAllPts;

    for (uint aKCnf=1; aKCnf < aVCnf.size(); aKCnf++)
    {
        cSetPMul1ConfigTPM * aCnf = aVCnf[aKCnf];
        if (uint (aCnf->NbIm()) == aVImgNum.size())
        {
            cout << "Cnf : " << aKCnf << " - Nb Imgs : " << aCnf->NbIm() << " - Nb Pts : " << aCnf->NbPts() << endl;
            std::vector<int> aVIdIm =  aCnf->VIdIm();

            for (uint aKPtCnf=0; aKPtCnf < uint(aCnf->NbPts());aKPtCnf++)
            {
                AllPts aAllPts;
                aAllPts.NumPt = aKPtCnf;
                std::vector<CamCoord> aVCamCoord;
                for (uint aKIm=0; aKIm < aVIdIm.size(); aKIm++)
                {
                    std::string oriNameFile = aOriIn+"Orientation-"+aVImg.at(aVImgNum.at(aKIm))+".xml";
                    if (!ELISE_fp::exist_file(oriNameFile)) continue;
                    CamStenope * cameraCourante = CamOrientGenFromFile(oriNameFile,aICNM);
                    Pt2dr coordCourant = aCnf->Pt(aKPtCnf,aKIm);                    
                    CamCoord aCamCoord;
                    aCamCoord.Cam = cameraCourante;
                    aCamCoord.coord2d=coordCourant;
                    aVCamCoord.push_back(aCamCoord);
                    cout << " bundle ++ " << endl;
                }
                aAllPts.CAC = aVCamCoord;
                aVAllPts.push_back(aAllPts);
            }
        }
    }

    cout << "Camera Pts size : " << aVAllPts.size() << endl;

    //Export results for the first and second group
    for (uint count = 0; count < 2; count ++)
    {
        std::string aOutput;
        uint aStart, aEnd;
        if (count == 0)
        {
            aOutput="Debut";
            aStart=0;
            aEnd=aDebut.size();
            cout << "Intersection for group 1" << endl;
        }
        else
        {
            aOutput="Fin";
            aStart=aDebut.size();
            aEnd=aVImgNum.size();
            cout << "Intersection for group 2" << endl;
        }

        //le vecteur des points 3d a exporter
        std::vector<Pt3dr> Pts3d;

        //boucle sur le nombre de points a projeter en 3d
        for(uint aHG=0 ; aHG < aVAllPts.size() ; aHG++)
        {
            AllPts aaAllPts = aVAllPts.at(aHG);

            //vecteur de cameras
            std::vector<CamStenope *> vCSPt;

            //vecteur de coordonnees 2d
            std::vector<Pt2dr> vC2d;


            for(uint aHF=aStart ; aHF < aEnd ; aHF++)
            {
                CamCoord aaCAC = aaAllPts.CAC.at(aHF);
                CamStenope * aCSPtC = aaCAC.Cam;
                vCSPt.push_back(aCSPtC);
                Pt2dr aCoordPtC = aaCAC.coord2d;
                vC2d.push_back(aCoordPtC);
            }
            Pt3dr aPt3d = IntersectionFaisceaux(vCSPt,vC2d);
            Pts3d.push_back(aPt3d);
        }

        cout << Pts3d << endl;

        //export en .txt
        if (!MMVisualMode)
        {

            FILE * aFP = FopenNN(aOutput+".txt","w","IntersectHomol_main");
            cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOutput);
            for(uint aVP=0; aVP<Pts3d.size(); aVP++)
            {
                fprintf(aFP,"%d %lf %lf %lf \n",aVP,Pts3d[aVP].x,Pts3d[aVP].y,Pts3d[aVP].z);
            }

            ElFclose(aFP);
            std::cout<< aOutput <<".txt written."<<std::endl;
        }


        //export en .ply
        if (!MMVisualMode)
        {

            FILE * aFP = FopenNN(aOutput+".ply","w","IntersectHomol_main");
            //cElemAppliSetFile aEASF(mDir + ELISE_CAR_DIR + aOut);
            fprintf(aFP,"ply\n");
            fprintf(aFP,"format ascii 1.0\n");
            fprintf(aFP,"element vertex %lu\n",Pts3d.size());
            fprintf(aFP,"property float x\n");
            fprintf(aFP,"property float y\n");
            fprintf(aFP,"property float z\n");
            fprintf(aFP,"property uchar red\n");
            fprintf(aFP,"property uchar green\n");
            fprintf(aFP,"property uchar blue\n");
            fprintf(aFP,"element face 0\n");
            fprintf(aFP,"property list uchar int vertex_indices\n");
            fprintf(aFP,"end_header\n");

            for(unsigned int aVP=0; aVP<Pts3d.size(); aVP++)
            {
                fprintf(aFP,"%lf %lf %lf 255 0 0\n",Pts3d[aVP].x,Pts3d[aVP].y,Pts3d[aVP].z);
            }

            ElFclose(aFP);
            std::cout<<aOutput<<".ply written."<<std::endl;
        }

        //export en .xml pour utiliser comme fichier de GCPs
        if(aXmlExport)
        {
            std::string aOutXml = StdPrefixGen(aOutput) + ".xml";

            cDicoAppuisFlottant aDico;

            for (int aKP=0 ; aKP<int(Pts3d.size()) ; aKP++)
            {
                cOneAppuisDAF aOAD;
                aOAD.Pt() = Pts3d[aKP];
                aOAD.NamePt() = ToString (aKP);
                aOAD.Incertitude() = aInc;

                aDico.OneAppuisDAF().push_back(aOAD);
            }

            MakeFileXML(aDico,aOutXml);
            std::cout<<aOutXml<<" written."<<std::endl;
        }
    }
    return EXIT_SUCCESS;
}

int ReechMAF_main (int argc, char ** argv)
{
    std::string aDir, aMAF, aNameMAF, aTempFile, aOut="ReechMAF.xml", aExt;


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aMAF, "Image measurement file", eSAM_IsExistFile)
                    << EAMC(aTempFile, "File of image temperature",eSAM_IsExistFile),
        LArgMain()  << EAM(aOut,"Out",true,"Output resampled MAF, Def = ReechMAF.xml")
                    << EAM(aExt,"Ext",true,"Extension of Imgs, Def = .thm.tif")
    );

    SplitDirAndFile(aDir,aNameMAF,aMAF);

    // read temperature file
    std::vector<ImgT> aVSIT = ReadImgTFile(aDir, aTempFile, aExt);
    cout << "Temperature file size : " << aVSIT.size() << endl;

    //input
    cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aMAF,SetOfMesureAppuisFlottants);
    std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();

    //output
    cSetOfMesureAppuisFlottants aDicoOut;
    std::list<cMesureAppuiFlottant1Im> aLMAFOut;
    std::list<cOneMesureAF1I> aMesOut;
    std::string aNameMapInit = "PolOfTXY-" + ToString(aVSIT.at(0).ImgTemp) + ".xml";
    cElMap2D * aMap = cElMap2D::FromFile(aNameMapInit);


    for (std::list<cMesureAppuiFlottant1Im>::iterator iT1 = aLMAF.begin() ; iT1 != aLMAF.end() ; iT1++)
    {

        for (uint iV=0; iV < aVSIT.size(); iV++)
        {
            if(aVSIT.at(iV).ImgName.compare(iT1->NameIm()) == 0)
            {
                std::string aNameMap = "PolOfTXY-" + ToString(aVSIT.at(iV).ImgTemp) + ".xml";
                aMap->FromFile(aNameMap);
                // * aMap->FromFile(aNameMap); Warn unused
            }
        }

        std::list<cOneMesureAF1I> & aMes = iT1->OneMesureAF1I();

        for (std::list<cOneMesureAF1I>::iterator iT2 = aMes.begin() ; iT2 != aMes.end() ; iT2++)
        {
            cOneMesureAF1I aOMAF1I;
            aOMAF1I.NamePt() = iT2->NamePt();
            aOMAF1I.PtIm() = Pt2dr(iT2->PtIm())*2- (*aMap)(Pt2dr(iT2->PtIm()));
            aMesOut.push_back(aOMAF1I);
        }
            cMesureAppuiFlottant1Im   aMAF1Im;
            aMAF1Im.NameIm() = "Reech_" + iT1->NameIm();
            aMAF1Im.OneMesureAF1I() = aMesOut;
            aLMAFOut.push_back(aMAF1Im);
            aMesOut.clear();
    }
    aDicoOut.MesureAppuiFlottant1Im() = aLMAFOut;
    MakeFileXML(aDicoOut,aOut);

    return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
