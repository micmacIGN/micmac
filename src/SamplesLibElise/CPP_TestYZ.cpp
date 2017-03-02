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

class cCalMECEsSim
{
public:
    cCalMECEsSim(string & mDir, string & mName, string & mNameX, string & mNameY);
    void CreateCom(string & aXml, int & aPas);
    void EsSim(string & aOutEsSim);
    void CalMEC();
    void StatIm(string & aOutStatIm);
    string & Dir () {return mDir;}
    string & Name () {return mName;}
    string & NameX () {return mNameX;}
    string & NameY () {return mNameY;}
    vector<string> & VDirMEC () {return mVDirMEC;}
    vector<string> & VComCalMEC () {return mVComCalMEC;}
private:
    string mDir;
    string mName;
    string mNameX;
    string mNameY;
    vector<string> mVDirMEC;
    vector<string> mVComCalMEC;
};

cCalMECEsSim::cCalMECEsSim(string & aDir, string & aName, string & aNameX, string & aNameY):
    mDir (aDir),
    mName (aName),
    mNameX (aNameX),
    mNameY (aNameY),
    mVDirMEC ()
{}

void cCalMECEsSim::CreateCom(string & aXml, int & aPas)
{
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    vector<string> aLFile = *(aICNM->Get(mName));

    cout << "File size = " << aLFile.size() << endl;

    for (uint aK=uint(aPas); aK<=aLFile.size()-uint(aPas); aK+=uint(aPas))
    {
        string aDirMEC = "MEC-" + aLFile.at(0) + "-" + aLFile.at(aK);
        MakeFileDirCompl(aDirMEC);
        mVDirMEC.push_back(aDirMEC);

        string aComCalMEC = MM3dBinFile("MICMAC")
                            + aXml
                            + " +WorkDir=" + mDir
                            + " +Im1=" + aLFile.at(0)
                            + " +Im2=" + aLFile.at(aK);
        mVComCalMEC.push_back(aComCalMEC);
    }
}

void cCalMECEsSim::CalMEC()
{
    for (uint aP=0; aP<mVComCalMEC.size(); aP++)
    {
        system_call(mVComCalMEC.at(aP).c_str());
    }
}

void cCalMECEsSim::EsSim(string & aOutEsSim)
{
    //Estimate the similarity and put the results in aOutEsSim.txt
    string aOut1 = aOutEsSim+".txt";
    FILE * aFP_EsSim = FopenNN(aOut1,"w","EsSim");
    cElemAppliSetFile aEASF(mDir+ELISE_CAR_DIR+aOut1);
    string Format = "#F=R1 R2 Tx Ty";
    fprintf(aFP_EsSim,"%s\n",Format.c_str());

    for (uint aF=0; aF < mVDirMEC.size(); aF++)
    {
        string aDirEsSim = mDir + mVDirMEC.at(aF);
        Tiff_Im aTifX (Tiff_Im::UnivConvStd(aDirEsSim  + mNameX));
        Tiff_Im aTifY (Tiff_Im::UnivConvStd(aDirEsSim  + mNameY));
        Pt2di aSz (aTifX.sz());

        Im2D<double,double> aImgX  (1,1); //initiation of vignette X
        Im2D<double,double> aImgY  (1,1); //initiation of vignette Y


        aImgX.Resize(aSz);
        aImgY.Resize(aSz);
        ELISE_COPY(aImgX.all_pts(),aTifX.in(),aImgX.out());
        ELISE_COPY(aImgY.all_pts(),aTifY.in(),aImgY.out());


        //Estimate the similarity
        L2SysSurResol aSys(4);
        double * aData1 = NULL;

        for(int aKx=0; aKx < aSz.x; aKx++)
        {
            for(int aKy=0;aKy < aSz.y;aKy++)
            {
                double coeffX[4] = {double(aKx), double(-aKy), 1.0, 0.0};
                double coeffY[4] = {double(aKy ), double(aKx), 0.0, 1.0};
                double delX = aImgX.GetR(Pt2di(aKx, aKy));
                double delY = aImgY.GetR(Pt2di(aKx, aKy));
                aSys.AddEquation(1.0, coeffX, delX);
                aSys.AddEquation(1.0, coeffY, delY);
            }
        }
        bool solveOK = true;
        Im1D_REAL8 aResol1 = aSys.GSSR_Solve(&solveOK);
        aData1 = aResol1.data();
        if (solveOK != false)
            fprintf(aFP_EsSim,"%f %f %f %f\n", aData1[0], aData1[1], aData1[2], aData1[3]);
        else
            cout<<"Can't estimate."<<endl;
    }
    ElFclose(aFP_EsSim);
}

void cCalMECEsSim::StatIm(string & aOutStatIm)
{
    Pt2di aP0 (0,0);
    string Format = "#F=ZMoy Sigma ZMin ZMax";

    string aOut2 = aOutStatIm + "X.txt";
    FILE * aFP_StatImX = FopenNN(aOut2,"w","EsSim");
    cElemAppliSetFile aEASFX(mDir+ELISE_CAR_DIR+aOut2);
    fprintf(aFP_StatImX,"%s\n",Format.c_str());

    for (uint aF=0; aF < mVDirMEC.size(); aF++)
    {
        string aDirEsSim = mDir + mVDirMEC.at(aF);
        Tiff_Im aTifX (Tiff_Im::UnivConvStd(aDirEsSim + mNameX));
        Pt2di aSzX (aTifX.sz());

        Symb_FNum aTFX (Rconv(aTifX.in()));

        double aSPX,aSomZX,aSomZX2,aZXMin,aZXMax;

        ELISE_COPY
        (
            rectangle(aP0,aP0+aSzX),
            Virgule(1,aTFX,Square(aTFX)),
            Virgule
            (
                 sigma(aSPX),
                 sigma(aSomZX)|VMax(aZXMax)|VMin(aZXMin),
                 sigma(aSomZX2)
            )
        );

        aSomZX /= aSPX;
        aSomZX2 /= aSPX;
        aSomZX2 -= ElSquare(aSomZX);

        fprintf(aFP_StatImX,"%f %f %f %f\n", aSomZX, sqrt(ElMax(0.0,aSomZX2)), aZXMin, aZXMax);
    }
    ElFclose(aFP_StatImX);

    string aOut3 = aOutStatIm + "Y.txt";
    FILE * aFP_StatImY = FopenNN(aOut3,"w","EsSim");
    cElemAppliSetFile aEASFY(mDir+ELISE_CAR_DIR+aOut3);
    fprintf(aFP_StatImY,"%s\n",Format.c_str());

    for (uint aF=0; aF < mVDirMEC.size(); aF++)
    {
        string aDirEsSim = mDir + mVDirMEC.at(aF);
        Tiff_Im aTifY (Tiff_Im::UnivConvStd(aDirEsSim + mNameY));
        Pt2di aSzY (aTifY.sz());

        Symb_FNum aTFY (Rconv(aTifY.in()));

        double aSPY,aSomZY,aSomZY2,aZYMin,aZYMax;

        ELISE_COPY
        (
            rectangle(aP0,aP0+aSzY),
            Virgule(1,aTFY,Square(aTFY)),
            Virgule
            (
                 sigma(aSPY),
                 sigma(aSomZY)|VMax(aZYMax)|VMin(aZYMin),
                 sigma(aSomZY2)
            )
        );

        aSomZY /= aSPY;
        aSomZY2 /= aSPY;
        aSomZY2 -= ElSquare(aSomZY);

        fprintf(aFP_StatImY,"%f %f %f %f\n", aSomZY, sqrt(ElMax(0.0,aSomZY2)), aZYMin, aZYMax);
    }
    ElFclose(aFP_StatImY);
}


int TestYZ_main(int argc, char ** argv)
{
    string aDir, aName, aXml, aOutEsSim="All_H2D", aNameX = "Px1_Num12_DeZoom1_LeChantier.tif", aNameY = "Px2_Num12_DeZoom1_LeChantier.tif", aOutStatIm="All_StatIm";
    int aPas(1), aFunc(0);

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Directory")
                    << EAMC(aName, "ImgPattern", eSAM_IsExistFile)
                    << EAMC(aXml, ".xml file", eSAM_IsExistFile),
        LArgMain()  << EAM(aFunc,"Function",true,"choice of functions to execute. Def=0[CalMEC &  EsSim], 1[CalMEC], 2[EsSim]")
                    << EAM(aPas,"Pas",true,"interval of image correlation; Def=1")
                    << EAM(aNameX,"NameX",true,"name of deplacement img of axis-x; Def=Px1_Num12_DeZoom1_LeChantier.tif")
                    << EAM(aNameY,"NameY",true,"name of deplacement img of axis-x; Def=Px1_Num12_DeZoom1_LeChantier.tif")
                    << EAM(aOutEsSim,"OutEsSim",true,"Output file name for A,B,C,D Helmert2D Params; Def=All_H2D")
                    << EAM(aOutStatIm,"OutStatIm",true,"Output file name for StatIm; Def=All_StatIm")
    );

    cCalMECEsSim  aCalMECEsSim(aDir, aName, aNameX, aNameY);
    aCalMECEsSim.CreateCom(aXml, aPas);

    if (aFunc == 0 || aFunc ==1)
        aCalMECEsSim.CalMEC();

    if (aFunc == 0 || aFunc ==2)
        aCalMECEsSim.EsSim(aOutEsSim);

    if (aOutStatIm != "")
        aCalMECEsSim.StatIm(aOutStatIm);

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
Footer-MicMac-eLiSe-25/06/2007*/
