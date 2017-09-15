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

class cCalMECEsSim
{
public:
    cCalMECEsSim(string & mDir, string & mName, string & mNameX, string & mNameY, int & mScale, int & mPas);
    void CreateCom();
    void CalMEC(string & aXml);
    void EsSim(string & aOutEsSim);
    void StatIm(string & aOutStatIm);
    void ScaleIm (bool & aIfScale);
    void Purge ();
    string & Dir () {return mDir;}
    string & Name () {return mName;}
    string & NameX () {return mNameX;}
    string & NameY () {return mNameY;}
    string & NameXScale () {return mNameXScale;}
    string & NameYScale () {return mNameYScale;}
    int & Scale () {return mScale;}
    int & Pas () {return mPas;}
    vector<string> & LFile () {return mLFile;}
    vector<string> & VDirMEC () {return mVDirMEC;}
    //vector<string> & VComCalMEC () {return mVComCalMEC;}
private:
    string mDir;
    string mName;
    string mNameX;
    string mNameY;
    string mNameXScale;
    string mNameYScale;
    int mScale;
    int mPas;
    vector<string> mLFile;
    vector<string> mVDirMEC;
    //vector<string> mVComCalMEC;
};

cCalMECEsSim::cCalMECEsSim(string & aDir, string & aName, string & aNameX, string & aNameY, int & aScale, int & aPas):
    mDir (aDir),
    mName (aName),
    mNameX (aNameX),
    mNameY (aNameY),
    mScale (aScale),
    mPas (aPas)
{}

void cCalMECEsSim::CreateCom()
{
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mLFile = *(aICNM->Get(mName));

    cout << "File size = " << LFile().size() << endl;

    for (uint aK=uint(mPas); aK<=mLFile.size()-uint(mPas); aK+=uint(mPas))
    {
        string aDirMEC = "MEC-" + mLFile.at(0) + "-" + mLFile.at(aK) + "/";
        mVDirMEC.push_back(aDirMEC);
        MakeFileDirCompl(aDirMEC);
    }
    cout << "Commands created" << endl;
}

void cCalMECEsSim::CalMEC(string & aXml)
{
      for (uint aK=uint(mPas); aK<=mLFile.size()-uint(mPas); aK+=uint(mPas))
      {
          cout << "CalMEC : Im1=" << mLFile.at(0) << " Im2=" << mLFile.at(aK) << endl;
            string aComCalMEC = MM3dBinFile("MICMAC")
                                + aXml
                                + " +WorkDir=" + mDir
                                + " +Im1=" + mLFile.at(0)
                                + " +Im2=" + mLFile.at(aK);
            system_call(aComCalMEC.c_str());
      }
}

void cCalMECEsSim::ScaleIm (bool & aIfScale)
{
    if(aIfScale)
    {
        cout << "Scale = " << mScale << endl;
        mNameXScale ="Px1_Scale_" + ToString(mScale) + ".tif";
        mNameYScale ="Px2_Scale_" + ToString(mScale) + ".tif";

        for (uint aS=0; aS < mVDirMEC.size(); aS++)
        {
            bool exist = false;
            cInterfChantierNameManipulateur * aICNMS = cInterfChantierNameManipulateur::BasicAlloc(mDir+mVDirMEC.at(aS));
            vector<string> aLFileS = *(aICNMS->Get(".*"));
            for (uint aL=0; aL < aLFileS.size(); aL++)
            {
                exist = (aLFileS.at(aL) == mNameXScale);
                if (exist)
                    break;
                cout << "scaled images existed!" << endl;
            }
            if (!exist)
            {
                cout << "Scale : " << mVDirMEC.at(aS) << endl;
                string aDirScaleX = mDir + mVDirMEC.at(aS)+mNameX;
                string aDirScaleY = mDir + mVDirMEC.at(aS)+mNameY;
                string aOutScaleX = mDir + mVDirMEC.at(aS)+mNameXScale;
                string aOutScaleY = mDir + mVDirMEC.at(aS)+mNameYScale;

                string aComScaleX = MM3dBinFile("ScaleIm")
                                    +aDirScaleX
                                    +" " + ToString(mScale)
                                    +" Out=" + aOutScaleX;

                string aComScaleY = MM3dBinFile("ScaleIm")
                                    +aDirScaleY
                                    +" " + ToString(mScale)
                                    +" Out=" + aOutScaleY;

                system_call(aComScaleX.c_str());
                system_call(aComScaleY.c_str());
            }
        }
    }

    else
    {
        mNameXScale = mNameX;
        mNameYScale = mNameY;
    }
}

void cCalMECEsSim::EsSim(string & aOutEsSim)
{
    //Estimate the similarity and put the results in aOutEsSim.txt
    string aOut1 = aOutEsSim+".txt";
    FILE * aFP_EsSim = FopenNN(aOut1,"w","EsSim");
    cElemAppliSetFile aEASF(mDir+ELISE_CAR_DIR+aOut1);
    string Format1 = "#F=R1 R2 Tx Ty";
    fprintf(aFP_EsSim,"%s\n",Format1.c_str());

    for (uint aF=0; aF < mVDirMEC.size(); aF++)
    {
        string aDirEsSim = mDir + mVDirMEC.at(aF);
        cout << "EsSim : " << mVDirMEC.at(aF) << endl;
        Tiff_Im aTifX (Tiff_Im::UnivConvStd(aDirEsSim  + mNameXScale));
        Tiff_Im aTifY (Tiff_Im::UnivConvStd(aDirEsSim  + mNameYScale));

        Pt2di aSz (aTifX.sz());

        Im2D<double,double> aImgX  (1,1); //initiation of vignette X
        Im2D<double,double> aImgY  (1,1); //initiation of vignette Y


        aImgX.Resize(aSz);
        aImgY.Resize(aSz);
        ELISE_COPY(aImgX.all_pts(),aTifX.in(),aImgX.out());
        ELISE_COPY(aImgY.all_pts(),aTifY.in(),aImgY.out());

        //Estimate the similarity
        L2SysSurResol aSys1(4);
        double * aData1 = NULL;

        for(int aKx=0; aKx < aSz.x; aKx++)
        {
            for(int aKy=0;aKy < aSz.y;aKy++)
            {
                double coeffXUL[4] = {(double(aKx)+0.5)*double(mScale)-1, -(double(aKy)+0.5)*double(mScale)+1, 1.0, 0.0};
                double coeffYUL[4] = {(double(aKy)+0.5)*double(mScale)-1, (double(aKx)+0.5)*double(mScale)-1, 0.0, 1.0};
                double coeffXDR[4] = {(double(aKx)+0.5)*double(mScale), -(double(aKy)+0.5)*double(mScale), 1.0, 0.0};
                double coeffYDR[4] = {(double(aKy)+0.5)*double(mScale), (double(aKx)+0.5)*double(mScale), 0.0, 1.0};
                double delX = aImgX.GetR(Pt2di(aKx, aKy));
                double delY = aImgY.GetR(Pt2di(aKx, aKy));
                aSys1.AddEquation(1.0, coeffXUL, delX);
                aSys1.AddEquation(1.0, coeffYUL, delY);
                aSys1.AddEquation(1.0, coeffXDR, delX);
                aSys1.AddEquation(1.0, coeffYDR, delY);
            }
        }
        bool solveOK1 = true;
        Im1D_REAL8 aResol1 = aSys1.GSSR_Solve(&solveOK1);
        aData1 = aResol1.data();

        if (solveOK1 != false)
            fprintf(aFP_EsSim,"%f %f %f %f\n", aData1[0], aData1[1], aData1[2], aData1[3]);
        else
            cout<<"Can't estimate."<<endl;

        //Calculate residual
        string str1 = mVDirMEC.at(aF).substr(4,17);
        string str2 = mVDirMEC.at(aF).substr(30,17);
        string aOut2 = "Res_"+str1+"_"+str2+".txt";
        FILE * aFP_Res = FopenNN(aOut2,"w","Res");
        cElemAppliSetFile aEASF(mDir+ELISE_CAR_DIR+aOut2);
//        string Format2 = "#F=X,Y,DeplX,DeplY,EsX,EsY,ResX,ResY";
//        fprintf(aFP_Res,"%s\n",Format2.c_str());
        for(int aKx=0; aKx < aSz.x; aKx++)
        {
            for(int aKy=0;aKy < aSz.y;aKy++)
            {
                double DeplX = aImgX.GetR(Pt2di(aKx, aKy));
                double DeplY = aImgY.GetR(Pt2di(aKx, aKy));
                double EsX = aData1[0]*aKx-aData1[1]*aKy+aData1[2];
                double EsY = aData1[1]*aKx+aData1[0]*aKy+aData1[3];
                double ResX = DeplX-EsX;
                double ResY = DeplY-EsY;
                //fprintf(aFP_Res,"%d %d %f %f %f %f %f %f\n", aKx, aKy, DeplX, DeplY, EsX, EsY, ResX, ResY);
                fprintf(aFP_Res,"%d %d %f %f\n", aKx, aKy, ResX, ResY);
            }
        }
        ElFclose(aFP_Res);
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

void cCalMECEsSim::Purge()
{

    for (uint aP=0; aP < mVDirMEC.size(); aP++)
    {
        cout << "Purge : " << mVDirMEC.at(aP) << endl;
        string aDirPurge = mDir+mVDirMEC.at(aP);
        cInterfChantierNameManipulateur * aICNMP = cInterfChantierNameManipulateur::BasicAlloc(aDirPurge);
        vector<string> aLFileP = *(aICNMP->Get(".*"));
        cout << "Size = " << aLFileP.size() << endl;


        for (uint aL=0; aL < aLFileP.size(); aL++)
        {
            bool rm = aLFileP.at(aL)!="Px2_Num12_DeZoom1_LeChantier.tif"
                        && aLFileP.at(aL)!= "Px2_Scale_10.tif"
                        && aLFileP.at(aL)!="Px1_Num12_DeZoom1_LeChantier.tif"
                        && aLFileP.at(aL)!= "Px1_Scale_10.tif";
            if (rm)
                ELISE_fp::RmFile(aDirPurge+aLFileP.at(aL));
        }

    }
}


int TestYZ_main(int argc, char ** argv)
{
    string aDir, aName, aXml, aOutEsSim="All_H2D", aNameX = "Px1_Num12_DeZoom1_LeChantier.tif", aNameY = "Px2_Num12_DeZoom1_LeChantier.tif", aOutStatIm="All_StatIm";
    int aPas(1), aFunc(0), aScale (10);
    bool aIfStat (false), aIfScale (false), aPurge(false);

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Directory")
                    << EAMC(aName, "ImgPattern", eSAM_IsExistFile),
        LArgMain()  << EAM(aFunc,"Func",true,"choice of functions to execute. Def=0[CalMEC &  EsSim], 1[CalMEC], 2[EsSim], 3[N/A]")
                    << EAM(aPas,"Pas",true,"interval of image correlation; Def=1")
                    << EAM(aXml, "XML", true,".xml File for image correlation.")
                    << EAM(aNameX,"NameX",true,"name of deplacement img of axis-x; Def=Px1_Num12_DeZoom1_LeChantier.tif")
                    << EAM(aNameY,"NameY",true,"name of deplacement img of axis-x; Def=Px1_Num12_DeZoom1_LeChantier.tif")
                    << EAM(aOutEsSim,"OutEsSim",true,"Output file name for A,B,C,D Helmert2D Params; Def=All_H2D")
                    << EAM(aIfStat,"IfStatIm",true,"execute StatIm or not; Def=false")
                    << EAM(aOutStatIm,"OutStatIm",true,"Output file name for StatIm; Def=All_StatIm")
                    << EAM(aIfScale,"IfScale",false,"execute ScaleIm or not; Def=false")
                    << EAM(aScale,"Scale",true,"Scale of image sampling before estimation; Def=10")
                    << EAM(aPurge,"Purge",false,"Purge unnecessary files; Def=false")
    );

    cCalMECEsSim  aCalMECEsSim(aDir, aName, aNameX, aNameY, aScale, aPas);
    aCalMECEsSim.CreateCom();

    if (aFunc == 0 || aFunc == 1)
        aCalMECEsSim.CalMEC(aXml);

    if (aFunc == 0 || aFunc == 2)
    {
        aCalMECEsSim.ScaleIm(aIfScale);
        aCalMECEsSim.EsSim(aOutEsSim);
    }


    if (aIfStat)
        aCalMECEsSim.StatIm(aOutStatIm);

    if (aPurge)
        aCalMECEsSim.Purge();

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
