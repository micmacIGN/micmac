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
#include <algorithm>
#include "hassan/reechantillonnage.h"

// Reading Homologous point associated to a set of images


int  Luc_test_ptshom_main(int argc,char ** argv)
{
  //=====================  PARAMETRES EN DUR ==============

   std::string aDir = "C:/Users/Luc Girod/Desktop/TFE/Vignettage/vignette_sift3/";
   std::string aPatIm = ".*NEF";
   std::string Prefix = "";
   // std::string Prefix =  "_SRes" ;
   std::string Extension = "dat";

  //=====================

    // Permet de manipuler les ensemble de nom de fichier
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

//On parcourt toutes les paires d'images différentes (->testé dans le if)
    for (int aK1=0 ; aK1<int(aSetIm->size()) ; aK1++)
    {
        cout<<(*aSetIm)[aK1]<<endl;
        for (int aK2=0 ; aK2<int(aSetIm->size()) ; aK2++)
        {
            if (aK1!=aK2)
            {
               std::string aNamePack =  aDir +  aICNM->Assoc1To2
                                        (
                                           "NKS-Assoc-CplIm2Hom@"+ Prefix + "@"+Extension,
                                           (*aSetIm)[aK1],
                                           (*aSetIm)[aK2],
                                           true
                                    );

               if (aK1==0)
               {
                   bool Exist = ELISE_fp::exist_file(aNamePack);
                   if (Exist)
                   {
                      ElPackHomologue aPack = ElPackHomologue::FromFile(aNamePack);
                      if (aK2==1)
                      {
                           int aNb=0;
                           for
                           (
                               ElPackHomologue::const_iterator itP=aPack.begin();
                               itP!=aPack.end();
                               itP++
                           )
                           {
                              if (aNb<10)
                                  std::cout  << itP->P1() << itP->P2() <<"\n";
                              aNb++;
                           }
                      }
                      std::cout << aNamePack  << " " << aPack.size() << "\n";
                   }
                   else
                      std::cout  << "     # NO PACK FOR  : " << aNamePack  << "\n";
               }
            }
        }
    }


    return 0;
}

void RotateImage(double alpha, Pt2di aSzOut, vector<Pt2dr> Pts , string aNameDir, string aNameIm)
{
    cout<<"Rotating "<<aNameIm<<endl;
    string aNameOut=aNameDir + "Croped_images/" + aNameIm + ".tif";

    //Reading the image and creating the objects to be manipulated
    Tiff_Im aTF= Tiff_Im::StdConvGen(aNameDir + aNameIm,1,false);

    Pt2di aSz = aTF.sz();
    Pt2dr P1Cor=Rot2D(alpha, Pts[0], Pts[1]); //P1Cor.x=cos(alpha)*(Pts[0].x-Pts[1].x)+sin(alpha)*(Pts[0].y-Pts[1].y)+Pts[1].x; P1Cor.y=-sin(alpha)*(Pts[0].x-Pts[1].x)+cos(alpha)*(Pts[0].y-Pts[1].y)+Pts[1].y;
    Pt2dr P3Cor=Rot2D(alpha, Pts[2], Pts[1]); //P3Cor.x=cos(alpha)*(Pts[2].x-Pts[1].x)+sin(alpha)*(Pts[2].y-Pts[1].y)+Pts[1].x; P3Cor.y=-sin(alpha)*(Pts[2].x-Pts[1].x)+cos(alpha)*(Pts[2].y-Pts[1].y)+Pts[1].y;

    Im2D_U_INT1  aImR(aSz.x,aSz.y);
    //Im2D_U_INT1  aImG(aSz.x,aSz.y);
    //Im2D_U_INT1  aImB(aSz.x,aSz.y);
    Im2D_U_INT1  aImROut(aSzOut.x,aSzOut.y);
    Im2D_U_INT1  aImGOut(aSzOut.x,aSzOut.y);
    Im2D_U_INT1  aImBOut(aSzOut.x,aSzOut.y);

    ELISE_COPY
    (
       aTF.all_pts(),
       aTF.in(),
       aImR.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
    );

    U_INT1 ** aDataR = aImR.data();
    //U_INT1 ** aDataG = aImG.data();
    //U_INT1 ** aDataB = aImB.data();
    U_INT1 ** aDataROut = aImROut.data();
    U_INT1 ** aDataGOut = aImGOut.data();
    U_INT1 ** aDataBOut = aImBOut.data();

    //Parcours des points de l'image de sortie et remplissage des valeurs
    Pt2dr ptOut;
    Pt2di imageTopCorner, imageBottomCorner;
    imageTopCorner.x = (int)(P1Cor.x + 0.5) + (int)(max(abs(P1Cor.x - Pts[1].x), abs(P1Cor.x - P3Cor.x)) - aSzOut.x) / 2;
    imageTopCorner.y = (int)(P1Cor.y + 0.5) + (int)(max(abs(P3Cor.y - Pts[1].y), abs(P1Cor.y - P3Cor.y)) - aSzOut.y) / 2;
    imageBottomCorner.x = imageTopCorner.x + aSzOut.x;
    imageBottomCorner.y = imageTopCorner.y + aSzOut.y;

    for (int aY=imageTopCorner.y ; aY<imageBottomCorner.y  ; aY++)
    {
        for (int aX=imageTopCorner.x ; aX<imageBottomCorner.x  ; aX++)
        {
            ptOut.x=cos(-alpha)*(aX-Pts[1].x)+sin(-alpha)*(aY-Pts[1].y)+Pts[1].x;
            ptOut.y=-sin(-alpha)*(aX-Pts[1].x)+cos(-alpha)*(aY-Pts[1].y)+Pts[1].y;
            aDataROut[aY-imageTopCorner.y][aX-imageTopCorner.x] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);
            aDataGOut[aY-imageTopCorner.y][aX-imageTopCorner.x] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);
            aDataBOut[aY-imageTopCorner.y][aX-imageTopCorner.x] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);
        }
    }

    Tiff_Im  aTOut
    (
         aNameOut.c_str(),
         aSzOut,
         GenIm::u_int1,
         Tiff_Im::No_Compr,
         Tiff_Im::RGB
    );


    ELISE_COPY
    (
        aTOut.all_pts(),
        Virgule(aImROut.in(),aImGOut.in(),aImBOut.in()),
        aTOut.out()
    );

    #if (ELISE_unix || ELISE_Cygwin || ELISE_MacOs)
    string aCom="convert ephemeral:" + aNameDir + "Croped_images/" + aNameIm + ".tif " + aNameDir + "Croped_images/" + aNameIm;
    system_call(aCom.c_str());
    #endif
    #if (ELISE_windows)
        string aCom=MMDir() + "binaire-aux/convert ephemeral:" + aNameDir + "Croped_images/" + aNameIm + ".tif " + aNameDir + "Croped_images/" + aNameIm;
        system_call(aCom.c_str());
    #endif

}

int  Luc_main_corner_crop(int argc,char ** argv){

    std::string aFullPattern, cornersTxt;
    //Reading the arguments
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullPattern,"Images Pattern")
                    << EAMC(cornersTxt,"Corner txt File"),
        LArgMain()
    );

    std::string aDir,aPatIm;
    SplitDirAndFile(aDir,aPatIm,aFullPattern);

    ELISE_fp::MkDirRec(aDir + "Croped_images/");

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

    std::vector<std::string> aVectIm=*aSetIm;
    int nbIm=(int)aVectIm.size();

    vector<vector<Pt2dr> > Pts;
    vector<int> SzX, SzY;
    std::ifstream file(cornersTxt.c_str(), ios::in);
    for(int i=0 ; i<nbIm ; i++)
    {
        vector<Pt2dr> PtsIm(3);
        string name;
        file >> name >> PtsIm[0].x >> PtsIm[0].y >> name >> PtsIm[1].x >> PtsIm[1].y >> name >> PtsIm[2].x >> PtsIm[2].y;
        Pts.push_back(PtsIm);
        SzX.push_back((int)euclid(PtsIm[0], PtsIm[1])); SzY.push_back((int)euclid(PtsIm[2], PtsIm[1]));
    }

    file.close();
    cout<<Pts<<endl;
    Pt2di aCrop; int border=10;
    aCrop.x=min(*min_element(SzX.begin(), SzX.end())-2*border,*min_element(SzY.begin(), SzY.end())-2*border);
    aCrop.y=aCrop.x;
    //aCrop.x=*min_element(std::begin(SzX), std::end(SzX))-2*border;
    //aCrop.y=*min_element(std::begin(SzY), std::end(SzY))-2*border;
    cout<<"Cropping to : "<<aCrop.x<<" "<<aCrop.y<<endl;

    for(int i=0 ; i<nbIm ; i++)
    {
        double alpha=(atan((Pts[i][0].y-Pts[i][1].y)/(Pts[i][0].x-Pts[i][1].x))+atan(-(Pts[i][2].x-Pts[i][1].x)/(Pts[i][2].y-Pts[i][1].y)))/2;
        cout<<"Alpha = "<<alpha<<endl;
        RotateImage(alpha, aCrop, Pts[i], aDir, aVectIm[i]);
    }

    //Pt2dr P1,P2,P3;
    //P1.x= 795 ; P1.y= 1064;
    //P2.x= 7401; P2.y= 926 ;
    //P3.x= 7518; P3.y= 7598;
    //cout<<(P1.y-P2.y)/(P1.x-P2.x)<<endl;
    //cout<<atan((P1.y-P2.y)/(P1.x-P2.x))<<endl;
    //cout<<(P3.x-P2.x)/(P3.y-P2.y)<<endl;
    //cout<<atan((P3.x-P2.x)/(P3.y-P2.y))<<endl;
    //double aT1=atan((P1.y-P2.y)/(P1.x-P2.x));
    //double aT2=atan(-(P3.x-P2.x)/(P3.y-P2.y));
    //cout<<aT1<<" + "<<aT2<< " = " <<(aT1+aT2)<<endl;


    return 0;
}

int Luc_main_XAlign(int argc,char ** argv)
{
    //MMD_InitArgcArgv(argc,argv,3);

    std::string aFilePtsIn;
    //Reading the arguments
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFilePtsIn,"Input file"),
        LArgMain()
    );

    std::string aFilePtsOut="GCP_xAligned.xml";

    std::ifstream file(aFilePtsIn.c_str(), ios::in);
    int nbIm;
    file >> nbIm;
    std::vector<Pt3dr> aVPts(nbIm);
    std::vector<Pt3dr> aVInc(nbIm);
    std::vector<std::string> aVName(nbIm,"");
    for(int i=0 ; i<nbIm ; i++)
    {
        string name;
        file >> aVName[i] >> aVPts[i].x >> aVPts[i].y >> aVPts[i].z >> aVInc[i].x >> aVInc[i].y >> aVInc[i].z;
    }

    file.close();
    //Least Square

    // Create L2SysSurResol to solve least square equation with 3 unknown
    L2SysSurResol aSys(2);

    //For Each SIFT point
    double sumX=0, sumY=0;
    for(int i=0;i<int(aVPts.size());i++){
        double aPds[2]={aVPts[i].x,1};
        double poids=1;
        aSys.AddEquation(poids,aPds,aVPts[i].y);
        sumX=sumX+aVPts[i].x;
        sumY=sumY+aVPts[i].y;
    }

    Pt2dr aRotCenter; aRotCenter.x=sumX/aVPts.size();aRotCenter.y=sumY/aVPts.size();

    bool Ok;
    Im1D_REAL8 aSol = aSys.GSSR_Solve(&Ok);

    double aAngle;
    if (Ok)
    {
        double* aData = aSol.data();
        aAngle=atan(aData[0]);
        cout<<"Angle = "<<aAngle<<endl<<"Rot Center = "<<aRotCenter<<endl;

    for(int i=0;i<int(aVPts.size());i++){
        Pt2dr aPt; aPt.x=aVPts[i].x; aPt.y=aVPts[i].y;
        aPt=Rot2D(aAngle, aPt, aRotCenter);aVPts[i].x=aPt.x;aVPts[i].y=aPt.y;
    }
    }

//End Least Square

    cDicoAppuisFlottant  aDico;
    for (int aKP=0 ; aKP<int(aVPts.size()) ; aKP++)
    {
        cOneAppuisDAF aOAD;
        aOAD.Pt() = aVPts[aKP];
        aOAD.NamePt() = aVName[aKP];
        aOAD.Incertitude() = aVInc[aKP];

        aDico.OneAppuisDAF().push_back(aOAD);
    }


    MakeFileXML(aDico,aFilePtsOut);

    return 0;
}
#if (0)
#endif


class SpatioTempImage
{
public:
    SpatioTempImage(){}
    ~SpatioTempImage(){}
    Im2D_REAL4 RChan;
    Im2D_REAL4 GChan;
    Im2D_REAL4 BChan;
    CamStenope *aCamera;
    Pt2di SZ;
};

/* pas utilis� et cause une erreur multiple definition de FindMaltEtape
int FindMaltEtape(int ResolModel, std::string aNameIm, std::string aPatModel)
{
    //Getting full image size
    Tiff_Im aTFforSize = Tiff_Im::StdConvGen(aNameIm, 1, false);
    int aSzX = aTFforSize.sz().x;

    std::string aDir, aPat;
    SplitDirAndFile(aDir, aPat, aPatModel);

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aSetModel = aICNM->Get(aPat);

    std::vector<std::string> aVectModel = *aSetModel;
    int nbModel = aVectModel.size();

    int aEtape = 0;
    for (int i = 0; i<nbModel; i++)
    {
        cElNuage3DMaille * info3D = cElNuage3DMaille::FromFileIm(aDir + aVectModel[i]);
        //cout<<info3D->SzGeom()<<endl;
        int ResolThisFile = float(aSzX) / float(info3D->SzGeom().x) + 0.5;
        //cout<<"ResolThisFile : "<<ResolThisFile<<endl;
        if (ResolModel == ResolThisFile){ aEtape = i + 1; }
    }
    cout << "MicMac step to be used = num" << aEtape << endl;

    return aEtape;
}
*/

vector<SpatioTempImage> LoadGrpImages(string aDir, std::string aPatIm, std::string aOri)
{
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);
    std::vector<std::string> aVectIm = *aSetIm;
    int nbIm = (int)aVectIm.size();

    vector<SpatioTempImage> aGrIm;

    for (int aK1 = 0; aK1<int(nbIm); aK1++)
    {
        SpatioTempImage aIm;

        //Loading the camera
        string aNameCam = "Ori-" + aOri + "/Orientation-" + aVectIm[aK1] + ".xml";
        CamStenope * aCam = CamOrientGenFromFile(aNameCam, aICNM);
        aIm.aCamera = aCam;

        //Loading the image
        Tiff_Im aTF1 = Tiff_Im::StdConvGen(aDir + aVectIm[aK1], 3, false);
        Pt2di aSz = aTF1.sz();
        Im2D_REAL4  aIm1R(aSz.x, aSz.y);
        Im2D_REAL4  aIm1G(aSz.x, aSz.y);
        Im2D_REAL4  aIm1B(aSz.x, aSz.y);
        ELISE_COPY
            (
            aTF1.all_pts(),
            aTF1.in(),
            Virgule(aIm1R.out(), aIm1G.out(), aIm1B.out())
            );

        aIm.RChan = aIm1R;
        aIm.GChan = aIm1G;
        aIm.BChan = aIm1B;
        aIm.SZ = aSz;
        aGrIm.push_back(aIm);
    }

    return aGrIm;
}

Im2D_U_INT1 Window_Maker(SpatioTempImage aIm, Pt2di aCtrPt, int aSzW)
{
    //cout << "Creating Window for " << aCtrPt << endl;
    Im2D_U_INT1 aWindow(1 + 2 * aSzW, 1 + 2 * aSzW, 0);
    float aWSum = 0;
    for (int i = -aSzW; i <= aSzW; i++)
    {
        vector<float> aLine;
        for (int j = -aSzW; j <= aSzW; j++)
        {
            aWindow.data()[i + aSzW][j + aSzW] =(unsigned char) aIm.RChan.data()[aCtrPt.y + j][aCtrPt.x + i];
            aWSum = aWSum + aIm.RChan.data()[aCtrPt.y + j][aCtrPt.x + i];
        }
    }
    //cout << "Normalization" << endl;
    //Normalization
    /*
    for (int i = 0; i < aWindow.sz().x ; i++)
    {
        for (int j = 0; j < aWindow.sz().y; j++)
        {
            aWindow.data()[i][j] = aWindow.data()[i][j] / aWSum;
        }
    }
    */
    //cout << aWindow << endl;
    return aWindow;
}

double Correlator(vector<vector<float> > aWindow1, vector<vector<float> > aWindow2)
{
    double aScore=0;
    int aSz = (int)aWindow1.size();
    for (int i = 0; i < aSz; i++)
    {
        for (int j = 0; j < aSz; j++)
        {
            aScore = aScore + abs(aWindow1[i][j] - aWindow2[i][j]);
        }

    }

    return aScore;
}


int Luc_main_truc(int argc, char ** argv)
{
    /*
    std::string aFullPattern, aOri, aNameOut="PointsCordinates.txt";
    //Reading the arguments
    ElInitArgMain
        (
        argc, argv,
        LArgMain()  << EAMC(aFullPattern, "Images Pattern 1", eSAM_IsPatFile)
                    << EAMC(aOri, "Orientation", eSAM_IsPatFile),
        LArgMain()  << EAM(aNameOut, "Out", true, "Output file (txt)")
        );

    string aPattern, aNameDir;
    SplitDirAndFile(aNameDir, aPattern, aFullPattern);

    //Reading input files
    list<string> ListIm = RegexListFileMatch(aNameDir, aPattern, 1, false);
    int nbIm = ListIm.size();

    string oriFileName = aNameDir + aNameOut;
    FILE *f = fopen(oriFileName.c_str(), "w");

    for (int i = 1; i <= nbIm; i++)
    {
        //Processing the image
        string aNameIm = ListIm.front();
        ListIm.pop_front();
        string aNameOut = aNameDir + aNameIm + ".tif";

        //Loading the camera
        string aNameCam = "Ori-" + aOri + "/Orientation-" + aNameIm + ".xml";
        cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
        CamStenope * aCam = CamOrientGenFromFile(aNameCam, anICNM);
        cout << aNameIm << " [ " << aCam->VraiOpticalCenter().x << " , " << aCam->VraiOpticalCenter().y << " , " << aCam->VraiOpticalCenter().z << " ]" << endl;
        fprintf(f, "%s %0.6f %0.6f %0.6f\n", aNameIm, aCam->VraiOpticalCenter().x, aCam->VraiOpticalCenter().y, aCam->VraiOpticalCenter().z);

    }
    fclose(f);



    */

    std::string aFullPattern1, aFullPattern2, aFile3D1, aFile3D2, aOri, aDirOut = "Visualisation/";
    int aSzW = 1;
    double aSzMovArea = 5;
    //Reading the arguments
    ElInitArgMain
        (
        argc, argv,
        LArgMain()	<< EAMC(aFullPattern1, "Images Pattern 1", eSAM_IsPatFile)
                    << EAMC(aFullPattern2, "Images Pattern 2", eSAM_IsPatFile)
                    << EAMC(aFile3D1, "File 3D 1", eSAM_IsPatFile)
                    << EAMC(aFile3D2, "File 3D 2", eSAM_IsPatFile)
                    << EAMC(aOri, "Orientation", eSAM_IsPatFile),
        LArgMain()	<< EAM(aDirOut, "Out", true, "Output folder (end with /) and/or prefix (end with another char)")
                    << EAM(aSzMovArea, "SzMovArea", true, "Max magnitude of movement in meters (def=5)")
        );

    //Creating vector of images
    std::string aDir, aPatIm1, aPatIm2;
    SplitDirAndFile(aDir, aPatIm1, aFullPattern1);
    SplitDirAndFile(aDir, aPatIm2, aFullPattern2);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aSetIm1 = aICNM->Get(aPatIm1);
    const std::vector<std::string> * aSetIm2 = aICNM->Get(aPatIm2);
    std::vector<std::string> aVectIm1 = *aSetIm1;
    std::vector<std::string> aVectIm2 = *aSetIm2;

    cout << "Set 1 size : " << aVectIm1.size() << endl;
    cout << "Set 1 : " << aVectIm1 << endl;
    cout << "Set 2 size : " << aVectIm2.size() << endl;
    cout << "Set 2 : " << aVectIm2 << endl;

    //loading 3D models
    cElNuage3DMaille * info3D1 = cElNuage3DMaille::FromFileIm(aFile3D1);
    cElNuage3DMaille * info3D2 = cElNuage3DMaille::FromFileIm(aFile3D2);
    cout << "Sz geom 1: " << info3D1->SzGeom() << endl;
    cout << "Sz geom 2: " << info3D2->SzGeom() << endl;
    cout << "Resol geom 1: " << info3D1->ResolSolGlob() << endl;
    cout << "Resol geom 2: " << info3D2->ResolSolGlob() << endl;

    //Loading images
    vector<SpatioTempImage> aGrIm1 = LoadGrpImages(aDir, aPatIm1, aOri);
    vector<SpatioTempImage> aGrIm2 = LoadGrpImages(aDir, aPatIm2, aOri);
    cout << "Loaded " << aGrIm1.size() << " images for group 1 and " << aGrIm2.size() << " for group 2" << endl;


    //Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);
    std::string aNameOut = "banane.tif";
    //Reading the image and creating the objects to be manipulated
    aNameOut = aDir + aDirOut + aNameOut;

    Pt2di aSz( info3D1->SzGeom().x , info3D1->SzGeom().y );

    Tiff_Im aTF = Tiff_Im(aNameOut.c_str(), aSz, GenIm::real4, Tiff_Im::No_Compr, Tiff_Im::BlackIsZero);

    Im2D_REAL4  aIm(aSz.x, aSz.y);

    ELISE_COPY
        (
        aTF.all_pts(),
        aTF.in(),
        aIm.out()
        );

    REAL4 ** aData = aIm.data();


    for (int aY = aSzW; aY < aSz.y - aSzW; aY++)
    {
        for (int aX = aSzW; aX < aSz.x - aSzW; aX++)
        {
            /*******************************
            HOW TO
            //transform XY pixel coordinates to terrain coordinates
            Pt2dr pos2DPtIm1; pos2DPtIm1.x = 10050 + aX*0.14999999999999999*2; pos2DPtIm1.y = 10350 - aY*0.14999999999999999*2;
            //Go back to pix coordinates
            pos2DPtIm1 = info3D->Plani2Index(pos2DPtIm1);
            ********************************/
            Pt2dr pos2DMNT( aX , aY );
            //Test if there is data
            if (info3D1->CaptHasData(pos2DMNT)){
                //Get 3D info at XY
                Pt3d<double> pos3DMNT = info3D1->PreciseCapteur2Terrain(pos2DMNT);
                //cout << "pos3DMNT1 = " << pos3DMNT << endl;
                //Get Im1(i,j)
                Pt2di pos2DIm1( int(aGrIm1[0].aCamera->Ter2Capteur(pos3DMNT).x) , int(aGrIm1[0].aCamera->Ter2Capteur(pos3DMNT).y) );
                cout << "pos2DIm1 = " << pos2DIm1 << endl;
                //Extracting window from Im1
                Im2D_U_INT1 aWindow1 = Window_Maker(aGrIm1[0], pos2DIm1, 1);

                Fonc_Num aScoreMin = 100;
                /*
                for (int bX = pos3DMNT.x - aSzMovArea; bX <= pos3DMNT.x + aSzMovArea; bX++)
                {
                    for (int bY = pos3DMNT.y - aSzMovArea; bY <= pos3DMNT.y + aSzMovArea; bY++)
                    {
                        Pt2di aPos(int(aGrIm2[0].aCamera->Ter2Capteur(pos3DMNT).x), int(aGrIm2[0].aCamera->Ter2Capteur(pos3DMNT).y));
                        //Extracting window from Im2
                        Im2D_U_INT1 aWindow2 = Window_Maker(aGrIm2[0], aPos, 1);
                        double aScore = Correlator(&aWindow1, &aWindow2);
                        if (aScore < aScoreMin)
                        {
                            //CACACACACACACACCACACACACA TESTER SI PT EXIST DANS INFO3D2=====================================================================================
                            aScoreMin = aScore;
                            Pt2dr aPos2D(bX, bY);
                            Pt3dr aPos3D = info3D2->PreciseCapteur2Terrain(info3D2->Plani2Index(aPos2D));
                            cout << "pos set 1 = " << pos3DMNT << endl;
                            cout << "pos set 2 = " << aPos3D << endl;
                            aData[aY][aX] = square_euclid(pos3DMNT, aPos3D);
                            cout << aData[aY][aX] << endl;
                        }
                    }
                }


                //Get Im2(i,j)
                Pt2di pos2DIm2( int(aGrIm2[0].aCamera->Ter2Capteur(pos3DMNT).x) , int(aGrIm2[0].aCamera->Ter2Capteur(pos3DMNT).y) );
                cout << "pos2DIm2 = " << pos2DIm2 << endl;
                //Define researsh area in pixels from input info in meters
                int aSzMovAreaPix = aSzMovArea / aGrIm2[0].aCamera->ResolutionSol();
                cout <<"Pouet"<<endl;
                double aScoreMin = 100;
                for (int bX = pos2DIm2.x - aSzMovAreaPix; bX <= pos2DIm2.x + aSzMovAreaPix; bX++)
                {
                    for (int bY = pos2DIm2.y - aSzMovAreaPix; bY <= pos2DIm2.y + aSzMovAreaPix; bY++)
                    {
                        Pt2di aPos( bX, bY );
                        //Extracting window from Im2
                        vector<vector<float> > aWindow2 = Window_Maker(aGrIm2[0], aPos, 1);
                        cout << aWindow2 << endl;
                        double aScore = Correlator(aWindow1, aWindow2);
                        cout << aScore << endl;
                        if (aScore < aScoreMin)
                        {
                            aScoreMin = aScore;
                            Pt2dr aPosR(aPos.x, aPos.y);
                            cout << "aPos " << aPos << endl;
                            cout << "aPosR " << aPosR << endl;

                            Pt3dr aPosPt3D = aGrIm2[0].aCamera->PreciseCapteur2Terrain(aPosR);
                            cout << aPosPt3D << endl;
                            Pt2dr aPos2DPtIm2 = info3D2->Plani2Index(aPosR);
                            cout << aPos2DPtIm2 << endl;
                            aData[aY][aX] = square_euclid(pos3DMNT, info3D2->PreciseCapteur2Terrain(aPos2DPtIm2));
                            cout << aData[aY][aX] << endl;
                        }
                    }
                }
                */

            }
            else
            {
                aData[aY][aX] = 1;
                //cout << "not in Masq" << endl;
            }
        }
    }

    Tiff_Im  aTOut
        (
        aNameOut.c_str(),
        aSz,
        GenIm::real4,
        Tiff_Im::No_Compr,
        Tiff_Im::BlackIsZero
        );

    ELISE_COPY
        (
        aTOut.all_pts(),
        aIm.in(),
        aTOut.out()
        );

    return 0;

}


int Luc_main_PSEUDORPC2D(int argc, char ** argv)
{
    //GET PSEUDO-RPC2D FOR ASTER FROM LATTICE POINTS
    std::string aTxtImage, aTxtCarto;
    std::string aFileOut = "RPC2D-params.xml";
    //Reading the arguments
    ElInitArgMain
        (
        argc, argv,
        LArgMain()
        << EAMC(aTxtImage, "txt file contaning the lattice point in the image coordinates", eSAM_IsPatFile)
        << EAMC(aTxtCarto, "txt file contaning the lattice point in the carto coordinates", eSAM_IsPatFile),
        LArgMain()
        << EAM(aFileOut, "Out", true, "Output xml file with RPC2D coordinates")
        );

    //Reading the files
    vector<Pt2dr> aPtsIm, aPtsCarto;
    {
        std::ifstream fic(aTxtImage.c_str());
        while (!fic.eof() && fic.good())
        {
            double X, Y;
            fic >> X >> Y;
            Pt2dr aPt(X, Y);
            if (fic.good())
            {
                aPtsIm.push_back(aPt);
            }
        }
        cout << "Read " << aPtsIm.size() << " points in image coordinates" << endl;
        //cout << aPtsIm << endl;
        std::ifstream fic2(aTxtCarto.c_str());
        while (!fic2.eof() && fic2.good())
        {
            double X, Y, Z;
            fic2 >> X >> Y >> Z;
            Pt2dr aPt(X, Y);
            if (fic2.good())
            {
                aPtsCarto.push_back(aPt);
            }
        }
        cout << "Read " << aPtsCarto.size() << " points in cartographic coordinates" << endl;
        //cout << aPtsCarto << endl;
    }

    //Finding normalization parameters
    //divide Pts into X and Y
    vector<double> aPtsCartoX, aPtsCartoY, aPtsImX, aPtsImY;
    for (u_int i = 0; i < aPtsCarto.size(); i++)
    {
        aPtsCartoX.push_back(aPtsCarto[i].x);
        aPtsCartoY.push_back(aPtsCarto[i].y);
        aPtsImX.push_back(aPtsIm[i].x);
        aPtsImY.push_back(aPtsIm[i].y);
    }

    Pt2dr aPtCartoMin(*std::min_element(aPtsCartoX.begin(), aPtsCartoX.end()), *std::min_element(aPtsCartoY.begin(), aPtsCartoY.end()));
    Pt2dr aPtCartoMax(*std::max_element(aPtsCartoX.begin(), aPtsCartoX.end()), *std::max_element(aPtsCartoY.begin(), aPtsCartoY.end()));
    Pt2dr aPtImMin(*std::min_element(aPtsImX.begin(), aPtsImX.end()), *std::min_element(aPtsImY.begin(), aPtsImY.end()));
    Pt2dr aPtImMax(*std::max_element(aPtsImX.begin(), aPtsImX.end()), *std::max_element(aPtsImY.begin(), aPtsImY.end()));
    Pt2dr aCartoScale((aPtCartoMax.x - aPtCartoMin.x) / 2, (aPtCartoMax.y - aPtCartoMin.y) / 2);
    Pt2dr aImScale((aPtImMax.x - aPtImMin.x) / 2, (aPtImMax.y - aPtImMin.y) / 2);
    Pt2dr aCartoOffset(aPtCartoMin.x + (aPtCartoMax.x - aPtCartoMin.x) / 2, aPtCartoMin.y + (aPtCartoMax.y - aPtCartoMin.y) / 2);
    Pt2dr aImOffset(aPtImMin.x + (aPtImMax.x - aPtImMin.x) / 2, aPtImMin.y + (aPtImMax.y - aPtImMin.y) / 2);

    //Parameters too get parameters of P1 and P2 in ---  Column=P1(X,Y)/P2(X,Y)  --- where (X,Y) are Carto coordinates (idem for Row)
    //Function is 0=Poly1(X,Y)-Column*Poly2(X,Y) ==> Column*k=a+bX+cY+dXY+eX^2+fY^2+gX^2Y+hXY^2+iX^3+jY^3-Column(lX+mY+nXY+oX^2+pY^2+qX^2Y+rXY^2+sX^3+tY^3)
    //k=1 to avoid sol=0
    L2SysSurResol aSysCol(19), aSysRow(19);

    //For all lattice points
    for (u_int i = 0; i<aPtsCarto.size(); i++){

        //NORMALIZATION
        double X = (aPtsCarto[i].x - aCartoOffset.x) / aCartoScale.x;
        double Y = (aPtsCarto[i].y - aCartoOffset.y) / aCartoScale.y;
        double COL = (aPtsIm[i].x - aImOffset.x) / aImScale.x;
        double ROW = (aPtsIm[i].y - aImOffset.y) / aImScale.y;

        double aEqCol[19] = {
            (1),
            (X),
            (Y),
            (X*Y),
            (pow(X, 2)),
            (pow(Y, 2)),
            (pow(X, 2)*Y),
            (X*pow(Y, 2)),
            (pow(X, 3)),
            (pow(Y, 3)),
            //(COL),
            (-COL*X),
            (-COL*Y),
            (-COL*X*Y),
            (-COL*pow(X, 2)),
            (-COL*pow(Y, 2)),
            (-COL*pow(X, 2)*Y),
            (-COL*X*pow(Y, 2)),
            (-COL*pow(X, 3)),
            (-COL*pow(Y, 3)),
        };
		aSysCol.AddEquation(1, aEqCol, COL);


        double aEqRow[19] = {
            (1),
            (X),
            (Y),
            (X*Y),
            (pow(X, 2)),
            (pow(Y, 2)),
            (pow(X, 2)*Y),
            (X*pow(Y, 2)),
            (pow(X, 3)),
            (pow(Y, 3)),
            //(ROW),
            (-ROW*X),
            (-ROW*Y),
            (-ROW*X*Y),
            (-ROW*pow(X, 2)),
            (-ROW*pow(Y, 2)),
            (-ROW*pow(X, 2)*Y),
            (-ROW*X*pow(Y, 2)),
            (-ROW*pow(X, 3)),
            (-ROW*pow(Y, 3)),
        };
		aSysRow.AddEquation(1, aEqRow, ROW);
    }

    //Computing the result
    bool Ok;
    Im1D_REAL8 aSolCol = aSysCol.GSSR_Solve(&Ok);
    Im1D_REAL8 aSolRow = aSysRow.GSSR_Solve(&Ok);
    double* aDataCol = aSolCol.data();
    double* aDataRow = aSolRow.data();

    //Outputting results
    {
        std::ofstream fic(aFileOut.c_str());
        fic << std::setprecision(15);
		fic << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>" << endl;
		fic << "<RPC2D>" << endl;
        fic << "\t<RFM_Validity>" << endl;
        fic << "\t\t<Direct_Model_Validity_Domain>" << endl;
        fic << "\t\t\t<FIRST_ROW>" << aPtImMin.x << "</FIRST_ROW>" << endl;
        fic << "\t\t\t<FIRST_COL>" << aPtImMin.y << "</FIRST_COL>" << endl;
        fic << "\t\t\t<LAST_ROW>" << aPtImMax.x << "</LAST_ROW>" << endl;
        fic << "\t\t\t<LAST_COL>" << aPtImMax.y << "</LAST_COL>" << endl;
        fic << "\t\t</Direct_Model_Validity_Domain>" << endl;
        fic << "\t\t<Inverse_Model_Validity_Domain>" << endl;
        fic << "\t\t\t<FIRST_X>" << aPtCartoMin.x << "</FIRST_X>" << endl;
        fic << "\t\t\t<FIRST_Y>" << aPtCartoMin.y << "</FIRST_Y>" << endl;
        fic << "\t\t\t<LAST_X>" << aPtCartoMax.x << "</LAST_X>" << endl;
        fic << "\t\t\t<LAST_Y>" << aPtCartoMax.y << "</LAST_Y>" << endl;
        fic << "\t\t</Inverse_Model_Validity_Domain>" << endl;

        fic << "\t\t<X_SCALE>" << aCartoScale.x << "</X_SCALE>" << endl;
        fic << "\t\t<X_OFF>" << aCartoOffset.x << "</X_OFF>" << endl;
        fic << "\t\t<Y_SCALE>" << aCartoScale.y << "</Y_SCALE>" << endl;
        fic << "\t\t<Y_OFF>" << aCartoOffset.y << "</Y_OFF>" << endl;

        fic << "\t\t<SAMP_SCALE>" << aImScale.x << "</SAMP_SCALE>" << endl;
        fic << "\t\t<SAMP_OFF>" << aImOffset.x << "</SAMP_OFF>" << endl;
        fic << "\t\t<LINE_SCALE>" << aImScale.y << "</LINE_SCALE>" << endl;
        fic << "\t\t<LINE_OFF>" << aImOffset.y << "</LINE_OFF>" << endl;

        fic << "\t</RFM_Validity>" << endl;

        for (int i = 0; i<10; i++)
        {
			fic << "<COL_NUMERATOR_" << i + 1 << ">" << aDataCol[i] << "</COL_NUMERATOR_" << i + 1 << ">" << endl;
        }
		fic << "<COL_DENUMERATOR_1>1</COL_DENUMERATOR_1>" << endl;
        for (int i = 10; i<19; i++)
        {
			fic << "<COL_DENUMERATOR_" << i - 8 << ">" << aDataCol[i] << "</COL_DENUMERATOR_" << i -8 << ">" << endl;
        }
		for (int i = 0; i<10; i++)
		{
			fic << "<ROW_NUMERATOR_" << i + 1 << ">" << aDataRow[i] << "</ROW_NUMERATOR_" << i + 1 << ">" << endl;
		}
		fic << "<ROW_DENUMERATOR_1>1</ROW_DENUMERATOR_1>" << endl;
		for (int i = 10; i<19; i++)
		{
			fic << "<ROW_DENUMERATOR_" << i - 8 << ">" << aDataRow[i] << "</ROW_DENUMERATOR_" << i - 8 << ">" << endl;
		}
		fic << "</RPC2D>" << endl;
    }
    cout << "Written functions in file " << aFileOut << endl;

    return 0;
}

int Luc_main_geodesy_geodeticvsgeocentric(int argc, char ** argv)
{

	//WGS84 ellipsoid
	double a = 6378137;
	double b = (1 - 1 / 298.257223563)*a;
	double e2 = 1 - (b * b) / (a * a);
	double WGSCorFact = 0.99330562;

	//Point 
	Pt3dr aPtECEF(-2741844.353494039736688137054443359375, -1664255.3605559789575636386871337890625, 6304189.845196328125894069671630859375); //ALASKA
	//Pt3dr aPtECEF(3310919.731059408746659755706787109375, 628192.44206570624373853206634521484375, 6219499.615476393140852451324462890625); //NORWAY
	//NEW
	Pt3dr aPtGeo;
	// Computing longitude (true transformation)
	aPtGeo.x = atan(aPtECEF.y / aPtECEF.x) * 180 / M_PI; //degrees
	cout << aPtGeo.x << endl;
	if (aPtECEF.y < 0)//"Western emisphere"
		aPtGeo.x = aPtGeo.x - 180;

	//Computing latitude (estimation)
	double r = sqrt(aPtECEF.x*aPtECEF.x + aPtECEF.y*aPtECEF.y + aPtECEF.z*aPtECEF.z);
	double p = sqrt(aPtECEF.x*aPtECEF.x + aPtECEF.y*aPtECEF.y);
	double latNow = atan(p / aPtECEF.z);//rad geocentric
	//loop
	for (u_int i = 0; i < 4; i++)
	{
		double Rn = a / sqrt(1 - e2*sin(latNow)*sin(latNow));
		double h = p / cos(latNow) - Rn;
		latNow = atan(aPtECEF.z / p * 1 / (1 - e2*Rn / (Rn + h)));
	}
	aPtGeo.y = latNow;

	//Computing Ellipsoid height
	double Rn = a / sqrt(1 - e2*sin(aPtGeo.y)*sin(aPtGeo.y));
	aPtGeo.z = p / cos(aPtGeo.y) - Rn;
	//Latitude to degrees
	aPtGeo.y = aPtGeo.y * 180 / M_PI;
	cout << "NEW solution :" << aPtGeo << endl;

	//OLD
	r = sqrt(aPtECEF.x*aPtECEF.x + aPtECEF.y*aPtECEF.y + aPtECEF.z*aPtECEF.z);
	aPtGeo.y = asin(aPtECEF.z / r) * 180 / M_PI; //degrees
	aPtGeo.x = acos(aPtECEF.x / (r*cos(aPtGeo.y * M_PI / 180))) * 180 / M_PI;//degrees
	if (aPtECEF.y < 0)//"Western emisphere"
	aPtGeo.x = -aPtGeo.x;
	aPtGeo.z = r - sqrt(a*a*b*b / (a*a*sin(aPtGeo.y * M_PI / 180)*sin(aPtGeo.y * M_PI / 180) + b*b*cos(aPtGeo.y * M_PI / 180)*cos(aPtGeo.y * M_PI / 180)));//(distance from point to earth center)-(distance from ellipsoide to earth center)
	//to geodetic
	aPtGeo.y = atan(tan(aPtGeo.y *M_PI / 180) / WGSCorFact) * 180 / M_PI;

	cout << "OLD solution :" << aPtGeo << endl;
	return 0;
}


int Luc_main(int argc, char ** argv)
{
	//GET PSEUDO-RPC2D FOR ASTER FROM LATTICE POINTS
	std::string aNameIm, aNameParallax;
	std::string aFileOut = "RPC2D-params.xml";
	//Reading the arguments
	ElInitArgMain
		(
		argc, argv,
		LArgMain()
		<< EAMC(aNameIm, "image to be corrected", eSAM_IsPatFile)
		<< EAMC(aNameParallax, "Paralax correction file", eSAM_IsPatFile),
		LArgMain()
		<< EAM(aFileOut, "Out", true, "Output xml file with RPC2D coordinates")
		);

	std::string aDir, aPatIm;
	SplitDirAndFile(aDir, aPatIm, aNameIm);

	cout << "Correcting " << aNameIm << endl;
	string aNameOut = aNameIm + "_corrected.tif";

	//Reading the image and creating the objects to be manipulated
	Tiff_Im aTF = Tiff_Im::StdConvGen(aDir + aNameIm, 1, false);

	Pt2di aSz = aTF.sz();
	Im2D_U_INT1  aIm(aSz.x, aSz.y);
	Im2D_U_INT1  aImOut(aSz.x, aSz.y);

	ELISE_COPY
		(
		aTF.all_pts(),
		aTF.in(),
		aIm.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
		);

	U_INT1 ** aData = aIm.data();
	U_INT1 ** aDataOut = aImOut.data();

	//Reading the parallax correction file
	Tiff_Im aTFPar = Tiff_Im::StdConvGen(aDir + aNameParallax, 1, false);
	Im2D_REAL8  aPar(aSz.x, aSz.y);
	ELISE_COPY
		(
		aTFPar.all_pts(),
		aTFPar.in(),
		aPar.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
		);
	REAL8 ** aDatPar = aPar.data();



	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			Pt2dr ptOut;
			ptOut.x = aX-aDatPar[aY][aX];
			ptOut.y = aY;
			aDataOut[aY][aX] = Reechantillonnage::biline(aData, aSz.x, aSz.y, ptOut);
		}
	}

	Tiff_Im  aTOut
		(
		aNameOut.c_str(),
		aSz,
		GenIm::u_int1,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
		);


	ELISE_COPY
		(
		aTOut.all_pts(),
		aImOut.in(),
		aTOut.out()
		);



	return 0;
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
