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
#include "kugelhupf.h"
#include "../uti_phgrm/NewOri/NewOri.h"
#include <algorithm>
#include "PHO_MI.h"
#if ELISE_windows
    #define uint unsigned
#endif



/**
 * TestPointHomo: read point homologue entre 2 image après Tapioca
 * Inputs:
 *  - Nom de 2 images
 *  - Fichier de calibration du caméra
 * Output:
 *  - List de coordonné de point homologue
 *  - serie1_Line$ mm3d PHO_MI "soussol_161015_001_0000[6-8].tif" Ori-Serie1/
 * */


void StdCorrecNameHomol_G(std::string & aNameH,const std::string & aDir)
{

    int aL = strlen(aNameH.c_str());
    if (aL && (aNameH[aL-1]==ELISE_CAR_DIR))
    {
        aNameH = aNameH.substr(0,aL-1);
    }

    if ((strlen(aNameH.c_str())>=5) && (aNameH.substr(0,5)==std::string("Homol")))
       aNameH = aNameH.substr(5,std::string::npos);

    std::string aTest =  ( isUsingSeparateDirectories()?MMOutputDirectory():aDir ) + "Homol"+aNameH+ ELISE_CAR_DIR;
}


vector<AbreHomol> creatAbreFromPattern(vector<string> aSetImages, string aNameHomol, string aFullPatternImages, string aOriInput)
{


    //=============Manip File Name=====================
      ELISE_fp::AssertIsDirectory(aNameHomol);

      // Initialize name manipulator & files
      std::string aDirImages, aPatImages;
      SplitDirAndFile(aDirImages,aPatImages,aFullPatternImages);

      StdCorrecNameOrient(aOriInput,aDirImages);//remove "Ori-" if needed

      cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);

      ELISE_ASSERT(aSetImages.size()>1,"Number of image must be > 1");
      vector<string> aSetImg = aSetImages;
    //======================================================
    vector<string> tempArbeRacine;
    vector<AbreHomol> Abre;
    for (uint i=0; i<aSetImg.size(); i++)
    {
        AbreHomol aAbre;
        for (uint j=0; j<aSetImg.size(); j++)
        {
            string nameImgB = aSetImg[j];
            string nameImgA = aSetImg[i];

            std::string aKey =   std::string("NKS-Assoc-CplIm2Hom@")
                    +  std::string(aNameHomol)
                    +  std::string("@")
                    +  std::string("dat");
            std::string aHomol = aICNM->Assoc1To2(aKey,nameImgA,nameImgB,true);
            StdCorrecNameHomol_G(aHomol,aDirImages);

            bool Exist = ELISE_fp::exist_file(aHomol);
            if (Exist)
            {
                //creer abre de Homol
                aAbre.ImgRacine = nameImgA;
                aAbre.ImgBranch.push_back(nameImgB);
                ElPackHomologue aPackIn =  ElPackHomologue::FromFile(aHomol);
                aAbre.NbPointHomo.push_back(aPackIn.size());
            }
        }
        Abre.push_back(aAbre);
        tempArbeRacine.push_back(aSetImg[i]);
    }
    for (uint k1=0; k1<Abre.size(); k1++)
    {
        for (uint k2=0; k2<Abre[k1].ImgBranch.size(); k2++)
        {
            vector <string> ColImg3eme;
            std::vector<string>::iterator it;
            it = std::find(tempArbeRacine.begin(), tempArbeRacine.end(), Abre[k1].ImgBranch[k2]);
            double p = std::distance( tempArbeRacine.begin(), it );
            for (uint k3=0; k3<Abre[p].ImgBranch.size(); k3++)
            {
                //search for common img b/w Abre[k1] and Abre[p]
                std::vector<string>::iterator it1;
                it1 = std::find(Abre[k1].ImgBranch.begin(), Abre[k1].ImgBranch.end(), Abre[p].ImgBranch[k3]);
                double p1 = std::distance( Abre[k1].ImgBranch.begin(), it1 );
                bool isPresent = (it1 != Abre[k1].ImgBranch.end());
                if (isPresent)
                {
                    ColImg3eme.push_back(Abre[k1].ImgBranch[p1]);//common img entre Abre[k1] and Abre[p]
                }
            }
            Abre[k1].Img3eme.push_back(ColImg3eme);
        }

    }
return Abre;
}

vector<string> displayAbreHomol(vector<AbreHomol> aAbre, bool disp)
{
    vector<string> result;
    for (uint i=0;i<aAbre.size();i++)
    {result.push_back(aAbre[i].ImgRacine);}
    if (disp)
    {
        for (uint i=0;i<aAbre.size();i++)
        {
            cout<<aAbre[i].ImgRacine<<endl;
            for(uint k=0; k<aAbre[i].ImgBranch.size(); k++)
            {
                cout<<" ++ "<<aAbre[i].ImgBranch[k]<<endl;
                for(uint l=0; l<aAbre[i].Img3eme[k].size(); l++)
                {
                    cout<<"   ++ "<<aAbre[i].Img3eme[k][l]<<endl;
                }
            }
        }
    }
    return result;
}

vector<bool> FiltreDe3img(string aNameImg1, string aNameImg2, string aNameImg3, string aNameHomol, string aDirImages, string aPatImages, string aOriInput, bool ExpTxt, Pt2dr centre_img, double diag, double aDistRepr, double aDistHom)
{
  //=============Manip File Name=====================
    ELISE_fp::AssertIsDirectory(aNameHomol);

    // Initialize name manipulator & files
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    std::string anExt = ExpTxt ? "txt" : "dat";
    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(aNameHomol)
            +  std::string("@")
            +  std::string(anExt);

    //===========================================================
    vector<bool> result;
    //==========import img1 img2 img3===========
    std::string aOri1 = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOriInput,aNameImg1,true);
    CamStenope * aCam1 = CamOrientGenFromFile(aOri1 , aICNM);
    std::string aOri2 = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOriInput,aNameImg2,true);
    CamStenope * aCam2 = CamOrientGenFromFile(aOri2 , aICNM);
    std::string aOri3 = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOriInput,aNameImg3,true);
    CamStenope * aCam3 = CamOrientGenFromFile(aOri3 , aICNM);

    std::string aHomoIn1_2 = aICNM->Assoc1To2(aKHIn, aNameImg1, aNameImg2,true);
    StdCorrecNameHomol_G(aHomoIn1_2,aDirImages);
    ElPackHomologue aPackIn1_2, aPackIn1_3, aPackIn2_3;
    bool Exist1_2 = ELISE_fp::exist_file(aHomoIn1_2);
    if (Exist1_2)
    {
     aPackIn1_2 =  ElPackHomologue::FromFile(aHomoIn1_2);
    }

    std::string aHomoIn1_3 = aICNM->Assoc1To2(aKHIn, aNameImg1, aNameImg3,true);
    StdCorrecNameHomol_G(aHomoIn1_3,aDirImages);
    bool Exist1_3 = ELISE_fp::exist_file(aHomoIn1_3);
    if (Exist1_3)
    {
     aPackIn1_3 =  ElPackHomologue::FromFile(aHomoIn1_3);
    }

    std::string aHomoIn2_3 = aICNM->Assoc1To2(aKHIn, aNameImg2, aNameImg3, true);
    StdCorrecNameHomol_G(aHomoIn2_3,aDirImages);
    bool Exist2_3 = ELISE_fp::exist_file(aHomoIn2_3);
    if (Exist2_3)
    {
     aPackIn2_3 =  ElPackHomologue::FromFile(aHomoIn2_3);
    }

    //================traite======================//
    double w=3;
    double count_pass_reproj=0;
    double countGoodTrip = 0;
    for (ElPackHomologue::const_iterator itP=aPackIn1_2.begin(); itP!=aPackIn1_2.end() ; itP++)
    {

        Pt2dr aP1 = itP->P1();  //Point img1
        Pt2dr aP2 = itP->P2();  //Point img2
        double d;
        bool pass_reproj, pass_corr;
        //=================verifier par reprojeter============
        Pt3dr PInter1_2= aCam1->ElCamera::PseudoInter(aP1, *aCam2, aP2, &d);	//use Point img1 & 2 to search point 3d
        Pt2dr PReproj3 = aCam3->ElCamera::R3toF2(PInter1_2);					//use point 3d to search Point img3
        double dist_centre = sqrt(pow((PReproj3.x - centre_img.x),2) + pow((PReproj3.y - centre_img.y),2));
        bool inside = (dist_centre < 0.67*diag/2) ? true : false;
        //chercher triplet
        const ElCplePtsHomologues  * aTriplet2_3 = aPackIn2_3.Cple_Nearest(aP2,true);
        const ElCplePtsHomologues  * aTriplet1_3 = aPackIn1_3.Cple_Nearest(aP1,true);
        double distP2 = sqrt(pow((aTriplet2_3->P1().x - aP2.x),2) + pow((aTriplet2_3->P1().y - aP2.y),2));
        double distP3 = sqrt(pow((aTriplet1_3->P2().x - aTriplet2_3->P2().x),2) + pow((aTriplet1_3->P2().y - aTriplet2_3->P2().y),2));
        if ( (distP2 < aDistHom) && (distP3 < aDistHom) )
        {
            Pt2dr aP3 = aTriplet2_3->P2();
            countGoodTrip ++;
            //check condition reproject
            double distRepr = sqrt(pow((aP3.x - PReproj3.x),2) + pow((aP3.y - PReproj3.y),2));

            if (distRepr < aDistRepr)
            {
                pass_reproj=1;
                count_pass_reproj++;
            }
            else
            {
                pass_reproj=0;
            }

        }
        else
        {
            pass_reproj=0;
        }
        result.push_back(pass_reproj);
    }
    //cout <<"   ++ => "<<count_pass_reproj<<" / "<<countGoodTrip<<" / "<<aPackIn1_2.size()<<endl;
    count_pass_reproj = 0;
    for (uint i=0;i<result.size(); i++)
    {
        if (result[i])
        {
            count_pass_reproj++;
        }
    }
    //cout <<"   ++ Verif => "<<count_pass_reproj<<" / "<<countGoodTrip<<" / "<<result.size()<<endl;
    return result;
}

void creatHomolFromPair(string aNameImg1, string aNameImg2, string aNameHomol, string aDirImages, string aPatImages, string aHomolOutput, bool ExpTxt, vector<bool> decision)
{
    cout<<"ecrit...";
    //=============Manip File Name=====================
    ELISE_fp::AssertIsDirectory(aNameHomol);

    // Initialize name manipulator & files
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    std::string anExt = ExpTxt ? "txt" : "dat";
    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(aNameHomol)
            +  std::string("@")
            +  std::string(anExt);

    //===========================================================
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(aHomolOutput)
                        +  std::string("@")
                        +  std::string("txt");

    std::string aKHOutDat =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(aHomolOutput)
                        +  std::string("@")
                        +  std::string("dat");

    std::string aHomoIn1_2 = aICNM->Assoc1To2(aKHIn, aNameImg1, aNameImg2,true);
    StdCorrecNameHomol_G(aHomoIn1_2,aDirImages);
    ElPackHomologue aPackIn1_2, aPackIn1_3, aPackIn2_3;
    bool Exist1_2 = ELISE_fp::exist_file(aHomoIn1_2);
    if (Exist1_2)
    {
     aPackIn1_2 =  ElPackHomologue::FromFile(aHomoIn1_2);
    }

    //creat name of homomogue file dans 2 sens
    ElPackHomologue Pair1_2, Pair1_2i;
    std::string NameHomolPair1 = aICNM->Assoc1To2(aKHOut, aNameImg1, aNameImg2, true);
    std::string NameHomolDatPair1 = aICNM->Assoc1To2(aKHOutDat, aNameImg1, aNameImg2, true);
    std::string NameHomolDatPair1i = aICNM->Assoc1To2(aKHOutDat, aNameImg2, aNameImg1 , true);
    double ind=0;
    cout<<NameHomolPair1;
    for (ElPackHomologue::const_iterator itP=aPackIn1_2.begin(); itP!=aPackIn1_2.end() ; itP++)
    {
        if(decision[ind])
        {
            Pair1_2.Cple_Add(ElCplePtsHomologues( itP->P1(), itP->P2() ));
            Pair1_2i.Cple_Add(ElCplePtsHomologues( itP->P2() , itP->P1() ));
        }
        ind++;
    }
    Pair1_2.StdPutInFile(NameHomolPair1);
    Pair1_2.StdPutInFile(NameHomolDatPair1);
    Pair1_2i.StdPutInFile(NameHomolDatPair1i);
    cout<<"..done !"<<endl;
}


VerifParRepr::VerifParRepr(vector<string> mListImg, string aDirImages, string aPatImages, string aNameHomol, string aOri, string aHomolOutput, bool ExpTxt)
{    
        this->mListImg = mListImg;
        this->mDirImages = aDirImages;
        this->mPatImages = aPatImages;
        this->mNameHomol = aNameHomol;
        this->mOri = aOri;
        this->mHomolOutput = aHomolOutput;
        this->mExpTxt = ExpTxt;
}

vector<AbreHomol> VerifParRepr::creatAbre()
{

    //=============Manip File Name=====================
      cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(this->mDirImages);
      ELISE_ASSERT(this->mListImg.size()>1,"Number of image must be > 1");
    //======================================================
    vector<string> tempArbeRacine;
    vector<AbreHomol> Abre;
    for (uint i=0; i<this->mListImg.size(); i++)
    {
        AbreHomol aAbre;
        for (uint j=0; j<this->mListImg.size(); j++)
        {
            string nameImgB = this->mListImg[j];
            string nameImgA = this->mListImg[i];

            std::string aKey =   std::string("NKS-Assoc-CplIm2Hom@")
                    +  std::string(this->mNameHomol)
                    +  std::string("@")
                    +  std::string("dat");
            std::string aHomol = aICNM->Assoc1To2(aKey,nameImgA,nameImgB,true);
            StdCorrecNameHomol_G(aHomol,this->mDirImages);

            bool Exist = ELISE_fp::exist_file(aHomol);
            if (Exist)
            {
                //creer abre de Homol
                aAbre.ImgRacine = nameImgA;
                aAbre.ImgBranch.push_back(nameImgB);
                ElPackHomologue aPackIn =  ElPackHomologue::FromFile(aHomol);
                aAbre.NbPointHomo.push_back(aPackIn.size());
            }
        }
        Abre.push_back(aAbre);
        tempArbeRacine.push_back(this->mListImg[i]);
    }
    for (uint k1=0; k1<Abre.size(); k1++)
    {
        for (uint k2=0; k2<Abre[k1].ImgBranch.size(); k2++)
        {
            vector <string> ColImg3eme;
            std::vector<string>::iterator it;
            it = std::find(tempArbeRacine.begin(), tempArbeRacine.end(), Abre[k1].ImgBranch[k2]);
            double p = std::distance( tempArbeRacine.begin(), it );
            for (uint k3=0; k3<Abre[p].ImgBranch.size(); k3++)
            {
                //search for common img b/w Abre[k1] and Abre[p]
                std::vector<string>::iterator it1;
                it1 = std::find(Abre[k1].ImgBranch.begin(), Abre[k1].ImgBranch.end(), Abre[p].ImgBranch[k3]);
                double p1 = std::distance( Abre[k1].ImgBranch.begin(), it1 );
                bool isPresent = (it1 != Abre[k1].ImgBranch.end());
                if (isPresent)
                {
                    ColImg3eme.push_back(Abre[k1].ImgBranch[p1]);//common img entre Abre[k1] and Abre[p]
                }
            }
            Abre[k1].Img3eme.push_back(ColImg3eme);
        }

    }
    this->mAbre = Abre;
    this->mtempArbeRacine = tempArbeRacine;
    Tiff_Im mTiffImg3(tempArbeRacine[0].c_str());

    Pt2dr centre_img(mTiffImg3.sz().x/2, mTiffImg3.sz().y/2);
    this->mcentre_img = centre_img;
    this->mdiag = sqrt(pow(mTiffImg3.sz().x,2) + pow(mTiffImg3.sz().y,2));
return Abre;
}

vector<string> VerifParRepr::displayAbreHomol(vector<AbreHomol> aAbre, bool disp)
{
    vector<string> result;
    for (uint i=0;i<aAbre.size();i++)
    {result.push_back(aAbre[i].ImgRacine);}
    if (disp)
    {
        for (uint i=0;i<aAbre.size();i++)
        {
            cout<<aAbre[i].ImgRacine<<endl;
            for(uint k=0; k<aAbre[i].ImgBranch.size(); k++)
            {
                cout<<" ++ "<<aAbre[i].ImgBranch[k]<<endl;
                for(uint l=0; l<aAbre[i].Img3eme[k].size(); l++)
                {
                    cout<<"   ++ "<<aAbre[i].Img3eme[k][l]<<endl;
                }
            }
        }
    }
    return result;
}

vector<bool> VerifParRepr::FiltreDe3img(string aNameImg1, string aNameImg2, string aNameImg3)
{
  //=============Manip File Name=====================
    ELISE_fp::AssertIsDirectory(this->mNameHomol);

    // Initialize name manipulator & files
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(this->mDirImages);
    std::string anExt = this->mExpTxt ? "txt" : "dat";
    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(this->mNameHomol)
            +  std::string("@")
            +  std::string(anExt);
    //===========================================================
    vector<bool> result;
    //==========import img1 img2 img3===========
    std::string aOri1 = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+this->mOri,aNameImg1,true);
    CamStenope * aCam1 = CamOrientGenFromFile(aOri1 , aICNM);
    std::string aOri2 = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+this->mOri,aNameImg2,true);
    CamStenope * aCam2 = CamOrientGenFromFile(aOri2 , aICNM);
    std::string aOri3 = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+this->mOri,aNameImg3,true);
    CamStenope * aCam3 = CamOrientGenFromFile(aOri3 , aICNM);


    std::string aHomoIn1_2 = aICNM->Assoc1To2(aKHIn, aNameImg1, aNameImg2,true);
    StdCorrecNameHomol_G(aHomoIn1_2,this->mDirImages);
    ElPackHomologue aPackIn1_2, aPackIn1_3, aPackIn2_3;
    bool Exist1_2 = ELISE_fp::exist_file(aHomoIn1_2);
    if (Exist1_2)
    {
     aPackIn1_2 =  ElPackHomologue::FromFile(aHomoIn1_2);
    }

    std::string aHomoIn1_3 = aICNM->Assoc1To2(aKHIn, aNameImg1, aNameImg3,true);
    StdCorrecNameHomol_G(aHomoIn1_3,this->mDirImages);
    bool Exist1_3 = ELISE_fp::exist_file(aHomoIn1_3);
    if (Exist1_3)
    {
     aPackIn1_3 =  ElPackHomologue::FromFile(aHomoIn1_3);
    }

    std::string aHomoIn2_3 = aICNM->Assoc1To2(aKHIn, aNameImg2, aNameImg3, true);
    StdCorrecNameHomol_G(aHomoIn2_3,this->mDirImages);
    bool Exist2_3 = ELISE_fp::exist_file(aHomoIn2_3);
    if (Exist2_3)
    {
     aPackIn2_3 =  ElPackHomologue::FromFile(aHomoIn2_3);
    }

    //================traite======================//
    double w=3;
    double count_pass_reproj=0;
    double countGoodTrip = 0;
    for (ElPackHomologue::const_iterator itP=aPackIn1_2.begin(); itP!=aPackIn1_2.end() ; itP++)
    {

        Pt2dr aP1 = itP->P1();  //Point img1
        Pt2dr aP2 = itP->P2();  //Point img2
        double d;
        bool pass_reproj, pass_corr;
        //=================verifier par reprojeter============
        Pt3dr PInter1_2= aCam1->ElCamera::PseudoInter(aP1, *aCam2, aP2, &d);	//use Point img1 & 2 to search point 3d
        Pt2dr PReproj3 = aCam3->ElCamera::R3toF2(PInter1_2);					//use point 3d to search Point img3
        double dist_centre = sqrt(pow((PReproj3.x - this->mcentre_img.x),2) + pow((PReproj3.y - this->mcentre_img.y),2));
        bool inside = (dist_centre < 0.67*this->mdiag/2) ? true : false;
        //chercher triplet
        const ElCplePtsHomologues  * aTriplet2_3 = aPackIn2_3.Cple_Nearest(aP2,true);
        const ElCplePtsHomologues  * aTriplet1_3 = aPackIn1_3.Cple_Nearest(aP1,true);
        double distP2 = sqrt(pow((aTriplet2_3->P1().x - aP2.x),2) + pow((aTriplet2_3->P1().y - aP2.y),2));
        double distP3 = sqrt(pow((aTriplet1_3->P2().x - aTriplet2_3->P2().x),2) + pow((aTriplet1_3->P2().y - aTriplet2_3->P2().y),2));
        if ( (distP2 < this->mDistHom) && (distP3 < this->mDistHom) )
        {
            Pt2dr aP3 = aTriplet2_3->P2();
            countGoodTrip ++;
            //check condition reproject
            double distRepr = sqrt(pow((aP3.x - PReproj3.x),2) + pow((aP3.y - PReproj3.y),2));

            if (distRepr < this->mDistRepr)
            {
                pass_reproj=1;
                count_pass_reproj++;
            }
            else
            {
                pass_reproj=0;
            }

        }
        else
        {
            pass_reproj=0;
        }
        result.push_back(pass_reproj);
    }
    //cout <<"   ++ => "<<count_pass_reproj<<" / "<<countGoodTrip<<" / "<<aPackIn1_2.size()<<endl;
    count_pass_reproj = 0;
    for (uint i=0;i<result.size(); i++)
    {
        if (result[i])
        {
            count_pass_reproj++;
        }
    }
    //cout <<"   ++ Verif => "<<count_pass_reproj<<" / "<<countGoodTrip<<" / "<<result.size()<<endl;
    return result;
}

void VerifParRepr::creatHomolFromPair(string aNameImg1, string aNameImg2, vector<bool> decision)
{
    cout<<"ecrit...";
    //=============Manip File Name=====================
    ELISE_fp::AssertIsDirectory(this->mNameHomol);

    // Initialize name manipulator & files
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(this->mDirImages);
    std::string anExt = this->mExpTxt ? "txt" : "dat";
    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(this->mNameHomol)
            +  std::string("@")
            +  std::string(anExt);

    //===========================================================
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(this->mHomolOutput)
                        +  std::string("@")
                        +  std::string("txt");

    std::string aKHOutDat =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(this->mHomolOutput)
                        +  std::string("@")
                        +  std::string("dat");

    std::string aHomoIn1_2 = aICNM->Assoc1To2(aKHIn, aNameImg1, aNameImg2,true);
    StdCorrecNameHomol_G(aHomoIn1_2,this->mDirImages);
    ElPackHomologue aPackIn1_2, aPackIn1_3, aPackIn2_3;
    bool Exist1_2 = ELISE_fp::exist_file(aHomoIn1_2);
    if (Exist1_2)
    {
     aPackIn1_2 =  ElPackHomologue::FromFile(aHomoIn1_2);
    }

    //creat name of homomogue file dans 2 sens
    ElPackHomologue Pair1_2, Pair1_2i;
    std::string NameHomolPair1 = aICNM->Assoc1To2(aKHOut, aNameImg1, aNameImg2, true);
    std::string NameHomolDatPair1 = aICNM->Assoc1To2(aKHOutDat, aNameImg1, aNameImg2, true);
    std::string NameHomolDatPair1i = aICNM->Assoc1To2(aKHOutDat, aNameImg2, aNameImg1 , true);
    double ind=0;
    cout<<NameHomolPair1;
    for (ElPackHomologue::const_iterator itP=aPackIn1_2.begin(); itP!=aPackIn1_2.end() ; itP++)
    {
        if(decision[ind])
        {
            Pair1_2.Cple_Add(ElCplePtsHomologues( itP->P1(), itP->P2() ));
            Pair1_2i.Cple_Add(ElCplePtsHomologues( itP->P2() , itP->P1() ));
        }
        ind++;
    }
    Pair1_2.StdPutInFile(NameHomolPair1);
    Pair1_2.StdPutInFile(NameHomolDatPair1);
    Pair1_2i.StdPutInFile(NameHomolDatPair1i);
    cout<<"..done !"<<endl;
}

void VerifParRepr::FiltragePtsHomo()
{
    vector<string> tempArbeRacine = this->mtempArbeRacine;

    vector<AbreHomol> aAbre = this->mAbre;

    Pt2dr centre_img=this->mcentre_img;
    double diag = this->mdiag;
    vector< vector<bool> > ColDec;

    double stat = 0;
    double all = 0;
    for (uint i=0;i<aAbre.size();i++)
    {
        cout<<aAbre[i].ImgRacine<<endl;
        string aNameImg1 = aAbre[i].ImgRacine;
        for(uint k=0; k<aAbre[i].ImgBranch.size(); k++)
        {
            cout<<" ++ "<<aAbre[i].ImgBranch[k]<<endl;
            string aNameImg2 = aAbre[i].ImgBranch[k];
            for(uint l=0; l<aAbre[i].Img3eme[k].size(); l++)
            {
                cout<<"   + Com + "<<aAbre[i].Img3eme[k][l]<<endl;
                //====Triplet Image==========//
                aNameImg1 = aAbre[i].ImgRacine;
                aNameImg2 = aAbre[i].ImgBranch[k];
                string aNameImg3 = aAbre[i].Img3eme[k][l];
                cout<<"   + Tri + "<<aNameImg1<<" "<<aNameImg2<<" "<<aNameImg3<<endl;
                vector<bool> result = FiltreDe3img( aNameImg1,  aNameImg2,  aNameImg3);
                ColDec.push_back(result);
            }
            //=====decision=====
            vector<bool> decision; vector<double> decPoint;

            for(uint m=0; m<ColDec[0].size(); m++)
            {
                bool dec = 0; double point=0;
                for (uint n=0; n<ColDec.size(); n++)
                {
                    dec = dec || ColDec[n][m];
                    if (ColDec[n][m])
                    {point++;}
                }
                decision.push_back(dec);
                decPoint.push_back(point);
            }
            //creat homol file with decision and pack homo b/w aNameImg1 aNameImg2
            //.....
            double totalImgCom = aAbre[i].Img3eme[k].size();
            double countVerif=0;
            for(uint o=0; o<decision.size(); o++)
            {
                if (decision[o] && ((decPoint[o]/totalImgCom) > 0.95) )
                {countVerif++; decision[o] = 1;}
                else
                {decision[o] = 0;}
            }
            cout<<"     *-*-*-*-*-*   "<<" NbPt Filtré = "<<countVerif<< " / "<<decision.size()<<endl<<endl;
            stat = stat + countVerif;
            all = all + decision.size();
            creatHomolFromPair(aNameImg1, aNameImg2, decision);
        }
    }
    cout<<endl<<endl<<"+- %Pt good + "<<stat/all*100<<"%"<<endl;

}

int PHO_MI_main(int argc,char ** argv)
{
    cout<<"*********************"<<endl;
    cout<<"* P : Points        *"<<endl;
    cout<<"* H : Homologues    *"<<endl;
    cout<<"* O : Observés sur  *"<<endl;
    cout<<"* M : Modèle        *"<<endl;
    cout<<"* I : Initial       *"<<endl;
    cout<<"*********************"<<endl;

    std::string aFullPatternImages = ".*.tif", aOriInput, aNameHomol="Homol/", aHomolOutput="_Filtered/", bStrategie = "6";
    double aDistRepr=2, aDistHom=2;
    bool ExpTxt = false;
    ElInitArgMain			//initialize Elise, set which is mandantory arg and which is optional arg
    (
    argc,                   //nb d’arguments
    argv,                   //chaines de caracteres contenants tous les arguments
    //mandatory arguments - arg obligatoires²
    LArgMain()  << EAMC(aFullPatternImages, "Pattern of images to compute",  eSAM_IsPatFile)
                << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri),
    //optional arguments - arg facultatifs
    LArgMain()  << EAM(aNameHomol, "HomolIn", true, "Name of input Homol foler, Homol par default")
                << EAM(ExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
                << EAM(aHomolOutput, "HomolOut" , true, "Output corrected Homologues folder")
                << EAM(bStrategie, "Strategie" , true, "Strategie de filtre les points homols")
                << EAM(aDistRepr, "Dist" , true, "Distant to verify reprojection point")
                << EAM(aDistHom, "DistHom" , true, "Distant to verify triplet")

    );
    if (MMVisualMode) return EXIT_SUCCESS;

    ELISE_fp::AssertIsDirectory(aNameHomol);

    // Initialize name manipulator & files
    std::string aDirImages, aPatImages;
    SplitDirAndFile(aDirImages,aPatImages,aFullPatternImages);

    StdCorrecNameOrient(aOriInput,aDirImages);//remove "Ori-" if needed

    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetImages = *(aICNM->Get(aPatImages));

    ELISE_ASSERT(aSetImages.size()>1,"Number of image must be > 1");
 //============================================================
    std::string anExt = ExpTxt ? "txt" : "dat";

    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(aHomolOutput)
                        +  std::string("@")
                        +  std::string("txt");

    std::string aKHOutDat =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(aHomolOutput)
                        +  std::string("@")
                        +  std::string("dat");

    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aNameHomol)
                       +  std::string("@")
                       +  std::string(anExt);

 //===========================================================


    if (bStrategie == "5")
    {
        vector<string> tempArbeRacine;
        vector<AbreHomol> aAbre = creatAbreFromPattern(aSetImages, aNameHomol, aFullPatternImages, aOriInput);
        tempArbeRacine = displayAbreHomol(aAbre, 0);

        Tiff_Im mTiffImg3(tempArbeRacine[0].c_str());
        Pt2dr centre_img(mTiffImg3.sz().x/2, mTiffImg3.sz().y/2);
        double diag = sqrt(pow(mTiffImg3.sz().x,2) + pow(mTiffImg3.sz().y,2));
        vector< vector<bool> > ColDec;

        double stat = 0;
        double all = 0;
        for (uint i=0;i<aAbre.size();i++)
        {
            cout<<aAbre[i].ImgRacine<<endl;
            string aNameImg1 = aAbre[i].ImgRacine;
            for(uint k=0; k<aAbre[i].ImgBranch.size(); k++)
            {
                cout<<" ++ "<<aAbre[i].ImgBranch[k]<<endl;
                string aNameImg2 = aAbre[i].ImgBranch[k];
                for(uint l=0; l<aAbre[i].Img3eme[k].size(); l++)
                {
                    cout<<"   + Com + "<<aAbre[i].Img3eme[k][l]<<endl;
                    //====Triplet Image==========//
                    aNameImg1 = aAbre[i].ImgRacine;
                    aNameImg2 = aAbre[i].ImgBranch[k];
                    string aNameImg3 = aAbre[i].Img3eme[k][l];
                    cout<<"   + Tri + "<<aNameImg1<<" "<<aNameImg2<<" "<<aNameImg3<<endl;
                    vector<bool> result = FiltreDe3img( aNameImg1,  aNameImg2,  aNameImg3,  aNameHomol,  aDirImages,  aPatImages,  aOriInput,  ExpTxt,  centre_img,  diag,  aDistRepr, aDistHom);
                    ColDec.push_back(result);
                }
                //=====decision=====
                vector<bool> decision; vector<double> decPoint;

                for(uint m=0; m<ColDec[0].size(); m++)
                {
                    bool dec = 0; double point=0;
                    for (uint n=0; n<ColDec.size(); n++)
                    {
                        dec = dec || ColDec[n][m];
                        if (ColDec[n][m])
                        {point++;}
                    }
                    decision.push_back(dec);
                    decPoint.push_back(point);
                }
                //creat homol file with decision and pack homo b/w aNameImg1 aNameImg2
                //.....
                double totalImgCom = aAbre[i].Img3eme[k].size();
                double countVerif=0;
                for(uint o=0; o<decision.size(); o++)
                {
                    if (decision[o] && ((decPoint[o]/totalImgCom) > 0.95) )
                    {countVerif++; decision[o] = 1;}
                    else
                    {decision[o] = 0;}
                }
                cout<<"     *-*-*-*-*-*   "<<" NbPt Filtré = "<<countVerif<< " / "<<decision.size()<<endl<<endl;
                stat = stat + countVerif;
                all = all + decision.size();
                creatHomolFromPair(aNameImg1, aNameImg2, aNameHomol, aDirImages, aPatImages, aHomolOutput, ExpTxt, decision);
            }
        }
        cout<<endl<<endl<<"+-+ "<<stat/all*100<<"%"<<endl;

        //creer abre de couple et des collection de 3eme images correspondant
        //parcourir abre 1 ; prendre image racine
        //parcourir les image branche corresponde with img racine
        //search for abre of image branch current, it's abre 2
        //out list of 2 abre to img3eme correspond with abre 1 abre 2
    }

    //===================================test===========================================
    if (bStrategie == "6")
    {
        VerifParRepr aImgVerif(aSetImages, aDirImages, aPatImages, aNameHomol, aOriInput, aHomolOutput, ExpTxt);
        aImgVerif.creatAbre();
        bool disp = 1;
        aImgVerif.displayAbreHomol(aImgVerif.mAbre, disp);
        aImgVerif.FiltragePtsHomo();
    }
    cout<<"use command SEL ./ img1 img2 KCpl=NKS-Assoc-CplIm2Hom@"<<aHomolOutput<< "@dat to view filtered point homomogues"<<endl;

    return EXIT_SUCCESS;
}




/*strategie à faire:
1)
Homol1_2 => P1 & P2, Homol2_3 => P2 & P3, triplet [P1, P2, P3].
Pt3d P3' reprojeter ves cam3 à partir P1 et P2
P3' = P3 ?
si OK => fabriquer Homol nouvel avec P1 P2 P3 pour tout les sens

2)
Homol1_2 => P1 & P2 => Pt3d ===reprojeter==> P3
Corellation entre P1 et P3
Corellation entre P2 et P3
si OK => garde P1 et P2
=> P3 juste pour validation point homo entre P1 et P2
Q: comment choisir le pose bien pour cam3 ? (avant ou apres cam1 et cam2)
    
3)
Homol cohérent, pas besoins l'orientation du caméra => si camera orientation est pas bonne, on peut eviter
Homol1_2 => P1 & P2, Homol2_3 => P2 & P3, triplet [P1, P2, P3].
P3 => chercher dans Homol3_1 => si trop loin avec P3 => pas bonne ????????????????????????????????????????
P3 => chercher dans Homol3_2 => si trop loin avec P3 => pas bonne couple P2 P3 ???????????????????????????
Comment faire une methode de validation plus efficace ?
*/

