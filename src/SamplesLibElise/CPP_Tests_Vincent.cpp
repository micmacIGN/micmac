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

// Main header in which a lot of libraries are included
#include "StdAfx.h"

// List  of classes

class cCMP_Ima;
class cCMP_Appli;

class cCMP_Ima
{
    public :
        cCMP_Ima(cCMP_Appli & anAppli,const std::string & aName, const std::string & aOri);
        Pt3dr ManipImage();
    //private :
        cCMP_Appli &   mAppli;
        CamStenope *    mCam;
        std::string mNameOri;
        std::string     mName;
};

class cCMP_Appli
{
    public :
		// Main function
        cCMP_Appli(int argc, char** argv);
        cInterfChantierNameManipulateur * ICNM() const {return mICNM;}
        std::string NameIm2NameOri(const std::string &, const std::string &) const;
    private :
        std::string mOri1;
        std::string mOri2;
        std::string mOut;
        std::string mFullName;
        std::string mPat1;
        std::string mPat2;
        std::string mDir1;
        std::string mDir2;
        std::list<std::string> mLFile1;
        std::list<std::string> mLFile2;
        cInterfChantierNameManipulateur * mICNM;
        std::vector<cCMP_Ima *>          mIms1;
        std::vector<cCMP_Ima *>          mIms2;
        // const std::string & Dir() const {return mDir;}
};

/********************************************************************/
/*                                                                  */
/*         cCMP_Ima                                                 */
/*                                                                  */
/********************************************************************/

cCMP_Ima::cCMP_Ima(cCMP_Appli & anAppli,const std::string & aName, const std::string & aOri) :
   mAppli  (anAppli),
   mName   (aName)
{
    mNameOri  = mAppli.NameIm2NameOri(mName,aOri);
    mCam      = CamOrientGenFromFile(mNameOri,mAppli.ICNM());

}

Pt3dr cCMP_Ima::ManipImage()
{
    Pt3dr mCentreCam = mCam->VraiOpticalCenter();
    return mCentreCam;
}


/********************************************************************/
/*                                                                  */
/*         cCMP_Appli                                               */
/*                                                                  */
/********************************************************************/


cCMP_Appli::cCMP_Appli(int argc, char** argv)
{
    // Initialisation of the arguments

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pat)")
                    << EAMC(mOri1,"First Orientation")
                    << EAMC(mOri2,"Second Orientation"),
        LArgMain()  << EAM(mOut,"Out",true,"Output result file")
    );

    // Initialize name manipulator & files
    SplitDirAndFile(mDir1,mPat1,mFullName);

    cout << "Ori1 : " << mOri1 << "\tOri2 : " << mOri2 << "\tOut : " << mOut << endl;
    // Get the list of files from the directory and pattern
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir1);
    mLFile1 = mICNM->StdGetListOfFile(mPat1);
	// If the users enters Ori-MyOrientation/, it will be corrected into MyOrientation
    StdCorrecNameOrient(mOri1,mDir1);

    SplitDirAndFile(mDir2,mPat2,mFullName);
    // Get the list of files from the directory and pattern
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir2);
    mLFile2 = mICNM->StdGetListOfFile(mPat2);
	// If the users enters Ori-MyOrientation/, it will be corrected into MyOrientation
    StdCorrecNameOrient(mOri2,mDir2);

    if (mOut == ""){mOut=mOri1+"_"+mOri2+".txt";};

    for (
              std::list<std::string>::iterator itS=mLFile1.begin();
              itS!=mLFile1.end();
              itS++
              )
     {
           cCMP_Ima * aNewIm1 = new  cCMP_Ima(*this,*itS,mOri1);
           cCMP_Ima * aNewIm2 = new  cCMP_Ima(*this,*itS,mOri2);

           mIms1.push_back(aNewIm1);
           mIms2.push_back(aNewIm2);
     }

     Pt3dr mDiffCentre;
     vector <Pt3dr> mVDCentre;
     for (unsigned int i = 0 ; i < mIms1.size() ; i++)
     {
        if (mIms1[i]->mName != mIms2[i]->mName) { cout << "!!!!!!!!! NOMS D'IMAGES INCOHÉRENTS !!!!!!!!!" << endl;}
        else
        {
           Pt3dr mCentre1 = mIms1[i]->ManipImage();
           Pt3dr mCentre2 = mIms2[i]->ManipImage();
           mDiffCentre = mCentre1 - mCentre2;
           cout << "Image : " << mIms1[i]->mName << " " << mDiffCentre << endl;
           mVDCentre.push_back(mDiffCentre);
        }
     }
/*
double dX(0),dY(0),dZ(0);
     for (unsigned int i = 0 ; i < mVDCentre.size() ; i++)
     {

        cout << "dX = " << dX  << "\tdY = " << dY << "\tdZ = " << dZ << endl;
        dX = dX + abs(mVDCentre[i].x);
        dY = dY + abs(mVDCentre[i].y);
        dZ = dZ + abs(mVDCentre[i].z);
     }
     cout << "Ecarts moyen absolus en X : " << dX/mVDCentre.size()
          << "\t en Y : " << dY/mVDCentre.size()
          << "\ten Z : " << dZ/mVDCentre.size() << endl;
*/
}


std::string cCMP_Appli::NameIm2NameOri(const std::string & aNameIm, const std::string & aOri) const
{
    return mICNM->Assoc1To1
    (
        "NKS-Assoc-Im2Orient@-"+aOri+"@",
        aNameIm,
        true
    );
}


/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/

// Main exercise
int Vincent_main1(int argc, char** argv)
{
/* Retourne, à partir de 2 orientations, les différences en X, Y, Z associées à chaque caméra */
   cCMP_Appli anAppli(argc,argv);

   return EXIT_SUCCESS;
}

int Vincent_main(int argc, char** argv)
{
/* Transforme résidus de GCPBascule (utiliser GCPBasc ... | tee myFile.txt) en un fichier "NamePt dX dY dZ sigmaX sY sZ eMoyPixel eMaxPixel")*/
    string mNameIn, mNameOut("");
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameIn,"Name of the residuals file"),
        LArgMain()  << EAM(mNameOut,"Out",true,"File to save the results")
    );
    ifstream fin (mNameIn.c_str());
    string mRead;
    int i=0;
    ofstream fout (mNameOut.c_str(),ios::out);
    while (fin >> mRead){

        if (mRead == "--NamePt"){i++;}
        if (i!=0){i++;}
        if (i==3){fout << mRead << " ";}
        if (i==6){fout << mRead << " ";}
        if (i==13){fout << mRead << " ";}
        if (i==20){fout << mRead << " ";}
        if (i==25){
            fout << mRead << "\n";}
        if (i!=0){cout << mRead << " at " << i << endl;}
        if (mRead == "For"){i=0;}
    }
    fout.close();
   return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
