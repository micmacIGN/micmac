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

#include "Pic.h"
#include "Detector.h"
#include <stdio.h>

extern vector<double> parse_dParam(vector<string> dParam);
string aFilePastisIn;

    /******************************************************************************
    The main function.
    ******************************************************************************/
int FAST_main(int argc,char ** argv)
{
    cout<<"**********************************************************"<<endl;
    cout<<"*Detector interest point (FAST, FAST_NEW, DIGEO, EXTREMA)*"<<endl;
    cout<<"**********************************************************"<<endl;
    cout<<"*******************Parameters detector********************"<<endl;
    cout<<"++ default is FAST , dParam=[20,3]  (threshold=20, radius=3pxl (fix))"<<endl;
    cout<<"++ FAST_NEW , dParam=[20,3,3]  (threshold=20, radius=3)"<<endl;
    cout<<"++ DIGEO & EXTREME - param default"<<endl<<endl;

        string aFullPattern;
        string aDirOut = "PtsInteret";
        string aTypeD="FAST";
        vector<string> dParam;
        vector<double> aParamD;
        bool display;
        double aZoomF;
        Pt3di mSzW;
        ElInitArgMain
                (
                    argc,argv,
                    //mandatory arguments
                    LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile),
                    //optional arguments
                    LArgMain()
                    << EAM(aTypeD, "aTypeD", true, "Type detector pts Interet (FAST, DIGEO, EXTREMA)")
                    << EAM(aDirOut, "aDirOut", true, "Output directory for pts interest file, default=PtsInteret")
                    << EAM(dParam, "dParam", true, "detector parameter")
                    << EAM(mSzW, "mSzW", true, "display [x,y,dZoom]")
                    << EAM(aFilePastisIn, "aPastisIn", true, "Input pastis file to draw over image")
                 );

        if (MMVisualMode) return EXIT_SUCCESS;

        if (EAMIsInit(&display) && !EAMIsInit(&aZoomF))
        {
            aZoomF=0.3;
        }
        if (!EAMIsInit(&display) && EAMIsInit(&aZoomF))
        {
            display=true;
        }
        if (!EAMIsInit(&dParam))
        {
            dParam.push_back("20");dParam.push_back("3");
        }
        string aDirImages, aPatIm;
        SplitDirAndFile(aDirImages, aPatIm, aFullPattern); //Working dir, Images pattern
        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
        vector<string>  aSetIm = *(aICNM->Get(aPatIm));
        ELISE_ASSERT(aSetIm.size()>0,"ERROR: No image found!");

        vector<pic*> aPtrListPic;
        for (uint i=0; i<aSetIm.size(); i++)
        {
            pic *aPic = new pic( &aSetIm[i], "NONE" , aICNM, i);
            aPtrListPic.push_back(aPic);
        }

        aParamD = parse_dParam(dParam);
        Video_Win * mW = 0;
        for (uint i=0; i<aPtrListPic.size(); i++)
        {
            pic* aPic=aPtrListPic[i];

            string aPtInteretIn = aICNM->Dir() + "PtsInteret/" +
                             aPic->getNameImgInStr() + "_" +
                             aTypeD + ".dat";
            ELISE_fp::RmFileIfExist(aPtInteretIn);
            vector<Pt2dr> lstPt;

            if (EAMIsInit(&aFilePastisIn) && ELISE_fp::exist_file(aFilePastisIn))
            {
                aPtInteretIn = aICNM->Dir() + aFilePastisIn;
                cout<<"Read Pts From :"<<aPtInteretIn<<endl;
                DigeoPoint fileDigeo;
                vector<DigeoPoint> listPtDigeo;
                bool ok = fileDigeo.readDigeoFile(aPtInteretIn, 1,listPtDigeo);
                for (uint i=0; i<listPtDigeo.size(); i++)
                {
                    lstPt.push_back(Pt2dr(listPtDigeo[i].x, listPtDigeo[i].y));
                }
                if (!ok)
                    cout<<" DIGEO File read error ! "<<endl;
                cout<<"Nb : "<<listPtDigeo.size()<<endl;
            }
            else
            {
                if(!EAMIsInit(&aFilePastisIn))
                {
                    Detector * aDecPic = new Detector( aTypeD, aParamD, aPic, aICNM );
                    aDecPic->detect();
                    aDecPic->getmPtsInterest(lstPt);
                    cout<<" ++ "<<aTypeD<<" : "<<lstPt.size()<<" pts detected!"<<endl;
                }
                else
                {
                    cout<<aFilePastisIn<<" not found !"<<endl;
                }
            }
            if (EAMIsInit(&mSzW))
            {
                if (aPic->mImgSz.x >= aPic->mImgSz.y)
                {
                    double scale =  double(aPic->mImgSz.x) / double(aPic->mImgSz.y) ;
                    mSzW.x = mSzW.x;
                    mSzW.y = round_ni(mSzW.x/scale);
                }
                else
                {
                    double scale = double(aPic->mImgSz.y) / double(aPic->mImgSz.x);
                    mSzW.x = round_ni(mSzW.y/scale);
                    mSzW.y = mSzW.y;
                }
                Pt2dr aZ(double(mSzW.x)/double(aPic->mImgSz.x) , double(mSzW.y)/double(aPic->mImgSz.y) );

                if (mW ==0)
                {
                    mW = Video_Win::PtrWStd(Pt2di(mSzW.x*mSzW.z, mSzW.y*mSzW.z), true, aZ*mSzW.z);
                    mW->set_sop(Elise_Set_Of_Palette::TheFullPalette());
                }
                if (mW)
                {
                    std::cout << "draw pt on Windows \n";
                    mW->set_title(aPic->getNameImgInStr().c_str());
                    ELISE_COPY(aPic->mPic_Im2D->all_pts(), aPic->mPic_Im2D->in(), mW->ogray());
                    for (uint aK=0; aK<lstPt.size(); aK++)
                    {
                        mW->draw_circle_loc(lstPt[aK],1.5,mW->pdisc()(P8COL::green));
                    }
                    mW->clik_in();

                }
            }
            return EXIT_SUCCESS;
    }
        return EXIT_SUCCESS;
}
