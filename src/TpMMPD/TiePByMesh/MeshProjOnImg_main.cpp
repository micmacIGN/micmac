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

#include "InitOutil.h"
#include <stdio.h>
#include "StdAfx.h"
#include "Triangle.h"
#include "Pic.h"
#include "DrawOnMesh.h"
#include "CorrelMesh.h"
#include "PHO_MI.h"

    /******************************************************************************
    The main function.
    ******************************************************************************/

int MeshProjOnImg_main(int argc,char ** argv)
{
    cout<<"********************************************************"<<endl;
    cout<<"*    		Reproject mesh on specific image		  *"<<endl;
    cout<<"********************************************************"<<endl;

    string pathPlyFileS ;
    string aFullPattern, aOriInput;
    double zoomF = 0.2;
    bool click = false;
    string PtsInteret="NO";string SH="NO";
    bool oriTri = false;

    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri)
                << EAMC(pathPlyFileS, "path to mesh(.ply) file - created by Inital Ori", eSAM_IsExistFile),
                //optional arguments
                LArgMain()
                << EAM(zoomF, "zoomF", true, "1 -> sz origin, 0.2 -> 1/5 size - default = 0.2")
                << EAM(oriTri, "oriTri", true, "consider orientation of triangle/cam")
                << EAM(click, "click", true, "true => draw each triangle by each click - default = false")
                << EAM(PtsInteret, "ptsInteret", true, "pack pts d'interet - give it with 1 img only")
                << EAM(SH, "SH", true, "pack homol - give it with 2 image only")
                );

    if (MMVisualMode) return EXIT_SUCCESS;
    InitOutil aChain(aFullPattern, aOriInput, SH);
    cout<<"Init all.."<<endl;
    aChain.read_file(pathPlyFileS);
    aChain.load_Im();
    aChain.load_tri();
    aChain.reprojectAllTriOnAllImg();

    vector<pic*> ptrPic = aChain.getmPtrListPic();
    vector<triangle*> ptrTri = aChain.getmPtrListTri();
    vector<Video_Win*> VWPic;
    for (uint i=0; i<ptrPic.size(); i++)
    {
        pic * aPic = ptrPic[i];
        Video_Win * aVW = 0;
        VWPic.push_back(display_image
                        (aPic->mPic_Im2D, "IMG" + intToString(i),
                         aVW, zoomF, false));
    }
    bool Exist = false;
    if (PtsInteret != "NO")
    {
        cout<<"Check pts interet file : "<<PtsInteret;
        Exist= ELISE_fp::exist_file(PtsInteret);
    }
    vector<Pt2dr> lstPtsInteret;
    bool ok = false;
    if (Exist)
    {
        cout<<" ...found"<<endl;
        DigeoPoint fileDigeo;
        vector<DigeoPoint> listPtDigeo;
        ok = fileDigeo.readDigeoFile(PtsInteret, 1,listPtDigeo);
        for (uint i=0; i<listPtDigeo.size(); i++)
        {
            lstPtsInteret.push_back(Pt2dr(listPtDigeo[i].x, listPtDigeo[i].y));
        }
        if (!ok)
            cout<<" DIGEO File read error ! "<<endl;
        if  (ptrPic.size() != 1)
        {
            cout<<"ERROR : 1 image for 1 file Pts Interest - not draw pts interet"<<endl;
            ok = false;
        }
    }
    else
        cout<<"...not found !"<<endl;
    vector<Pt2dr> lstPtHomo1;
    vector<Pt2dr> lstPtHomo2;
    bool ExistHomo = false;
    string HomoIn;
    if (SH != "NO")
    {
        string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                           +  std::string(SH)
                           +  std::string("@")
                           +  std::string("dat");
        HomoIn = aChain.getPrivmICNM()->Assoc1To2(aKHIn,
                                         ptrPic[0]->getNameImgInStr(),
                                         ptrPic[1]->getNameImgInStr(), true);
        ExistHomo= ELISE_fp::exist_file(HomoIn);
        if (!ExistHomo)
        {
            cout<<"Pack HOMO not found "<<HomoIn<<endl;
            StdCorrecNameHomol_G(HomoIn, aChain.getPrivmICNM()->Dir());
            HomoIn = aChain.getPrivmICNM()->Assoc1To2(aKHIn,
                                                 ptrPic[0]->getNameImgInStr(),
                                                 ptrPic[1]->getNameImgInStr(), true);
            ExistHomo= ELISE_fp::exist_file(HomoIn);
            if (!ExistHomo)
            {
                cout<<"Pack HOMO not found "<<HomoIn<<endl;
                HomoIn = SH + "/Pastis"+ ptrPic[0]->getNameImgInStr()
                        +"/"+ptrPic[1]->getNameImgInStr()+".dat";
                ExistHomo= ELISE_fp::exist_file(HomoIn);
                if (!ExistHomo)
                    cout<<"Pack HOMO not found "<<HomoIn<<endl;
            }
        }
    }
    if (ExistHomo)
    {
         ElPackHomologue aPackIn;
         aPackIn =  ElPackHomologue::FromFile(HomoIn);
         cout<<" + Found Pack Homo "<<aPackIn.size()<<" pts"<<endl;
         for (ElPackHomologue::const_iterator itP=aPackIn.begin(); itP!=aPackIn.end() ; itP++)
         {
             lstPtHomo1.push_back(itP->P1());
             lstPtHomo2.push_back(itP->P2());
         }
    }
    for (uint i=0; i<ptrPic.size(); i++)
    {
        pic * aPic = ptrPic[i];
        Video_Win * aVW = VWPic[i];
        for (uint j=0; j<ptrTri.size(); j++)
        {
            triangle * aTri = ptrTri[j];
            Pt3dr P1 = aTri->getSommet(0);
            Pt3dr P2 = aTri->getSommet(1);
            Pt3dr P3 = aTri->getSommet(2);

            Tri2d aTri2D = *aTri->getReprSurImg()[aPic->mIndex];
            bool visible =
                    aPic->mOriPic->PIsVisibleInImage(P1) &&
                    aPic->mOriPic->PIsVisibleInImage(P2) &&
                    aPic->mOriPic->PIsVisibleInImage(P3);
            double signTri = (aTri2D.sommet1[0]-aTri2D.sommet1[1])^(aTri2D.sommet1[0]-aTri2D.sommet1[2]);

            if (oriTri == false)
            {
                 if (signTri < 0 && visible)
                    draw_polygon_onVW(aTri2D, aVW, Pt3di(0,255,0), true, click);
            }
            else
            {
                if (signTri > 0 && visible)
                    draw_polygon_onVW(aTri2D, aVW, Pt3di(0,255,0), true, click);
            }

        }
        if (lstPtsInteret.size() > 0)
            draw_pts_onVW(lstPtsInteret,aVW);
        if ((lstPtHomo1.size() > 0) && (lstPtHomo2.size() > 0))
        {
            if (i==0)
                draw_pts_onVW(lstPtHomo1,aVW);
            if (i==1)
                draw_pts_onVW(lstPtHomo2,aVW);
        }
        if (i==ptrPic.size()-1)
            aVW->clik_in();
    }
    return EXIT_SUCCESS;
}



