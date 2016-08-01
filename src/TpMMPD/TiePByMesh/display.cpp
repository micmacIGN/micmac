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

#include <stdio.h>
#include "StdAfx.h"
#include "Triangle.h"
#include "Pic.h"
#include "../kugelhupf.h"
    /******************************************************************************
    Display function.
    ******************************************************************************/

Video_Win *mW, *mW1, *mWImgetGlobMas, *mWImagetGlobSlav, *mW6, *mW7;

/*=====Affiche 2 fenetre pour imagette maitre et second======
 *1. Affiche en 2 cas: fenetre sans pts interet et fenetre avec pts interet
 *2. *mWImgetGlobMas, *mWImagetGlobSlav => fenetre sans pts interet
 *3. *mW, *mW1 : fentre avec pts interet
 *4. Si il y a deja une fentre ouvert avec meme pointer => check taille.
 *5. Si taille du fenetre = taille du imagette => garder la fenetre et affichier nouveau imagette
*/
void display(cCorrelImage * Imgette_Maitre, cCorrelImage * Imagette_2nd, bool imagetteGlobal)
{
    int zoomF=2;
    if (imagetteGlobal)
    {   //imagette global
        cout<<"check display sz: "<<Imgette_Maitre->getIm()->sz()<<" "<<Imagette_2nd->getIm()->sz()<<endl;
        if (mWImgetGlobMas==0)
            {mWImgetGlobMas = Video_Win::PtrWStd(Imgette_Maitre->getIm()->sz()*zoomF, 1, Pt2dr(zoomF,zoomF));}
        if (mWImgetGlobMas != 0)
        {
            if((mWImgetGlobMas->sz().x/zoomF) != Imgette_Maitre->getIm()->sz().x)
            {
                mWImgetGlobMas=0;
                mWImgetGlobMas = Video_Win::PtrWStd(Imgette_Maitre->getIm()->sz()*zoomF, 1, Pt2dr(zoomF,zoomF));
            }
        }
        mWImgetGlobMas->set_title("Imgette Global");
        mWImgetGlobMas->clear();
        ELISE_COPY(mWImgetGlobMas->all_pts(), Imgette_Maitre->getIm()->in_proj()  ,mWImgetGlobMas->ogray());
        if (mWImagetGlobSlav==0)
        {
            mWImagetGlobSlav = new Video_Win(*mWImgetGlobMas,Video_Win::eDroiteH,Imagette_2nd->getIm()->sz()*zoomF);
        }
        if (mWImagetGlobSlav != 0)
        {
            if((mWImagetGlobSlav->sz().x/zoomF) != Imgette_Maitre->getIm()->sz().x)
            {
                mWImagetGlobSlav=0;
                mWImagetGlobSlav = new Video_Win(*mWImgetGlobMas,Video_Win::eDroiteH,Imagette_2nd->getIm()->sz()*zoomF);
            }
        }
        mWImagetGlobSlav->clear();
        ELISE_COPY(mWImagetGlobSlav->all_pts(), Imagette_2nd->getIm()->in()[Virgule(FX/zoomF,FY/zoomF)],mWImagetGlobSlav->ogray());
        mWImagetGlobSlav->clik_in();
    }
    else
    {   //imagette pour pts interet
        cout<<"check display sz: "<<Imgette_Maitre->getIm()->sz()<<" "<<Imagette_2nd->getIm()->sz()<<endl;
        if (mW==0)
        {
            mW = Video_Win::PtrWStd(Imgette_Maitre->getIm()->sz()*zoomF,1,Pt2dr(zoomF,zoomF));
        }
        if (mW != 0)
        {
            if((mW->sz().x/zoomF) > Imgette_Maitre->getIm()->sz().x)
            {
                mW=0;
                mW = Video_Win::PtrWStd(Imgette_Maitre->getIm()->sz()*zoomF,1,Pt2dr(zoomF,zoomF));
            }
        }
        mW->set_title("Imagette of pts interest");
        mW->clear();
        ELISE_COPY(mW->all_pts(), Imgette_Maitre->getIm()->in_proj() ,mW->ogray());
        if (mW1==0)
        {
            mW1 = new Video_Win(*mW,Video_Win::eDroiteH,Imagette_2nd->getIm()->sz()*zoomF);
        }
        if (mW1 != 0)
        {
            if((mW1->sz().x/zoomF) > Imgette_Maitre->getIm()->sz().x)
            {
                mW1=0;
                mW1 = new Video_Win(*mW,Video_Win::eDroiteH,Imagette_2nd->getIm()->sz()*zoomF);
            }
        }
        mW1->clear();
        ELISE_COPY(mW1->all_pts(), Imagette_2nd->getIm()->in()[Virgule(FX/zoomF,FY/zoomF)] ,mW1->ogray());
        mW1->clik_in();
    }
}

/*=====Affichier triangle et pts d'interet bleu sur image=========
 * 1. zoomF pour réduire résolution d'image (choisir en respect résolution d'ecran)
 * 2. Affiche 2 image maitre et slave en 2 fenetre separer mW6 mW7
 * 3. Affiche pts interet sur image maitre
 * 4. Affiche triangle process sur 2 image
 * 5. click sur image slave pour continue le programme
*/
extern void dispTriSurImg(Tri2d TriMaitre, pic * ImgMaitre ,Tri2d Tri2nd, pic * Img2nd, Pt2dr centre, double size, vector<Pt2dr> & listPtsInteret, bool dispAllPtsInteret = false)
{
    int zoomF = 5;
    //double scale = 1/zoomF;
    Disc_Pal Pdisc = Disc_Pal::P8COL();
    Gray_Pal Pgr (30);
    Circ_Pal Pcirc = Circ_Pal::PCIRC6(30);
    RGB_Pal Prgb (255,1,1);
    Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));
    Line_St lstLineG(Pdisc(P8COL::green),1);
    Line_St lstLineB(Pdisc(P8COL::blue),1);
    Line_St lstLineR(Pdisc(P8COL::red),1);

    cout<<"Verif triangle 1 "<<TriMaitre.sommet1[0]<<TriMaitre.sommet1[1]<<TriMaitre.sommet1[2]<<endl;
    cout<<"Verif triangle 2 "<<Tri2nd.sommet1[0]<<Tri2nd.sommet1[1]<<Tri2nd.sommet1[2]<<endl;
    vector<Pt2dr> sommetMaitre;
    sommetMaitre.push_back(TriMaitre.sommet1[0]);
    sommetMaitre.push_back(TriMaitre.sommet1[1]);
    sommetMaitre.push_back(TriMaitre.sommet1[2]);
    vector<Pt2dr> sommet2nd;
    sommet2nd.push_back(Tri2nd.sommet1[0]);
    sommet2nd.push_back(Tri2nd.sommet1[1]);
    sommet2nd.push_back(Tri2nd.sommet1[2]);
    if (mW6==0)
    {
        mW6 = Video_Win::PtrWStd(ImgMaitre->mPic_Im2D->sz()/zoomF, 1, Pt2dr(0.2,0.2));   //coherent avec zoomF
        //mW6 = Video_Win::PtrWStd(ImgMaitre->pic_Im2D->sz()/zoomF, 1); //pas reduit la taille image
        mW6->set_title("Image Master");
    }
    mW6->clear();
    //ELISE_COPY(mW6->all_pts(), ImgMaitre->pic_Im2D->in()[Virgule(FX*zoomF, FY*zoomF)] ,mW6->ogray());
    ELISE_COPY(mW6->all_pts(), ImgMaitre->mPic_Im2D->in_proj() ,mW6->ogray());


    if (mW7==0)
    {
        mW7 = Video_Win::PtrWStd(Img2nd->mPic_Im2D->sz()/zoomF, 1, Pt2dr(0.2,0.2));
       // mW7 = Video_Win::PtrWStd(Img2nd->pic_Im2D->sz()/zoomF, 1);
        mW7->set_title("Image 2nd");
    }
    mW7->clear();
    //ELISE_COPY(mW7->all_pts(), Img2nd->pic_Im2D->in()[Virgule(FX*zoomF, FY*zoomF)] ,mW7->ogray());
    ELISE_COPY(mW7->all_pts(), Img2nd->mPic_Im2D->in_proj() ,mW7->ogray());
    mW6->set_sop(SOP);    mW7->set_sop(SOP);
    if (dispAllPtsInteret)
    {
        for (uint i=0; i<listPtsInteret.size(); i++)
        {
            Pt2dr ptsDraw(listPtsInteret[i].x, listPtsInteret[i].y);
            mW6->draw_circle_loc(listPtsInteret[i], 2 ,lstLineB);
        }
    }
    mW6->draw_poly(sommetMaitre, lstLineG, 1);
    mW6->draw_rect((centre - Pt2dr(size,size)), (centre + Pt2dr(size,size)), lstLineR);
    mW7->draw_poly(sommet2nd, lstLineG, 1);
    mW7->clik_in();

}

/*=====Affichier pts d'interet Green sur imaget maitre et dans le triangle process d'image maitre=========
 * 1. aVWin est pointer au fenetre à dessiner les pts (vers imagette maitre)
 * 2. mW6 est fenetre image maitre
*/
void dispPtsSurImageMaster(const Pt2dr pts, Pt2dr centre , double SzDemiCote, Video_Win * aVWin,  bool dispSurImageGrand)
{
    Disc_Pal Pdisc = Disc_Pal::P8COL();
    Gray_Pal Pgr (30);
    Circ_Pal Pcirc = Circ_Pal::PCIRC6(30);
    RGB_Pal Prgb (255,1,1);
    Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));
    Line_St lstLineG(Pdisc(P8COL::green),1);
    Line_St lstLineB(Pdisc(P8COL::blue),1);
    Line_St lstLineR(Pdisc(P8COL::red),1);

    aVWin->set_sop(SOP);    mW6->set_sop(SOP);

    Pt2dr ptsOnImagette = pts - (centre - Pt2dr(SzDemiCote,SzDemiCote));
    cout<<ptsOnImagette<<endl;
    aVWin->draw_circle_loc(ptsOnImagette, 1, lstLineG);   //draw pts interet on Imgette Maitre Global
    if (dispSurImageGrand)
        mW6->draw_circle_loc(pts, 2, lstLineG);             //draw pts on image matraisse
}

/*=====Affichier pts d'interet bleu sur image mW6 (image maitre)=========
*/
void dispAllPtsInteret(vector<Pt2dr> listPtsInteret)
{
    /*===Display all pts in listPtsInteret on image mW6 ====*/
    Disc_Pal Pdisc = Disc_Pal::P8COL();
    Gray_Pal Pgr (30);
    Circ_Pal Pcirc = Circ_Pal::PCIRC6(30);
    RGB_Pal Prgb (255,1,1);
    Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));
    Line_St lstLineG(Pdisc(P8COL::green),1);
    Line_St lstLineB(Pdisc(P8COL::blue),1);
    Line_St lstLineR(Pdisc(P8COL::red),1);
    mW6->set_sop(SOP);
    for (uint i=0; i<listPtsInteret.size(); i++)
    {
        Pt2dr ptsDraw(listPtsInteret[i].x, listPtsInteret[i].y);
        mW6->draw_circle_loc(ptsDraw, 2 ,lstLineB);
    }
}
