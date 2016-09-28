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


#include "../../uti_phgrm/NewOri/NewOri.h"
#include "Fast.h"

//using namespace cv;

const static Pt2di pxlPosition[17]=
{
/*
       11 12 13
     10 - -  - 14
    9 - - -  - - 15
    8 - - 16 - - 0
    7 - - -  - - 1
      6 - -  - 2
        5 4  3
*/
    Pt2di(0,3),	//0
    Pt2di(1,3),
    Pt2di(2,2),
    Pt2di(3,1),
    Pt2di(3,0),
    Pt2di(3,-1),
    Pt2di(2,-2),
    Pt2di(1,-3),
    Pt2di(0,-3),
    Pt2di(-1,-3),
    Pt2di(-2,-2),
    Pt2di(-3,-1),
    Pt2di(-3,0),
    Pt2di(-3,1),
    Pt2di(-2,2),
    Pt2di(-1,3),
    Pt2di(0,0)	//16
};

Video_Win *mWTest;
Fast::Fast(double threshold, int radius = 1)
{
    this->threshold = threshold;
    this->radius = radius;
}

bool Fast::isCorner(Pt2di pxlCenter, Im2D<unsigned char, int> &pic, double & threshold)
{
    bool valid;
    int nbpxlValid = 0;
    vector <int> pxlValidIndx;
    double valPxlCenter =   pic.GetI(pxlCenter);
    for (uint i=0; i<16; i++)
    {
        Pt2di pxlAutourCurrent = pxlCenter.operator +(pxlPosition[i]);
        if ( abs(pic.GetI(pxlAutourCurrent)-valPxlCenter) > threshold)
        {
            nbpxlValid++;
            pxlValidIndx.push_back(i);
        }
    }
    if (nbpxlValid > 12 && isConsecutive(pxlValidIndx))
        valid = true;
    else
    {
        /* GIANG Strategie */
//        if (nbpxlValid > 8 && isConsecutive(pxlValidIndx))
//            {valid = true;}
//        else
//            {valid = false;}

        /* Autre Strategie */
        valid = false;
    }
    return valid;
}

bool Fast::isConsecutive(vector<int> &pxlValidIndx)
{
    bool validCW = true;
    //check if all pixel is consecutive, sens CW
    for (uint i=0; i<pxlValidIndx.size(); i++)
    {
        if ((pxlValidIndx[i] - pxlValidIndx[i+1]) != 1)
        {
            validCW = false;
            break;
        }
    }
    bool validCCW = true;
    //check if all pixel is consecutive, sens CW
    for (uint i=pxlValidIndx.size()-1; i>0; i--)
    {
        if ((pxlValidIndx[i] - pxlValidIndx[i-1]) != 1)
        {
            validCCW = false;
            break;
        }
    }
    return (validCW || validCCW);
}

void Fast::detect(Im2D<U_INT1,INT4> &pic, std::vector<Pt2dr> &resultCorner)
{
    //cv::KeyPoint aPts;
    for(int i=this->radius; i<pic.sz().x-this->radius; i++)
    {
        for (int j=this->radius; j<pic.sz().y-this->radius; j++)
        {
            if (this->isCorner( Pt2di(i,j), pic, this->threshold))
            {
                resultCorner.push_back(Pt2dr(i,j));
                //aPts.pt.x=i; aPts.pt.y=j;
                //keypoints_FASTG.push_back(aPts);
            }
        }
    }
}

void Fast::outElPackHomo(vector<Pt2dr> &packIn, ElPackHomologue & packOut)
{
    for (uint i=0; i<packIn.size(); i++)
    {
        Pt2dr pts(packIn[i].x, packIn[i].y);
        ElCplePtsHomologues aCouple(pts,pts);
        packOut.Cple_Add(aCouple);
    }
}

void Fast::dispPtIntertFAST(Im2D<U_INT1,INT4> pic, vector<Pt2dr> pts, int zoomF, string filename)
{
    Disc_Pal Pdisc = Disc_Pal::P8COL();
    Gray_Pal Pgr (30);
    Circ_Pal Pcirc = Circ_Pal::PCIRC6(30);
    RGB_Pal Prgb (255,1,1);
    Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));
    Line_St lstLine(Pdisc(P8COL::green),1);
    if (mWTest == 0)
        {mWTest = Video_Win::PtrWStd(pic.sz()/int(2), 1, Pt2dr(0.5,0.5)); }
    mWTest->clear();
    mWTest->set_title("FAST Point");
    mWTest->set_sop(SOP);
    ELISE_COPY(mWTest->all_pts(), pic.in_proj() ,mWTest->ogray());
    for (uint i=0; i<pts.size(); i++)
        {
            Pt2dr thisPts(pts[i].x, pts[i].y);
            mWTest->draw_circle_loc( thisPts , 2, lstLine);
        }
    /* ===== write image to disk =======
    string fileOut = filename + "_FASTDETECTEUR";
    ELISE_COPY
    (
        mWTest->all_pts(),
        mWTest->inside() ,
        Tiff_Im(
            fileOut.c_str(),
            mWTest->sz(),
            GenIm::u_int1,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero,
            Tiff_Im::Empty_ARG ).out()
    );
    */
}


