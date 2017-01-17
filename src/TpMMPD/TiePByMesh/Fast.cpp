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
       11  12 13
     10 -  -  - 14
    9 - -  -  - - 15
    8 - -  16 - - 0
    7 - -  -  - - 1
      6 -  -  - 2
        5  4  3
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
Fast::Fast(double threshold, double radius = 2)
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
    for(int i=this->radius; i<pic.sz().x-this->radius; i++)
    {
        for (int j=this->radius; j<pic.sz().y-this->radius; j++)
        {
            if (this->isCorner( Pt2di(i,j), pic, this->threshold))
            {
                resultCorner.push_back(Pt2dr(i,j));
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
    /* ===== write image to disk // save image // ecrire ecrit image =======
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

FastNew::FastNew(const TIm2D<tPxl, tPxl> & anIm, double threshold, double radius, const TIm2DBits<1> & anMasq):
    mThres     (threshold),
    mRad       (radius),
    mImInit    (anIm._the_im),
    mTImInit   (anIm),
    mImMasq    (anMasq._the_im),
    mTImMasq   (mImMasq)
{
    cout<<"Info FAST : "<<
          endl<< "  ++Img: "<<mImInit.sz()<<" -Thres: "<<mThres<<" -Radius: "<< mRad<<endl;


//    Video_Win * mW;
//    if (mW ==0)
//    {
//         int aZ = 2;
//         mW = Video_Win::PtrWStd(ToPt2di(mImInit.sz()/aZ),true,Pt2dr(1/double(aZ),1/double(aZ)));
//    }
//    cout<<"Win Created!"<<endl;
//    if (mW)
//    {
//         ELISE_COPY(mImInit.all_pts(), mImInit.in(), mW->ogray());
//         mW->clik_in();
//    }

    getVoisinInteret(mRad);
    detect(mTImInit, mTImMasq, mLstPt);

//    for (uint i=0; i<mLstPt.size(); i++)
//    {
//        mW->draw_circle_loc(mLstPt[i],2.0,mW->pdisc()(P8COL::green));
//    }
//    mW->clik_in();
}

void FastNew::sortCWfor_mVoisin(vector<Pt2di> & mVoisin)
{
    deque<Pt2di> aVoisin;
    for (uint aK=0; aK<mRad; aK++)
        aVoisin.push_back(mVoisin[aK]);
    for (uint aK=mRad; aK<mVoisin.size()-mRad; aK=aK+2)
    {
        aVoisin.push_front(mVoisin[aK]);
        aVoisin.push_back(mVoisin[aK+1]);
    }
    for (uint aK=mVoisin.size()-1; aK>=mVoisin.size()-mRad; aK--)
        aVoisin.push_back(mVoisin[aK]);
    mVoisin.clear();
    for (uint aK=0; aK<aVoisin.size(); aK++)
        mVoisin.push_back(aVoisin[aK]);
}


void FastNew::getVoisinInteret(double radius)
{
    Pt2di aP;
    if (radius >= 2)
    {
        for (aP.x=-radius; aP.x<=radius; aP.x++)
        {
            for (aP.y=-radius; aP.y<=radius; aP.y++)
            {
                double dst=euclid(aP);
                if (dst<=radius+0.5 && dst>radius-0.5)
                    mVoisin.push_back(aP);
            }
        }
        sortCWfor_mVoisin(mVoisin);
    }
    else
        cout<<"FAST Err : radius < 2"<<endl;
}

#define tBrighter 2
#define tDarker   1
#define tSame     0

bool FastNew::isContinue(vector<int> & label , int typeExtreme)
{
    for (uint aK=0; aK<label.size(); aK++)
    {
        if (label[aK] == typeExtreme)
        {
            bool isCont = true;
            for (uint offset=0; offset<round(label.size()*3/4); offset++)
            {
                int indexVoisin = aK+offset;
                if (indexVoisin >= int(label.size()))
                    indexVoisin = label.size() - indexVoisin;
                if(label[indexVoisin] != typeExtreme)
                    {isCont = false;break;}
            }
            if (isCont == true)
                return true;
        }
    }
    return false;
}

void FastNew::detect(
                        const TIm2D<tPxl, tPxl> & anIm,
                        const TIm2DBits<1> anMasq,
                        vector<Pt2dr> & lstPt
                    )
{
    Pt2di aP;
    Pt2di szIm = anIm.sz();
    for (aP.x=mRad; aP.x<szIm.x-mRad; aP.x++)
    {
        for (aP.y=mRad; aP.y<szIm.y-mRad; aP.y++)
        {
            if ( anMasq.get(aP) )
            {
                vector<double> diffCentreVoisin;
                vector<int> label;
                double nbBrighter=0; double nbDarker=0; double nbSame=0;
                for (uint aK=0; aK<mVoisin.size(); aK++)
                {
                    double subVal = anIm.get(aP) - anIm.get(aP+mVoisin[aK]);
                    diffCentreVoisin.push_back(subVal);
                    if (subVal > mThres)
                        {nbBrighter++;label.push_back(tBrighter);}
                    else if (subVal < -mThres)
                        {nbDarker++;label.push_back(tDarker);}
                    else
                        {nbSame++;label.push_back(tSame);}
                }
                bool isExtreme = (
                                     nbBrighter > round(diffCentreVoisin.size()*3/4)
                                  || nbDarker   > round(diffCentreVoisin.size()*3/4)
                                 ) ? true : false;
                if (isExtreme)
                {
                    int  typeExtreme = (nbBrighter > diffCentreVoisin.size()*3/4) ? tBrighter : tDarker;
                    if (isContinue(label, typeExtreme))
                        lstPt.push_back(ToPt2dr(aP));
                }
            }
        }
    }
}





