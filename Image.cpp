#include "StdAfx.h"
#include "Image.h"
// define  a class where the colour is stored
/********************************************************************/
/*                                                                  */
/*             ColorImg                                             */
/*                                                                  */
/********************************************************************/
// Methods are to be specified


ColorImg::ColorImg(std::string filename) :
  mImgName(filename)
{
    Tiff_Im mTiffImg(mImgName.c_str());
    //GenIm::type_el aType = mTiffImg.type_el();

   // std::cout<<"Image channels "<<mTiffImg.nb_chan()<<endl;

   /* cout << "Types = "
         << (INT) aType <<  " "
         << (INT) GenIm::int2 <<  " "
         << (INT) GenIm::u_int2 << "\n";*/


    this->setChannels(mTiffImg.nb_chan());
    mImgSz.x=mTiffImg.sz().x;
    mImgSz.y=mTiffImg.sz().y;
    mImgR=new Im2D<U_INT2,INT4>(mImgSz.x,mImgSz.y);
    mImgG=new Im2D<U_INT2,INT4>(mImgSz.x,mImgSz.y);
    mImgB=new Im2D<U_INT2,INT4>(mImgSz.x,mImgSz.y);
    mImgRT=new TIm2D<U_INT2,INT4>(*mImgR);
    mImgGT=new TIm2D<U_INT2,INT4>(*mImgG);
    mImgBT=new TIm2D<U_INT2,INT4>(*mImgB);
    ELISE_COPY(mImgR->all_pts(),mTiffImg.in(),mImgR->out());
    ELISE_COPY(mImgG->all_pts(),mTiffImg.in(),mImgG->out());
    ELISE_COPY(mImgB->all_pts(),mTiffImg.in(),mImgB->out());
}


ColorImg::ColorImg(Pt2di sz) :
  mImgName(""),
  mImgSz(sz)
{
    mImgR=new Im2D<U_INT2,INT4>(mImgSz.x,mImgSz.y);
    mImgG=new Im2D<U_INT2,INT4>(mImgSz.x,mImgSz.y);
    mImgB=new Im2D<U_INT2,INT4>(mImgSz.x,mImgSz.y);
    mImgRT=new TIm2D<U_INT2,INT4>(*mImgR);
    mImgGT=new TIm2D<U_INT2,INT4>(*mImgG);
    mImgBT=new TIm2D<U_INT2,INT4>(*mImgB);
}

ColorImg::~ColorImg()
{
    delete mImgR;
    delete mImgG;
    delete mImgB;
    delete mImgRT;
    delete mImgGT;
    delete mImgBT;
}

Color ColorImg::get(Pt2di pt) // the method get() return an objet "color" point
{
    return Color(mImgRT->get(pt,0),mImgGT->get(pt,0),mImgBT->get(pt,0));
}

Color ColorImg::getr(Pt2dr pt)
{
                                // get (pt, 0) est plus robuste que get (pt), retourne 0 si le point est hors images
        return Color(mImgRT->getr(pt,0),mImgGT->getr(pt,0),mImgBT->getr(pt,0));
}


int ColorImg::getChannnels()
{
    return mChannels;
}


void ColorImg::setChannels(int chan)
{
 mChannels=chan;
}

void ColorImg::set(Pt2di pt, Color color)
{
    U_INT2 ** aImRData=mImgR->data();
    U_INT2 ** aImGData=mImgG->data();
    U_INT2 ** aImBData=mImgB->data();
    aImRData[pt.y][pt.x]=color.r();
    aImGData[pt.y][pt.x]=color.g();
    aImBData[pt.y][pt.x]=color.b();
}


void ColorImg::write(std::string filename)
{
    if (this->getChannnels()>1)
    {
        ELISE_COPY
        (
            mImgR->all_pts(),
            Virgule( mImgR->in(), mImgG->in(), mImgB->in()) ,
            Tiff_Im(
                filename.c_str(),
                mImgSz,
                GenIm::u_int2,
                Tiff_Im::No_Compr,
                Tiff_Im::RGB,
                Tiff_Im::Empty_ARG ).out()
        );
    }

    else
    {
        ELISE_COPY
        (
            mImgG->all_pts(),
            mImgG->in() ,
            Tiff_Im(
                filename.c_str(),
                mImgSz,
                GenIm::u_int2,
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero,
                Tiff_Im::Empty_ARG ).out()
        );
    }
}


ColorImg  ColorImg::ResampleColorImg(double aFact)
{
   Pt2di aSzR = round_up(Pt2dr(mImgSz)/aFact);

   ColorImg aResampled(aSzR);

   Fonc_Num aFInR = StdFoncChScale
                 (
                       this->mImgR->in_proj(),
                       Pt2dr(0,0),
                       Pt2dr(aFact,aFact)
                 );
  Fonc_Num aFInG = StdFoncChScale(this->mImgG->in_proj(),Pt2dr(0,0),Pt2dr(aFact,aFact));
  Fonc_Num aFInB = StdFoncChScale(this->mImgB->in_proj(),Pt2dr(0,0),Pt2dr(aFact,aFact));

    ELISE_COPY(aResampled.mImgR->all_pts(),aFInR,aResampled.mImgR->out());
    ELISE_COPY(aResampled.mImgG->all_pts(),aFInG,aResampled.mImgG->out());
    ELISE_COPY(aResampled.mImgB->all_pts(),aFInB,aResampled.mImgB->out());
   return aResampled;
}
