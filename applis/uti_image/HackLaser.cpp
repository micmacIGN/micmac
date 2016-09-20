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
#include "general/all.h"
#include "private/all.h"
#include <algorithm>

/*
    mF3 :

      ZZZZ : 2211149
      Sz    :  564  x 3914
               565 x 3914 = 2211410
       2211149 - 2211410 =  261
*/

/*
   SDC
[30][3][0][0][5][0][2][0]packed record
TIME        : DOUBLE;   // seconds of the day or of the week, depending on GPS string
RANGE       : SINGLE;   // range value of point in scanner's own coordinate system
THETA       : SINGLE;   // theta value
X, Y, Z     : SINGLE;   // x, y, z value of point in scanner's own coordinate system
AMPLITUDE   : WORD;     // linearized amplitude
WIDTH       : WORD;     // width of target return
TARGETTYPE  : BYTE;     // 0 .. COG, 1 .. PAR, 2 .. GAUSS
TARGET      : BYTE;     // index of target (1 .. NUMTARGET)
NUMTARGET   : BYTE;     // total number of targets
RGINDEX     : WORD;     // range gate index of measurement
CHANNELDESC : BYTE;     // channel descriptor (Bit6: 0=low, 64=high power channel)
CLASSID     : BYTE;     // class identifier (ground, vegatation, ...)
end
*/

#define OFSET_SDC  798

const int SzElSDC = sizeof(double)+5*sizeof(float) + 3 * sizeof(short) + 5 * sizeof(unsigned char);

struct cElSDC
{
     void ReadOnFile(ELISE_fp &  aFile);
     double  mTime;
     float   mRange;
     float   mTheta;
     float   mX;
     float   mY;
     float   mZ;
     short   mAmpl;
     short   mWidth;
     unsigned char mType;
     unsigned char mTarget;
     unsigned char mNbTarget;
     short         mIndex;
     unsigned char mChDesc;
     unsigned char mClassif;

     
};

void cElSDC::ReadOnFile(ELISE_fp &  aFile)
{
   mTime  = aFile.read_REAL8();
   mRange = aFile.read_REAL4();
   mTheta = aFile.read_REAL4();
   mX     = aFile.read_REAL4();
   mY     = aFile.read_REAL4();
   mZ     = aFile.read_REAL4();
   mAmpl  = aFile.read_U_INT2();
   mWidth = aFile.read_U_INT2();
   mType  = aFile.read_U_INT1();
   mTarget = aFile.read_U_INT1();
   mNbTarget = aFile.read_U_INT1();
   mIndex    = aFile.read_U_INT2();
   mChDesc   = aFile.read_U_INT1();
   mClassif  = aFile.read_U_INT1();
}

class cSdrSdc
{
    public :
        cSdrSdc
        (
            const std::string & aName,
            bool SauteEntete
        ) :
           mName (aName),
           mSDF  ((aName+".sdf").c_str(),ELISE_fp::READ),
           mSDC  ((aName+".sdc").c_str(),ELISE_fp::READ),
           mSzF  ( sizeofile((aName+".sdf").c_str())),
           mSzC  ( sizeofile((aName+".sdc").c_str())),
           mNbElC ((mSzC-OFSET_SDC) / SzElSDC)
        {
            std::cout << mSzC << " " << (OFSET_SDC + SzElSDC * mNbElC) << "\n";


        }

        void Inspect_FW();

        void TestSDC();
        void TestSDF();
        ELISE_fp & FP(bool sdf)
        {
              return sdf ? mSDF : mSDC;
        }
        void InitSDC()
        {
             mSDC.seek(OFSET_SDC,ELISE_fp::sbegin);
             mLast.ReadOnFile(mSDC);
        }
        void  MakeImage();
        void  MakeIndexe_FW(Pt2di aSz);
    private:

        std::string mName;
        ELISE_fp  mSDF;
        ELISE_fp  mSDC;
        int       mSzF;
        int       mSzC;
        int       mNbElC;
        cElSDC    mCur;
        cElSDC    mLast;
};

void cSdrSdc::Inspect_FW()
{
     Pt2di aSzWIm(100,100);
     Pt2di aPMil = aSzWIm / 2;
     Tiff_Im aFRange = Tiff_Im::StdConv(mName+ "_RangeShade.tif");

     Tiff_Im aFInd = Tiff_Im::StdConv(mName+ "_IndxFW.tif");
     Pt2di aSzInd = aFInd.sz();
     Im2D_INT4 aImInd(aSzInd.x,aSzInd.y);
     ELISE_COPY(aFInd.all_pts(),aFInd.in(),aImInd.out());

     Video_Win aWIm = Video_Win::WStd(aSzWIm,3.0);
     
     Pt2di aSzWFw(500,260);
     Video_Win aWFw(aWIm,Video_Win::eBasG,aSzWFw);

      while (1)
      {
           int x,y;
           cin >> x  >> y;

           Pt2di aP0 = Pt2di(x,y) - aPMil;
           ELISE_COPY
           (
                aWIm.all_pts(),
                trans(aFRange.in(0),aP0),
                aWIm.orgb()
           );

            int aB=1;
            while (aB==1)
            {
               Clik aCl = aWIm.clik_in();
               aB = aCl._b;
               aWIm.draw_circle_loc(aCl._pt,1,aWIm.pdisc()(P8COL::red));
               Pt2di aPIm = aCl._pt + aP0;

               if ((aPIm.x>=0) && (aPIm.y>=0)&& (aPIm.x<aSzInd.x)&& (aPIm.y<aSzInd.y))
               {
                  int aInd = aImInd.data()[aPIm.y][aPIm.x];
                  if (aInd > 0)
                  {
                      mSDF.seek(aInd,ELISE_fp::sbegin);
                      bool aCont = true;
                      std::vector<int> aVC;
                      while (aCont)
                      {
                           int aC = mSDF.fgetc();
                           aVC.push_back(aC);
                           int aNB= aVC.size();
                           if  (
                                   (aNB>=4)
                                && (aVC[aNB-1]== 'Z')
                                && (aVC[aNB-2]== 'Z')
                                && (aVC[aNB-3]== 'Z')
                                && (aVC[aNB-4]== 'Z')
                            )
                            {
                                aCont = false;
                            }
                       }
                       int aNB = aVC.size();
                       aWFw.clear();
                       for (int aK=0; aK<aNB ; aK++)
                       {
                           std::cout << aVC[aK] << " ";
                           if (aK > 0)
                           {
                               Pt2di aP1(2*aK,258-aVC[aK]);
                               Pt2di aP2(2*(aK-1),258-aVC[aK-1]);
                               aWFw.draw_seg(aP1,aP2,aWFw.pdisc()(P8COL::red));
                           }
                       }
                       std::cout << "\n";
                       std::cout << aVC.size() << "\n";
                  }
               }
            }
      }


}


void  cSdrSdc::MakeImage()
{
     InitSDC();

     int aNbTest = 10000;
     double aMinT = 1e9;
     double aMaxT = -1e9;

     std::vector<double> mVEcarts;
     for (int aK=0 ; aK< aNbTest ; aK++)
     {
         mCur.ReadOnFile(mSDC);
         ElSetMax(aMaxT,mCur.mTheta);
         ElSetMin(aMinT,mCur.mTheta);
         mVEcarts.push_back(mCur.mTheta-mLast.mTheta);
         mLast = mCur;
     }
     std::sort(mVEcarts.begin(),mVEcarts.end());
     double anEcMed = mVEcarts[aNbTest/2];
     std::cout << "EC MED = " << anEcMed 
               << " ; Interv Teta " << aMinT << " - " << aMaxT << "\n";
     
             /***********************************/

     InitSDC();
     double aLargTeta = aMaxT-aMinT;
     
     int aNbEcTeta = 0;
     int aNbLigne = 0;
     int aMaxNbTeta = 0;
     double  aRgMax = 0;
     for (int aK=0 ; aK< (mNbElC-1) ; aK++)
     {
         mCur.ReadOnFile(mSDC);
         ElSetMax(aRgMax ,mCur.mRange);
         ElSetMax(aMaxT,mCur.mTheta);
         ElSetMin(aMinT,mCur.mTheta);
         double aDTeta = mCur.mTheta-mLast.mTheta;
         if (aDTeta < - aLargTeta/3.0)
         {
             std::cout << "NbT["<< aNbLigne <<"] = " << aNbEcTeta 
                       <<  " Max " << aMaxNbTeta << "\n";
             ElSetMax(aMaxNbTeta,aNbEcTeta);
             aNbEcTeta =0;
             aNbLigne++;
         }
         else
         {
             aDTeta /= anEcMed;
             int iDT = round_ni(aDTeta);
             aNbEcTeta += iDT;
             if (aNbEcTeta > 900)
             {
                  std::cout << aNbEcTeta << " " << iDT << "\n";
                  getchar();
             }
         }
         mLast = mCur;
     }

     Pt2di aSz(aMaxNbTeta+1,aNbLigne+1);
     Im2D_REAL4 aIRange(aSz.x,aSz.y,aRgMax);

     std::cout << "NB Last Teta "  << aNbEcTeta << "\n";
     std::cout << "Nb Ligne = " << aSz.y  << " " << aSz.x << "\n";
     std::cout << "Max2 = " << 1+(aMaxT-aMinT)/anEcMed << "\n";

             /***********************************/

     InitSDC();
     aNbLigne = 0;
     for (int aK=0 ; aK< (mNbElC-1) ; aK++)
     {
         mCur.ReadOnFile(mSDC);
         double aDTeta = mCur.mTheta-mLast.mTheta;
         if (aDTeta < - aLargTeta/3.0)
         {
             if ((aNbLigne%100)==0)
                std::cout << "Ligne = " << aNbLigne << "\n";
             aNbLigne++;
         }

         int aITeta = round_ni((mCur.mTheta-aMinT)/anEcMed);
         if ((aITeta>=0) && (aITeta<aSz.x) && (aNbLigne<aSz.y))
            aIRange.data()[aNbLigne][aITeta] = mCur.mRange;

         mLast = mCur;
     }
 
     Tiff_Im::CreateFromIm(aIRange,mName+ "_Range.tif");
}


void cSdrSdc::TestSDC()
{
     mSDC.seek(OFSET_SDC,ELISE_fp::sbegin);

     std::cout << "RATIO " << mSzF / double(mNbElC) << "\n";
     int aNbTT = 0;
     cElSDC anEl;

             
// std::cout << "NbEl " << mNbElC << "\n";
     cElSDC LastEl;
     int aH[1000];
     for (int aK=0 ; aK< 1000 ; aK++)
        aH[aK] = 0;

     std::vector<cElSDC> aVEl;
     int aNbFirtL = 0;
     for (int aK=0 ; aK< mNbElC ; aK++)
     {
                  
         anEl. ReadOnFile(mSDC);
         if (aK==0)
            LastEl = anEl;
         aNbTT += anEl.mWidth;
         if (anEl.mTarget==1) 
         {
            aH[anEl.mNbTarget]++;
            aNbFirtL++;
         }

         if (1)
         {
            double aDz = LastEl.mZ - anEl.mZ;

            std::cout    <<  anEl.mTime << " " << anEl.mRange << " "  << anEl.mTheta 
                      << " A " << anEl.mAmpl 
                      << " W " << anEl.mWidth 
                    << " XYZ " << anEl.mX << " " << anEl.mY << " " << anEl.mZ << " "
                 << anEl.mIndex << " " << int(anEl.mTarget) << " " << int(anEl.mNbTarget) 
                 << " "   <<  (anEl.mIndex /  anEl.mRange) * 700  << " " << "\n";

             //if (anEl.mNbTarget > 1) getchar();

             if (1)
             {
                 if (ElAbs(aDz) > 10)
                 {
                    std::cout << " NB = " << aVEl.size() 
                              << " ; First = " << aNbFirtL 
                              << " ;K = " << aK 
                              << "\n";
                    getchar();
                    aVEl.clear();
                    aNbFirtL=0;
                 }
                 aVEl.push_back(anEl);
             }
             // getchar();
          }
                 LastEl = anEl;
     }
     std::cout << "TT " << aNbTT << " SZF " << mSzF << "\n";
     std::cout <<  "TT / NbEl " << aNbTT / double(mNbElC) << "\n";
     std::cout <<  "SZF / aNbTT " << mSzF / double(aNbTT) << "\n";
     std::cout << "\n";
    for (int aK=0 ; aK < 1000 ;aK++)
        if (aH[aK] !=0)
           std::cout << "NbTarg[" << aK << "]=" << aH[aK] << "\n";
}

void  cSdrSdc::MakeIndexe_FW(Pt2di aSz)
{
    Im2D_INT4 aImInd(aSz.x,aSz.y,-1);
    mSDF.seek(0,ELISE_fp::sbegin);
    std::vector<int> aVC;

    int aCpZZZ=0;
    int aNbC = 0;

    int aC;
    while ((aC= mSDF.fgetc()) != EOF)
    {
         aVC.push_back(aC);
         int aNB= aVC.size();
         if  (
                 (aNB>=4)
              && (aVC[aNB-1]== 'Z')
              && (aVC[aNB-2]== 'Z')
              && (aVC[aNB-3]== 'Z')
              && (aVC[aNB-4]== 'Z')
          )
          {
              aCpZZZ++;
              aVC.clear();
              if (aCpZZZ >=2)
              {
                  int aY = (aCpZZZ-2)/aSz.x;
                  int aX = (aCpZZZ-2)%aSz.x;
                  aImInd.data()[aY][aX] = aNbC;
              }
              if ((aCpZZZ%100000) == 0) 
                  std::cout << "ZZ " << aCpZZZ << "\n";
          }

          aNbC++;
    }
    Tiff_Im::CreateFromIm(aImInd,mName+ "_IndxFW.tif");
}

void  cSdrSdc::TestSDF()
{
    Video_Win *aW = Video_Win::PtrWStd(Pt2di(400,260));
    std::vector<int> aVC;
    mSDC.seek(OFSET_SDC,ELISE_fp::sbegin);
    int aCpZZZ = 0;
    int aCpZZZ152 = 0;
    int aCpZZZ88 = 0;
    cElSDC anEl;

    int aH[1000];
    for (int aK=0 ; aK<1000 ; aK++)
        aH[aK] = 0;
    int aC;
    while ((aC= mSDF.fgetc()) != EOF)
    {
         aVC.push_back(aC);
         int aNB= aVC.size();
         if  (
                 (aNB>=4)
              && (aVC[aNB-1]== 'Z')
              && (aVC[aNB-2]== 'Z')
              && (aVC[aNB-3]== 'Z')
              && (aVC[aNB-4]== 'Z')
          )
          {
              if  (0) //(aCpZZZ >2)
              {
                  anEl. ReadOnFile(mSDC);
                  aW->clear();
                  for (int aK=0; aK<aNB ; aK++)
                  {
                      std::cout << aVC[aK] << " ";
                      if (aK > 0)
                      {
                          Pt2di aP1(2*aK,258-aVC[aK]);
                          Pt2di aP2(2*(aK-1),258-aVC[aK-1]);
                          aW->draw_seg(aP1,aP2,aW->pdisc()(P8COL::red));
                      }
                  }
                  std::cout << "\n";
                  std::cout << "NB = " << aVC.size()  << " " << int(anEl.mNbTarget) << "\n";
                  getchar();
              } 
              aVC.clear();
              aCpZZZ++;
              aH[ElMin(999,aNB)] ++;
              if (aNB==152)
                aCpZZZ152++;
              if (aNB==88)
                aCpZZZ88++;

              if ((aCpZZZ % 10000)==0)  
                  std::cout << "Z " << aCpZZZ  <<  " " << mNbElC << "\n";
          }
    }
    for (int aK=0 ; aK < 1000 ;aK++)
        if (aH[aK] !=0)
           std::cout << "H[" << aK << "]=" << aH[aK] << "\n";
    std::cout << "ZZZ " << aCpZZZ << " NbC " << mNbElC <<  "\n";
    std::cout << " 152 " << aCpZZZ152 
              << " 88 " << aCpZZZ88 
              << " " <<  (aCpZZZ152 + aCpZZZ88*0.5) << "\n";
}

class c4SDRSDC
{
    public :
       c4SDRSDC(const std::string & aName,int aNum,bool SauteEntete) :
          mF1 (aName+ToString(aNum+0),SauteEntete),
          mF2 (aName+ToString(aNum+1),SauteEntete),
          mF3 (aName+ToString(aNum+2),SauteEntete),
          mF4 (aName+ToString(aNum+3),SauteEntete)
       {
       }

       cSdrSdc   mF1;
       cSdrSdc   mF2;
       cSdrSdc   mF3;
       cSdrSdc   mF4;

       void TestEntete(bool sdf);
};

void c4SDRSDC::TestEntete(bool sdf)
{
    bool AllAscii = true;
    bool AllEqual = true;
    for (int aK=0 ; (aK< 10000) ; aK++)
    {
          int aC1  = mF1.FP(sdf).fgetc();
          int aC2  = mF2.FP(sdf).fgetc();
          int aC3  = mF3.FP(sdf).fgetc();
          int aC4  = mF4.FP(sdf).fgetc();

          AllAscii =  
                       isascii(aC1)
                    && isascii(aC2)
                    && isascii(aC3)
                    && isascii(aC4);

           bool ThisAllEqual =    (aC1==aC2)
                               && (aC1==aC3)
                               && (aC1==aC4);

           bool FirstNotEq = AllEqual && (! ThisAllEqual);
           AllEqual =    AllEqual  && ThisAllEqual;

           if (FirstNotEq)
              std::cout << "[##" <<  aK << "##]";

           int aC = aC2;
           if (isprint(aC) )
              std::cout << (unsigned char ) aC ;
           else  if (aC==180)
                 std::cout << "'";
           else  if (aC==13)
                 std::cout << "\n";
           else 
               std::cout << "[" <<  aC << "]";
   // If (aK%100==0) getchar();
    }
    std::cout << "----------------------\n";
    getchar();
}

int main(int argc,char ** argv)
{
    c4SDRSDC a4File("/home/mpd/Data/LaserFullWave/sdc/2006-04-19-Saar-5_00",20,false);

     // a4File.TestEntete(true);
     // a4File.mF3.TestSDC();
     // a4File.mF3.MakeIndexe_FW(Pt2di(565,3914));
     // a4File.mF3.TestSDF();
    a4File.mF3.Inspect_FW();
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
