#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdio.h>
#include "InitOutil.h"
#include "Fast.h"
#include "Pic.h"
#include "../include/general/ptxd.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
/*
template <class Type> int  CmpValAndDec(const Type & aV1,const Type & aV2, const Pt2di & aDec)
{
   //    aV1 =>   aV1 + eps * aDec.x + eps * esp * aDec

   if (aV1 < aV2) return -1;
   if (aV1 > aV2) return  1;

   if (aDec.x<0)  return -1;
   if (aDec.x>0)  return  1;

   if (aDec.y<0)  return -1;
   if (aDec.y>0)  return  1;

   return 0;
}
*/
class Detector
{
    public:
        Detector();
        Detector( string typeDetector, vector<double> paramDetector,
                        string nameImg,
                        Im2D<unsigned char, int> * img,
                        InitOutil * aChain
                        );
        Detector(InitOutil * aChain , pic * pic1 , pic * pic2);   //from pack homo Init
        Detector(string typeDetector,
                 vector<double> paramDetector,
                 pic * aPic,
                 InitOutil * aChain
                );
        Detector(
                    string typeDetector,
                    vector<double> paramDetector,
                    pic * aPic,
                    cInterfChantierNameManipulateur * aICNM
                 );
        int detect(bool useTypeFileDigeo = true);
        int readResultDetectFromFileDigeo(string filename);
        int readResultDetectFromFileElHomo(string filename);
        vector<Pt2dr> importFromHomolInit(pic* pic2);
        void saveToPicTypeVector(pic* aPic);
        void saveToPicTypePackHomol(pic* aPic);
        void saveResultToDiskTypeDigeo           (string aDir="./");
        void saveResultToDiskTypeElHomo          (string aDir="./");
        void getmPtsInterest(vector<Pt2dr> & ptsInteret);
    protected:
        string mNameImg;
        string mTypeDetector;           //FAST, DIGEO, HOMOLINIT
        vector<double> mParamDetector;
        vector<Pt2dr> mPtsInterest;
        Im2D<unsigned char, int> *mImg;
        InitOutil * mChain;
        pic* mPic2;
        cInterfChantierNameManipulateur * mICNM;
};


class ExtremePoint
{
public:
   ExtremePoint (double radiusVoisin);

   void detect  (
                    const TIm2D<unsigned char,int> &anIm,
                    vector<Pt2dr> &lstPt,
                    const TIm2DBits<1> * aMasq = NULL
                );

    // detect extrema point + compute Fast quality (like TiepTri)
   template <typename Type, typename Type_Base> int detect
   (
           TIm2D<Type,Type_Base> & anIm,
           TIm2DBits<1> & aMasq,
           vector<cIntTieTriInterest> & lstPt
   )
   {
       Pt2di aP;
       Pt2di aSzIm = anIm.sz();
       for (aP.x=0 ; aP.x<aSzIm.x ; aP.x++)
       {
           for (aP.y=0 ; aP.y<aSzIm.y ; aP.y++)
           {

               bool get;
               get = aMasq.get(aP);
               if (get && isAllVoisinInside(anIm, aP, mVoisin))
               {
                   int aCmp0 =  ExtremePoint::IsExtrema(anIm,aP);
                   if (aCmp0)
                   {
                       eTypeTieTri aType = (aCmp0==1)  ? eTTTMax : eTTTMin;
                       cIntTieTriInterest aPt(aP, aType, 0);
                       lstPt.push_back(aPt);
                   }
               }
           }
       }
       return lstPt.size();
   }

private:
   /*
   bool                isAllVoisinInside( const TIm2D<unsigned char,int> &anIm,
                                           Pt2di aP,
                                           vector<Pt2di> &  aVE
                                        );
                                        */
   template <typename Type, typename Type_Base> bool isAllVoisinInside
                                      (
                                         const TIm2D<Type, Type_Base> &anIm,
                                         Pt2di aP,
                                         vector<Pt2di> &  aVE
                                       )
   {
       for (uint aK=0; aK<aVE.size(); aK++)
       {
           if ( !anIm.inside(aP + aVE[aK]) )
           {
               return false;
               break;
           }
       }
       return true;
   }
    /*
   int                 IsExtrema(const TIm2D<unsigned char,int> & anIm,Pt2di aP);
   */


   template <typename Type, typename Type_Base> int IsExtrema
                   (
                       const TIm2D<Type,Type_Base> & anIm,
                       Pt2di aP
                   )
   {
       const std::vector<Pt2di> &  aVE = this->mVoisin;
       int aCmp0 =0;
       for (int aKP=0 ; aKP<int(aVE.size()) ; aKP++)
       {
           int aCmp = CmpValAndDec(anIm.get(aP),anIm.get(aP+aVE[aKP]),aVE[aKP]);
           if (aKP==0)
           {
               aCmp0 = aCmp;
               if (aCmp0==0) return 0;
           }

           if (aCmp!=aCmp0) return 0;
       }
       return aCmp0;
   }

   vector<Pt2di>       getVoisinInteret(double minR, double maxR);
   vector<Pt2di>       mVoisin;
};

#endif

