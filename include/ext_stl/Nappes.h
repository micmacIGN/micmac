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


/* 
     Classes faites initialement, dans le cadre de MICMAC,
   pour memoriser  de l'information  en chaque point et
   chaque paralaxe.
*/


#ifndef _EL_NAPPES
#define _EL_NAPPES

// Strtucture pour gerer des napppes telle qu'utilise
// en Cox-Roy et Progdyn 2D


template <class T> class cDynTplNappe2D
{
     public :
          typedef T * tTPtr ;

          cDynTplNappe2D() 
          {
          }

          void Resize
               (
                    signed short * aZMin,
                    signed short * aZMax,
                    int             aSz,
                    int             aRab,
                    int             aMul
               )  
          {
                mNbElem = 0;
                mZMin   = aZMin;
                mZMax   = aZMax;
                mSz     = aSz;
                mRab    = aRab;
                mMul    = aMul;

                for (int aX=0 ; aX<mSz; aX++)
                {
                    ELISE_ASSERT(LengthCol(aX)>=0 ,"cTplNappe2D Min>Max!!");
                    mNbElem += LengthCol(aX);
                }
                mVDataLin.resize(mNbElem);
                mVData.resize(mSz);

                mDataLin  = VData(mVDataLin);
                mData = VData(mVData);
                int aCumul = 0;

                for (int aX=0 ; aX<mSz; aX++)
                {
                    mData[aX] = mDataLin + aCumul -ZMin(aX);
                    aCumul += LengthCol(aX);
                }
          }
          int LengthCol(int aX) const
          {
               return   (ZMax(aX)- ZMin(aX));
          }
          T ** Data() {return mData;}
          int  ZMax(int aX) const {return mRab +  mMul * (mZMax?mZMax[aX]:0);}
          int  ZMin(int aX) const {return         mMul * (mZMin?mZMin[aX]:0);}
     private :
          cDynTplNappe2D(const cDynTplNappe2D<T> &) ; // N.I.

          int     mNbElem;
          std::vector<T> mVDataLin;
          std::vector<T*> mVData;


          T  **   mData;
          T  *    mDataLin;
          signed short *  mZMin;
          signed short *  mZMax;
          int             mSz;
          int             mRab;
          int             mMul;

};


///  !!!    Data[Y][X][Z]  !!!!
///  CONVENTIONS "Standard" sur les intervalles de nappes
///       aImZMin  <=    Z  < aImZMax

template <class T> class cDynTplNappe3D
{
    public :
          typedef  T **             tPtrPtrT;
          cDynTplNappe3D
          (
               Im2D_INT2       aImZMin,
               Im2D_INT2       aImZMax,
               int             aRab,
               int             aMul
          )   :
                mSz      (aImZMin.sz()),
                mImZMin  (aImZMin),
                mTZMin   (mImZMin),
                mImZMax  (aImZMax),
                mTZMax   (mImZMax),
                mZMin    (aImZMin.data()),
                mZMax    (aImZMax.data()),
                mRab     (aRab),
                mMul     (aMul),
                mNap2D   (  new  cDynTplNappe2D<T>  [mSz.y]),
                mData    (  new tPtrPtrT[mSz.y])
          {
              for (int aY=0 ; aY<mSz.y ; aY++)
              {
                   mNap2D[aY].Resize(mZMin[aY],mZMax[aY],mSz.x,aRab,aMul);
                   mData[aY] = mNap2D[aY].Data(); 
              }
          }
          ~cDynTplNappe3D()
          {
               delete [] mNap2D ;
               delete [] mData ;
          }
 
          T*** Data() {return mData;}
          Im2D_INT2 IZMin(){return mImZMin;}
          Im2D_INT2 IZMax(){return mImZMax;}
          INT2 ** ZMin() {return mZMin;}
          INT2 ** ZMax() {return mZMax;}

    private :
          cDynTplNappe3D(const cDynTplNappe3D<T> &) ; // N.I.

          Pt2di                 mSz;
          Im2D_INT2             mImZMin;
          TIm2D<INT2,INT>       mTZMin;
          Im2D_INT2             mImZMax;
          TIm2D<INT2,INT>       mTZMax;
          signed short **       mZMin;
          signed short **       mZMax;
          int                   mRab;
          int                   mMul;
          cDynTplNappe2D<T>  *  mNap2D;
          T ***                 mData;
};


// Class pour preparer la creation d'une nappe d'objet de taille non connue a l'avance
//   NON TESTEE CAR ABANDONNEE EN ROUTE !!!!
template <class Type> class cNappeSizeUndef
{
     public :
         cNappeSizeUndef (Box2di  aBox);
         // void InitParcour();
         bool next(Pt2di &,int & I0,int & I1);

         Im2D_INT2       Cpt()            {return mCpt;}
         const std::vector<Type> & Objs() {return mObjs;}
         void PushCur(const std::vector<Type> &);

     private :
         cNappeSizeUndef(const cNappeSizeUndef<Type> &); // N.I. 

         void AssertInCreate(){ELISE_ASSERT(mInCreate,"cNappeSizeUndef no more in create mode")};
         void AssertInRead(){ELISE_ASSERT((!mInCreate) && (!mFinish) ,"cNappeSizeUndef no tn read mode")};

         TFlux_Rect2d       Flux();

         Box2di            mBox;
         Pt2di             mSz;
         std::vector<Type> mObjs;
         Im2D_INT2         mCpt;
         TIm2D<INT2,INT>   mTCpt;
         TFlux_Rect2d      mFlux;
         Pt2di             mPCur;
         int               mICur;
         bool              mInCreate;
         bool              mFinish;
};

template <class Type> cNappeSizeUndef<Type>::cNappeSizeUndef(Box2di aBox) :
    mBox         (aBox),
    mSz          (mBox.sz()),
    mCpt         (mSz.x,mSz.y,-1),
    mTCpt        (mCpt),
    mFlux        (Flux()),
    mPCur        (mFlux.PtsInit()),
    mInCreate    (true),
    mFinish      (false)
{
}
template <class Type> TFlux_Rect2d cNappeSizeUndef<Type>::Flux()
{
   return TFlux_Rect2d(mBox._p0,mBox._p1);
}
template <class Type> void cNappeSizeUndef<Type>::PushCur(const std::vector<Type> & aV)
{
   AssertInCreate();
   mInCreate = mFlux.next(mPCur);
   mTCpt.oset(mPCur-mBox._p0,aV.size());
   for (int aK=0 ; aK<int(aV.size()) ; aK++)
      mObjs.push_back(aV[aK]);
   if (!mInCreate)
   {
      mPCur = mFlux.PtsInit();
      mICur = 0;
   }
}

template <class Type> bool cNappeSizeUndef<Type>::next(Pt2di & aP,int & I0,int & I1)
{
    AssertInRead();
    mFinish = mFlux.next(mPCur);
    aP = mPCur;
    I0 = mICur;
    I1 = I0 + mTCpt.get(aP-mBox._p0);
    mICur = I1;
    return mFinish;
}









#endif //  _EL_NAPPES

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
