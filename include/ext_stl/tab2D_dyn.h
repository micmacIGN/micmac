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



#ifndef _EL_TAB2D_DYN_
#define _EL_TAB2D_DYN_

template <class T> class cElTab2DResizeable
{
    public :

        typedef T value_type;
		typedef T* tPtr_value_type;

        ~cElTab2DResizeable()
        {
             for (INT y = 0; y<mSzMax.y ; y++)
                  delete [] mData[y];
             delete [] mData;
        }

        cElTab2DResizeable(Pt2di aSz)
        {
             mSz = aSz;
             mSzMax = mSz;

             mData = new tPtr_value_type [mSzMax.y];
             for (INT y = 0; y<mSzMax.y ; y++)
             {
                 mData[y] =  NewLine(mSzMax.x);
             }
        }

        void SetSize(Pt2di aSz)
        {
            if (aSz.x > mSzMax.x)
            {
                for (INT y = 0; y<mSzMax.y ; y++)
                {
                    delete [] mData[y];
                    mData[y] = NewLine(aSz.x);
                }
                mSzMax.x = aSz.x;
            }
            if (aSz.y > mSzMax.y)
            {
                 T** aD = new tPtr_value_type [aSz.y];

                 for (INT y = 0; y<mSzMax.y ; y++)
                     aD[y] = mData[y];
                 for (INT y = mSzMax.y; y<aSz.y ; y++)
                     aD[y] = NewLine(mSzMax.x);

                 delete [] mData;
                 mData = aD;
                 mSzMax.y = aSz.y;
            }
            mSz = aSz;
        }

        T & operator() (const Pt2di & aP)
        {
            ELISE_ASSERT(inside(aP),"Outside in cElTab2DResizeable()");
            return PrivateGet(aP);
        }

        T & operator() (const Pt2di & aP,T& aDef)
        {
             return inside(aP) ? PrivateGet(aP) : aDef;
        }

        bool inside (const Pt2di & aP) const
        {
            return    (aP.x >= 0)
                   && (aP.y >= 0)
                   && (aP.x <  mSz.x)
                   && (aP.y <  mSz.y) ;
        }
        Pt2di Sz() const {return mSz;}


    private :

        T * NewLine(INT aSz)
        {
            T* aRes = new T [aSz];
            for (INT x=0; x<aSz ; x++)
                aRes[x] = T();
            return aRes;
        }

        T & PrivateGet(const Pt2di & aP) {return mData[aP.y][aP.x]; }

        Pt2di mSz;
        Pt2di mSzMax; // Taille max atteint au cours de "l'historique"
        T **  mData;
};

template <class T> class cElBoxTab2DResizeable
{
    public :

        typedef T value_type;

        cElBoxTab2DResizeable(Box2di aBox) :
             mTab(aBox.sz()),
             mP0 (aBox._p0)
        {
        }

        void SetSize(Box2di aBox)
        {
            mTab.SetSize(aBox.sz());
            mP0 = aBox._p0;
        }

        T & operator() (const Pt2di & aP)
        {
            return mTab(aP-mP0);
        }

        T & operator() (const Pt2di & aP,T& aDef)
        {
             return mTab(aP-mP0,aDef);
        }

        bool inside (const Pt2di & aP) const { return mTab.inside(aP-mP0); }
        Pt2di P0() const {return mP0;}
        Pt2di P1() const {return mP0 + mTab.Sz();}
        Box2di Box() const {return Box2di(P0(),P1());}


    private :

        cElTab2DResizeable<T>  mTab;
        Pt2di                  mP0;
};


typedef std::vector<Pt2di> tT2drCrNE;

template <class T> class cElBoxTab2DResizeableCreux
{
          typedef T * tPtr;
          typedef std::vector<tPtr>     tReserve;

     public :

          cElBoxTab2DResizeableCreux(Box2di aBox,bool WithNonEmpty) :
              mTab (aBox),
              mWNE (WithNonEmpty)
          {
          }

          ~cElBoxTab2DResizeableCreux();

          Pt2di P0() const {return  mTab.P0();}
          Pt2di P1() const {return  mTab.P1();}
          Box2di Box() const {return Box2di(P0(),P1());}

          bool  IsOccuped(Pt2di aP) 
          {
               return mTab(aP) != 0;
          }
          T * GetPtr (Pt2di aP)
          {
               return mTab(aP);
          }
          // T & operator () (Pt2di aP)
          T & Get (Pt2di aP)
          {
              tPtr & aPtr = mTab(aP);
              if (aPtr == 0)
              {
                 if (mWNE)
                    mNonEmpty.push_back(aP);
                 if ( ! mReserve.empty())
                 {
                    aPtr = mReserve.back();
                    mReserve.pop_back();
                    *aPtr = T();
                 }
                 else
                    aPtr = new T;
              }
              return  * aPtr;
          }

          bool inside (Pt2di aP)
          {
              return mTab.inside(aP) && mTab(aP) != 0;
          }

          void SetSize(Box2di aBox)
          {
               for 
               (
                    tT2drCrNE::const_iterator itP=mNonEmpty.begin(); 
                    itP!=mNonEmpty.end() ; 
                    itP++
               )
               {
                    tPtr & aPtr = mTab(*itP);
                    if (aPtr !=0)
                       mReserve.push_back(aPtr);
                    aPtr = 0;
               }
               mTab.SetSize(aBox);
               mNonEmpty.clear();
          }
          const tT2drCrNE  & NonEmpty() {return mNonEmpty;}
          bool WNE() const {return mWNE;}
          
 
     private :
          cElBoxTab2DResizeable<T*>     mTab;
          tT2drCrNE                     mNonEmpty;
          tReserve                      mReserve;
          bool                          mWNE;
};


template <class Tel,class TAction>
void  ParseElT2dRC
      (
           cElBoxTab2DResizeableCreux<Tel> & aTab,
           TAction                         & anAction
      )
{
  if (aTab.WNE())
  {
      const tT2drCrNE  &  aNE = aTab.NonEmpty();
      for (tT2drCrNE::const_iterator itP=aNE.begin(); itP!=aNE.end(); itP++)
      {
          anAction(aTab.Get(*itP),*itP);
      }
  }
  else
  {
     Box2di aBoxPx = aTab.Box();
     Pt2di aPP;
     for (aPP.y =aBoxPx._p0.y; aPP.y<aBoxPx._p1.y ; aPP.y++)
     {
         for (aPP.x=aBoxPx._p0.x; aPP.x<aBoxPx._p1.x ; aPP.x++)
         {
             Tel * anEl = aTab.GetPtr(aPP);
             if (anEl)
                anAction(*anEl,aPP);
         }
     }
  }

}

template <class T> class cDeletEl_TRC
{
    public :
       void operator () (T & aT,const Pt2di & aP)
       {
            delete &aT;
       }
};

template <class T> 
cElBoxTab2DResizeableCreux<T>::~cElBoxTab2DResizeableCreux()
{
    cDeletEl_TRC<T> aDel;
    ParseElT2dRC(*this,aDel);
}


#endif  // _EL_TAB2D_DYN_




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
