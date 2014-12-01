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


#ifndef _EL_APPLI_TAB_
#define _EL_APPLI_TAB_


#include "GpGpu/GpGpu.h"


// Matrice d'origine variable, a priori petite
// mais cela reflete plus l'utilisation dans le
// correlateur qu'une contrainte forte

template <class T> class cSmallMatrixOrVar
{
        typedef T * tPtr;
     public :
        void DeleteMem()
        {
           delete [] mDataLin;
           delete [] mDataInit;
        }

        ~cSmallMatrixOrVar()
        {
            DeleteMem();
        }
        cSmallMatrixOrVar()     : 
            mDataLin(0),
            mDataInit(0)
        {
        }
        void  SetMem(Box2di aBox,const T & aVinit)      
        {
           DeleteMem();
           int aS = aBox.surf();
           mDataLin  = new T [aS];
           for (int aK=0; aK<aS ; aK++)
               mDataLin[aK] = aVinit;
           mDataInit = new tPtr [aBox.hauteur()];
           SetBox(aBox);
        }
        void SetBox(Box2di aBox)
        {
            mBox = aBox;
            mData = mDataInit - aBox._p0.y;
            mData[aBox._p0.y] = mDataLin -aBox._p0.x;
            for (int aY= aBox._p0.y +1 ; aY<aBox._p1.y ; aY++)
                mData[aY] =  mData[aY-1]+aBox.largeur();
        }
        T & operator[](const Pt2di & aPt) { return mData[aPt.y][aPt.x]; }
        const T&operator[](const Pt2di& aPt)const{ return mData[aPt.y][aPt.x];}
        T*  operator[](int anY) { return mData[anY]; }
        const Box2di & Box() const {return mBox;}

        Pt2di PClipedIntervC(const Pt2di & aPt) const
        {
            return  BoxPClipedIntervC(mBox,aPt);
        }
        const T&  GetClipedIntervC(const Pt2di & aPt)  const
        {
             return (*this)[PClipedIntervC(aPt)];
        }


     private :

         cSmallMatrixOrVar(const cSmallMatrixOrVar<T> &); 
         Box2di mBox;
         T  *   mDataLin;
         T  **  mDataInit;
         T  **  mData;
};


template <class T> class cVectOfSMV
{
    public :
        cVectOfSMV () :
            mVectInit (0)
        {
        }
        ~cVectOfSMV()
        {
            delete [] mVectInit;
        }

        void SetMem
        (
              int  aX0,
              int  aX1,
              INT2 * aBoxXMin,
              INT2 * aBoxYMin,
              INT2 * aBoxXMax,
              INT2 * aBoxYMax,
              const T & aVinit
        ) 
        {
            this->~cVectOfSMV();
            mVectInit = new cSmallMatrixOrVar<T> [aX1-aX0];
            mVect = mVectInit - aX0;
            for (int anX=aX0 ; anX<aX1 ; anX++)
            {
                mVect[anX].SetMem
                (
                     Box2di
                     (
                         Pt2di(aBoxXMin[anX],aBoxYMin ? aBoxYMin[anX] : 0),
                         Pt2di(aBoxXMax[anX],aBoxYMax ? aBoxYMax[anX] : 1)
                     ),
                     aVinit
                );
            }
        }

#if CUDA_ENABLED
        void SetMem
        (
                int  aX0,
                int  aX1,
                INT2 * aBoxXMin,
                INT2 * aBoxYMin,
                INT2 * aBoxXMax,
                INT2 * aBoxYMax,
                const T & aVinit,
                sMatrixCellCost<ushort>    &poInitCost,
                uint2                   &ptTer
        )
        {
            this->~cVectOfSMV();
            mVectInit = new cSmallMatrixOrVar<T> [aX1-aX0];
            mVect = mVectInit - aX0;
            for (int anX=aX0 ; anX<aX1 ; anX++)
            {
                ptTer.x         =  anX;
                poInitCost.PointIncre(ptTer,make_short2(aBoxXMin[anX],aBoxXMax[anX]));

                mVect[anX].SetMem
                (
                     Box2di
                     (
                         Pt2di(aBoxXMin[anX],aBoxYMin ? aBoxYMin[anX] : 0),
                         Pt2di(aBoxXMax[anX],aBoxYMax ? aBoxYMax[anX] : 1)
                     ),
                     aVinit
                );
            }
        }
#endif
        cSmallMatrixOrVar<T> & operator[](int anX)
        {
            return mVect[anX];
        }
    private :
         cSmallMatrixOrVar<T> * mVect;
         cSmallMatrixOrVar<T> * mVectInit;
};

template <class T> class cMatrOfSMV
{
    public :
        cMatrOfSMV
        (
              Box2di aBox,
              INT2 ** aBoxXMin,
              INT2 ** aBoxYMin,
              INT2 ** aBoxXMax,
              INT2 ** aBoxYMax,
              const T & aVinit
        )
        {
            mMatrInit = new cVectOfSMV<T>   [aBox.hauteur()];
            mMatr = mMatrInit - aBox._p0.y;
            for (int anY=aBox._p0.y; anY<aBox._p1.y ; anY++)
            {
                mMatr[anY].SetMem
                (
                    aBox._p0.x,aBox._p1.x,
                    aBoxXMin[anY], aBoxYMin ? aBoxYMin[anY] : 0,
                    aBoxXMax[anY], aBoxYMax ? aBoxYMax[anY] : 0,
                    aVinit
                );
            }
        }        
#if CUDA_ENABLED
        cMatrOfSMV
        (
              Box2di aBox,
              INT2 ** aBoxXMin,
              INT2 ** aBoxYMin,
              INT2 ** aBoxXMax,
              INT2 ** aBoxYMax,
              const T & aVinit,
              sMatrixCellCost<ushort>    &poInitCost
        )
        {
            mMatrInit = new cVectOfSMV<T>   [aBox.hauteur()];

            poInitCost.ReallocPt(make_uint2(abs(aBox._p1.x-aBox._p0.x),abs(aBox._p1.y-aBox._p0.y)));
            uint2 ptTer;

            mMatr = mMatrInit - aBox._p0.y;
            for (int anY=aBox._p0.y; anY<aBox._p1.y ; anY++)
            {
                ptTer.y = anY;
                mMatr[anY].SetMem
                (
                            aBox._p0.x,aBox._p1.x,
                            aBoxXMin[anY], aBoxYMin ? aBoxYMin[anY] : 0,
                            aBoxXMax[anY], aBoxYMax ? aBoxYMax[anY] : 0,
                            aVinit,
                            poInitCost,
                            ptTer
                );
            }
        }
#endif
        ~cMatrOfSMV()
        {
            delete [] mMatrInit;
        }
        cVectOfSMV<T> & operator[](int anY)
        {
            return mMatr[anY];
        }

        cSmallMatrixOrVar<T> & operator[] (const Pt2di & aP)
        {
            return (*this)[aP.y][aP.x];
        }

    private :
        cVectOfSMV<T> * mMatr;
        cVectOfSMV<T> * mMatrInit;
};

#endif //  _EL_APPLI_TAB_

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
