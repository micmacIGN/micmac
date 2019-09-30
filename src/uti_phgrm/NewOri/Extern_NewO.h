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
   Classes generique pour faire des objets reallouables.
   En fait pas utilise aujourd'hui

*/

#ifndef _ELISE_EXTERN_NEW_ORI_H
#define _ELISE_EXTERN_NEW_ORI_H



class cBaseSwappable
{
     public :
         int & Swap_HeapIndex() {return mHeapIndex;}
         const int & Swap_HeapIndex() const {return mHeapIndex;}
         _INT8 & Swap_IndLastLoaded() {return mIndLastLoaded;}
         const _INT8 & Swap_IndLastLoaded() const {return mIndLastLoaded;}
         bool& Swap_Deletable() {return mDeletable;}
         int & Swap_NbCreate() {return mNbCreate;}

        cBaseSwappable() :
             mHeapIndex (HEAP_NO_INDEX),
             mDeletable (false),
             mNbCreate (0)
        {
        }

      private:
         int mHeapIndex;
         _INT8 mIndLastLoaded;
         bool mDeletable;
         int  mNbCreate;
};



template <class Type> class cHeapSwapIndex
{
     public :
        static void SetIndex(Type * aP,int i) { aP->Swap_HeapIndex()=i;}
        static int  Index(const Type *aP) { return aP->Swap_HeapIndex(); }
};

template <class Type> class cHeapSwapCmpIndLoad
{
    public :
      bool operator() (const Type * aP1,const Type * aP2) {return aP1->Swap_IndLastLoaded() < aP2->Swap_IndLastLoaded();}
};




template <class Type> class cMemorySwap
{
    public :

        typedef typename Type::Swap_tArgCreate  tCreate;
        cMemorySwap(double aSzRam);
        void  ReAllocateObject(Type *,const tCreate & anArg);
    
    private :
        _INT8       mSzRam;
        _INT8       mSzLoaded;
        cHeapSwapCmpIndLoad<Type> mCmp;
        ElHeap<Type *,cHeapSwapCmpIndLoad<Type>,cHeapSwapIndex<Type> > mHeap;
        unsigned long int     mCpt;
};






template <class Type>    cMemorySwap<Type>::cMemorySwap(double aSzRam) :
    mSzRam         (aSzRam),
    mSzLoaded      (0),
    mCmp           (),
    mHeap          (mCmp),
    mCpt           (0)
{
}

template <class Type> void cMemorySwap<Type>::ReAllocateObject(Type * anObj,const tCreate & anArg)
{
    anObj->Swap_IndLastLoaded() =  mCpt++;
    int aCurCpt =  anObj->Swap_IndLastLoaded() ;
    mHeap.MajOrAdd(anObj);
    

    if (anObj->Swap_Loaded())
    {
         return;
    }

    anObj->Swap_Create(anArg);
    anObj->Swap_NbCreate()++;
    anObj->Swap_Deletable() = false;
    mSzLoaded += anObj->Swap_SizeOf();

    bool Cont = true;
    while (Cont)
    {
       if (mSzLoaded <mSzRam)
       {
          Cont = false;
       }
       else
       {
          Type * aPop;
          bool aGot = mHeap.pop(aPop);
          ELISE_ASSERT(aGot,"Heap empty in cMemorySwap<Type>::ReAllocateObject");
          if (aPop->Swap_Deletable())
          {
              mSzLoaded -= anObj->Swap_SizeOf();
              anObj->Swap_Delete();
          }
          else
          {
               if (aPop->Swap_IndLastLoaded() >= aCurCpt)
               {
                   Cont = false;
               }
               aPop->Swap_IndLastLoaded() =  mCpt++;
               mHeap.MajOrAdd(aPop);
          }
       }
    }
}


class cGenGaus3D
{
    public :
        cGenGaus3D(const cXml_Elips3D & anEl );
        const double & ValP(int aK) const;
        const Pt3dr  & VecP(int aK) const;
        const Pt3dr  & CDG() const {return mCDG;}

        //distribution de points selon e1,e2,e3 
        //indiqué par (2*aN1+1),(2*aN2+1),(2*aN3+1) et Gauss
        void GetDistribGaus(std::vector<Pt3dr> & aVPts,int aN1,int aN2,int aN3);
        
        //distribution de points selon e1,e2,e3 
        //indiqué par aN1,aN2,aN3
        void GetDistribGausNSym(std::vector<Pt3dr> & aVPts,int aN1,int aN2,int aN3,bool aAddPts=false);

        //RANDOM distribution of points 
        void GetDistribGausRand(std::vector<Pt3dr> & aVPts,int aN);

		//5-pts distribution
        void GetDistr5Points(std::vector<Pt3dr> & aVPts,double aRedFac=1.0);

    private :
		void GetDistr5PointsFromVP(Pt3dr aFact1,Pt3dr aFact2,Pt3dr aFact3,std::vector<Pt3dr> & aVPts);

        Pt3dr mCDG;
        double mVP[3];
        Pt3dr mVecP[3];

};

class cGenGaus2D
{
    public :
        cGenGaus2D(const cXml_Elips2D & anEl );
        const double & ValP(int aK) const;
        const Pt2dr  & VecP(int aK) const;

        void GetDistribGaus(std::vector<Pt2dr> & aVPts,int aN1,int aN2);

	//3-pts distribution
        void GetDistr3Points(std::vector<Pt2dr> & aVPts);

    private :
        Pt2dr  mCDG;
        double mVP[2];
        Pt2dr  mVecP[2];

};

void RazEllips(cXml_Elips3D & anEl);
void RazEllips(cXml_Elips2D & anEl);
void AddEllips(cXml_Elips3D & anEl,const Pt3dr & aP,double aPds);
void AddEllips(cXml_Elips2D & anEl,const Pt2dr & aP,double aPds);
void NormEllips(cXml_Elips3D & anEl);
void NormEllips(cXml_Elips2D & anEl);





#endif //  _ELISE_EXTERN_NEW_ORI_H



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
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
