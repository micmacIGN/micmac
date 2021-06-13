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



#ifndef _ELISE_EX_OPER_ASSOC_EXTERN_H_
#define _ELISE_EX_OPER_ASSOC_EXTERN_H_

class cELise_InterfaceIm2D : public NS_TestOpBuf::cInterfaceIm2D
{
     public :

        U_INT1 GetValue(const std::complex<int> & aP) const 
        {
            return mIm.getproj(Std2Elise(aP));
        }
        void  SetValue(const std::complex<int> & aP,const U_INT1 & aVal) 
        {
           mIm.oset(Std2Elise(aP),aVal);
        }

        cELise_InterfaceIm2D(Im2D<U_INT1,int> anIm) :
          mIm (anIm)
        {
        }
     private :
        TIm2D<U_INT1,int> mIm;
};


inline int EleNeutre ( int *) {return 0;}
inline double EleNeutre (double *) {return 0.0;}
inline Pt2dr EleNeutre (Pt2dr *) {return Pt2dr(0,0);}

template <class Type> class  cCumulScal
{
     public :
           cCumulScal () :
                mSom (EleNeutre((Type *) 0))
           {
           }
           void AddCumul(int aSigne,const cCumulScal<Type> & aCSI)
           {
              mSom +=  aCSI.mSom * aSigne;
           }
           void AddElem(int aSigne,const Type & anEl)
           {
              mSom +=  anEl * aSigne;
           }

           const Type & Som() const {return mSom;}
     private :
         Type mSom;
};



template <class Type> class  cElise_ImBuf
{
    public :
         typedef typename El_CTypeTraits<Type>::tBase     tBase;
         typedef Im2D<Type,tBase>                         tIm;
         typedef TIm2D<Type,tBase>                        tTIm;

         typedef  tBase               tElem;
         typedef cCumulScal<tBase>    tCumul;

         void Init(const std::complex<int> & aP,tElem & anEl)
         {
              anEl = mTIm.get(Pt2di(aP.real(),aP.imag()),mDef);
         }

         void UseAggreg(const std::complex<int> & aP,const cCumulScal<tBase> & aCVC)
         {
         }

         void OnNewLine(int anY) {}

         cElise_ImBuf(tIm anIm,const tBase & aDef) :
            mIm (anIm),
            mTIm (mIm),
            mDef (aDef)
          {
          }

    private :

         tIm    mIm;
         tTIm   mTIm;
         tBase   mDef;
};


template <class Type,class TypeBase> void Binarise(Im2D<Type,TypeBase> anIm,Box2di aBox)
{
   TIm2D<Type,TypeBase> aTIm(anIm);
   Pt2di aP;
   for (aP.x=aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)
   {
       for (aP.y=aBox._p0.y ; aP.y<aBox._p1.y ; aP.y++)
       {
            aTIm.oset(aP,aTIm.get(aP)!=0);
       }
   }
}

template <class Type,class TypeBase> void Binarise(Im2D<Type,TypeBase> anIm)
{
   Binarise(anIm,Box2di(Pt2di(0,0),anIm.sz()));
}


// Image doit etre > 0, pour que la fonction soit valide (on teste le !=0 de la somme);
template <class Type,class TypeBase> void  Dilate
                                           (
                                                 Im2D<Type,TypeBase> anIm,
                                                 Im2D<Type,TypeBase> aRes,
                                                 const Pt2di &aSzV,
                                                 const Box2di & aBox,
                                                 TypeBase aDef=0
                                           )
{

    // Binarise(anIm);
    cElise_ImBuf<Type> aBuf(anIm,aDef);
    TIm2D<Type,TypeBase> aTIm(anIm);
    TIm2D<Type,TypeBase> aTRes(aRes);

    cTplOpbBufImage<cElise_ImBuf<Type> > anOp
                                           (
                                                aBuf,
                                                Elise2Std(aBox._p0),
                                                Elise2Std(aBox._p1),
                                                Elise2Std(-aSzV),
                                                Elise2Std(aSzV)
                                           );
    // int aNb = (1+2*aSzV.x) * (1+2*aSzV.y);

    cCumulScal<TypeBase> * aCum =0;
    while (  (aCum=anOp.GetNext()) != 0)
    {
         Pt2di aP = Std2Elise(anOp.CurPtOut());
         aTRes.oset(aP,aCum->Som() != 0);
    }
}

template <class Type,class TypeBase> void  SelfDilate(Im2D<Type,TypeBase> anIm, const Pt2di & aSzV)
{
    SelfDilate(anIm,aSzV,Box2di(Pt2di(0,0),anIm.sz()));
}


// Image doit etre binarise, pour que la fonction soit valide (on teste somme = (1+2Nb)^2 
template <class Type,class TypeBase> void  SelfErode(Im2D<Type,TypeBase> anIm,const Pt2di &aSzV,const Box2di & aBox,TypeBase aDef=0)
{

    // Binarise(anIm);
    cElise_ImBuf<Type> aBuf(anIm,aDef);
    TIm2D<Type,TypeBase> aTIm(anIm);

    cTplOpbBufImage<cElise_ImBuf<Type> > anOp
                                           (
                                                aBuf,
                                                Elise2Std(aBox._p0),
                                                Elise2Std(aBox._p1),
                                                Elise2Std(-aSzV),
                                                Elise2Std(aSzV)
                                           );
    int aNbV = (1+2*aSzV.x) * (1+2*aSzV.y);

    cCumulScal<TypeBase> * aCum =0;
    while (  (aCum=anOp.GetNext()) != 0)
    {
         Pt2di aP = Std2Elise(anOp.CurPtOut());
         aTIm.oset(aP,aCum->Som() == aNbV);
    }
}

template <class Type,class TypeBase> void  QMoyenne
                                           (
                                                 Im2D<Type,TypeBase> anIm,
                                                 Im2D<Type,TypeBase> aRes,
                                                 const Pt2di &aSzV,
                                                 const Box2di & aBox,
                                                 TypeBase aDef=0
                                           )
{

    // Binarise(anIm);
    cElise_ImBuf<Type> aBuf(anIm,aDef);
    TIm2D<Type,TypeBase> aTIm(anIm);
    TIm2D<Type,TypeBase> aTRes(aRes);

    cTplOpbBufImage<cElise_ImBuf<Type> > anOp
                                           (
                                                aBuf,
                                                Elise2Std(aBox._p0),
                                                Elise2Std(aBox._p1),
                                                Elise2Std(-aSzV),
                                                Elise2Std(aSzV)
                                           );
    // int aNb = (1+2*aSzV.x) * (1+2*aSzV.y);

    cCumulScal<TypeBase> * aCum =0;
    double aPds = (1+2*aSzV.x) * (1+2*aSzV.y);
    while (  (aCum=anOp.GetNext()) != 0)
    {
         Pt2di aP = Std2Elise(anOp.CurPtOut());
         aTRes.oset(aP,aCum->Som() / aPds);
    }
}

template <class Type,class TypeBase> void  SelfQMoyenne(Im2D<Type,TypeBase> anIm, const Pt2di & aSzV)
{
    QMoyenne(anIm,anIm,aSzV,Box2di(Pt2di(0,0),anIm.sz()));
}



   //  ================================================================
   //  ================================================================
   //  ================================================================

template <class Type> class  cCumulM2
{
     public :
           cCumulM2 () :
                mSom (EleNeutre((Type *) 0)),
                mSom2 (mSom)
           {
           }
           void AddCumul(int aSigne,const cCumulM2<Type> & aCSI)
           {
              mSom +=  aCSI.mSom * aSigne;
              mSom2 +=  aCSI.mSom2 * aSigne;
           }
           void AddElem(int aSigne,const Type & anEl)
           {
              mSom +=  anEl * aSigne;
              mSom2 +=  ElSquare(anEl) * aSigne;
           }

           const Type & Som() const  {return mSom;}
           const Type & Som2() const {return mSom2;}
     private :
         Type mSom;
         Type mSom2;
};

template <class Type> class  cElise_ImBufM2
{
    public :
         typedef typename El_CTypeTraits<Type>::tBase     tBase;
         typedef Im2D<Type,tBase>                         tIm;
         typedef TIm2D<Type,tBase>                        tTIm;

         typedef  tBase               tElem;
         typedef  cCumulM2<tBase>    tCumul;

         void Init(const std::complex<int> & aP,tElem & anEl)
         {
              anEl = mTIm.get(Pt2di(aP.real(),aP.imag()),mDef);
         }

         void UseAggreg(const std::complex<int> & aP,const cCumulM2<tBase> & aCVC)
         {
         }

         void OnNewLine(int anY) {}

         cElise_ImBufM2(tIm anIm,const tBase & aDef) :
            mIm (anIm),
            mTIm (mIm),
            mDef (aDef)
          {
          }

    private :

         tIm    mIm;
         tTIm   mTIm;
         tBase   mDef;
};


template <class Type,class TypeBase> void  MomOrdre2
                                           (
                                                Im2D<Type,TypeBase> anIm,
                                                Im2D<Type,TypeBase> aISom,
                                                Im2D<Type,TypeBase> aISom2,
                                                const Pt2di &aSzV,
                                                const Box2di & aBox,
                                                TypeBase aDef=0
                                           )
{
    cElise_ImBufM2<Type>   aBuf(anIm,aDef);
    //TIm2D<Type,TypeBase> aTIm(anIm);
    TIm2D<Type,TypeBase> aTSom(aISom);
    TIm2D<Type,TypeBase> aTS2(aISom2);

    cTplOpbBufImage<cElise_ImBufM2<Type> > anOp
                                           (
                                                aBuf,
                                                Elise2Std(aBox._p0),
                                                Elise2Std(aBox._p1),
                                                Elise2Std(-aSzV),
                                                Elise2Std(aSzV)
                                           );

    cCumulM2<TypeBase> * aCum =0;
    while (  (aCum=anOp.GetNext()) != 0)
    {
         Pt2di aP = Std2Elise(anOp.CurPtOut());
         aTSom.oset(aP,aCum->Som());
         aTS2.oset(aP,aCum->Som2());
    }
}



template <class Type,class TypeBase> void  MomOrdre2_Creux
                                           (
                                                Im2D<Type,TypeBase> anIm,
                                                Im2D<Type,TypeBase> aISom,
                                                Im2D<Type,TypeBase> aISom2,
                                                const Pt2di &aSzV,
                                                const Box2di & aBox
                                           )
{
    TIm2D<Type,TypeBase> aTSom(aISom);
    TIm2D<Type,TypeBase> aTS2(aISom2);

    Pt2di  aP;
    for (aP.y=aBox._p0.y ; aP.y<aBox._p1.y; aP.y++)
    {
        Type * aLPrec = anIm.data()[aP.y-aSzV.y] +aBox._p0.x;
        Type * aD0 = aLPrec-aSzV.x;
        Type * aD1 = aLPrec       ;
        Type * aD2 = aLPrec+aSzV.x;

        Type * aLCur  = anIm.data()[aP.y] +aBox._p0.x;
        Type * aD3 = aLCur-aSzV.x;
        Type * aD4 = aLCur       ;
        Type * aD5 = aLCur+aSzV.x;


        Type * aLNext = anIm.data()[aP.y+aSzV.y] +aBox._p0.x;
        Type * aD6 = aLNext-aSzV.x;
        Type * aD7 = aLNext       ;
        Type * aD8 = aLNext+aSzV.x;


        for (aP.x=aBox._p0.x ; aP.x<aBox._p1.x; aP.x++)
        {
           Type aV0 = *(aD0++);
           Type aV1 = *(aD1++);
           Type aV2 = *(aD2++);
           Type aV3 = *(aD3++);
           Type aV4 = *(aD4++);
           Type aV5 = *(aD5++);
           Type aV6 = *(aD6++);
           Type aV7 = *(aD7++);
           Type aV8 = *(aD8++);
           aTSom.oset(aP,aV0+aV1+aV2+aV3+aV4+aV5+aV6+aV7+aV8);
           aTS2.oset(aP,  aV0*aV0 + aV1*aV1 + aV2*aV2
                        + aV3*aV3 + aV4*aV4 + aV5*aV5
                        + aV6*aV6 + aV7*aV7 + aV8*aV8);
        }
    }
}



   //  ================================================================
   //  ================================================================
   //  ================================================================


template <class Type> class  cPairVal
{
    public :
          Type mV1;
          Type mV2;
};

template <class Type> class  cCumulM2M12
{
     public :
           cCumulM2M12 () :
                mSom2   (EleNeutre((Type *) 0)),
                mSom22  (mSom2),
                mSom12  (mSom2)
           {
           }
           void AddCumul(int aSigne,const cCumulM2M12<Type> & aCSI)
           {
              mSom2 +=  aCSI.mSom2   * aSigne;
              mSom22 +=  aCSI.mSom22 * aSigne;
              mSom12 +=  aCSI.mSom12 * aSigne;
           }
           void AddElem(int aSigne,const cPairVal<Type> & anEl)
           {
              mSom2  +=  anEl.mV2 * aSigne;
              mSom22 +=  ElSquare(anEl.mV2) * aSigne;
              mSom12 +=  anEl.mV1 * anEl.mV2 * aSigne;
           }

           const Type & Som12() const  {return mSom12;}
           const Type & Som2() const {return mSom2;}
           const Type & Som22() const {return mSom22;}
     private :
           Type mSom2;
           Type mSom22;
           Type mSom12;
};

template <class Type> class  cElise_ImBufM2M12
{
    public :
         typedef typename El_CTypeTraits<Type>::tBase     tBase;
         typedef Im2D<Type,tBase>                         tIm;
         typedef TIm2D<Type,tBase>                        tTIm;

         typedef  cPairVal<tBase>       tElem;
         typedef  cCumulM2M12<tBase>    tCumul;

         void Init(const std::complex<int> & aP,tElem & anEl)
         {
              anEl.mV1 = mTIm1.get(Pt2di(aP.real(),aP.imag()),mDef);
              anEl.mV2 = mTIm2.get(Pt2di(aP.real(),aP.imag()),mDef);
         }

         void UseAggreg(const std::complex<int> & aP,const cCumulM2<tBase> & aCVC)
         {
         }

         void OnNewLine(int anY) {}

         cElise_ImBufM2M12(tIm anIm1,tIm anIm2,const tBase & aDef) :
            mIm1 (anIm1),
            mTIm1 (mIm1),
            mIm2 (anIm2),
            mTIm2 (mIm2),
            mDef (aDef)
          {
          }

    private :

         tIm    mIm1;
         tTIm   mTIm1;
         tIm    mIm2;
         tTIm   mTIm2;
         tBase   mDef;
};

template <class Type,class TypeBase> void  Mom12_22
                                           (
                                                Im2D<Type,TypeBase> anIm1,
                                                Im2D<Type,TypeBase> anIm2,
                                                Im2D<Type,TypeBase> aISom12,
                                                Im2D<Type,TypeBase> aISom2,
                                                Im2D<Type,TypeBase> aISom22,
                                                const Pt2di &aSzV,
                                                const Box2di & aBox,
                                                TypeBase aDef=0
                                           )
{
    cElise_ImBufM2M12<Type>   aBuf(anIm1,anIm2,aDef);
    TIm2D<Type,TypeBase>      aTSom12(aISom12);
    TIm2D<Type,TypeBase>      aTSom2(aISom2);
    TIm2D<Type,TypeBase>      aTSom22(aISom22);

    cTplOpbBufImage<cElise_ImBufM2M12<Type> > anOp
                                           (
                                                aBuf,
                                                Elise2Std(aBox._p0),
                                                Elise2Std(aBox._p1),
                                                Elise2Std(-aSzV),
                                                Elise2Std(aSzV)
                                           );

    cCumulM2M12<TypeBase> * aCum =0;
    while (  (aCum=anOp.GetNext()) != 0)
    {
         Pt2di aP = Std2Elise(anOp.CurPtOut());
         aTSom12.oset(aP,aCum->Som12());
         aTSom2.oset(aP,aCum->Som2());
         aTSom22.oset(aP,aCum->Som22());
    }
}


template <class Type,class TypeBase> void  Mom12_22_creux
                                           (
                                                Im2D<Type,TypeBase> anIm1,
                                                Im2D<Type,TypeBase> anIm2,
                                                Im2D<Type,TypeBase> aISom12,
                                                Im2D<Type,TypeBase> aISom2,
                                                Im2D<Type,TypeBase> aISom22,
                                                const Pt2di &aSzV,
                                                const Box2di & aBox
                                           )
{
    TIm2D<Type,TypeBase> aTSom12(aISom12);
    TIm2D<Type,TypeBase> aTS2(aISom2);
    TIm2D<Type,TypeBase> aTS22(aISom22);

    Pt2di  aP;
    for (aP.y=aBox._p0.y ; aP.y<aBox._p1.y; aP.y++)
    {
        Type * aLPrec1 = anIm1.data()[aP.y-aSzV.y] +aBox._p0.x;
          Type * a1D0 = aLPrec1-aSzV.x;
          Type * a1D1 = aLPrec1       ;
          Type * a1D2 = aLPrec1+aSzV.x;
        Type * aLCur1  = anIm1.data()[aP.y] +aBox._p0.x;
          Type * a1D3 = aLCur1-aSzV.x;
          Type * a1D4 = aLCur1       ;
          Type * a1D5 = aLCur1+aSzV.x;
        Type * aLNext1 = anIm1.data()[aP.y+aSzV.y] +aBox._p0.x;
          Type * a1D6 = aLNext1-aSzV.x;
          Type * a1D7 = aLNext1       ;
          Type * a1D8 = aLNext1+aSzV.x;

        Type * aLPrec2 = anIm2.data()[aP.y-aSzV.y] +aBox._p0.x;
          Type * a2D0 = aLPrec2-aSzV.x;
          Type * a2D1 = aLPrec2       ;
          Type * a2D2 = aLPrec2+aSzV.x;
        Type * aLCur2  = anIm2.data()[aP.y] +aBox._p0.x;
          Type * a2D3 = aLCur2-aSzV.x;
          Type * a2D4 = aLCur2       ;
          Type * a2D5 = aLCur2+aSzV.x;
        Type * aLNext2 = anIm2.data()[aP.y+aSzV.y] +aBox._p0.x;
          Type * a2D6 = aLNext2-aSzV.x;
          Type * a2D7 = aLNext2       ;
          Type * a2D8 = aLNext2+aSzV.x;

        for (aP.x=aBox._p0.x ; aP.x<aBox._p1.x; aP.x++)
        {
           Type a2V0 = *(a2D0++);
           Type a2V1 = *(a2D1++);
           Type a2V2 = *(a2D2++);
           Type a2V3 = *(a2D3++);
           Type a2V4 = *(a2D4++);
           Type a2V5 = *(a2D5++);
           Type a2V6 = *(a2D6++);
           Type a2V7 = *(a2D7++);
           Type a2V8 = *(a2D8++);
           aTS2.oset(aP,a2V0+a2V1+a2V2+a2V3+a2V4+a2V5+a2V6+a2V7+a2V8);
           aTS22.oset(aP,  a2V0*a2V0 + a2V1*a2V1 + a2V2*a2V2
                        + a2V3*a2V3 + a2V4*a2V4 + a2V5*a2V5
                        + a2V6*a2V6 + a2V7*a2V7 + a2V8*a2V8);
           aTSom12.oset(aP,    a2V0**(a1D0++) + a2V1**(a1D1++) + a2V2**(a1D2++)
                            +  a2V3**(a1D3++) + a2V4**(a1D4++) + a2V5**(a1D5++)
                            +  a2V6**(a1D6++) + a2V7**(a1D7++) + a2V8**(a1D8++));
        }
    }
}




#endif // _ELISE_EX_OPER_ASSOC_EXTERN_H_




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
