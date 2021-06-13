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



#ifndef _ELISE_EXT_STL_NUMERICS_H
#define _ELISE_EXT_STL_NUMERICS_H




template <class Type> void  Rank3(int * aRnk, const Type & aN0,const Type & aN1,const Type & aN2)
{
     int Inf10 = (aN1<aN0);
     int Inf20 = (aN2<aN0);
     int Inf21 = (aN2<aN1);
     aRnk[0] = Inf10 + Inf20;
     aRnk[1] = (1-Inf10) +  Inf21;
     aRnk[2] = (1-Inf20) + (1-Inf21);
/*
     aRnk[0] = (aN0>aN1)  +  (aN0>aN2);
     aRnk[1] = (aN0<=aN1) +  (aN1>aN2);
     aRnk[2] = (aN0<=aN2)  +  (aN1<=aN2);
*/
}

template <class Type> class cTplTriplet
{
     public :
            cTplTriplet(const Type & aV0,const Type & aV1,const Type &aV2) :
                 mV0 (aV0),
                 mV1 (aV1),
                 mV2 (aV2)
            {
                  if (mV1<mV0) ElSwap(mV0,mV1);
                  if (mV2<mV0) ElSwap(mV0,mV2);
                  if (mV2<mV1) ElSwap(mV1,mV2);
            }

            bool operator < (const cTplTriplet<Type> & aT2) const
            {
                if (mV0 < aT2.mV0) return true;
                if (aT2.mV0 < mV0) return false;
                if (mV1 < aT2.mV1) return true;
                if (aT2.mV1 < mV1) return false;
                return mV2 < aT2.mV2;
            }

            bool operator == (const cTplTriplet<Type> & aT2) const
            {
                  return  (mV0==aT2.mV0) &&  (mV1==aT2.mV1) && (mV2==aT2.mV2);
            }

            Type  mV0;
            Type  mV1;
            Type  mV2;
};
//  !!!  NE PEUX PAS ETRE COPIEE .....
template <class Type> class cTplTripletByRef
{
     public :
            cTplTripletByRef(const Type & aV0,const Type & aV1,const Type &aV2) :
                 mV0 (&aV0),
                 mV1 (&aV1),
                 mV2 (&aV2)
            {
                  if (*mV1<*mV0) ElSwap(mV0,mV1);
                  if (*mV2<*mV0) ElSwap(mV0,mV2);
                  if (*mV2<*mV1) ElSwap(mV1,mV2);
            }

            const Type * mV0;
            const Type * mV1;
            const Type * mV2;
};


// ElMedian directement pompes sur __median de G++-stl,
// car je ne suis pas sur que ce soit standard

template <class T>
inline const T& ElMedian(const T& a, const T& b, const T& c) {
  if (a < b)
  {
    if (b < c)
      return b;
    else if (a < c)
      return c;
    else
      return a;
  }
  else if (a < c)
    return a;
  else if (b < c)
    return c;
  else
    return b;
}

template <class T, class Compare>
inline const T& ElMedian(const T& a, const T& b, const T& c, Compare comp) {
  if (comp(a, b))
  {
    if (comp(b, c))
      return b;
    else if (comp(a, c))
      return c;
    else
      return a;
  }
  else if (comp(a, c))
    return a;
  else if (comp(b, c))
    return c;
  else
    return b;
}



template <class TVal,class tGetVal>
typename tGetVal::tValue     GenValPercentile
         (
             const std::vector<TVal> & aVec,
             const double & aPerc,
             const tGetVal&  aGetV
         )
{
    int aNBV = (int)aVec.size();
    ELISE_ASSERT(aNBV,"No Val in ValPercentile");
    if (aNBV==1)
       return  aGetV(aVec[0]);

   double aPerc0 = (0.5/aNBV) * 100;
   double aPercLast = ((aNBV-0.5)/aNBV) * 100;

   double aRang =  ((aPerc-aPerc0)/(aPercLast-aPerc0)) * (aNBV-1);

   if (aRang<0)
      return aGetV(aVec[0]);
   else if (aRang>=aNBV-1)
       return aGetV(aVec[aNBV-1]);

   int aR0 = round_down(aRang);
   double aP1 = aRang-aR0;
   double aP0 = 1-aP1;
   return aGetV(aVec[aR0])*aP0+ aGetV(aVec[aR0+1])*aP1 ;
}

template <class TVal> class  cOperatorIdentite
{
    public :
      const TVal & operator ()(const TVal & aVal) const{ return aVal;}
      typedef TVal tValue;
};

template <class TVal>
TVal     ValPercentile
         (
             const std::vector<TVal> & aVec,
             const double & aPerc
         )
{
   cOperatorIdentite<TVal> anId;
   return  GenValPercentile(aVec,aPerc,anId);
}


template <class TVal,class tGetPds> double  SomPerc(const std::vector<TVal> & aVec,const tGetPds & aGetP)
{
   double aSom = 0.0;
   for (int aKV=0 ; aKV<int(aVec.size()) ; aKV++)
     aSom += aGetP(aVec[aKV]);

   return aSom;
}

template <class TVal,class tGetVal,class tGetPds>
typename tGetVal::tValue     GenValPdsPercentile
         (
             const std::vector<TVal> & aVec,
             const double & aPerc,
             const tGetVal&  aGetV,
             const tGetPds & aGetP,
             double aSom
         )
{
    int aNBV = (int)aVec.size();
    ELISE_ASSERT(aNBV,"No Val in ValPercentile");
    if (aNBV==1)
       return  aGetV(aVec[0]);


   double aMul = 50.0 / aSom;
   double aLastP = aGetP(aVec[0]);

   // double aPerc0 = (0.5 * (aGetP(aVec[0]) / aSom) ) * 100;
   double aPerc0  = aLastP * aMul;

   if (aPerc <= aPerc0)  
      return aGetV(aVec[0]);

   double aCumPerc = aPerc0;

   for (int aK=1 ; aK< aNBV ; aK++)
   {
      double aNewP = aGetP(aVec[aK]);
      double aNewCum = aCumPerc + (aLastP+aNewP) * aMul;
      if ((aPerc>=aCumPerc) && (aPerc <= aNewCum))
      {
           ELISE_ASSERT(aNewCum>aCumPerc,"Equal value in cum :: GenValPdsPercentile");
           typename tGetVal::tValue aV0 = aGetV(aVec[aK-1]);
           typename tGetVal::tValue aV1 = aGetV(aVec[aK]);
           double aP0 = (aNewCum-aPerc) / (aNewCum-aCumPerc);
           return  aV0 * aP0 + aV1 * (1-aP0);
      }
      aCumPerc  = aNewCum;
      aLastP = aNewP;
   }
   

   return aGetV(aVec[aNBV-1]);

}

template <class TVal> class  cOperator2Double
{
    public :
      double operator ()(const TVal & aVal) const{ return aVal;}
      typedef double tValue;
};


template <class Type> class cVectorFoncteur
{
    public:
       const std::vector<Type> &  mVH;
       const Type  & operator()(const int & anInd) const {return mVH[anInd];}
       cVectorFoncteur(const std::vector<Type> & aVH) : mVH(aVH) {}
};
template <class Type> double GetValPercOfHisto
                      (
                            const std::vector<Type> & aVH,
                            double aPerc
                      )
{
    //cOperatorIdentite<int> anOI;
    std::vector<double> aVInd;
    int aNbV= (int)aVH.size();
    for (int aK=0 ; aK<aNbV ; aK++)
       aVInd.push_back(aK);

    cVectorFoncteur<Type> aVF(aVH);

    return GenValPdsPercentile
    (
         aVInd,
         aPerc,
         // cOperatorIdentite<int>(),
         cOperator2Double<int>(),
         aVF,
         SomPerc(aVH,cOperatorIdentite<Type>())
    );

}

inline bool IsOkData(const double & aVal)
{
    return ! std_isnan(aVal);
}

template <class TVal> void VerifData(TVal * Data,int aNb)
{
    for (int aK=0 ; aK<aNb ; aK++)
       if (! IsOkData(Data[aK]))
          ELISE_ASSERT(false,"VerifData");
}

/*
*/

// Permute les valeur de telle maniere que les [0,K[  premiere soit inferieure ou 
// egale aux suivante
template <class TVal> void SplitArrounKthValue(TVal * Data,int aNb,int aKth)
{
   if (aNb==0) return;
   if (aKth<=0) return;
   if (aKth>=aNb) return;
 // std::cout << " SplitArrounKthValue " << aNb << " " << aKth << "\n";
   // On calcule la moyenne
   TVal aMoy(0);

   for (int aKv=0 ; aKv<aNb ; aKv++)
      aMoy = aMoy+Data[aKv];
   aMoy = aMoy / aNb;

   // On permut de maniere a ce que les valeur du debut soit < Moy  et celle de la fin >=Moy
   int aK0 =0;
   int aK1 = aNb-1;
   while (aK0 < aK1)
   {
        while ((aK0<aK1) && (Data[aK0] <  aMoy)) aK0++;
        while ((aK0<aK1) && (Data[aK1] >= aMoy)) aK1--;
        if (aK0 < aK1) 
        {
           ElSwap(Data[aK0],Data[aK1]);
        }
   }
   ELISE_ASSERT(aK0==aK1,"Verif in SplitArrounKthValue");

   // Si le cas, on n'a pas progresse, toute les valeur sont egale
   if  (aK0==0)
   {
       return;
   }

   // Cas degenere, on teste toutes les valeurs egales
   {
      int aNbV0 = 0;
      TVal aV0 = Data[0];
      for (int aKv=0 ; aKv<aNb ; aKv++)
      {
          aNbV0 += (Data[aKv] == aV0);
      }
      if (aNbV0==aNb)
      {
          return;
      }
   }

   // A la fin K0=K1 et elle sont bien splite autour de la moyenne
   if (aK0 == aKth)  return;


   if (aK0 < aKth)  SplitArrounKthValue(Data+aK0,aNb-aK0,aKth-aK0);
   else             SplitArrounKthValue(Data,aK0,aKth);
}

template <class TVal> TVal Moy(TVal * Data,int aNb)
{
    TVal aSom=0;
    for (int aK=0 ; aK<aNb ; aK++)
        aSom += Data[aK];

    return aSom/aNb;
}

template <class TVal> TVal MaxTab(TVal * Data,int aNb)
{
    TVal aMax=Data[0];
    for (int aK=0 ; aK<aNb ; aK++)
        if (Data[aK]> aMax)
           aMax = Data[aK];

    return aMax;
}

template <class TVal> TVal MinTab(TVal * Data,int aNb)
{
    TVal aMin=Data[0];
    for (int aK=0 ; aK<aNb ; aK++)
        if (Data[aK]< aMin)
           aMin = Data[aK];

    return aMin;
}


template <class TVal> TVal KthVal(TVal * Data,int aNb,int aKth)
{
   ELISE_ASSERT(aKth>=0 && (aKth<=aNb-1),"KthVal");
   VerifData(Data,aNb);
   SplitArrounKthValue(Data,aNb,aKth);
   return MinTab(Data+aKth,aNb-aKth);
}

template <class TVal> TVal KthValGen(TVal * Data,int aNb,int aKth,const TVal & aDef)
{
    if (aNb==0) return aDef;
    if (aKth <=0) return  MinTab(Data,aNb);
    if (aKth >= (aNb-1)) return  MaxTab(Data,aNb);
    return KthVal(Data,aNb,aKth);
}



template <class TVal> void SplitArrounKthValue(std::vector<TVal> & aV,int aKth)
{
   VerifData(VData(aV),(int)aV.size());
   SplitArrounKthValue(VData(aV), (int)aV.size(),aKth);
}

template <class TVal> TVal MoyKPPVal(std::vector<TVal> & aV,int aKth)
{
   SplitArrounKthValue(aV,aKth);
   return Moy(VData(aV),aKth);
}

template <class TVal> TVal KthVal(std::vector<TVal> & aV,int aKth)
{
    return KthVal(VData(aV),(int)aV.size(),aKth);
}

template <class TVal> TVal MedianeSup(std::vector<TVal> & aV)
{
    return KthVal(aV,(int)(aV.size()/2));
}

template <class TVal> TVal KthValProp(std::vector<TVal> & aV,double aProp)
{
    return KthVal(VData(aV),(int)aV.size(),ElMax(0,ElMin(int(aV.size()-1),round_ni(aV.size()*aProp))));
}

double MedianPond(std::vector<Pt2df> &  aV,int * aKMed=0);

template <class Type,class TFctr> Pt3dr  P3DMed(const Type & aCont,const TFctr & aFctr)
{
     std::vector<double>  mCX;
     std::vector<double>  mCY;
     std::vector<double>  mCZ;

     for (const auto & anObj : aCont)
     {
         Pt3dr aP = aFctr(anObj);
         mCX.push_back(aP.x);
         mCY.push_back(aP.y);
         mCZ.push_back(aP.z);
     }
     return Pt3dr(MedianeSup(mCX),MedianeSup(mCY),MedianeSup(mCZ));
}



#endif  // _ELISE_EXT_STL_NUMERICS_H






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
