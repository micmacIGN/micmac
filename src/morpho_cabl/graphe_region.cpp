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



#include "StdAfx.h"




/**************************************************/
/*                                                */
/*            ElImplemDequantifier                */
/*                                                */
/**************************************************/


extern void ShowPoint(const std::vector<Pt2di> & aV,int aCoul,int aModeCoul);


template <class TIm1,class TIm2,class Container>  class cCompCnx
{
    public :

        typedef typename El_CTypeTraits<TIm1>::tBase tBase1;
        typedef Im2D<TIm1,tBase1>           tIm1;
        typedef TIm2D<TIm1,tBase1>          tTim1;

        typedef typename El_CTypeTraits<TIm2>::tBase tBase2;
        typedef Im2D<TIm2,tBase2>           tIm2;
        typedef TIm2D<TIm2,tBase2>          tTim2;

static void DoCC(Container & aCont,bool V8,const Pt2di & aSeed,tIm1 anImValue,tIm2 anImMarq,int  anUpdateMarq,ElGrowingSetInd * aSetVois,bool Reset)
{
   if (aSetVois)
       aSetVois->clear();
   aCont.clear();
   tTim2 aTIMarq(anImMarq);
   tBase2 aMarq = aTIMarq.get(aSeed);

   if (aMarq==anUpdateMarq) 
      return;
   

   aCont.push_back(aSeed);
   tTim1 aTIVal(anImValue);

   tBase1 aVal = aTIVal.get(aSeed);
   aTIMarq.oset(aSeed,anUpdateMarq);

   int aK0 = 0;

   int aNbV = V8 ? 8 : 4;
   Pt2di * TabV = V8 ? TAB_8_NEIGH : TAB_4_NEIGH;

   while (aK0 != int(aCont.size()))
   {
       Pt2di aP = aCont[aK0];

       for (int aKV=0 ; aKV<aNbV ; aKV++)
       {
            Pt2di aQ = aP +  TabV[aKV];
            int aMarqV = aTIMarq.get(aQ);
            if (aMarqV==aMarq) 
            {
                 if(aTIVal.get(aQ)==aVal) 
                 {
                     aCont.push_back(aQ);
                     aTIMarq.oset(aQ,anUpdateMarq);
                 }
            }
            else
            {
                if (aSetVois && (aMarqV>=0) && (aMarqV!=anUpdateMarq))
                {
                      aSetVois->insert(aMarqV);
                }
            }

       }
       aK0++;
   }

   if (Reset)
      for (int aK=0 ; aK<int(aCont.size()) ; aK++)
          aTIMarq.oset(aCont[aK],aMarq);
}
};


template  <class TIm1,class TIm2,class Container>
         void CompCnx (Container & aCont,bool V8,const Pt2di & aSeed,TIm1 anImValue,TIm2 anImMarq,int anUpdateMarq,ElGrowingSetInd *aSV=0)
{
    cCompCnx<typename TIm1::tElem,typename TIm2::tElem,Container>::DoCC(aCont,V8,aSeed,anImValue,anImMarq,anUpdateMarq,aSV,false);
}


template  <class TIm1,class TIm2,class Container>
         void CompCnxCste (Container & aCont,bool V8,const Pt2di & aSeed,TIm1 anImValue,TIm2 anImMarq)
{
    cCompCnx<typename TIm1::tElem,typename TIm2::tElem,Container>::DoCC(aCont,V8,aSeed,anImValue,anImMarq,anImMarq.GetI(aSeed)+1,0,true);
}





class cGR_AttrSom
{
   public :
     cGR_AttrSom();
     cGR_AttrSom(Pt2di aP0,int aValRegion,int aNumRegion,int aSz);
     int  ValR() const {return mValR;}
     bool & DoneValR() {return mDoneValR;}
     bool & Valid()    {return mValid;}
     int Sz() const {return mSz;}
     const Pt2di &  P0() const {return mP0;}
   private :
     Pt2di mP0;
     int   mValR;
     int   mNumR;
     int   mSz;
     bool  mValid;
     bool  mDoneValR;
};

cGR_AttrSom::cGR_AttrSom(Pt2di aP0,int aValR,int aNumR,int aSz):
  mP0     (aP0),
  mValR   (aValR),
  mNumR   (aNumR),
  mSz     (aSz),
  mValid  (false),
  mDoneValR  (false)
{
}

cGR_AttrSom::cGR_AttrSom() :
   mSz (-1)
{
}


template <class TCont> int SomSz(const TCont & aCont)
{
   int aRes = 0;
   for (int aK=0 ; aK<int(aCont.size()) ;  aK++)
      aRes += aCont[aK]->attr().Sz() ;
   return aRes;
}


template <class TCont,class TIm1,class TIm2>  void ShowComp(const TCont & aCont,bool V8,int aCoul,TIm1 anImValue,TIm2 anImMarq)
{
   for (int aK=0 ; aK<int(aCont.size()) ;  aK++)
   {
       std::vector<Pt2di> aVPts;
       CompCnxCste(aVPts,V8,aCont[aK]->attr().P0(),anImValue,anImMarq);
       ELISE_ASSERT(int(aVPts.size())==aCont[aK]->attr().Sz(),"ShowComp");
       ShowPoint(aVPts,aCoul,2);
   }
}



class cGR_AttrArc
{
   public :
};


typedef ElSom<cGR_AttrSom,cGR_AttrArc>    tGRSom;
typedef ElGraphe<cGR_AttrSom,cGR_AttrArc> tGRGraf;

class cSubGrInterv : public ElSubGraphe<cGR_AttrSom,cGR_AttrArc>
{
    public :
        cSubGrInterv(int aNbMin,int aNbMax) :
          mNbMin (aNbMin),
          mNbMax (aNbMax)
        {
        }
        bool inS(TSom & aS)  {return (aS.attr().ValR()>=mNbMin) &&  (aS.attr().ValR()<=mNbMax) ;}
    private :
        int mNbMin;
        int mNbMax;
};

class cSubGrSzSup : public ElSubGraphe<cGR_AttrSom,cGR_AttrArc>
{
     public :
         cSubGrSzSup(int aSzMin) :
             mSzMin (aSzMin)
         {
         }
         bool inS(TSom & aS)  {return (aS.attr().Sz()>=mSzMin);}
    private :
        int mSzMin;
};

class cParamGrReg 
{
     public :
        cParamGrReg(int aSzMinInit,int aSeuilValRDelta,int aSeuilZonWS,int aSzBarb) :
            mSzMinInit       (aSzMinInit),
            mSeuilValRDelta  (aSeuilValRDelta),
            mSeuilZonWS      (aSeuilZonWS),      // Water Shade
            mSzBarb          (aSzBarb)
        {
        }

        int mSzMinInit;
        int mSeuilValRDelta;
        int mSeuilZonWS;
        int mSzBarb;
};

template <class TypeIm,class TypeBase> Im2D_Bits<1> CreateGr(tGRGraf & mGr,Im2D<TypeIm,TypeBase> anIm,int aLabelOut,const cParamGrReg & aPGR)
{
    bool V8=true;
    //int aNbV = V8 ? 8 : 4;
    //Pt2di * TabV = V8 ? TAB_8_NEIGH : TAB_4_NEIGH;

    TIm2D<TypeIm,TypeBase> aTIm(anIm);
    Pt2di aSz = anIm.sz();

    Im2D_INT4 aImLabel(aSz.x,aSz.y,-1);
    TIm2D<INT4,INT4> aTL(aImLabel);
    ELISE_COPY(aImLabel.border(1),-2,aImLabel.out());

    Pt2di aP0;
    int aCpt=0;
    ElGrowingSetInd aSV(1000);
    std::vector<tGRSom *> aVSom;
    for (aP0.y=0 ; aP0.y<aSz.y ; aP0.y++)
    {
        for (aP0.x=0 ; aP0.x<aSz.x ; aP0.x++)
        {
             int aLabel = aTIm.get(aP0);
             if ((aTL.get(aP0)==-1) && (aLabel != aLabelOut))
             {
                std::vector<Pt2di> aVPts;
                CompCnx(aVPts,V8,aP0,anIm,aImLabel,INT4(aCpt),&aSV);
                int aNbPts = (int)aVPts.size();
                if (aNbPts >= aPGR.mSzMinInit)
                {
                   cGR_AttrSom anAttr(aP0,aLabel,aCpt,aNbPts);
                 
                   tGRSom &  aSom = mGr.new_som(anAttr);
                   aVSom.push_back(&aSom);

                   for (ElGrowingSetInd::const_iterator itV=aSV.begin(); itV!=aSV.end() ; itV++)
                   {
                        tGRSom * aS2 = aVSom[*itV];
                        if (aS2)
                        {
                            cGR_AttrArc anAA;
                            mGr.add_arc(aSom,*aS2,anAA);
                        }
                   }
                }
                else
                {
                     aVSom.push_back(0);
                }
                aCpt++;
             }
        }
    }

    //std::cout << "BGIN FLAG MONT-DESC\n"; getchar();
    // Calcul des flag montant et descandant 
    int aFlagArcMont = mGr.alloc_flag_arc();
    int aFlagArcDesc = mGr.alloc_flag_arc();
    ElSubGraphe<cGR_AttrSom,cGR_AttrArc> aSubAll;
    for (int aKS=0 ; aKS<int(aVSom.size()) ; aKS++)
    {
        tGRSom * aSom = aVSom[aKS];
        if (aSom)
        {
            tGRSom * aSMax = aSom;
            tGRSom * aSMin = aSom;
            for (tGRSom::TArcIter itA=aSom->begin(aSubAll); itA.go_on(); itA++)
            {
                tGRSom * aS2 = &(itA->s2());
                if (    (aS2->attr().ValR() > aSMax->attr().ValR())
                     || ((aS2->attr().ValR()==aSMax->attr().ValR()) && (aS2->attr().Sz() > aSMax->attr().Sz()))
                   )
                {
                    aSMax = aS2;
                }

                if (    (aS2->attr().ValR() < aSMin->attr().ValR())
                     || ((aS2->attr().ValR()==aSMin->attr().ValR()) && (aS2->attr().Sz() > aSMin->attr().Sz()))
                   )
                {
                    aSMin = aS2;
                }
            }

            if (aSMax != aSom)
               mGr.arc_s1s2(*aSom,*aSMax)->sym_flag_set_kth_true(aFlagArcMont);
            if (aSMin != aSom)
               mGr.arc_s1s2(*aSom,*aSMin)->sym_flag_set_kth_true(aFlagArcDesc);
        }
    }
    // std::cout << "EeenDD  FLAG MONT-DESC\n";

    //  Analyse zone water shade
    for (int aKMD=0 ; aKMD<2 ; aKMD++)
    {
         bool isMont = (aKMD==0);
         int aFlagArc = (isMont) ? aFlagArcMont : aFlagArcDesc;
         ElPartition<tGRSom * > aPart;
         ElSubGraphe<cGR_AttrSom,cGR_AttrArc> aSubAll;
         cSubGrFlagArc< ElSubGraphe<cGR_AttrSom,cGR_AttrArc> > aSub(aSubAll,aFlagArc);
         PartitionCC(aPart,mGr,aSub);


         std::vector<tGRSom *> aVBarbs;
         cSubGrSzSup aGrCons(aPGR.mSzBarb);
         Ebarbule(mGr,aSub,aGrCons,aVBarbs);
         std::set<tGRSom *> aSetB(aVBarbs.begin(),aVBarbs.end());

         for (int aKC=0 ; aKC< aPart.nb() ; aKC++)
         {
             ElSubFilo<tGRSom *> aCC = aPart[aKC];
             int aSzTot = SomSz(aCC);
             bool Ok = aSzTot >= aPGR.mSeuilZonWS ;
             if (Ok)
             {
                 for (int aKS=0 ; aKS<aCC.size() ; aKS++)
                 {
                     if (aSetB.find(aCC[aKS])==aSetB.end())
                     {
                        aCC[aKS]->attr().Valid() = true;
                     }
                     else
                     {
                     }
                 }
             }
         }
    }
    // std::cout << "EeenDD  PARTITION\n"; getchar();

 
    double aSomT=0;
    double aMaxT=0;

    int aDelta = 1;  // 1 
    for (int aKS=0 ; aKS<int(aVSom.size()) ; aKS++)
    {
        tGRSom * aSom = aVSom[aKS];
        if (aSom && (! aSom->attr().DoneValR()))
        {
            ElTimer aChrono;
            cSubGrInterv aSG(aSom->attr().ValR()-aDelta,aSom->attr().ValR()+aDelta);
            ElFifo<tGRSom *> aCC;
            comp_connexe_som(aCC,aSom,aSG);
            int aSzCC = SomSz(aCC);
            bool Ok = aSzCC >= aPGR.mSeuilValRDelta;

            for (int aKC=0 ; aKC<int(aCC.size()) ; aKC++)
            {
                  tGRSom * aSom2 = aCC[aKC];
                  if (aSom2->attr().ValR() == aSom->attr().ValR())
                     aSom2->attr().DoneValR() = true;
                  if (Ok)
                     aSom2->attr().Valid() = true;
            }
            // ShowComp(aCC,V8,(Ok?255:0), anIm,aImLabel);

            double aT = aChrono.uval();
            aSomT += aT;
            if (aT > aMaxT)
            {
               aMaxT = aT;
            }
        }
    }

/*
*/

   //   EXPORT DES RESULTATS FINALS 
    Im2D_Bits<1>  aImRes(aSz.x,aSz.y,0);
    TIm2DBits<1>  aTImRes(aImRes);
    for (int aKS=0 ; aKS<int(aVSom.size()) ; aKS++)
    {
        tGRSom * aSom = aVSom[aKS];
        if (aSom  && aSom->attr().Valid())
        {
           std::vector<Pt2di> aVPts;
           CompCnxCste(aVPts,V8,aSom->attr().P0(),anIm,aImLabel);
           for (int aKP=0 ; aKP<int(aVPts.size()) ; aKP++)
           {
              aTImRes.oset(aVPts[aKP],1);
           }
        }
    }
    return  aImRes;
}



//template  Im2D_INT4 CreateGr(cGR_Graphe&,Im2D<INT2,INT>);

Im2D_Bits<1>  TestLabel(Im2D_INT2 aILabel,INT aLabelOut)
{
    //cCompCnx<INT2,INT,std::vector<Pt2di> > aCC;
    cParamGrReg aPGR(2,150,1000,10);
    tGRGraf aGr;
    return CreateGr(aGr,aILabel,aLabelOut,aPGR);
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
