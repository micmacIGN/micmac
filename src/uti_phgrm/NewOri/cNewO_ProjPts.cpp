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

#include "NewOri.h"


/***********************************************/
/*           FONCTIONS GLOBALES                */
/***********************************************/





class cCdtSelect
{
     public :
        cCdtSelect(int aNum,const double & aPds) :
            mNum      (aNum),
            mPds0     (aPds),
            mPdsOccup (0.0),
            mDMin     (1e10),
            mTaken    (false)
         {
         }
        int    mNum;
        double mPds0;
        double mPdsOccup;
        double mDMin;
        bool   mTaken;
};


static double DefDistArret = -1;

template<class TypePt> class   cTplSelecPt
{
    public :
        cTplSelecPt(const std::vector<TypePt> & aVPres,const std::vector<double> * = 0);
        void InitPresel(int aNbPres);
        void CalcPond();
        const TypePt & PSel(const int & aK) const
        {
              return (*mVPts)[mVPresel[aK].mNum];
        }
        const TypePt & PSel(const cCdtSelect & aCdt) const
        {
              return (*mVPts)[aCdt.mNum];
        }

        void UpdateDistPtsAdd(const TypePt & aNewP);
        void SelectN(int aN,double aDistArret = DefDistArret);
        const std::vector<int>  & VSel() const {return mVSel;}

        double DistMinSelMoy();
        cResIPR & Res() {return mRes;}

        std::vector<double> VDistMin();

    private :
        double Pds(int aK) {return mVPds ? (*mVPds)[aK] : 1.0;}
 
        double mDistType;
        const std::vector<TypePt> * mVPts;
        const std::vector<double> * mVPds;
        std::vector<cCdtSelect>     mVPresel;
        int                         mNbPts;
        int                         mNbPres;
        cResIPR                     mRes;
        std::vector<int> &          mVSel;
        
};

template<class TypePt> cTplSelecPt<TypePt>::cTplSelecPt(const std::vector<TypePt> & aVPts,const std::vector<double> * aVPds) :
     mVPts  (&aVPts),
     mVPds  (aVPds),
     mNbPts ((int)mVPts->size()),
     mVSel  (mRes.mVSel)
{
}


template<class TypePt> void cTplSelecPt<TypePt>::InitPresel(int aNbPres)
{
    RMat_Inertie aMat;
    {
       cRandNParmiQ aSelec(aNbPres,mNbPts);
       for (int aK=0 ; aK<mNbPts ; aK++)
       {
            if (aSelec.GetNext())
            {
               const  TypePt & aPt = (*mVPts)[aK];
               aMat.add_pt_en_place(aPt.x,aPt.y);
               mVPresel.push_back(cCdtSelect(aK,Pds(aK)));
            }
       }
    }
    aMat = aMat.normalize();
    mNbPres = int(mVPresel.size());

    double aSurfType  =  sqrt (aMat.s11()* aMat.s22() - ElSquare(aMat.s12()));
    mDistType = sqrt(aSurfType/mNbPres);
}


template<class TypePt> void cTplSelecPt<TypePt>::CalcPond()
{
    for (int aKS1 = 0 ; aKS1 <mNbPres ; aKS1++)
    {
        const TypePt & aP1 = PSel(aKS1);
        for (int aKS2 = aKS1 ; aKS2 <mNbPres ; aKS2++)
        {
           double aDist = euclid( aP1-PSel(aKS2));
           // sqrt pour attenuer la ponderation
           double aPds = sqrt(1 / (mDistType+aDist));
           mVPresel[aKS1].mPdsOccup += aPds;
           mVPresel[aKS2].mPdsOccup += aPds;
        }
    }
    for (int aKSom = 0 ; aKSom<mNbPres ; aKSom++)
    {
       mVPresel[aKSom].mPdsOccup *= mVPresel[aKSom].mPds0;
    }
}



template<class TypePt> void cTplSelecPt<TypePt>::UpdateDistPtsAdd(const TypePt & aNewP)
{
   for (int aKSom = 0 ; aKSom <mNbPres ; aKSom++)
   {
       cCdtSelect & aCdt = mVPresel[aKSom];
       aCdt.mDMin = ElMin(aCdt.mDMin,euclid(PSel(aCdt)-aNewP));
   }
}


template<class TypePt> void cTplSelecPt<TypePt>::SelectN(int aTargetNbSel,double aDistArret)
{

    int aNbSomSel = ElMin(mNbPres,aTargetNbSel);
    mVSel.clear();
    bool Cont = true;




    for (int aKSel=0 ; (aKSel<aNbSomSel) && Cont ; aKSel++)
    {
         // Recherche du cdt le plus loin
         double aMaxDMin = -1;
         cCdtSelect * aBest = 0;
         for (int aKSom = 0 ; aKSom <mNbPres ; aKSom++)
         {
             cCdtSelect & aCdt = mVPresel[aKSom];
             double aDist = aCdt.mDMin *  aCdt.mPdsOccup;

             if ((!aCdt.mTaken) &&  (aDist > aMaxDMin))
             {
                 aMaxDMin = aDist;
                 aBest = & aCdt;
             }
         }

         ELISE_ASSERT(aBest!=0,"::SelectN");
         aBest->mTaken = true;

         UpdateDistPtsAdd(PSel(*aBest));
         mVSel.push_back(aBest->mNum);

         //    aKSom>50 => pour que la dist puisse etre fiable;  aKSom%10 pour gagner du temps
         if ( (aDistArret>0) && (aKSel>50)  && ((aKSel%10)==0) )
         {
             Cont = (DistMinSelMoy() > aDistArret);
         }
    }
}

template<class TypePt> std::vector<double> cTplSelecPt<TypePt>::VDistMin() 
{
   std::vector<double> aVDist;
   for (int aKS=0 ; aKS<int(mVSel.size()) ; aKS++)
   {
       aVDist.push_back(mVPresel[aKS].mDMin);
   }
   return aVDist;
}


template<class TypePt> double cTplSelecPt<TypePt>::DistMinSelMoy() 
{
    double aSom = 0.0;
    int aNb = 0;
    for (int aKS=0 ; aKS<int(mVSel.size()) ; aKS++)
    {
        double aD = mVPresel[aKS].mDMin;
        if (aD>0)
        {
            aSom += aD;
            aNb++;
        }
    }
    mRes.mMoyDistNN = aSom / ElMax(1,aNb);
    return mRes.mMoyDistNN;
}

cResIPR cResIPRIdent(int aNb)
{
   cResIPR aRes;
   for (int aK=0 ; aK< aNb ; aK++)
       aRes.mVSel.push_back(aK);
   aRes.mMoyDistNN = 0;
   return aRes;

}

template<class TypePt> cResIPR  TplIndPackReduit
                                (
                                     const std::vector<TypePt> & aVPts,
                                     int aNbMaxInit,
                                     int aNbFin, 
                                     const cResIPR * aResExist = 0,
                                     const std::vector<TypePt> * aVPtsExist = 0
                                )
{
    // risque d'avoir des degeneresnce
    if (aVPts.size() <= 5)
    {
        return cResIPRIdent((int)aVPts.size());
    }


    cTplSelecPt<TypePt> aSel(aVPts);
    aSel.InitPresel(aNbMaxInit);
    aSel.CalcPond();

    double aDistArret = DefDistArret;
    if (aResExist)
    {
       for (int aK=0 ; aK<int(aResExist->mVSel.size()) ; aK++)
       {
           aSel.UpdateDistPtsAdd((*aVPtsExist)[aResExist->mVSel[aK]]);
       }
       aDistArret = aResExist->mMoyDistNN;
    }


    aSel.SelectN(aNbFin,aDistArret);
    aSel.DistMinSelMoy();
    aSel.Res().mMoyDistNN *= sqrt(aSel.VSel().size()/double(aNbFin));

    return aSel.Res();
}



cResIPR  IndPackReduit(const std::vector<Pt2dr> & aV,int aNbMaxInit,int aNbFin)
{
   return TplIndPackReduit(aV,aNbMaxInit,aNbFin);
}

cResIPR  IndPackReduit(const std::vector<Pt2df> & aV,int aNbMaxInit,int aNbFin)
{
   return TplIndPackReduit(aV,aNbMaxInit,aNbFin);
}

cResIPR  IndPackReduit(const std::vector<Pt2df> & aV,int aNbMaxInit,int aNbFin,const cResIPR & aResExist,const std::vector<Pt2df> & aVPtsExist)
{
   cResIPR aRes = TplIndPackReduit(aV,aNbMaxInit,aNbFin,&aResExist,&aVPtsExist);
   return aRes;
}





void cNewO_OrInit2Im::TestNewSel(const ElPackHomologue & aPack)
{
     std::vector<Pt2dr> aVPts;
     for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
     {
         aVPts.push_back(itP->P1());
     }

     for (int aK=0 ; aK<int(aVPts.size()) ; aK++)
        mW->draw_circle_abs(ToW(aVPts[aK]),2.0,mW->pdisc()(P8COL::green));

     std::vector<int>  aVSel = TplIndPackReduit(aVPts,1500,500).mVSel;
     for (int aK=0 ; aK<int(aVSel.size()) ; aK++)
     {
         Pt2dr aP = ToW(aVPts[aVSel[aK]]);
         mW->draw_circle_abs(aP,4.0,mW->pdisc()(P8COL::red));
         mW->draw_circle_abs(aP,6.0,mW->pdisc()(P8COL::red));
     }
     mW->clik_in();
}
   
ElPackHomologue PackReduit(const ElPackHomologue & aPackIn,int aNbMaxInit,int aNbFin)
{
   std::vector<Pt2dr> aVP1;
   std::vector<Pt2dr> aVP2;
   for (ElPackHomologue::const_iterator itP=aPackIn.begin() ; itP!=aPackIn.end() ; itP++)
   {
       aVP1.push_back(itP->P1());
       aVP2.push_back(itP->P2());
   }

   ElPackHomologue aRes;
   std::vector<int>  aVSel = TplIndPackReduit(aVP1,aNbMaxInit,aNbFin).mVSel;
   for (int aK=0 ; aK<int(aVSel.size()) ; aK++)
   {
         ElCplePtsHomologues aCple(aVP1[aVSel[aK]],aVP2[aVSel[aK]]);
         aRes.Cple_Add(aCple);
   }


   return aRes;
}






ElPackHomologue PackReduit(const ElPackHomologue & aPackIn,int aNbFin)
{
    return PackReduit(aPackIn,aPackIn.size(),aNbFin);
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
