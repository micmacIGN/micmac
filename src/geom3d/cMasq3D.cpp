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

#if (ELISE_QT_VERSION >= 4)

#include "StdAfx.h"
#include "../saisieQT/include_QT/3DObject.h"

/***************************************************************/
/*                                                             */
/*                     cMasqBooleen3D                          */
/*                                                             */
/***************************************************************/

class cMasqBooleen3D
{
    public :
        // virtual bool HasAnswer(const Pt3dr & aP) const = 0;
        const SELECTION_MODE & ModeSel() const;
    protected :
        cMasqBooleen3D(SELECTION_MODE aMode); 
        SELECTION_MODE mModeSel;
};

cMasqBooleen3D::cMasqBooleen3D(SELECTION_MODE aMode) :
   mModeSel (aMode)
{
}

/***************************************************************/
/*                                                             */
/*                     cMasq3DOrthoRaster                      */
/*                                                             */
/***************************************************************/

class cMasq3DOrthoRaster : public cMasqBooleen3D
{
     public :
        static cMasq3DOrthoRaster ByPolyg3D(SELECTION_MODE aModeSel,const std::vector<Pt3dr> aPolygone,double aNbPix);
        cMasq3DOrthoRaster(SELECTION_MODE aSel,Pt2dr aP0,double aScal,ElRotation3D aE2P,Im2D_Bits<1> aImMasq);

        Pt2dr ToIm(const Pt3dr & aP3) const
        {
            return (Proj(mRE2P.ImAff(aP3))-mP0)  * mScale;
        }
        bool HasAnswer(const Pt3dr & aP) const 
        {
              return mTMasq.get(round_ni(ToIm(aP)),0);
        }

        void Test(const std::vector<Pt3dr> aPol3);
        
     public :
        Pt2dr  mP0;
        double mScale;
        ElRotation3D mRE2P;
        Im2D_Bits<1> mMasq;
        TIm2DBits<1> mTMasq;
};

cMasq3DOrthoRaster::cMasq3DOrthoRaster(SELECTION_MODE aModeSel,Pt2dr aP0,double aScal,ElRotation3D aE2P,Im2D_Bits<1> aImMasq) :
     cMasqBooleen3D (aModeSel),
     mP0 (aP0),
     mScale (aScal),
     mRE2P (aE2P),
     mMasq (aImMasq),
     mTMasq (mMasq)
{
}

void cMasq3DOrthoRaster::Test(const std::vector<Pt3dr> aPol3)
{
    Video_Win *  aW = Video_Win::PtrWStd(mMasq.sz());
    ELISE_COPY(aW->all_pts(),mMasq.in(),aW->odisc());

    for (int aKP=0 ; aKP<int(aPol3.size()) ; aKP++)
    {
        Pt2dr aP2 = ToIm(aPol3[aKP]);
        aW->draw_circle_loc(aP2,3.0,aW->pdisc()(P8COL::red));
    }
}

cMasq3DOrthoRaster cMasq3DOrthoRaster::ByPolyg3D(SELECTION_MODE aModeSel,const std::vector<Pt3dr> aPol3,double aNbPix)
{

    cElPlan3D aPlan(aPol3,0,0);

    ElRotation3D aP2E = aPlan.CoordPlan2Euclid();
    ElRotation3D aE2P = aP2E.inv();

    std::cout << "\n\n  TEST cMasq3DOrthoRaster \n";
    std::vector<Pt2dr> aVP2;
    Pt2dr aMin(1e20,1e20);
    Pt2dr aMax(-1e20,-1e20);
    double aZAM = 0; // Z Abs Max
    for (int aKP=0 ; aKP<int(aPol3.size()); aKP++)
    {
        Pt3dr  aQ3 = aE2P.ImAff(aPol3[aKP]);
        Pt2dr aP2(aQ3.x,aQ3.y);
        aVP2.push_back(aP2);
        aMax.SetSup(aP2);
        aMin.SetInf(aP2);
        aZAM = ElMax(aZAM,ElAbs(aQ3.z));
        std::cout << "Plan Coord " << aE2P.ImAff(aPol3[aKP]) << "\n";
    }
    ELISE_ASSERT(aZAM<1e-5,"Planarity in cMasq3DOrthoRaster::ByPolyg3D");

    Pt2dr aSzR = aMax-aMin;
    double aLarg = ElMax(aSzR.x,aSzR.y);
    //  Plan = > Ras  :  (aP-aMin) * aScal;
    double aScal = aNbPix / aLarg;

    Pt2di aSzI = round_up(aSzR*aScal);

    Im2D_Bits<1> aMasq(aSzI.x,aSzI.y,0);

    
    std::vector<Pt2di> aVP2I;
    for (int aKP=0 ; aKP<int(aVP2.size()); aKP++)
    {
         aVP2I.push_back(round_ni((aVP2[aKP]-aMin)*aScal));
    }
// quick_poly

    ELISE_COPY(polygone(ToListPt2di(aVP2I)),1,aMasq.out());


    
    if (0)
    {
        Video_Win *  aW = Video_Win::PtrWStd(aSzI);
        ELISE_COPY(aW->all_pts(),aMasq.in(),aW->odisc());

        std::cout << " aZAMaZAM " << aZAM  << " SzI " << aSzI  << "\n";
        std::cout << " MiMax " <<  aMin << " " << aMax << "\n";
    }


    cMasq3DOrthoRaster aRes(aModeSel,aMin,aScal,aE2P,aMasq);
    aRes.Test(aPol3);
    return aRes;
}


#include "MatrixManager.h"
void Test3dQT()
{
   // QString filename = "/home/marc/TMP/EPI/Champs/AperiCloud_All2_selectionInfo.xml";
   std::string Name = "/media/data2/Jeux-Test/Soldat-Temple-Hue/AperiCloud_AllRel_selectionInfo.xml";
   QString filename = Name.c_str();
   HistoryManager *HM = new HistoryManager();
   MatrixManager *MM = new MatrixManager();
   bool Ok = HM->load(filename);
   if (!Ok)
   {
       std::cout << "For File " << Name << "\n";
       ELISE_ASSERT(false,"Cannot load for 3D mask");
   }
   QVector <selectInfos> vInfos = HM->getSelectInfos();
std::cout << "BBBBBBB " << vInfos.size() << "\n";
   for (int aK=0; aK< vInfos.size();++aK)
   {
std::cout << "CCCCCC\n";
      selectInfos &Infos = vInfos[aK];
      MM->importMatrices(Infos);
      std::cout << "INFOS MODE " << Infos.selection_mode << "\n";
      // SELECTION_MODE aS = (SELECTION_MODE) Infos.selection_mode;
      std::vector<Pt3dr> aVP3;
      for (int bK=0;bK < Infos.poly.size();++bK)
      {
         QPointF pt = Infos.poly[bK];
         Pt3dr q0;
         MM->getInverseProjection(q0, pt, 0.0);
         aVP3.push_back(q0);
/*
         Pt3dr q1;
         MM->getInverseProjection(q1, pt, 1.f);
         std::cout << q0 << q1 <<   q0-q1 << "\n";
*/
         
      }

      cMasq3DOrthoRaster::ByPolyg3D((SELECTION_MODE) Infos.selection_mode,aVP3,300.0);
   }
   getchar();
}

#endif







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
