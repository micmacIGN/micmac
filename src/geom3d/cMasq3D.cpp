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

#if (ELISE_QT_VERSION >= 4)

#include "../saisieQT/include_QT/3DObject.h"

class cMasqBin3D
{
     public :
        virtual bool IsInMasq(const Pt3dr &) const = 0;
        virtual ~cMasqBin3D();
        static cMasqBin3D * FromSaisieMasq3d(const std::string & aName);

        Im2D_Bits<1>  Mas2DPointInMasq3D(const cElNuage3DMaille &);

     private :
};


bool IsModeGlobal(SELECTION_MODE aMode)
{
   return (aMode==ALL) || (aMode==NONE) || (aMode==INVERT);
}
bool IsModeConst(SELECTION_MODE aMode)
{
   return (aMode==ALL) || (aMode==NONE) ;
}
bool IsModeAdditif(SELECTION_MODE aMode)
{
   return (aMode==ALL) || (aMode==ADD_INSIDE) || (aMode==ADD_OUTSIDE) ;
}

SELECTION_MODE ModeInverse(SELECTION_MODE aMode)
{
   switch (aMode)
   {
        case  SUB_INSIDE : return ADD_INSIDE;
        case  ADD_INSIDE : return SUB_INSIDE;
        case  SUB_OUTSIDE : return ADD_OUTSIDE;
        case  ADD_OUTSIDE : return SUB_OUTSIDE;
        case  ALL : return NONE;
        case  NONE : return ALL;

        default : ;
   }
   ELISE_ASSERT(false,"Cannot make ModeInverse");
   return INVERT;
}




class cMasq3DPartiel;
class cMasq3DOrthoRaster;


class cMasq3DPartiel
{
    public :
        virtual bool HasAnswer(const Pt3dr & aP) const = 0;
        const SELECTION_MODE & ModeSel() const;
        const bool &  Additif() const;
        virtual ~cMasq3DPartiel() ;
        void Invert();
    protected :
        cMasq3DPartiel(SELECTION_MODE aMode);
        SELECTION_MODE mModeSel;
        bool           mAdditif;

};

class cMasq3DConst : public cMasq3DPartiel
{
      public :
           cMasq3DConst(SELECTION_MODE aMode);
           virtual ~cMasq3DConst();
           bool HasAnswer(const Pt3dr & aP) const  ;
      private :
};

class cMasq3DOrthoRaster : public cMasq3DPartiel
{
     public :
        virtual ~cMasq3DOrthoRaster() ;
        static cMasq3DOrthoRaster * ByPolyg3D(SELECTION_MODE aModeSel,const std::vector<Pt3dr> aPolygone,double aNbPix);
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


class cMasq3DEmpileMasqPart : public  cMasqBin3D
{
     public :

        virtual bool IsInMasq(const Pt3dr &) const ;
        virtual ~cMasq3DEmpileMasqPart();
        cMasq3DEmpileMasqPart(const std::vector<cMasq3DPartiel *> & aVM);
        static cMasq3DEmpileMasqPart * FromSaisieMasq3d(const std::string & aName);
     private  :

         std::vector<cMasq3DPartiel *> mVM;  // Est inverse / au masque

};

/***************************************************************/
/*                                                             */
/*                     cMasq3DPartiel                          */
/*                                                             */
/***************************************************************/


cMasq3DPartiel::cMasq3DPartiel(SELECTION_MODE aMode) :
   mModeSel (aMode),
   mAdditif (IsModeAdditif(mModeSel))
{
}

cMasq3DPartiel::~cMasq3DPartiel()
{
}


const SELECTION_MODE & cMasq3DPartiel::ModeSel() const {return mModeSel;}
const bool & cMasq3DPartiel::Additif() const {return mAdditif;}


void cMasq3DPartiel::Invert()
{
    mModeSel =  ModeInverse(mModeSel);
}

/***************************************************************/
/*                                                             */
/*                     cMasq3DConst                            */
/*                                                             */
/***************************************************************/

cMasq3DConst::cMasq3DConst(SELECTION_MODE aMode) :
    cMasq3DPartiel(aMode)
{
}
cMasq3DConst::~cMasq3DConst()
{
}
bool cMasq3DConst::HasAnswer(const Pt3dr & aP) const
{
   return true;
}

/***************************************************************/
/*                                                             */
/*                     cMasq3DOrthoRaster                      */
/*                                                             */
/***************************************************************/


cMasq3DOrthoRaster::cMasq3DOrthoRaster(SELECTION_MODE aModeSel,Pt2dr aP0,double aScal,ElRotation3D aE2P,Im2D_Bits<1> aImMasq) :
     cMasq3DPartiel (aModeSel),
     mP0 (aP0),
     mScale (aScal),
     mRE2P (aE2P),
     mMasq (aImMasq),
     mTMasq (mMasq)
{
}

cMasq3DOrthoRaster::~cMasq3DOrthoRaster()
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
    getchar();
}

cMasq3DOrthoRaster * cMasq3DOrthoRaster::ByPolyg3D(SELECTION_MODE aModeSel,const std::vector<Pt3dr> aPol3,double aNbPix)
{

    cElPlan3D aPlan(aPol3,0,0);

    ElRotation3D aP2E = aPlan.CoordPlan2Euclid();
    ElRotation3D aE2P = aP2E.inv();

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




    cMasq3DOrthoRaster * aRes = new cMasq3DOrthoRaster(aModeSel,aMin,aScal,aE2P,aMasq);
    // aRes.Test(aPol3);
    return aRes;
}

/***************************************************************/
/*                                                             */
/*                   cMasq3DEmpileMasqPart                     */
/*                                                             */
/***************************************************************/

cMasq3DEmpileMasqPart::~cMasq3DEmpileMasqPart()
{
}

bool cMasq3DEmpileMasqPart::IsInMasq(const Pt3dr & aP) const
{
   for (int aK=0 ; aK<int(mVM.size()) ; aK++)
   {
      if (mVM[aK]->HasAnswer(aP))
          return mVM[aK]->Additif();
   }
   ELISE_ASSERT(false,"cMasq3DEmpileMasqPart::IsInMasq");
   return false;
}

cMasq3DEmpileMasqPart::cMasq3DEmpileMasqPart(const std::vector<cMasq3DPartiel *> & aVM)
{
   bool doInvert = false;
   bool HasConst = false;
   bool LastAdd = false;
   for (int aK=aVM.size()-1 ; (aK>=0) && (!HasConst) ; aK--)
   {
        cMasq3DPartiel * aMk = aVM[aK];
        SELECTION_MODE aMode = aMk->ModeSel();
        if (aMode == INVERT)
        {
            doInvert = ! doInvert ;
        }
        else
        {
             mVM.push_back(aVM[aK]);
             if (doInvert)
                mVM.back()->Invert();
             if (IsModeConst(aMode))
                HasConst = true;
             else
             {
                 LastAdd = IsModeAdditif(aMode);
             }
        }
   }
   if (! HasConst)
   {
       SELECTION_MODE aMode = LastAdd ?  NONE : ALL ;

       if (doInvert) aMode = ModeInverse(aMode);
       mVM.push_back(new cMasq3DConst(aMode));
   }
}

/*
*/

/*
bool cMasq3DEmpileMasqPart::IsInMasq(const Pt3dr & aP) const
{
   for (int aK=mVM.size() -1 ; aK>=1 ; aK++)
   {
        if (mVM[aK]->HasAnswer(aP))
        {
        }
        virtual bool HasAnswer(const Pt3dr & aP) const = 0;
        const SELECTION_MODE & ModeSel() const;
   }
}
*/


#include "MatrixManager.h"
cMasq3DEmpileMasqPart * cMasq3DEmpileMasqPart::FromSaisieMasq3d(const std::string & aName)
{
   // QString filename = "/home/marc/TMP/EPI/Champs/AperiCloud_All2_selectionInfo.xml";
   QString filename = aName.c_str();
   HistoryManager *HM = new HistoryManager();
   MatrixManager *MM = new MatrixManager();
   bool Ok = HM->load(filename);
   if (!Ok)
   {
       std::cout << "For File " << aName << "\n";
       ELISE_ASSERT(false,"Cannot load for 3D mask");
   }
   QVector <selectInfos> vInfos = HM->getSelectInfos();

   bool Cont=true;
   int aK0 =  vInfos.size()-1;
   for ( ; (aK0>=0) && Cont ; aK0--)
   {
      SELECTION_MODE aMode = (SELECTION_MODE) vInfos[aK0].selection_mode;
      if (IsModeConst(aMode))
         Cont = false;
   }
   aK0++;

   std::vector<cMasq3DPartiel * > aVMP;
   for (int aK= aK0; aK< vInfos.size();++aK)
   {
      selectInfos &Infos = vInfos[aK];
      MM->importMatrices(Infos);
      SELECTION_MODE aMode = (SELECTION_MODE) Infos.selection_mode;
      if (IsModeGlobal(aMode))
      {
           aVMP.push_back(new cMasq3DConst(aMode));
      }
      else
      {
          std::vector<Pt3dr> aVP3;
          for (int bK=0;bK < Infos.poly.size();++bK)
          {
             QPointF pt = Infos.poly[bK];
             Pt3dr q0;
             MM->getInverseProjection(q0, pt, 0.0);
             aVP3.push_back(q0);
          }

          aVMP.push_back(cMasq3DOrthoRaster::ByPolyg3D((SELECTION_MODE) Infos.selection_mode,aVP3,300.0));
      }
   }
   return new cMasq3DEmpileMasqPart(aVMP);
}

void Test3dQT()
{
   std::string aNameMasq3D  = "/media/data2/Jeux-Test/Soldat-Temple-Hue/AperiCloud_AllRel_selectionInfo.xml";
   std::string aNameNuage = "/media/data2/Jeux-Test/Soldat-Temple-Hue/MTD-Image-IMGP7048.JPG/Fusion_NuageImProf_LeChantier_Etape_1.xml";
   std::string aNameSh= StdPrefix(aNameNuage) + "Shade.tif";

   cMasqBin3D * aM3D = cMasq3DEmpileMasqPart::FromSaisieMasq3d(aNameMasq3D);
   cElNuage3DMaille * aNuage = cElNuage3DMaille::FromFileIm(aNameNuage);

   Im2D_Bits<1>  aMasq = aM3D->Mas2DPointInMasq3D(*aNuage);

   Pt2dr aSzMax(1200,800);
   Pt2di aSzI = aMasq.sz();
   double aRatio = ElMin(aSzMax.x/aSzI.x,aSzMax.y/aSzI.y);

   Pt2dr aSzW =  Pt2dr(aSzI) * aRatio;

   Video_Win * aW = Video_Win::PtrWStd(round_ni(aSzW),true,Pt2dr(aRatio,aRatio));

   Tiff_Im aImS(aNameSh.c_str());
   Symb_FNum aSSh(aImS.in(0));

   ELISE_COPY
   (
       aMasq.all_pts(),
       Virgule(aMasq.in()*aSSh,aSSh,aSSh),
       aW->orgb()
   );
   getchar();
}




/***************************************************************/
/*                                                             */
/*                       cMasqBin3D                            */
/*                                                             */
/***************************************************************/

cMasqBin3D::~cMasqBin3D()
{
}


Im2D_Bits<1>  cMasqBin3D::Mas2DPointInMasq3D(const cElNuage3DMaille & aNuage)
{
    Pt2di aSz = aNuage.SzUnique();
    Im2D_Bits<1> aRes(aSz.x,aSz.y,0);
    TIm2DBits<1> aTRes(aRes);

    Pt2di aP;
    for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
        {
             if (aNuage.IndexHasContenu(aP))
             {
                  Pt3dr aP3 = aNuage.PtOfIndex(aP);
                  if (IsInMasq(aP3))
                     aTRes.oset(aP,1);
             }
        }
    }

    return aRes;
}


int Masq3Dto2D_main(int argc,char ** argv)
{
    std::string aNameMasq3D;
    std::string aNameNuage;
    std::string aNameRes;
    std::string aNameMasq="";

    bool AcceptNew2d=false;
    int aDilate=2;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameMasq3D,"Name of 3D masq")
                    << EAMC(aNameNuage,"Name of Raster Nuage")
                    << EAMC(aNameRes,"Name of Resulting 2D masq"),
        LArgMain()  << EAM(AcceptNew2d,"OkNew2d",true, "Accept New 2D Image, Def=false")
                    << EAM(aDilate,"Dilate",true, "Dilatation of masq")
                    << EAM(aNameMasq,"MasqNuage",true, "Masq of Nuage if dif of XML File")
    );

   cMasqBin3D * aM3D = cMasq3DEmpileMasqPart::FromSaisieMasq3d(aNameMasq3D);

   cXML_ParamNuage3DMaille aXmlPN = StdGetFromSI(aNameNuage,XML_ParamNuage3DMaille);
   cElNuage3DMaille * aNuage = cElNuage3DMaille::FromParam
                               (
                                    aXmlPN,
                                    DirOfFile(aNameNuage),
                                    aNameMasq,
                                    1.0
                               );

   if (! ELISE_fp::exist_file(aNameRes))
   {
      if (!AcceptNew2d)
      {
          ELISE_ASSERT(false,"Masq3Dto2D file res do not exist, set AcceptNew2d=true for create")
      }
      Tiff_Im aFileRes(aNameRes.c_str(),aNuage->SzUnique(),GenIm::u_int1,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
      ELISE_COPY(aFileRes.all_pts(),1,aFileRes.out());
   }

   Im2D_Bits<1>  aImMasq2D =  aM3D->Mas2DPointInMasq3D(*aNuage);

   Tiff_Im aFileRes(aNameRes.c_str());

   ELISE_COPY
   (
       aFileRes.all_pts(),
       dilat_32(aImMasq2D.in(0),2*aDilate),
       aFileRes.out()
   );
   return 1;
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
