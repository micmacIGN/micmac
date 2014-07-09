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
#include "general/all.h"
#include "private/all.h"
#include "MICMAC.h"
#include "im_tpl/image.h"

using namespace NS_ParamMICMAC;

class cSuperpMM;
class cOneImageSup;

/*****************************************/
/*                                       */
/*          DECLARATIONS                 */
/*                                       */
/*****************************************/

class cOneImageSup
{
    public :
      cOneImageSup
      (
           cSuperpMM & ,
           cPriseDeVue * aPDV,
           cOneImageSup * aPDVRef,
           cElRegex_Ptr ,
           std::string
      );
      void LoadIm(bool First);
      Fonc_Num ColBal(); // Couleur "balancee des blancs"

      // Valeur "vraie", utilise les calcul de MEC
      Pt2dr  LocFromRef(Pt2dr aP);

      // Valeur "fausse", utilise la saisie initiale

    private :
      Pt2dr  FromRef(Pt2dr aP);
      Pt2dr  FromRefByGrid(Pt2dr aP);
      Pt2dr  FromRefByHom(Pt2dr aP);


      cSuperpMM &    mSup;
      const cSuperpositionImages  & mSupIm;
      cPriseDeVue *  mPDV;
      cOneImageSup * mRef;
      cAppliMICMAC & mAppli;
      cDbleGrid *    mDbleGr;
      cDbleGrid *    mCalib;
      Pt2di          mSzRef;
      Im2D_REAL4     mImGeoRef;
      Im2D_REAL4     mImBrute;
      double         mValB;  // Val de blanc
      double         mValN;  // Val de blanc
      Pt2di          mP0Im;
      cElHomographie  mHomFromRef;
};

class cSuperpMM
{
    public :
        cSuperpMM(int argc,char ** argv);
        void NoOp() {}
        Pt2di  P0() const {return mP0;}
        Pt2di  Sz() const {return mSz;}
        Box2dr Box() {return Box2dr(P0(),P0()+Sz());}
        cAppliMICMAC & Appli() {return mAppli;}
 
        Pt2di  PBB() const {return mPBB;}
        void LoadIm(bool First);
        Fonc_Num  RGB();
        const cSuperpositionImages  & SupIm() const {return mSup;}
    private :
       cAppliMICMAC & mAppli;
       std::vector<cPriseDeVue *>  mPDVS;
       cSuperpositionImages        mSup;
       Pt2di                       mP0;
       Pt2di                       mSz;
       std::vector<cOneImageSup *> mIms;
       Pt2di                       mPBB; // Pt Loc de Balance Blanc
       Pt3di                       mOC;
       Video_Win *                 mW;
       
};

/*****************************************/
/*                                       */
/*          cSuperpMM                    */
/*                                       */
/*****************************************/

cSuperpMM::cSuperpMM(int argc,char ** argv) :
    mAppli (*(cAppliMICMAC::Alloc(argc,argv,eAllocAM_Surperposition))),
    mPDVS  (mAppli.AllPDV()),
    mSup   (mAppli.SuperpositionImages().Val()),
    mP0    (mSup.P0Sup().ValWithDef(Pt2di(0,0))),
    mPBB   (-1,-1),
    mOC    (mSup.OrdreChannels()),
    mW     (0)
{
    if (mSup.SzSup().IsInit())
    {
         mSz = mSup.SzSup().Val();
         if (mSup.PtBalanceBlancs().IsInit())
         {
            mP0 = mSup.PtBalanceBlancs().Val() - mSz/2;
            mPBB = mSup.PtBalanceBlancs().Val() - mP0;
         }
    }
    else
    {
          mSz =  Std2Elise(mPDVS[0]->IMIL()->Sz(1))-mP0;
    }

    mIms.push_back(new cOneImageSup(*this,mPDVS[0],0,0,""));

    cElRegex_Ptr anAutom = mSup.PatternSelGrid();
    std::string aName = mSup.PatternNameGrid();
    if (mPDVS.size() > 1)
       mIms.push_back(new cOneImageSup(*this,mPDVS[1],mIms[0],anAutom,aName));

    if (mPDVS.size() > 2)
       mIms.push_back(new cOneImageSup(*this,mPDVS[2],mIms[0],anAutom,aName));


    mW = Video_Win::PtrWStd(mSz);

    LoadIm(true);
}

void cSuperpMM::LoadIm(bool First)
{
   for (int aK=0 ; aK<int(mIms.size()) ; aK++)
       mIms[aK]->LoadIm(First);

   if (mW)
   {
       ELISE_COPY
       (
          mW->all_pts(),
          RGB(),
          mW->orgb()
       );
   }
   if (mSup.GenFileImages().Val())
   {
       std::string aName =   mAppli.FullDirResult()
                           + "TestSuperposition.tif";
       Tiff_Im::Create8BFromFonc
       (
           aName,
           mSz,
           RGB()
             // Tiff_Im::Empty_ARG
           // + Arg_Tiff(Tiff_Im::ANoStrip())
       );
   }
   getchar();
}


Fonc_Num  cSuperpMM::RGB()
{
   return Virgule
          (
               mIms[mOC.x]->ColBal(),
               mIms[mOC.y]->ColBal(),
               mIms[mOC.z]->ColBal()
          );
}
/*****************************************/
/*                                       */
/*          cOneImageSup                 */
/*                                       */
/*****************************************/

cOneImageSup::cOneImageSup
(
           cSuperpMM &   aSup,
           cPriseDeVue * aPDV,
           cOneImageSup * aRef,
           cElRegex_Ptr   anAutom,
           std::string    aMotif
)  :
   mSup       (aSup),
   mSupIm     (mSup.SupIm()),
   mPDV       (aPDV),
   mRef       (aRef),
   mAppli     (mSup.Appli()),
   mDbleGr    (0),
   mCalib     (0),
   mSzRef     (mSup.Sz()),
   mImGeoRef  (mSzRef.x,mSzRef.y),
   mImBrute   (1,1),
   mValB        (-1),
   mValN        (0.0),
   mHomFromRef  (cElHomographie::Id())
{
    if (!mSupIm.ColorimetriesCanaux().empty())
    {
        int aNbMatch = 0;
        for 
        (
           std::list<cColorimetriesCanaux>::const_iterator 
                     itCC=mSupIm.ColorimetriesCanaux().begin();
           itCC != mSupIm.ColorimetriesCanaux().end();
           itCC++
        )
        {
            if (itCC->CanalSelector()->Match(aPDV->Name()))
            {
               aNbMatch++;
               if (itCC->ValBlanc().IsInit())
               {
                  mValB = itCC->ValBlanc().Val();
               }
               mValN = itCC->ValNoir().Val();
               std::cout << "NOIR=" << mValN << "\n";
            }
        }
        if (aNbMatch!=1)
        {
            std::cout << "Name=" << aPDV->Name() 
                      << " NbMatch=" << aNbMatch <<"\n";
            ELISE_ASSERT(false,"Nb Match in ColorimetriesCanaux");
        }
    }
    cDbleGrid::cXMLMode aXMM;
    std::cout << aPDV->NameGeom() << "\n";
    mCalib  =  new cDbleGrid(aXMM,mAppli.FullDirGeom(),aPDV->NameGeom());

    if (aRef != 0)
    {
       cPriseDeVue * aPDVRef = mRef->mPDV;
       std::string aCatName = aPDVRef->Name() + "@" + aPDV->Name();
       std::string aNameGr = MatchAndReplace(*anAutom,aCatName,aMotif);
       std::cout << "aNameGr " << aNameGr  << "\n";

       mDbleGr = new cDbleGrid(aXMM,mAppli.FullDirGeom(),aNameGr);
       
       std::string aNameHom =   mAppli.FullDirGeom()
                              + mAppli.NamePackHom(aPDVRef->Name(),aPDV->Name());

       cElXMLTree aTree(aNameHom);
       ElPackHomologue aPack = aTree.GetPackHomologues("ListeCpleHom");
       ElPackHomologue aPackN;
       for 
       (
            ElPackHomologue::tIter itP=aPack.begin();
            itP!=aPack.end();
            itP++
       )
       { 
          ElCplePtsHomologues aCple
                              (
                                  mRef->mCalib->Direct(itP->P1()),
                                  mCalib->Direct(itP->P2())
                              );
          aPackN.Cple_Add(aCple);
       }
       mHomFromRef = cElHomographie(aPackN,true);

       for 
       (
            ElPackHomologue::tIter itP=aPack.begin();
            itP!=aPack.end();
            itP++
       )
          std::cout << itP->P2() << " " << FromRefByHom(itP->P1()) << "\n";

    }

}


Fonc_Num cOneImageSup::ColBal()
{
    Fonc_Num aRes =  (mImGeoRef.in()-mValN)/(mValB-mValN);
    aRes = Max(0.0,Min(1.0,aRes));

    aRes = pow(aRes,1/mSupIm.GammaCorrection().Val());

    return aRes * 255.0;

}

Pt2dr  cOneImageSup::FromRefByGrid(Pt2dr aP)
{
     return mDbleGr->Direct(aP);
}

Pt2dr  cOneImageSup::FromRefByHom(Pt2dr aP)
{
    return  mCalib->Inverse(mHomFromRef.Direct(mRef->mCalib->Direct(aP)));
}

Pt2dr  cOneImageSup::FromRef(Pt2dr aP)
{
     // return FromRefByHom(aP);

      return FromRefByGrid(aP);
}


Pt2dr  cOneImageSup::LocFromRef(Pt2dr aP)
{
     return FromRef(aP+Pt2dr(mSup.P0()))-Pt2dr(mP0Im);
}

void cOneImageSup::LoadIm(bool First)
{
   if (mDbleGr!=0)
   {
       Pt2dr aC[4];
       mSup.Box().Corners(aC);
       Pt2dr aQ0 = FromRef(aC[0]);
       Pt2dr aQ1 =aQ0;
       for (int aK=1 ; aK<4 ; aK++)
       {
           aQ0 = Inf(aQ0,FromRef(aC[aK]));
           aQ1 = Sup(aQ1,FromRef(aC[aK]));
       }

       
       Box2dr aBox(aQ0,aQ1);


       mP0Im = round_down(aBox._p0);
       Pt2di aSzIm = round_up(aBox._p1-Pt2dr(mP0Im));


       mImBrute = Im2D_REAL4(aSzIm.x,aSzIm.y);
       LoadAllImCorrel(mImBrute,mPDV->IMIL(),1,mP0Im);

       TIm2D<REAL4,REAL8> mTBr(mImBrute);
       TIm2D<REAL4,REAL8> mGR(mImGeoRef);


       Pt2di aPRef;
       for (aPRef.x=0 ; aPRef.x<mSzRef.x ; aPRef.x++)
       {
           for (aPRef.y=0 ; aPRef.y<mSzRef.y ; aPRef.y++)
           {
              Pt2dr aPBr =  LocFromRef(Pt2dr(aPRef));
              mGR.oset(aPRef,mTBr.getr(aPBr,0.0));
           }
       }
   }
   else
   {
       LoadAllImCorrel(mImGeoRef,mPDV->IMIL(),1,mSup.P0());
   }

   if (First)
   {
      if (mValB < 0)
      {
         if (mSup.PBB().x >=0)
         {
            mValB= mImGeoRef.data()[mSup.PBB().y][mSup.PBB().x];
         }
         else
         {
              ELISE_COPY
              (
                 mImGeoRef.all_pts(),
                 mImGeoRef.in(),
                 VMax(mValB)
              );
         }
      }
   }
   mValB *= mSupIm.MultiplicateurBlanc().Val();
   std::cout << "ValB = " << mValB << "\n";
   std::cout << "Done \n";

   if (mSupIm.GenFileImages().Val())
   {
       std::string aName =   mAppli.FullDirResult()
                           + "TestSup_"
                           +  mPDV->Name();
       Tiff_Im::Create8BFromFonc
       (
           aName,
           mImGeoRef.sz(),
           ColBal()
       );
   }
}

/*****************************************/
/*                                       */
/*          main                         */
/*                                       */
/*****************************************/

int main(int argc,char ** argv)
{
   cSuperpMM aSup(argc,argv);
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
