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


#include "Ortho.h"

Pt2di aPB(198,158);


cLoadedIm::cLoadedIm (cAppli_Ortho & anAppli,int anInd,Tiff_Im & aTF0,const Pt2di aSz) :
     mAppli      (anAppli),
     mCurOrtho   (0),
     mInd        (anInd),
     mImIncH     (aSz.x,aSz.y),
     mImPCBrute  (aSz.x,aSz.y),
     mImPC       (aSz.x,aSz.y),
     mTImPC      (mImPC),
     mIms        (aTF0.VecOfIm(aSz)),
     mChCor      (mIms.size() /2 ) // Ca fait V dans RVB ....
{
}

void cLoadedIm::TransfertIm(const std::vector<Im2DGen *> & aVIm,const Pt2di& aPLoc,const Pt2di& aPGlob)
{
     std::vector<double> aVV = Vals(aPLoc);

     cER_OneIm * anERI = mCurOrtho->ERIPrinc();

// TEST A ACTIVER POUR VERIF QUE EGAL PHYS EST BIEN CALC...
if (0)
{
static int aCpt=0; aCpt++;
if ((aCpt%378)==0)
std::cout << "xxxxxxxxxxxxxxxxxxWWWWWW " << aCpt << "\n ";
  anERI = mCurOrtho->ERIPhys() ;
}

     if (anERI)
     {
         Pt2dr aP =  Pt2dr(aPLoc + DecLoc());
         std::vector<double> aVCor;
         anERI->ValueGlobCorr(aP,aVCor,aVV,Pt3dr(aPGlob.x,aPGlob.y,0.0));
         // anERI->ValueLoc(aP,aVCor,aVV);
         aVV = aVCor;
     }

     for (int aK=0; aK<int(aVIm.size()) ; aK++)
     {
        double aVK = aVV[aK] * mAppli.DynGlob();
        //  if (mAppli.ValMasqMesure( aPLoc)) aVK= 255- aVK;
        // aVIm[aK]->SetI(aPLoc,aVV[aK]);

        aVIm[aK]->SetI(aPLoc,aVK);
        mAppli.SetRMax(aVK,aK);

     }
        // aVIm[aK]->SetI(aP,mIms[aK]->GetI(aP));
}


bool cLoadedIm::Init
     (
          cOneImOrhto &anOrtho,
          const Box2di & aBoxOut,
          const Box2di & aBoxIn
     )
{
  
   mCurOrtho = & anOrtho;
   mSz = aBoxIn.sz();
   mDecLoc = aBoxIn._p0-anOrtho.mBox._p0;
   mImPC.Resize(mSz);
   mTImPC = TIm2D<U_INT1,INT>(mImPC);
   mImPCBrute.Resize(mSz);

   Fonc_Num aFoncPC = anOrtho.mTifPC.in(255);
   ELISE_COPY
   (
       mImPC.all_pts(),
       trans(aFoncPC,mDecLoc),
       mImPCBrute.out()
   );

   int aNbOK;
   ELISE_COPY
   (
       rectangle(aBoxOut._p0-aBoxIn._p0,aBoxOut._p1-aBoxIn._p0),
       mImPCBrute.in() != 255,
       sigma(aNbOK)
   );
   if (aNbOK==0)
      return false;

   aFoncPC =  mImPCBrute.in_proj();
   int aNbDil = mAppli.CO().SzDilatPC().Val();
   int aNbOuv = mAppli.CO().SzOuvPC().Val();

   if (aNbDil+aNbOuv)
       aFoncPC = rect_max(aFoncPC,aNbDil+aNbOuv);
   if (aNbOuv)
       aFoncPC = rect_min(aFoncPC,aNbOuv);

   ELISE_COPY ( mImPC.all_pts(), aFoncPC, mImPC.out());



   mFactRed = anOrtho.mMdPC.SsResolIncH().Val();
   mSzRed = round_up(Pt2dr(mSz)/mFactRed)+Pt2di(1,1);
   mImIncH.Resize(mSzRed);
   Pt2dr aRDL = Pt2dr(mDecLoc) /mFactRed  ;//  CET PAS CA
   mRedDecLoc = round_down(aRDL);
   mPR_RDL = aRDL -Pt2dr(mRedDecLoc);

   ELISE_COPY
   (
       mImIncH.all_pts(),
       trans(anOrtho.mTifIncH.in_proj(),mRedDecLoc),
          mImIncH.out() 
       |  sigma(mIncMoy)
   );
   mIncMoy  /= mSzRed.x * mSzRed.y;

   for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
   {
        mIms[aKIm]->Resize(mSz);
   }
   ELISE_COPY
   (
       mImPC.all_pts(),
       trans(anOrtho.mTif.in(0),mDecLoc),
       StdOut(mIms)
   );


/*
   ELISE_COPY
   (
      mImPC.all_pts(),
      Max(0,Min(255,500*FoncInc())),
      mAppli.W()->ogray()
   );
   getchar();

   std::cout << mInd << "  " << mCurOrtho->Name() << " ";
   for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
   {
        std::cout << mIms[aKIm]->GetI(aPB) << " " ;
   }
   std::cout << "\n";

   Video_Win * aW = mAppli.W();
   if (aW)
   {
        ELISE_COPY
        (
            aW.all_pts(),
            mImIncH
        );
   }
*/

   return true;
}

Im2D_U_INT1     cLoadedIm::ImPCBrute()
{
  return  mImPCBrute;
}

Fonc_Num cLoadedIm::FoncInc()
{
   return mImIncH.in(0) [ Virgule(FX/mFactRed-mPR_RDL.x,FY/mFactRed-mPR_RDL.y) ];
}

int cLoadedIm::Ind() const
{
   return mInd;
}

cOneImOrhto * cLoadedIm::CurOrtho()
{
   return mCurOrtho;
}

void cLoadedIm::UpdateNadirIndex(Im2D_REAL4 aScoreNadir,Im2D_INT2 aIndexNadir)
{

    double  aPriorite = mCurOrtho->Priorite();
    Pt2di aP;
    TIm2D<U_INT1,INT> aTPC(mImPC);
    TIm2D<INT2,INT> aTInd(aIndexNadir);
    TIm2D<REAL4,REAL> aTScMin(aScoreNadir);
    TIm2D<REAL4,REAL> mTScLoc(mImIncH);


    for (aP.x =0 ; aP.x < mSz.x ; aP.x++)
    {
         for (aP.y =0 ; aP.y < mSz.y ; aP.y++)
         {  
// if (aP==aPB) std::cout << " --== " <<  aTPC.get(aP) << "\n";
              if (aTPC.get(aP) != 255)
              {
                 Pt2dr aPR = Pt2dr(aP)/mFactRed-mPR_RDL;
                 double aScore = mTScLoc.getr(aPR,1e5) + 1e3 * aPriorite;
                 if (aScore<aTScMin.get(aP))
                 {
                      aTScMin.oset(aP,aScore);
                      aTInd.oset(aP,mInd);
                 }
              }
// if (aP==aPB) std::cout << "    " <<  aTInd.get(aP) << "\n";
         }
    }
}

int cLoadedIm::ValeurPC(const Pt2di & aP) const
{
    return mTImPC.get(aP,255);
}

const Pt2di&  cLoadedIm::DecLoc() const
{
   return mDecLoc;
}
 
std::vector<double> cLoadedIm::Vals(const Pt2di & aP) const
{
   std::vector<double> aV;
   for (int aK=0 ; aK<int(mIms.size()) ; aK++)
   {
       aV.push_back(mIms[aK]->GetR(aP));
   }

   return aV;
}

double cLoadedIm::ValCorrel(const Pt2di & aP) const
{
    return mIms[mChCor]->GetR(aP);
}


bool   cLoadedIm::OkCorrel(const Pt2di & aP ,int aSzV) const
{
    return    (aP.x >= aSzV)
           && (aP.y>= aSzV)
           && (aP.x<mSz.x-aSzV)
           && (aP.y<mSz.y-aSzV);
}


double cLoadedIm::Correl(const Pt2di & aP,cLoadedIm * aL2,const  std::vector<std::vector<Pt2di> > & aVV,int aSz)
{
    if ((!OkCorrel(aP,aSz)) || (!aL2->OkCorrel(aP,aSz)))
       return -1;

    double aRes = 1.0;

    RMat_Inertie aMat;
    for (int aKV=0 ; aKV<int(aVV.size()) ; aKV++)
    {
         const std::vector<Pt2di>&  aV = aVV[aKV];
         for (int aKP=0 ; aKP<int(aV.size()) ; aKP++)
         {
             Pt2di aQ = aP+ aV[aKP] ;
             aMat.add_pt_en_place(ValCorrel(aQ),aL2->ValCorrel(aQ));
         }
         aRes = ElMin(aRes,aMat.correlation(1.0));
    }

    return aRes;
}

void cLoadedIm::TestDiff(cLoadedIm * aL2)
{
    Tiff_Im::Create8BFromFonc
    (
        "TestDif.tif",
        mIms[mChCor]->sz(),
        Max(0,Min(255,128+mIms[mChCor]->in()-aL2->mIms[mChCor]->in()))
    );
}

bool    cLoadedIm::Im2Test() const
{
    return mCurOrtho->Im2Test();
}
// bool OK=



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
