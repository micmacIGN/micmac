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
#include "Digeo.h"

namespace NS_ParamDigeo
{


/****************************************/
/*                                      */
/*             cImDigeo                 */
/*                                      */
/****************************************/

cImDigeo::cImDigeo
(
   int                 aNum,
   const cImageDigeo & aIMD,
   const std::string & aName,
   cAppliDigeo &       anAppli
) :
  mName        (aName),
  mAppli       (anAppli),
  mIMD         (aIMD),
  mNum         (aNum),
  mTifF        (new Tiff_Im(Tiff_Im::StdConv(mAppli.DC()+mName))),
  mSzGlob      (mTifF->sz()),
  mBoxIm       (mIMD.BoxIm().ValWithDef(Box2di(Pt2di(0,0),mSzGlob))),
  mSzMax       (0,0),
  mVisu        (0),
  mG2MoyIsCalc (false),
  mDyn         (1.0)
{
   mBoxIm = Inf(mBoxIm,Box2di(Pt2di(0,0),mSzGlob));
   //Provisoire
   ELISE_ASSERT(! aIMD.PredicteurGeom().IsInit(),"Asservissement pas encore gere");

   // Verification de coherence
   if (aNum==0)
   {
        ELISE_ASSERT(! aIMD.PredicteurGeom().IsInit(),"Asservissement sur image maitresse ?? ");
   }
   else
   {
       if ( aIMD.PredicteurGeom().IsInit())
       {
          //Provisoire
          ELISE_ASSERT(!aIMD.BoxIm().IsInit()," Asservissement et Box Im sec => redondant ?");
       }
       else
       {
          ELISE_ASSERT
          (
             ! mAppli.DigeoDecoupageCarac().IsInit(),
             "Decoupage+Multimage => Asservissement requis"
          );
          
       }

/*
 Redondant
       ELISE_ASSERT
       (
            !(mAppli.DigeoDecoupageCarac().IsInit()&&aIMD.BoxIm().IsInit()),
            "Decoupage + Box Im sec"
       );
*/
   }
   
}

Box2di cImDigeo::BoxIm() const
{
   return mBoxIm;
}


void cImDigeo::NotifUseBox(const Box2di & aBox)
{
  if (mIMD.PredicteurGeom().IsInit())
  {
       ELISE_ASSERT(false,"NotifUseBox :: Asservissement pas encore gere");
  }
  else
  {
      mSzMax.SetSup(aBox.sz());
  }
}



GenIm::type_el  cImDigeo::TypeOfDeZoom(int aDZ) const
{
   GenIm::type_el aRes = mTifF->type_el();
   int aDZMax = 0;
   for 
   (
       std::list<cTypeNumeriqueOfNiv>::const_iterator itP=mAppli.TypeNumeriqueOfNiv().begin();
       itP!=mAppli.TypeNumeriqueOfNiv().end();
       itP++
   )
   {
      if  ((itP->Niv()>=aDZMax) && (itP->Niv()<=aDZ))
      {
         aRes = Xml2EL(itP->Type());
      }
   }
   return aRes;
}


void cImDigeo::AllocImages()
{
   Pt2di aSz = mSzMax;
   mNiv=0;

   const cTypePyramide & aTP = mAppli.TypePyramide();
   if (aTP.NivPyramBasique().IsInit())
      mNiv = aTP.NivPyramBasique().Val();
   else if (aTP.PyramideGaussienne().IsInit())
      mNiv = aTP.PyramideGaussienne().Val().NivOctaveMax();
   else
   {
        ELISE_ASSERT(false,"cImDigeo::AllocImages PyramideImage");
   }

   int aNivDZ = 0;
   for (int aDz = 1 ; aDz <=mNiv ; aDz*=2)
   {
       cOctaveDigeo * anOct = cOctaveDigeo::Alloc(TypeOfDeZoom(aDz),*this,aDz,aSz);
       mOctaves.push_back(anOct);
       if (aTP.NivPyramBasique().IsInit())
       {
          // mVIms.push_back(cImInMem::Alloc (*this,aSz,TypeOfDeZoom(aDz), *anOct, 1.0));
 // C'est l'image Bas qui servira
 //         mVIms.push_back(anOct->AllocIm(1.0,0));
 
       }
       else if (aTP.PyramideGaussienne().IsInit())
       {
            const cPyramideGaussienne &  aPG = aTP.PyramideGaussienne().Val();
            int aNbIm = aPG.NbByOctave().Val();
            if (mAppli.ModifGCC())
               aNbIm = mAppli.ModifGCC()->NbByOctave();

            if (aPG.NbInLastOctave().IsInit() && (aDz*2>mNiv))
               aNbIm = aPG.NbInLastOctave().Val();
            int aK0 = 0;
            if (aDz==1)
               aK0 = aPG.IndexFreqInFirstOctave().Val();
           anOct->SetNbImOri(aNbIm);
            for (int aK=aK0 ; aK< aNbIm+3 ; aK++)
            {
                double aSigma = aPG.Sigma0().Val() * pow(2.0,aK/double(aNbIm));
                //mVIms.push_back(cImInMem::Alloc (*this,aSz,TypeOfDeZoom(aDz), *anOct,aSigma));
                mVIms.push_back((anOct->AllocIm(aSigma,aK,aNivDZ*aNbIm+(aK-aK0))));
            }
                
       }
       aSz = (aSz+Pt2di(1,1)) /2 ;
       aNivDZ++;
   }

   for (int aK=1 ; aK<int(mVIms.size()) ; aK++)
   {
      mVIms[aK]->SetMere(mVIms[aK-1]);
   }
}



void cImDigeo::LoadImageAndPyram(const Box2di & aBox)
{
    ElTimer aChrono;
    mSzCur = aBox.sz();


    mOctaves[0]->ImBase()->LoadFile(*mTifF,aBox);
    // mVIms[0]->LoadFile(*mTifF,aBox);

    double aTLoad = aChrono.uval();
    aChrono.reinit();
   
    const cTypePyramide & aTP = mAppli.TypePyramide();
/*
    if (aTP.PyramideGaussienne().IsInit())
    {
         mVIms[0]->MakeConvolInit(aTP.ConvolFirstImage().Val());
         mVIms[0]->SauvIm();
    }
*/


    
    for (int aK=0 ; aK< int(mVIms.size()) ; aK++)
    {
       if (aTP.NivPyramBasique().IsInit())
       {
          if (aK>0)
             mVIms[aK]->VMakeReduce_121(*(mVIms[aK-1]));
       }
       else if (aTP.PyramideGaussienne().IsInit())
       {
            mVIms[aK]->ReduceGaussienne();
       }
       mVIms[aK]->SauvIm();
    }

    for (int aKOct=0 ; aKOct<int(mOctaves.size()) ; aKOct++)
    {
        mOctaves[aKOct]->PostPyram();
    }


    double aTPyram = aChrono.uval();
    aChrono.reinit();

    if (mAppli.ShowTimes().Val())
    {
        std::cout << "Time,  load : " << aTLoad << " ; Pyram : " << aTPyram << "\n";
    }
}

void cImDigeo::DoExtract()
{
    if (mIMD.VisuCarac().IsInit())
    {
        const cParamVisuCarac & aPVC = mIMD.VisuCarac().Val();
        mVisu = new cVisuCaracDigeo
                    (
                       mAppli,
                       mSzCur,
                       aPVC.Zoom().Val(),
                       mOctaves[0]->ImBase()->Im().in_proj() * aPVC.Dyn(),
                       aPVC
                    );
    }
    ElTimer aChrono;

    DoSiftExtract();

    if (mAppli.ShowTimes().Val())
    {
        std::cout << "Time,  Extrema : " << aChrono.uval() << "\n";
    }

    if (mVisu)
    {
       mVisu->Save(mName);
       delete mVisu;
       mVisu = 0;
    }
}

void cImDigeo::DoCalcGradMoy(int aDZ)
{
   if (mG2MoyIsCalc)
      return;

   mG2MoyIsCalc = true;

   if (mAppli.MultiBloc())
   {
      ELISE_ASSERT(false,"DoCalcGradMoy : Multi Bloc a gerer");
   }

   ElTimer aChrono;
   mG2Moy = GetOctOfDZ(aDZ).ImBase()->CalcGrad2Moy();

   std::cout << "Grad = " << sqrt(mG2Moy)/Dyn() <<  " Time =" << aChrono.uval() << "\n";
}


void cImDigeo::DoSiftExtract()
{
    if (!mAppli.SiftCarac().IsInit())
       return;
    const cSiftCarac &  aSC = mAppli.SiftCarac().Val();
    DoCalcGradMoy(aSC.NivEstimGradMoy().Val());
    ELISE_ASSERT(mAppli.PyramideGaussienne().IsInit(),"Sift require Gauss Pyr");
    for (int aKoct=0; aKoct<int(mOctaves.size());aKoct++)
    {
         mOctaves[aKoct]->DoSiftExtract(aSC);
    }
    
}

cOctaveDigeo * cImDigeo::SVPGetOctOfDZ(int aDZ)
{
   for (int aK=0 ; aK<int(mOctaves.size()) ; aK++)
   {
      if (mOctaves[aK]->Niv() == aDZ)
      {
          return mOctaves[aK];
      }
   }
   return 0;
}

cOctaveDigeo & cImDigeo::GetOctOfDZ(int aDZ)
{
   cOctaveDigeo * aRes = SVPGetOctOfDZ(aDZ);

   ELISE_ASSERT(aRes!=0,"cAppliDigeo::GetOctOfDZ");

   return *aRes;
}


double cImDigeo::Dyn() const
{
    return mDyn;
}

void cImDigeo::SetDyn(double aDyn)
{
    mDyn = aDyn;
}



const std::string  &  cImDigeo::Name() const {return mName;}
cAppliDigeo &  cImDigeo::Appli() {return mAppli;}
const cImageDigeo &  cImDigeo::IMD() {return mIMD;}
cVisuCaracDigeo  *   cImDigeo::CurVisu() {return mVisu;}
double cImDigeo::G2Moy() const {return mG2Moy;}



};



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
