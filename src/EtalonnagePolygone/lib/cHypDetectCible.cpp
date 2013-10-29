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

/************************************************************/
/*                                                          */
/*                    cSetHypDetectCible                    */
/*                                                          */
/************************************************************/

void cSetHypDetectCible::AddHyp
     (
          const cCiblePolygoneEtal & aCible,
	  const cCamIncEtalonage &   aCam,
          Pt3dr DirU,
          Pt3dr DirV
     )
{
    cHypDetectCible * anHyp = new cHypDetectCible(this,aCible,aCam,DirU,DirV);

    for 
    (
       std::list<cHypDetectCible *>::iterator itH =   mHyps.begin();
       itH !=  mHyps.end();
       itH++
    )
    {
       cHypDetectCible * H2 = *itH;
       if (    (anHyp->DistCentre(*H2) < mDistConf)
            && (anHyp->DistForme(*H2)  < mFactConfEll)
	  )
       {
           anHyp->SetConfusionPossible(true);
           H2->SetConfusionPossible(true);
       }
    }

    mHyps.push_back(anHyp);
}

cSetHypDetectCible::cSetHypDetectCible
(
     const  cParamEtal &      aParam,
     const cPolygoneEtal &    aPol,
     const cCamIncEtalonage & aCam,
     REAL aDConf,
     REAL aFactConf,
     Pt3dr DirU,
     Pt3dr DirV

)  :
   mDistConf    (aDConf),
   mFactConfEll (aFactConf)
{
    cPolygoneEtal::tContCible  lC = aPol.ListeCible();

    INT IdSel = aParam.CibleDeTest();
    for
    (
        cPolygoneEtal::tContCible::iterator iT = lC.begin();
        iT != lC.end() ;
        iT++
    )
    {
	    if ((IdSel==-1) || (IdSel==(*iT)->Ind()))
	        AddHyp(**iT,aCam,DirU,DirV);
    }
}

const std::list<cHypDetectCible *>  & cSetHypDetectCible::Hyps() const
{
   return mHyps;
}


REAL cSetHypDetectCible::DistConfusionCentre() const
{
	return mDistConf;
}
REAL cSetHypDetectCible::DistConfusionShape()  const
{
	return mFactConfEll;
}

void cSetHypDetectCible::SauvFile
     (
             const cEtalonnage & mEt,
             const cSetPointes1Im & aSetPointes,
             const cParamRechCible & aPRC ,
             const std::string & aName,
	     bool Complet
     )
{
    FILE * fp = ElFopen(aName.c_str(),"w");
    ELISE_ASSERT(fp!=0,"Cannot Open file in cSetHypDetectCible::SauvFile");

    for 
    (
       std::list<cHypDetectCible *>::iterator itH =   mHyps.begin();
       itH !=  mHyps.end();
       itH++
    )
    {
       cHypDetectCible & aH = **itH;

       cPointeEtalonage *  aPE = const_cast<cSetPointes1Im &>(aSetPointes).PointeOfIdSvp(aH.Cible().Ind());

       if (aH.OkDetec() || ((aPRC.mUseCI != eUCI_Jamais) && (aPE!=0)))
       {
           Pt2dr aP ;
           if (aPE)
	   {
	      if ((aPRC.mUseCI ==eUCI_Toujours) || (aPRC.mUseCI ==eUCI_Only))
	         aP = mEt.FromPN(aPE->PosIm());
	      else
	         aP = aH.OkDetec() ? aH.CentreFinal() : mEt.FromPN(aPE->PosIm());
	   }
	   else
	   {
              aP = aH.CentreFinal();
	   }



           fprintf(fp,"%d %lf %lf",aH.Cible().Ind(),aP.x,aP.y);
           if (Complet)
               fprintf(fp," %f",aH.OkDetec() ? aH.Largeur() : -1e9);
           fprintf(fp,"\n");
       }
    }
    ElFclose(fp);
}


/************************************************************/
/*                                                          */
/*                    cHypDetectCible                       */
/*                                                          */
/************************************************************/



cHypDetectCible::cHypDetectCible
(
    const cSetHypDetectCible * aSet,
    const cCiblePolygoneEtal & aCible,
    const cCamIncEtalonage &   aCam,
    Pt3dr                      aDirU,
    Pt3dr                      aDirV  
)  :
   pSet      (aSet),
   mOkDetec  (false),
   mConfPos  (false),
   mCible    (&aCible),
   mCam      (&aCam),
   mRay      (aCible.Mire().KthDiam(0)/2000.0),
   mCentr0   (Terrain2ImageGlob(aCible.Pos())),
   mCentreFinal    (-1111,-66666),
   mLargeur        (-1),
   mCorrel         (-1),
   mDistCentreInit (-1),
   mDistShapeInit  (-1),
   mPosFromPointe  (false)
{
   // la valeur initiale de aDirU/aDirV devient inutile, calculees avec les
   // normale
   aDirU = vunit(OneDirOrtho(aCible.CC()->Normale()));
   aDirV = vunit(aCible.CC()->Normale() ^ aDirU);

   //
   Pt2dr aQ1 = Terrain2ImageGlob(aCible.Pos()+aDirU*mRay)-mCentr0;
   Pt2dr aQ2 = Terrain2ImageGlob(aCible.Pos()+aDirV*mRay)-mCentr0;
   if (mCam->UseDirectPointeManuel())
   {
      cPointeEtalonage *  aPE = aCam.PointeInitial()->PointeOfIdSvp(mCible->Ind());
      if (aPE) 
      {
          mPosFromPointe = true;
	  mCentr0 = aPE->PosIm();
      }
   }
   ImRON2ParmEllipse(mA0,mB0,mC0,aQ1,aQ2);
   EllipseEq2ParamPhys(mGdAxe,mPtAxe,mDirGA,mA0,mB0,mC0);


   Pt2di aSz = aCam.Tiff().sz();
   REAL aDC = pSet->DistConfusionCentre();
   mInsideImage =    (mCentr0.x >-aDC)
                  && (mCentr0.y >-aDC)
                  && (mCentr0.x < aSz.x+aDC)
                  && (mCentr0.y < aSz.y+aDC);

}

bool  cHypDetectCible::PosFromPointe() const
{
   return mPosFromPointe;
}

const cSetHypDetectCible & cHypDetectCible::Set() const
{
    return *pSet;
}
REAL cHypDetectCible:: GdAxe() const
{
     return ElMax(mGdAxe,mPtAxe);
}

REAL cHypDetectCible::Largeur() const
{
   return mLargeur;
}

Pt2dr cHypDetectCible::Terrain2ImageGlob(Pt3dr aP) const
{
   return mCam->Terrain2ImageGlob(aP);
}

REAL  cHypDetectCible::A0() const     {return mA0;}
REAL  cHypDetectCible::B0() const     {return mB0;}
REAL  cHypDetectCible::C0() const     {return mC0;}
Pt2dr cHypDetectCible::Centr0() const {return mCentr0;}
Pt2dr cHypDetectCible::CentreFinal() const {return mCentreFinal;}

const cCamIncEtalonage & cHypDetectCible::Cam() const {return *mCam;}

void cHypDetectCible::SetConfusionPossible(bool isCP)
{
	mConfPos = isCP;
}
bool cHypDetectCible::ConfPos() const
{
    return mConfPos;
}
REAL cHypDetectCible::DistCentre(const cHypDetectCible & aH2) const
{
   return euclid(mCentr0,aH2.mCentr0);
}
REAL cHypDetectCible::DistForme(const cHypDetectCible & aH2) const
{
   return SimilariteEllipse(mA0,mB0,mC0,aH2.mA0,aH2.mB0,aH2.mC0);
}

bool cHypDetectCible::InsideImage() const
{
	return mInsideImage;
}

const cCiblePolygoneEtal & cHypDetectCible::Cible()
{
    return *mCible;
}

void cHypDetectCible::SetResult
     (
         Pt2dr aCentre,
         REAL  aLarg,
         bool Ok,
         REAL Correl,REAL DistC,REAL DistShape
      )
{
	mCentreFinal = aCentre;
	mLargeur = aLarg;
	mOkDetec = Ok;
	mCorrel  = Correl;
	mDistCentreInit = DistC;
	mDistShapeInit = DistShape;
}

bool cHypDetectCible::OkDetec() const
{
	return mOkDetec;
}


bool cHypDetectCible::OkForDetectInitiale(bool RequirePerferct) const
{
   return      mInsideImage
	   &&  (! mConfPos)
	   &&  ((! RequirePerferct) || (mCible->Qual() == cCiblePolygoneEtal::ePerfect));
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
