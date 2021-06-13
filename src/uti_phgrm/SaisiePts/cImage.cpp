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




/*************************************************/
/*                                               */
/*                XXXXXXX                        */
/*                                               */
/*************************************************/


cImage::cImage(const std::string & aName,cAppli_SaisiePts & anAppli,bool Visualizable) :
   mAppli (anAppli),
   mName  (aName),
   mTif   (0),
   mCapt3d   (0),
   mCaptElCam  (0),
   mCaptNuage(0),
   mSPIm  (0),
   mSzIm  (-1,-1),
   mWAff  (0),
   mPrio  (0),
   mInitCamNDone (false),
   mCptAff       (0),
   mVisualizable (Visualizable),
   mPImsNuage         (0),
   mPNSeuilAlti       (25.0),
   mPNSeuilPlani      (5.0),
   mLastLoaded        (false),
   mCurLoaded         (false)
{
}

void cImage::SetMemoLoaded()
{
   mLastLoaded = mCurLoaded;
   mCurLoaded  = false;
}
void  cImage::SetLoaded()
{
   mCurLoaded = true;
}

void cImage::OnModifLoad()
{

   if (mCurLoaded==mLastLoaded) return;

   if (mCurLoaded)
   {
      cMMByImNM * aMMI =   mAppli.PIMsFilter ();
      if (aMMI)
      {
           std::string aNameDepth = aMMI->NameFileXml(eTMIN_Depth,mName);
           mPImsNuage = cElNuage3DMaille::FromFileIm(aNameDepth);

      }
   }
   else
   {
       delete mPImsNuage;
       mPImsNuage =0;
   }
}

bool  cImage::PIMsValideVis(const Pt3dr & aPTer)
{
   if (mPImsNuage==0) return true;

   Pt2dr aPIm = mPImsNuage->Terrain2Index(aPTer);
   Pt2di aIPIm = round_ni(aPIm);


   double aProfMax= -1e20;
   double aProfMin= 1e20;
   int aDil = round_up(mPNSeuilPlani);

   for (int aDx= -aDil ; aDx <= aDil ; aDx++)
   {
       for (int aDy= -aDil ; aDy <= aDil ; aDy++)
       {
            Pt2di aPVois(aIPIm.x+aDx,aIPIm.y+aDy);
            if ((mPImsNuage->IndexHasContenu(aPVois)) && (euclid(Pt2dr(aPVois)-aPIm)<mPNSeuilPlani))
            {
                double aProfNu = mPImsNuage->ProfEnPixel(aPVois);
                ElSetMax(aProfMax,aProfNu);
                ElSetMin(aProfMin,aProfNu);
            }
       }
   }


   Pt3dr aTerInNu = mPImsNuage->Euclid2ProfPixelAndIndex(aPTer);
   double aProfTer = aTerInNu.z;

    return (aProfTer >= (aProfMin-mPNSeuilAlti)) && (aProfTer<=(aProfMax+mPNSeuilAlti));

/*
std::cout << "Valll   BBBBBBB\n";
   if (! aEnv->IndexHasContenuForInterpol(aPIm))
   {
       // A Parametrer eventuellement
       return false;
   }

   Pt3dr aTerInNu = aEnv->Euclid2ProfPixelAndIndex(aPTer);
   double aProfTer = aTerInNu.z;
   double aProfNu = aEnv->ProfInterpEnPixel(aPIm);

   bool isDessus = (aProfTer >= aProfNu);

std::cout << "Valll   CCCCCCC " << aProfTer << " " << aProfNu << " \n";

   return aMin ?  isDessus : (! isDessus);
*/

  return false;

}



bool  cImage::Visualizable() const
{
   return mVisualizable;
}


int & cImage::CptAff() {return mCptAff;}

Pt2di  cImage::SzIm() const
{
   if (mSzIm.x <=0)
      mSzIm = Tif().sz();

  return mSzIm;
}

bool cImage::InImage(const Pt2dr & aP)
{
   return (aP.x >0) && (aP.y >0) && (aP.x <SzIm().x) && (aP.y < SzIm().y);
}


Pt2dr cImage::PointArbitraire()  const
{

   Pt2dr aSz = Pt2dr(SzIm());
   Box2dr aBox(aSz*0.25,aSz*0.75);

#if (ELISE_X11)
   if (mWAff)
   {
       aBox = mWAff->BoxImageVisible();
   }
#endif


   return aBox.RandomlyGenereInside();
}


const std::string & cImage::Name() const { return mName; }
cWinIm *            cImage::WAff() const { return mWAff; }


void cImage::SetWAff(cWinIm * aWAff)
{
   mWAff = aWAff;
}

Tiff_Im &  cImage::Tif() const
{

   if (mTif==0 )
   {
        mTif  = new  Tiff_Im(  Tiff_Im::StdConvGen(mAppli.DC()+mName,mAppli.Param().ForceGray().Val() ? 1 : -1,false,true));
   }
   return *mTif;
}

void cImage::InitCameraAndNuage()
{
   if (mInitCamNDone)
      return;
   mInitCamNDone = true;

   if (! mAppli.HasOrientation())
      return;

   std::string aKey = mAppli.Param().KeyAssocOri().Val();


   // C'EST LALA
   cResulMSO aRMso =  mAppli.ICNM()->MakeStdOrient(aKey,true,&mName,true);


   if (aRMso.Capt3d())
      mCapt3d = aRMso.Capt3d();

   if (aRMso.Cam())
      mCaptElCam = aRMso.Cam();

   if (aRMso.Nuage())
   {
      mCaptNuage = aRMso.Nuage();
      // std::cout << "mCaptNuage " << mName << " " << mCaptNuage
   }

   if (0)
   {
      std::cout << "ICAN "  << mName << " Cap3D " << mCapt3d  << " Cam " << mCaptElCam << " Nuage " << mCaptNuage << "\n";
   }
}


ElCamera * cImage::ElCaptCam()
{
    InitCameraAndNuage();
    return mCaptElCam;
}
cBasicGeomCap3D * cImage::Capt3d()
{
    InitCameraAndNuage();
    return mCapt3d;
}
cElNuage3DMaille * cImage::CaptNuage()
{
    InitCameraAndNuage();
    return mCaptNuage;
}

void cImage::SetSPIM(cSaisiePointeIm * aSPIM)
{
    ELISE_ASSERT(mSPIm==0,"Multiple cImage::SetSPIM");
    mSPIm = aSPIM;
}

const std::vector<cSP_PointeImage *> &  cImage::VP()
{
    return mVP;
}

cSP_PointeImage * cImage::PointeOfNameGlobSVP(const std::string & aNameGlob)
{
    std::map<std::string,cSP_PointeImage *>::iterator anIt = mPointes.find(aNameGlob);
    if  (anIt == mPointes.end()) return 0;
    return anIt->second;
}

void cImage::AddAImPointe(cOneSaisie * anOS,cSP_PointGlob * aPG,bool FromFile)
{
   if (PointeOfNameGlobSVP(aPG->PG()->Name()))
       return;

   if (mSPIm==0)
   {
      mAppli.SOSPI().SaisiePointeIm().push_back(cSaisiePointeIm());
      mSPIm =  &(mAppli.SOSPI().SaisiePointeIm().back());
      mSPIm->NameIm() = mName;
   }

   if (! FromFile)
   {
       mSPIm->OneSaisie().push_back(*anOS);
       anOS = & mSPIm->OneSaisie().back();
   }
   cSP_PointeImage * aPIm = new cSP_PointeImage(anOS,this,aPG);
   mPointes[aPG->PG()->Name()] = aPIm;
   mVP.push_back(aPIm);
   aPG->AddAGlobPointe(aPIm);
}


double  cImage::CalcPriority(cSP_PointGlob * aPP,bool UseCpt) const
{
   double aRes = 0;
   for (int aKS=0; aKS<int(mVP.size()) ; aKS++)
   {
       cSP_PointeImage * aPIm = mVP[aKS];
       double aPrio = mAppli.StatePriority(aPIm->Saisie()->Etat());
       if (     (aPIm->Gl() == aPP)
             && (  mAppli.Visible(*aPIm))
          )
          aPrio = (1+aPrio) * 1e4;
       aRes += aPrio;
   }

   if (UseCpt)
   {
       aRes = aRes - mCptAff * 1e7 - mCurLoaded*1e10;
   }

   return aRes;
}

double cImage::Prio() const
{
    return mPrio;
}


void  cImage::SetPrio(double aPrio)
{
   mPrio = aPrio;
}

bool cImage::PtInImage(const Pt2dr aP)
{
   if (mCapt3d && (!mCapt3d->CaptHasData(aP))) return false; 
   return    (aP.x>0)
          && (aP.y>0)
          && (aP.x<SzIm().x)
          && (aP.y<SzIm().y);
}

// CREATE point ground From Pointe Mono
//
cSP_PointGlob * cImage::CreatePGFromPointeMono(Pt2dr  aPtIm,eTypePts aType,double aSz,cCaseNamePoint * aCNP)
{

    //bool PIsInit = false;
    Pt3dr aPt(0,0,0);
    bool PInit = false;
    // Pt3dr aP1(0,0,0);
    // Pt3dr aP2(0,0,0);

    cBasicGeomCap3D *  aCapt3d = Capt3d();
    if (aCapt3d)
    {
       if (aCapt3d->HasRoughCapteur2Terrain())
       {
          //  aPt = aCamera->ImEtProf2Terrain(aPtIm,aCamera->GetProfondeur());
          aPt = aCapt3d->RoughCapteur2Terrain(aCapt3d->ImRef2Capteur(aPtIm));
          PInit = true;
       }
    }

            //PIsInit = true;
    cPointGlob aPG;
    std::pair<int,std::string> anId = mAppli.Interface()->IdNewPts(aCNP);
    if (anId.second=="NONE")
       return NULL;
    mAppli.Interface()->ChangeFreeNamePoint(anId.second,false);

    aPG.Type() = aType;
    aPG.Name() = anId.second;
    aPG.NumAuto().SetVal(anId.first);
    aPG.ContenuPt().SetNoInit();  // OK
    aPG.SzRech().SetVal(aSz);  // OKK

    if (PInit)
        aPG.P3D().SetVal(aPt);
    else
        aPG.P3D().SetNoInit();

    cSP_PointGlob * aSPG = mAppli.AddPointGlob(aPG,true);

    if (aSPG==0) //already exists
    {
       //ELISE_ASSERT(aSPG!=0,"Incoherence (1) in cImage::CreatePGFromPointeMono");
       return NULL;
    }

    aSPG->SuprDisp();

    mAppli.AddPGInAllImages(aSPG);


    cSP_PointeImage * aPIm = PointeOfNameGlobSVP(aSPG->PG()->Name());
    if (aPIm==0)
    {
        std::cout << "For-Name " << aSPG->PG()->Name()  << " SzDico = " << mPointes.size() << "\n";
        ELISE_ASSERT(aPIm!=0,"Incoherence (2) in cImage::CreatePGFromPointeMono");
    }
    aPIm->Saisie()->Etat() = eEPI_Valide;
    aPIm->Saisie()->PtIm() = aPtIm;
    mAppli.HighLightSom(aSPG);

    aSPG->ReCalculPoints();

    mAppli.RedrawAllWindows();
    mAppli.Save();

    return aSPG;

}




double LargeGeoCube = 0.01;  // +ou-  10 cm

Fonc_Num  cImage::FilterImage(Fonc_Num aFonc,eTypePts aType,cPointGlob * aPG)
{
    InitCameraAndNuage();

    if (! mCapt3d)
       return aFonc;
    double aResol = (aPG && aPG->P3D().IsInit()) ? mCapt3d->ResolSolOfPt(aPG->P3D().Val()) : mCapt3d->GlobResol() ;

    aResol *= mCapt3d->ResolImRefFromCapteur();
    // std::cout <<  "FILIM : " << eToString(aType) <<  "  aPG " << aPG<<  " R=" << aResol << "\n";

    // if) (aType == eNSM_GeoCube)
    if ( aPG && aPG->LargeurFlou().IsInit() )
    {
        double aLarg = aPG->LargeurFlou().Val();
        double aNbPix = (aLarg /  aResol)/2.0;

        std::cout << " NbPix " << aNbPix  << " L = " << aLarg << "\n";

        aFonc = MoyByIterSquare(aFonc,aNbPix,3);
    }


    return aFonc;
}

void cImage::UpdateMapPointes(const std::string& aName)
{
    std::map<std::string,cSP_PointeImage *>::iterator It = mPointes.begin();
    for ( ; It != mPointes.end(); It++)
    {
        if (It->second->Saisie()->NamePt() == aName)
        {
            std::string oldName = It->first;
            cSP_PointeImage* aPIm = It->second;

            mPointes.erase(oldName);
            mPointes[aName] = aPIm;
        }
    }
}


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
