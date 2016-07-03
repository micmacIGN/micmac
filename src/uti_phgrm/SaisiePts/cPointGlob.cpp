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

cSP_PointGlob::cSP_PointGlob(cAppli_SaisiePts & anAppli,cPointGlob * aPG) :
    mAppli       (anAppli),
    mPG          (aPG),
    mHighLighted (false)
{
}

bool & cSP_PointGlob::HighLighted() {return mHighLighted;}

cPointGlob * cSP_PointGlob::PG()
{
   return mPG;
}

void cSP_PointGlob::AddAGlobPointe(cSP_PointeImage * aPIm)
{
    ELISE_ASSERT
    (
        mPointes.find(aPIm->Image()->Name())==mPointes.end(),
        "Multiple add AddAGlobPointe"
    );
    mPointes[aPIm->Image()->Name()] = aPIm;
}


void cSP_PointGlob::SetKilled()
{
    mPG->Disparu().SetVal(true);
    eEtatPointeImage aState = mPG->FromDico().ValWithDef(false) ?
                              eEPI_Refute                       :
                              eEPI_Disparu                      ;

    for
    (
       std::map<std::string,cSP_PointeImage *>::iterator it=mPointes.begin();
       it!=mPointes.end();
       it++
    )
    {
       it->second->Saisie()->Etat() = aState;
    }
    // Ce qui precede n'est pas suffisant car il se peut qu'il existe
    // des pointes non modifies car existant dans des images non chargees
    mAppli.GlobChangStatePointe(mPG->Name(),aState);
}


bool cSP_PointGlob::IsPtAutom() const
{
    return mPG->NumAuto().IsInit();
}

void  cSP_PointGlob::SuprDisp()
{
    for
    (
       std::map<std::string,cSP_PointeImage *>::iterator it=mPointes.begin();
       it!=mPointes.end();
       it++
    )
    {
         cSP_PointeImage * aPIm = it->second;
         eEtatPointeImage & aState = aPIm->Saisie()-> Etat();

         aState = eEPI_NonSaisi;
    }
}

bool  cSP_PointGlob::Has3DValue() const
{
   return     HasStrong3DValue()
          || mPG->P3D().IsInit() ;
}

bool  cSP_PointGlob::HasStrong3DValue() const
{
   return     mPG->FromDico().ValWithDef(false)
          ||  mPG->Mes3DExportable().ValWithDef(false);
}





Pt3dr cSP_PointGlob::Best3dEstim() const 
{
   if (mPG->Mes3DExportable().ValWithDef(false))
   {
      ELISE_ASSERT(mPG->P3D().IsInit(),"P3D :: cSP_PointGlob::Best3dEstim");
      return mPG->P3D().Val();
   }
   if (mPG->FromDico().ValWithDef(false))
   {
/*
      ELISE_ASSERT(mPG->Pt3DFromDico().IsInit(),"Pt3DFromDico :: cSP_PointGlob::Best3dEstim");
      return mPG->Pt3DFromDico().Val();
*/
     // Modif MPD pour compatibilite avec anciens fichiers deja crees avant masq3D
      if (mPG->Pt3DFromDico().IsInit())
      {
         return mPG->Pt3DFromDico().Val();
      }
   }
   if (mPG->P3D().IsInit())
   {
      return  mPG->P3D().Val();
   }

   ELISE_ASSERT(false,"cSP_PointGlob::Best3dEstim No Pt\n");
   return Pt3dr(0,0,0);
}

extern bool ERupnik_MM();

void cSP_PointGlob::ReCalculPoints()
{

    if (! IsPtAutom())
    {
        return;
    }

    if (! mAppli.HasOrientation() )
    {
       return;
    }

    Pt3dr aP0 = mPG->P3D().ValWithDef(Pt3dr(1234.67,1.56e69,-6.87e24));

    mPG->Mes3DExportable().SetVal(false);


    std::vector<cSP_PointeImage *> aVOK;
    for
    (
       std::map<std::string,cSP_PointeImage *>::iterator it=mPointes.begin();
       it!=mPointes.end();
       it++
    )
    {
        cSP_PointeImage * aPIm = it->second;
        eEtatPointeImage  aState = aPIm->Saisie()->Etat();
        cImage &          anIm = *(aPIm->Image());
        if (
                (anIm.Capt3d()!=0)
                && (
                    (aState==eEPI_Valide)
                    || (aState==eEPI_Douteux)
                    )
                )
        {
            aVOK.push_back(aPIm);
        }
    }

    if (aVOK.size() == 0)
    {
       return;
    }

    if (aVOK.size() == 1)
    {
        cSP_PointeImage & aPointeIm = *(aVOK[0]);
        cImage &          anIm = *(aPointeIm.Image());
        cBasicGeomCap3D *      aCap3d =  anIm.Capt3d();
        ELISE_ASSERT(aCap3d!=0,"Internal problem in cSP_PointGlob::ReCalculPoints");

        Pt2dr             aPIm = aCap3d->ImRef2Capteur(aPointeIm.Saisie()->PtIm());
       if (! aCap3d->CaptHasData(aPIm))
       {
            std::cout << "For Pts= " << aPIm << "\n";
            ELISE_ASSERT(aCap3d->CaptHasData(aPIm),"Internal pb, no data in sensor for required point");
        }


        Pt3dr aPt = aP0;
        // cElNuage3DMaille * aNuage = anIm.CaptNuage();
        ElCamera * aCamera = anIm.ElCaptCam();

        if (aCap3d->HasPreciseCapteur2Terrain())
        {
            aPt = aCap3d->PreciseCapteur2Terrain(aPIm);
            mPG->P3D().SetVal(aPt);
            mPG->Mes3DExportable().SetVal(true);
        }
        else if ( aCamera && aCamera->ProfIsDef())
        {
            double aProf = aCamera->GetProfondeur();
            double aInc = 1+ mAppli.Param().IntervPercProf().Val()/100.0;

            aPt = aCamera->ImEtProf2Terrain(aPIm,aProf);

            mPG->P3D().SetVal(aPt);
            // mPG->PS1().SetVal(aCamera->ImEtProf2Terrain(aPIm,aProf*aInc));
            // mPG->PS2().SetVal(aCamera->ImEtProf2Terrain(aPIm,aProf/aInc));

            int aNbKMoins = 20;
            int aNbKPlus = 20;
            std::vector<Pt3dr> aVPt;
            for (int aK= -aNbKMoins ; aK<= aNbKPlus ; aK++)
            {
                double aProfK = aProf * pow(aInc,aK);
                aVPt.push_back(aCamera->ImEtProf2Terrain(aPIm,aProfK));
            }
            mPG->VPS() = aVPt;
            mPG->PS1().SetVal(aVPt[aNbKMoins+1]);
            mPG->PS2().SetVal(aVPt[aNbKPlus-1]);
        }
        else if (aCap3d->HasRoughCapteur2Terrain())
        {

            //double aProf = aCap3d->ProfondeurDeChamps(aCap3d->PMoyOfCenter());
            double aProf = aCap3d->ProfondeurDeChamps(aCap3d->RoughCapteur2Terrain(aPIm));
            // std::cout << "PIMMM " << aPIm   << " " << aCap3d->RoughCapteur2Terrain(aPIm) << "\n";

            double aMul  = pow(1 + aCap3d->GetVeryRoughInterProf(),2) ;


            aPt = aCap3d->ImEtProf2Terrain(aPIm,aProf);

            mPG->P3D().SetVal(aPt);
            // mPG->PS1().SetVal(aCamera->ImEtProf2Terrain(aPIm,aProf*aInc));
            // mPG->PS2().SetVal(aCamera->ImEtProf2Terrain(aPIm,aProf/aInc));

            int aNbSeg = 20;
            std::vector<Pt3dr> aVPt;
            for (int aK= -aNbSeg ; aK<= aNbSeg ; aK++)
            {
                double aProfK = aProf * pow(aMul,aK/double(aNbSeg));
                aVPt.push_back(aCap3d->ImEtProf2Terrain(aPIm,aProfK));

                if ((MPD_MM() || ERupnik_MM()) && (aK==0))
                {
                    // std::cout << "Check-reproj " << aPIm - aCap3d->Ter2Capteur(aVPt.back()) 
                              // <<  " " << aPIm -aCap3d->ImRef2Capteur(aPIm) << "\n";
                }
            }
            mPG->VPS() = aVPt;
            mPG->PS1().SetVal(aVPt.back());
            mPG->PS2().SetVal(aVPt[0]);
        }





        if (euclid(aPt-aP0)< 1e-9) 
        {
            return;
        }
    }

    if (aVOK.size() > 1)
    {
        std::vector<ElSeg3D> aVSeg;
        std::vector<Pt3dr>   aVPts;

        for (int aK=0 ;aK<int(aVOK.size()) ; aK++)
        {
            cSP_PointeImage & aPointeIm = *(aVOK[aK]);
            cImage &          anIm = *(aPointeIm.Image());
            cBasicGeomCap3D *      aCap3d =  anIm.Capt3d();
            ELISE_ASSERT(aCap3d!=0,"Internal problem in cSP_PointGlob::ReCalculPoints");
            Pt2dr             aPIm = aCap3d->ImRef2Capteur(aPointeIm.Saisie()->PtIm());
            aVSeg.push_back(aCap3d->Capteur2RayTer(aPIm));
            if (aCap3d->HasPreciseCapteur2Terrain())
            {
                 double aPrec = 2.0;  // Arbitraire, par rapport a precision sur seg
                 Pt3dr aPtPrec(aPrec,aPrec,aPrec);
                 aVPts.push_back(aCap3d->PreciseCapteur2Terrain(aPIm));
                 aVPts.push_back(aPtPrec);
            }
        }

        Pt3dr aPt = ElSeg3D::L2InterFaisceaux
                    (
                         (const std::vector<double> *) 0,
                         aVSeg,
                         (bool *) 0,
                         (const cRapOnZ *) 0,
                         (cResOptInterFaisceaux *) 0,
                         &aVPts
                    );

        mPG->P3D().SetVal(aPt);
        mPG->PS1().SetNoInit();
        mPG->PS2().SetNoInit();
        mPG->VPS().clear();
        mPG->Mes3DExportable().SetVal(true);

        if (1)
        {
            std::cout << " ---------------- Pt=" << mPG->Name() << " -------------\n";
            for (int aK=0 ;aK<int(aVOK.size()) ; aK++)
            {
                cSP_PointeImage & aPointeIm = *(aVOK[aK]);
                cImage &          anIm = *(aPointeIm.Image());
                cBasicGeomCap3D *      aCap3d =  anIm.Capt3d();
                ELISE_ASSERT(aCap3d!=0,"Internal problem in cSP_PointGlob::ReCalculPoints");
                Pt2dr             aPIm = aCap3d->ImRef2Capteur(aPointeIm.Saisie()->PtIm());
                Pt2dr aProj = aCap3d->Ter2Capteur(aPt);
                std::cout << "DIST-Reproj= " << euclid(aPIm-aProj) << " for im=" << anIm.Name()<< "\n";
            }
        }

        if (euclid(aPt-aP0)< 1e-9)
        {
            return;
        }
    }
    mAppli.AddPGInAllImages(this);
    mAppli.RedrawAllWindows();
}



int cAppli_SaisiePts::GetCptMax() const
{
    int aCptMax=-1;
    for (int aKP=0 ; aKP<int(mPG.size()) ; aKP++)
    {
        const cPointGlob & aPG=*(mPG[aKP]->PG());
        if (aPG.NumAuto().IsInit())
        {
            aCptMax = ElMax(aCptMax,aPG.NumAuto().Val());
        }
    }
    return aCptMax;
}

void cSP_PointGlob::Rename(const std::string & aNewName)
{
     PG()->Name() = aNewName;
     for
     (
          std::map<std::string,cSP_PointeImage *>::iterator itM=mPointes.begin();
          itM!=mPointes.end();
          itM++
     )
     {
          itM->second->Saisie()->NamePt() = aNewName;
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
